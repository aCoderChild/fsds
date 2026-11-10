#!/usr/bin/env python3
"""Verify the local, hash-bound Kaggle smoke notebook before publishing."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
NOTEBOOK = HERE / "llama31_8b_runtime_smoke.ipynb"
METADATA = HERE / "kernel-metadata.json"
BUILD = json.loads((HERE / "notebook_manifest.json").read_text(encoding="utf-8"))
PUBLISH_DIR = HERE / "publish_v8"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) != 2:
        raise ValueError("Expected one scope cell and one executable runner cell")
    cell = "".join(notebook["cells"][1].get("source", []))
    ast.parse(cell)
    if sha(NOTEBOOK.read_bytes()) != BUILD.get("notebook_sha256"):
        raise ValueError("Notebook does not match its build manifest")
    if metadata.get("enable_gpu") is not False or metadata.get("enable_internet") is not False:
        raise ValueError("Published notebook must remain GPU-off and Internet-off by default")
    if metadata.get("dataset_sources") != []:
        raise ValueError("Runtime smoke must not attach research datasets")
    for flag in ("RUN_ACCESS_PREFLIGHT = False", "RUN_MASK_PREFLIGHT = False", "RUN_SMOKE = False"):
        if flag not in cell:
            raise ValueError(f"Notebook safe default missing: {flag}")

    expected_sources = {
        "smoke_source": "runner_sha256",
        "masking_source": "assistant_masking_sha256",
        "access_source": "access_preflight_sha256",
        "objective_source": "npo_objective_sha256",
        "mask_preflight_source": "mask_preflight_sha256",
    }
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(cell))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    for variable, manifest_key in expected_sources.items():
        source = assignments.get(variable)
        if source is None or sha(source.encode("utf-8")) != BUILD.get(manifest_key):
            raise ValueError(f"Embedded source does not match its hash: {variable}")

    runner = (SCRIPTS / "kaggle_8b_runtime_smoke.py").read_text(encoding="utf-8")
    if runner.index("encode_assistant_only(tokenizer, messages)") > runner.index("AutoModelForCausalLM.from_pretrained("):
        raise ValueError("The assistant mask must be validated before the large model checkpoint loads")

    published_notebook = PUBLISH_DIR / NOTEBOOK.name
    published_metadata = PUBLISH_DIR / METADATA.name
    publish_manifest = json.loads((PUBLISH_DIR / "publish_manifest.json").read_text(encoding="utf-8"))
    if published_notebook.read_bytes() != NOTEBOOK.read_bytes():
        raise ValueError("Publish notebook differs from the verified local notebook")
    if published_metadata.read_bytes() != METADATA.read_bytes():
        raise ValueError("Publish metadata differs from the verified local metadata")
    if sha(published_notebook.read_bytes()) != publish_manifest.get("notebook_sha256"):
        raise ValueError("Publish notebook hash mismatch")
    if sha(published_metadata.read_bytes()) != publish_manifest.get("metadata_sha256"):
        raise ValueError("Publish metadata hash mismatch")
    if (publish_manifest.get("run_access_preflight_default") is not False
            or publish_manifest.get("run_mask_preflight_default") is not False
            or publish_manifest.get("run_smoke_default") is not False
            or publish_manifest.get("research_data_loaded") is not False
            or publish_manifest.get("training_started") is not False):
        raise ValueError("Publish manifest does not preserve safe defaults")

    print(json.dumps({
        "status": "VERIFIED_PREPARED_NOTEBOOK_SAFE_TO_PUBLISH",
        "notebook_sha256": sha(NOTEBOOK.read_bytes()),
        "runner_sha256": BUILD["runner_sha256"],
        "mask_preflight_sha256": BUILD["mask_preflight_sha256"],
        "mask_validation_precedes_weights": True,
        "gpu_enabled_at_publish": False,
        "internet_enabled_at_publish": False,
        "research_data_attached": False,
        "run_flags_safe": True,
    }, indent=2))


if __name__ == "__main__":
    main()
