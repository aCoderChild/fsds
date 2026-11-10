#!/usr/bin/env python3
"""Build a CPU-only Kaggle version that runs the hash-bound HF access preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SMOKE_DIR = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = SMOKE_DIR / "llama31_8b_runtime_smoke.ipynb"
SOURCE_METADATA = SMOKE_DIR / "kernel-metadata.json"
OUT = SMOKE_DIR / "access_preflight_publish_v6"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing publish directory: {OUT}")

    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(code_cells) != 1:
        raise ValueError(f"Expected exactly one code cell, found {len(code_cells)}")
    cell_source = "".join(code_cells[0].get("source", []))
    if cell_source.count("RUN_ACCESS_PREFLIGHT = False") != 1:
        raise ValueError("Expected one safe-default access-preflight flag")
    if cell_source.count("RUN_SMOKE = False") != 1:
        raise ValueError("The runtime smoke must remain disabled")
    cell_source = cell_source.replace("RUN_ACCESS_PREFLIGHT = False", "RUN_ACCESS_PREFLIGHT = True", 1)
    code_cells[0]["source"] = cell_source.splitlines(keepends=True)

    kaggle_meta = notebook.setdefault("metadata", {}).setdefault("kaggle", {})
    kaggle_meta["isInternetEnabled"] = True
    kaggle_meta["accelerator"] = "None"
    notebook_bytes = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")

    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata["enable_internet"] = True
    metadata["enable_gpu"] = False
    metadata["dataset_sources"] = []
    metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / SOURCE_NOTEBOOK.name).write_bytes(notebook_bytes)
    (OUT / SOURCE_METADATA.name).write_bytes(metadata_bytes)
    manifest = {
        "status": "PREFLIGHT_ONLY_PUBLISH_READY",
        "kernel_id": metadata["id"],
        "source_notebook_sha256": sha(SOURCE_NOTEBOOK.read_bytes()),
        "published_notebook_sha256": sha(notebook_bytes),
        "published_metadata_sha256": sha(metadata_bytes),
        "run_access_preflight": True,
        "run_runtime_smoke": False,
        "gpu_enabled": False,
        "internet_enabled": True,
        "datasets_attached": False,
        "expected_scope": "retrieve Kaggle HF_TOKEN and fetch only pinned model config.json",
        "hub_dependency_policy": "use Kaggle image dependency; do not pin/downgrade huggingface_hub",
    }
    (OUT / "publish_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
