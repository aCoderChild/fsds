#!/usr/bin/env python3
"""Verify the hash-bound v6 preflight-only Kaggle publish artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISH_DIR = HERE / "access_preflight_publish_v6"
ARCHIVE_DIR = PUBLISH_DIR / "archive"
LOCAL = ARCHIVE_DIR / "source_notebook.ipynb"
PUBLISHED = PUBLISH_DIR / "llama31_8b_runtime_smoke.ipynb"
METADATA = PUBLISH_DIR / "kernel-metadata.json"
PUBLISH_MANIFEST = json.loads((PUBLISH_DIR / "publish_manifest.json").read_text(encoding="utf-8"))
BUILD_MANIFEST = json.loads((ARCHIVE_DIR / "build_manifest.json").read_text(encoding="utf-8"))
# Verify the immutable v6 artifact against the exact source embedded when it
# was published. The live runner evolves independently for future versions.
RUNNER = ARCHIVE_DIR / "kaggle_8b_runtime_smoke.py"
ACCESS_PREFLIGHT = ARCHIVE_DIR / "kaggle_8b_access_preflight.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    local = json.loads(LOCAL.read_text(encoding="utf-8"))
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    local_cells, published_cells = local.get("cells", []), published.get("cells", [])
    if len(local_cells) != 2 or len(published_cells) != 2:
        raise ValueError("Expected exactly two notebook cells")
    if [c.get("id") for c in local_cells] != ["scope", "smoke-runner"]:
        raise ValueError("Local notebook cell IDs are missing or unexpected")
    if [c.get("id") for c in published_cells] != [c.get("id") for c in local_cells]:
        raise ValueError("Published cell IDs differ from local notebook")
    runner_cell = "".join(published_cells[1].get("source", []))
    if sha(LOCAL) != BUILD_MANIFEST.get("notebook_sha256"):
        raise ValueError("Local notebook does not match the build manifest")
    if sha(LOCAL) != PUBLISH_MANIFEST.get("source_notebook_sha256"):
        raise ValueError("Local notebook does not match the v6 publish source hash")
    if sha(PUBLISHED) != PUBLISH_MANIFEST.get("published_notebook_sha256"):
        raise ValueError("Published notebook does not match the v6 publish hash")
    if sha(METADATA) != PUBLISH_MANIFEST.get("published_metadata_sha256"):
        raise ValueError("Kaggle metadata does not match the v6 publish hash")
    if sha(RUNNER) != BUILD_MANIFEST.get("runner_sha256") or BUILD_MANIFEST["runner_sha256"] not in runner_cell:
        raise ValueError("Embedded Kaggle runner does not match the hash-bound local runner")
    if (sha(ACCESS_PREFLIGHT) != BUILD_MANIFEST.get("access_preflight_sha256")
            or BUILD_MANIFEST["access_preflight_sha256"] not in runner_cell):
        raise ValueError("Embedded access-preflight runner does not match its hash-bound source")
    if ("RUN_ACCESS_PREFLIGHT = True" not in runner_cell or "RUN_SMOKE = False" not in runner_cell
            or "huggingface_hub==0.36.2" in runner_cell):
        raise ValueError("v6 does not select only the corrected access preflight")
    if (BUILD_MANIFEST.get("research_data_loaded") is not False or BUILD_MANIFEST.get("run_training") is not False
            or BUILD_MANIFEST.get("enable_gpu_at_publish") is not False
            or PUBLISH_MANIFEST.get("gpu_enabled") is not False
            or PUBLISH_MANIFEST.get("datasets_attached") is not False
            or metadata.get("enable_gpu") is not False
            or metadata.get("dataset_sources") != []):
        raise ValueError("Notebook build manifest does not preserve safe-only scope")
    report = {
        "status": "VERIFIED_V6_ACCESS_PREFLIGHT_PUBLISH_ARTIFACT",
        "kernel_id": PUBLISH_MANIFEST["kernel_id"],
        "local_notebook_sha256": sha(LOCAL),
        "published_notebook_sha256": sha(PUBLISHED),
        "published_metadata_sha256": sha(METADATA),
        "hash_bound_runners_match": True,
        "access_preflight_enabled": True,
        "runtime_smoke_enabled": False,
        "hub_downgrade_present": False,
        "internet_enabled": metadata.get("enable_internet"),
        "weights_downloaded": False,
        "gpu_runtime_smoke": False,
        "research_data_loaded": False,
        "training_started": False,
    }
    output = PUBLISH_DIR / "verification.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
