#!/usr/bin/env python3
"""Verify a Kaggle access-probe run and distinguish secret binding from HF access."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = json.loads((HERE / "build_manifest.json").read_text(encoding="utf-8"))
LOCAL_SOURCE = (HERE / "probe.py").read_text(encoding="utf-8")
REMOTE_NOTEBOOK = HERE / "remote_output/access_probe_remote.ipynb"
RECEIPT = Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else HERE / "remote_output/r1_8b_access_probe.json"
LOG = Path(__import__("sys").argv[2]) if len(__import__("sys").argv) > 2 else HERE / "remote_output/synthworlds-llama31-8b-access-probe-2026-11-04.log"
REPORT = HERE / "remote_output/verification.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_embedded_source(notebook: dict) -> str:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "source" for t in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    return value
    raise ValueError("Remote notebook did not contain the embedded probe source")


def main() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    log = LOG.read_text(encoding="utf-8")
    remote_notebook = json.loads(REMOTE_NOTEBOOK.read_text(encoding="utf-8"))
    embedded_source = extract_embedded_source(remote_notebook)
    if sha(LOCAL_SOURCE.encode("utf-8")) != BUILD["probe_sha256"]:
        raise ValueError("Local probe source differs from build manifest")
    if embedded_source != LOCAL_SOURCE or sha(embedded_source.encode("utf-8")) != BUILD["probe_sha256"]:
        raise ValueError("Kaggle-pulled notebook does not embed the exact local probe")
    if receipt.get("model_id") != BUILD["model_id"] or receipt.get("expected_revision") != BUILD["expected_revision"]:
        raise ValueError("Receipt model identity/revision mismatch")
    if (receipt.get("gpu_required") is not False or receipt.get("weights_downloaded") is not False
            or receipt.get("research_data_loaded") is not False or receipt.get("training_started") is not False
            or receipt.get("token_value_logged") is not False):
        raise ValueError("Probe scope or secret-handling metadata is unsafe/misreported")
    if "HTTP Error 400: Bad Request" not in log or "kaggle_secrets.py" not in log or "get_secret" not in log:
        raise ValueError("Run log does not establish the Kaggle secret-service failure point")
    if receipt.get("status") != "FAIL_ACCESS_ONLY" or receipt.get("error_type") != "ConnectionError":
        raise ValueError("Expected this archived run to fail at secret retrieval")
    if "config_sha256" in receipt or "resolved_revision" in receipt:
        raise ValueError("Receipt claims model file access despite secret retrieval failure")
    report = {
        "status": "VERIFIED_BLOCKED_AT_KAGGLE_SECRET_RETRIEVAL",
        "kernel_id": "bunnoob2005/synthworlds-llama31-8b-access-probe-2026-11-04",
        "probe_sha256": sha(LOCAL_SOURCE.encode("utf-8")),
        "remote_embedded_probe_sha256": sha(embedded_source.encode("utf-8")),
        "receipt_sha256": sha(RECEIPT.read_bytes()),
        "log_sha256": sha(LOG.read_bytes()),
        "model_id": receipt["model_id"],
        "expected_revision": receipt["expected_revision"],
        "observed_failure": "Kaggle secret retrieval raised ConnectionError; run log shows HTTP 400 at kaggle_secrets.get_secret before any Hugging Face config request.",
        "model_access_verified": False,
        "weights_downloaded": False,
        "gpu_used": False,
        "training_started": False,
        "research_data_loaded": False,
        "token_value_logged": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
