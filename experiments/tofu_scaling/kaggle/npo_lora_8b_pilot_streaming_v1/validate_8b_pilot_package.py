#!/usr/bin/env python3
"""Offline integrity checks for the self-contained Kaggle 8B pilot package."""

from __future__ import annotations

import hashlib
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    errors: list[str] = []
    protocol_path = ROOT / "npo_lora_8b_pilot_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        errors.append("protocol is not marked preregistered")
    if protocol.get("protocol_id") != "tofu_8b_npo_lora_pilot_v1" or protocol.get("model_family") != "8b":
        errors.append("protocol identity or model family is not the locked 8B pilot")
    if protocol.get("seed") != 42 or protocol.get("adaptation", {}).get("epochs") != 10:
        errors.append("seed/epoch settings differ from the locked pilot")
    if protocol.get("adaptation", {}).get("effective_paired_batch_size") != 8:
        errors.append("effective paired batch is not 8")

    baseline_dir = HERE / "baseline_input"
    manifest_path = baseline_dir / "npo_8b_baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS" or manifest.get("source_independent_verification_status") != "PASS":
        errors.append("baseline manifest lacks verified source status")
    for relative, expected in manifest.get("files", {}).items():
        path = baseline_dir / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"baseline file missing or hash mismatch: {relative}")

    notebook_path = HERE / "tofu_8b_npo_lora_pilot.ipynb"
    notebook = json.loads(notebook_path.read_text())
    mode_cell = "".join(notebook["cells"][1]["source"])
    expected_flag = f"RUN_FULL_TRAINING = {'False' if args.mode == 'smoke' else 'True'}"
    if expected_flag not in mode_cell:
        errors.append(f"notebook does not match expected {args.mode} mode")
    if args.mode == "smoke" and HERE.name != "npo_lora_8b_pilot":
        errors.append("smoke mode must use the canonical smoke package")
    if args.mode == "full" and HERE.name != "npo_lora_8b_pilot_full":
        errors.append("full mode must use the separate full-pilot package")
    protocol_cell = "".join(notebook["cells"][12]["source"])
    protocol_prefix = "%%writefile /kaggle/working/npo_lora_8b_pilot_protocol.json\n"
    if not protocol_cell.startswith(protocol_prefix):
        errors.append("notebook protocol cell has an unexpected format")
    else:
        try:
            embedded_protocol = json.loads(protocol_cell[len(protocol_prefix):])
            if embedded_protocol != protocol:
                errors.append("embedded protocol differs from the preregistered local protocol")
        except json.JSONDecodeError as exc:
            errors.append(f"embedded protocol is invalid JSON: {exc}")
    for index, filename in {
        5: "tofu_official_eval.py",
        6: "verify_tofu_official_eval.py",
        7: "npo_objective.py",
        8: "verify_npo_objective.py",
        9: "npo_lora_train.py",
        10: "evaluate_npo_adapter.py",
        11: "verify_npo_run.py",
    }.items():
        source = "".join(notebook["cells"][index]["source"])
        prefix = f"%%writefile /kaggle/working/{filename}\n"
        if not source.startswith(prefix):
            errors.append(f"notebook cell {index} does not write {filename}")
            continue
        local = ROOT / filename
        embedded = source[len(prefix):]
        if hashlib.sha256(embedded.encode()).hexdigest() != sha256(local):
            errors.append(f"embedded source differs from local {filename}")
    metadata = json.loads((HERE / "kernel-metadata.json").read_text())
    if metadata.get("id") != "bunnoob2005/tofu-8b-npo-lora-pilot-2026-11-08":
        errors.append("Kaggle kernel metadata slug mismatch")
    if metadata.get("dataset_sources") != ["bunnoob2005/tofu-8b-npo-baseline-inputs-2026-11-08"]:
        errors.append("Kaggle dataset source does not point to the locked baseline inputs")
    source_manifest_path = HERE / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("notebook_sha256") != sha256(notebook_path):
        errors.append("source manifest notebook hash does not match the package notebook")
    if source_manifest.get("protocol_sha256") != sha256(protocol_path):
        errors.append("source manifest protocol hash does not match the locked protocol")
    if source_manifest.get("notebook_mode") != args.mode or source_manifest.get("full_training_enabled") != (args.mode == "full"):
        errors.append("source manifest mode does not match the requested package mode")

    with tempfile.TemporaryDirectory(prefix="tofu8b_baseline_verify_") as temp:
        temp_root = Path(temp)
        eval_relpaths = {
            "llama_3_1_8b_full_official_eval.json": "llama_3_1_8b_full_official_eval.json",
            "llama_3_1_8b_retain90_official_eval.json": "llama_3_1_8b_retain90_official_eval.json",
            "tofu_readiness_audit.json": "audit/tofu_readiness_audit.json",
            "official_eval_status.json": "official_eval_status.json",
            "source_verification_receipt.json": "source_verification/official_eval_verification.json",
        }
        for relative in manifest["files"]:
            source = baseline_dir / relative
            target = temp_root / eval_relpaths[relative]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        audit = temp_root / "audit" / "tofu_readiness_audit.json"
        cmd = [
            sys.executable,
            str(ROOT / "verify_tofu_official_eval.py"),
            str(temp_root),
            "--audit",
            str(audit),
            "--allow-partial",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            errors.append("pinned independent TOFU baseline verifier failed: " + result.stdout[-2000:] + result.stderr[-1000:])
        else:
            receipt = json.loads((temp_root / "official_eval_verification.json").read_text())
            if receipt.get("overall_status") != "PARTIAL":
                errors.append(f"expected a PARTIAL receipt for the two 8B baselines, got {receipt.get('overall_status')}")
            if receipt.get("missing_models") != ["llama_3_2_3b_full", "llama_3_2_3b_retain90"]:
                errors.append(f"unexpected missing baseline list: {receipt.get('missing_models')}")
            for key in ("llama_3_1_8b_full", "llama_3_1_8b_retain90"):
                if receipt.get("models", {}).get(key, {}).get("status") != "PASS":
                    errors.append(f"independent baseline verification failed for {key}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "protocol_sha256": sha256(protocol_path),
        "notebook_sha256": sha256(notebook_path),
        "baseline_manifest_sha256": sha256(manifest_path),
        "errors": errors,
    }
    report_path = ROOT / "diagnostics" / "npo_lora_8b_pilot_2026-11-08" / f"package_verification_{args.mode}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
