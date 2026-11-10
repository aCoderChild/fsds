#!/usr/bin/env python3
"""Run the preregistered IDK+AP development diagnostic and fixed stop rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def run_stage(name: str, command: list[str], root: Path, env: dict | None = None) -> tuple[int, Path]:
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"{name}.log"
    started = time.monotonic()
    print(json.dumps({"stage": name, "status": "STARTED", "command": command}), flush=True)
    with path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    print(json.dumps({"stage": name, "status": "PASS" if process.returncode == 0 else "ERROR",
                      "returncode": process.returncode, "elapsed_seconds": round(time.monotonic() - started, 2),
                      "log": str(path)}), flush=True)
    if process.returncode:
        tail = path.read_text(encoding="utf-8", errors="replace")[-5000:]
        print(tail, flush=True)
    return process.returncode, path


def zip_run(root: Path, zip_path: Path) -> dict:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(root.parent))
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
    if bad:
        raise RuntimeError(f"Artifact ZIP CRC failed at {bad}")
    return {"path": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": sha256(zip_path), "crc": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "idk_ap_lora_protocol.json")
    parser.add_argument("--idk", type=Path, default=HERE / "idontknow.jsonl")
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--reference-support-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-zip", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256(args.protocol)
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(args.upstream_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_commit = protocol["objective"]["source_commit"]
    if upstream_commit != expected_commit:
        raise RuntimeError(f"Official IDK+AP source checkout is not pinned: {upstream_commit}")
    upstream_losses = args.upstream_repo / protocol["objective"]["loss_source_path"]
    if sha256(upstream_losses) != protocol["objective"]["loss_source_sha256"]:
        raise RuntimeError("Pinned official objective source file hash mismatch")
    snapshot_sources = [
        args.protocol, args.idk, *HERE.glob("*.py"), HERE / "requirements.txt",
        ROOT / "tofu_official_eval.py", ROOT / "evaluate_npo_profile_cohort.py", upstream_losses,
    ]
    for source in snapshot_sources:
        if source.is_file():
            name = "official_losses_pinned.py" if source == upstream_losses else source.name
            destination = root / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    receipt = {
        "receipt_schema_version": 2,
        "status": "RUNNING",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "upstream_source_commit": upstream_commit,
        "upstream_losses_sha256": sha256(upstream_losses),
        "kaggle_gpu_names": [],
        "gpu_names_by_cohort": {},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "development_training_started": False,
        "final_training_started": False,
        "development_gate_status": None,
    }
    receipt_path = root / "run_receipt.json"
    write_json(root / "idk_ap_lora_protocol.json", protocol)
    write_json(root / "source_provenance.json", {
        "protocol_sha256": protocol_hash,
        "idk_response_file_sha256": sha256(args.idk),
        "upstream_commit": upstream_commit,
        "upstream_loss_sha256": sha256(upstream_losses),
        "source_snapshot_sha256": {
            path.name: sha256(path) for path in sorted((root / "source_snapshot").iterdir()) if path.is_file()
        },
    })
    write_json(receipt_path, receipt)
    try:
        audit_path = root / "idk_ap_data_audit.json"
        command = [sys.executable, str(HERE / "validate_protocol.py"), "--protocol", str(args.protocol), "--idk", str(args.idk), "--output", str(audit_path)]
        if run_stage("pinned-data-profile-audit", command, root)[0]:
            raise RuntimeError("Pinned data/profile audit failed")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS_IDK_AP_PROTOCOL_DATA_AUDIT":
            raise RuntimeError("Pinned data/profile audit receipt failed")
        objective_path = root / "idk_ap_objective_equivalence.json"
        command = [sys.executable, str(HERE / "verify_objective.py"), "--upstream-losses", str(upstream_losses),
                   "--protocol", str(args.protocol), "--output", str(objective_path)]
        if run_stage("official-objective-value-gradient-equivalence", command, root)[0]:
            raise RuntimeError("Official objective equivalence check failed")
        objective = json.loads(objective_path.read_text(encoding="utf-8"))
        if objective.get("status") != "PASS":
            raise RuntimeError("Official objective equivalence receipt failed")

        from evaluate import verify_reference_support
        reference_path = verify_reference_support(args.reference_support_dir, protocol)
        reference_copy = root / "reference_support" / "full_reference_profile_eval.json"
        reference_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference_path, reference_copy)
        shutil.copy2(args.reference_support_dir / "reference_support_manifest.json", root / "reference_support" / "reference_support_manifest.json")

        data_audit_path = audit_path
        dev_dir = root / "dev"
        receipt["training_started"] = True
        receipt["development_training_started"] = True
        write_json(receipt_path, receipt)
        command = [sys.executable, str(HERE / "train.py"), "--protocol", str(args.protocol),
                   "--objective-check", str(objective_path), "--data-audit", str(data_audit_path),
                   "--idk", str(args.idk), "--cohort", "dev", "--seed", str(protocol["training"]["seed"]),
                   "--output-dir", str(dev_dir / "training")]
        if run_stage("dev-idk-ap-training", command, root)[0]:
            raise RuntimeError("Development IDK+AP training failed")
        train_receipt_path = dev_dir / "training" / "training_receipt.json"
        train_receipt = json.loads(train_receipt_path.read_text(encoding="utf-8"))
        if train_receipt.get("status") != "TRAINING_COMPLETE" or train_receipt.get("optimizer_steps") != protocol["training"]["expected_optimizer_steps"]:
            raise RuntimeError("Development training did not complete the locked schedule")
        dev_gpu_names = train_receipt.get("gpus")
        if dev_gpu_names != ["Tesla T4", "Tesla T4"]:
            raise RuntimeError(f"Development training receipt has unexpected GPUs: {dev_gpu_names}")
        receipt["gpu_names_by_cohort"]["dev"] = dev_gpu_names
        receipt["kaggle_gpu_names"] = dev_gpu_names
        write_json(receipt_path, receipt)
        command = [sys.executable, str(HERE / "evaluate.py"), "--protocol", str(args.protocol),
                   "--cohort", "dev", "--adapter-dir", str(dev_dir / "training" / "adapter"),
                   "--training-receipt", str(train_receipt_path), "--reference-support-dir", str(args.reference_support_dir),
                   "--idk", str(args.idk), "--output-dir", str(dev_dir / "evaluation")]
        if run_stage("dev-idk-ap-evaluation-and-gates", command, root)[0]:
            raise RuntimeError("Development evaluation or gate calculation failed")
        dev_comparison_path = dev_dir / "evaluation" / "dev_idk_ap_comparison.json"
        dev_comparison = json.loads(dev_comparison_path.read_text(encoding="utf-8"))
        receipt["development_gate_status"] = dev_comparison["status"]
        receipt["development_comparison"] = str(dev_comparison_path.relative_to(root))
        write_json(receipt_path, receipt)
        if not dev_comparison.get("all_gates_pass"):
            receipt.update({
                "status": "DEV_GATES_FAILED_STOP",
                "final_training_started": False,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            write_json(receipt_path, receipt)
            return

        final_dir = root / "final"
        receipt["final_training_started"] = True
        write_json(receipt_path, receipt)
        command = [sys.executable, str(HERE / "train.py"), "--protocol", str(args.protocol),
                   "--objective-check", str(objective_path), "--data-audit", str(data_audit_path),
                   "--idk", str(args.idk), "--cohort", "final", "--seed", str(protocol["training"]["seed"]),
                   "--output-dir", str(final_dir / "training")]
        if run_stage("final-idk-ap-training", command, root)[0]:
            raise RuntimeError("Final IDK+AP training failed")
        final_receipt_path = final_dir / "training" / "training_receipt.json"
        final_receipt = json.loads(final_receipt_path.read_text(encoding="utf-8"))
        if final_receipt.get("status") != "TRAINING_COMPLETE" or final_receipt.get("optimizer_steps") != protocol["training"]["expected_optimizer_steps"]:
            raise RuntimeError("Final training did not complete the locked schedule")
        final_gpu_names = final_receipt.get("gpus")
        if final_gpu_names != ["Tesla T4", "Tesla T4"] or final_gpu_names != dev_gpu_names:
            raise RuntimeError(f"Development/final GPU receipts do not match: dev={dev_gpu_names}, final={final_gpu_names}")
        receipt["gpu_names_by_cohort"]["final"] = final_gpu_names
        write_json(receipt_path, receipt)
        command = [sys.executable, str(HERE / "evaluate.py"), "--protocol", str(args.protocol),
                   "--cohort", "final", "--adapter-dir", str(final_dir / "training" / "adapter"),
                   "--training-receipt", str(final_receipt_path), "--reference-support-dir", str(args.reference_support_dir),
                   "--idk", str(args.idk), "--output-dir", str(final_dir / "evaluation")]
        if run_stage("final-idk-ap-evaluation-and-gates", command, root)[0]:
            raise RuntimeError("Final evaluation or gate calculation failed")
        final_comparison_path = final_dir / "evaluation" / "final_idk_ap_comparison.json"
        final_comparison = json.loads(final_comparison_path.read_text(encoding="utf-8"))
        receipt.update({
            "status": final_comparison["status"],
            "final_comparison": str(final_comparison_path.relative_to(root)),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(receipt_path, receipt)
    except Exception as error:
        receipt.update({
            "status": "ERROR",
            "error": f"{type(error).__name__}: {error}",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(receipt_path, receipt)
        raise
    finally:
        zipped = zip_run(root, args.artifact_zip)
        archive_manifest = args.artifact_zip.with_suffix(args.artifact_zip.suffix + ".manifest.json")
        write_json(archive_manifest, {"artifact_archive": zipped, "run_status": receipt.get("status"),
                                     "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash})
        cache_root = Path(os.environ.get("TOFU_MODEL_CACHE_ROOT", ""))
        if str(cache_root) not in {"", "."}:
            shutil.rmtree(cache_root, ignore_errors=True)
        print(json.dumps({"run_receipt": receipt, "artifact_archive_manifest": str(archive_manifest),
                          "artifact_archive": zipped}, indent=2), flush=True)


if __name__ == "__main__":
    main()
