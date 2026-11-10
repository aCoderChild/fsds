#!/usr/bin/env python3
"""Run the locked TOFU NPO-LoRA alpha-selection and held-out profile follow-up."""

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


def run_stage(name: str, command: list[str], root: Path) -> int:
    log = root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===", flush=True)
    print("COMMAND:", " ".join(command), flush=True)
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip("\n"), flush=True)
            sink.write(line)
            sink.flush()
        return_code = process.wait()
    print(f"STAGE {name} exit={return_code}; elapsed_seconds={time.monotonic() - start:.1f}", flush=True)
    return return_code


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def archive(root: Path, destination: Path) -> dict:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != destination:
                zf.write(path, path.relative_to(root.parent))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/kaggle/working/tofu_npo_profile_recovery"))
    parser.add_argument("--protocol", type=Path, default=HERE / "npo_lora_recovery_protocol.json")
    parser.add_argument("--upstream-src", type=Path, default=Path("/kaggle/working/open-unlearning/src"))
    parser.add_argument("--objective-check", type=Path, default=Path("/kaggle/working/npo_recovery_objective_check.json"))
    parser.add_argument("--artifact-zip", type=Path, default=Path("/kaggle/working/tofu_npo_profile_recovery_artifacts.zip"))
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    for path in (
        HERE / "npo_lora_recovery_protocol.json",
        HERE / "validate_npo_recovery_protocol.py",
        HERE / "npo_lora_train.py",
        HERE / "npo_objective.py",
        HERE / "verify_npo_objective.py",
        HERE / "tofu_official_eval.py",
        HERE / "evaluate_npo_profile_cohort.py",
        HERE / "requirements-npo-pilot.txt",
    ):
        shutil.copy2(path, root / path.name)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        raise RuntimeError("Recovery protocol is not locked before GPU training")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "") and not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    import torch
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(gpu_names) != 2 or gpu_names.count("Tesla T4") != 2:
        raise RuntimeError(f"Recovery protocol requires exactly two Tesla T4 GPUs, found {gpu_names}")

    receipt = {
        "status": "RUNNING",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_names": gpu_names,
        "training_started": False,
        "candidate_attempts": [],
        "selected_alpha_retain": None,
        "final_training_started": False,
    }
    receipt_path = root / "recovery_run_receipt.json"
    write_json(receipt_path, receipt)
    try:
        audit_path = root / "profile_cohort_data_audit.json"
        command = [sys.executable, str(HERE / "validate_npo_recovery_protocol.py"), "--protocol", str(args.protocol), "--output", str(audit_path)]
        if run_stage("data-and-profile-audit", command, root):
            raise RuntimeError("Pinned source data/profile cohort audit failed")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS_PROFILE_COHORT_DATA_AUDIT":
            raise RuntimeError("Profile cohort audit did not pass")

        command = [sys.executable, str(HERE / "verify_npo_objective.py"), "--upstream-src", str(args.upstream_src), "--protocol", str(args.protocol), "--output", str(args.objective_check)]
        if run_stage("objective-equivalence", command, root):
            raise RuntimeError("Pinned objective loss/gradient equivalence failed")
        objective = json.loads(args.objective_check.read_text(encoding="utf-8"))
        if objective.get("status") != "PASS":
            raise RuntimeError("Objective receipt did not pass")
        shutil.copy2(args.objective_check, root / "objective_equivalence.json")

        reference_dir = root / "reference_eval"
        command = [sys.executable, str(HERE / "evaluate_npo_profile_cohort.py"), "--protocol", str(args.protocol), "--mode", "reference", "--output-dir", str(reference_dir)]
        if run_stage("full-checkpoint-reference-evaluation", command, root):
            raise RuntimeError("Full-checkpoint profile reference evaluation failed")
        reference_path = reference_dir / "full_reference_profile_eval.json"
        if not reference_path.is_file():
            raise RuntimeError("Reference evaluation receipt is missing")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference.get("status") != "PASS_PROFILE_COHORT_EVAL":
            raise RuntimeError("Full-checkpoint profile reference did not pass")

        smoke_dir = root / "smoke_alpha_5"
        command = [
            sys.executable, str(HERE / "npo_lora_train.py"), "--model", "3b", "--seed", "42",
            "--protocol", str(args.protocol), "--objective-check", str(args.objective_check),
            "--cohort", "dev", "--alpha-retain", "5", "--output-dir", str(smoke_dir), "--smoke",
        ]
        if run_stage("one-step-dev-cohort-smoke", command, root):
            raise RuntimeError("Real-data profile-cohort smoke failed")
        smoke = json.loads((smoke_dir / "training_receipt.json").read_text(encoding="utf-8"))
        if smoke.get("status") != "SMOKE_PASS" or smoke.get("global_steps") != 1 or smoke.get("smoke_max_abs_lora_parameter_update", 0) <= 0:
            raise RuntimeError("Real-data smoke receipt failed update checks")
        if smoke.get("runtime", {}).get("cpu_or_disk_offload") is not False:
            raise RuntimeError("Real-data smoke used CPU or disk offload")
        receipt["smoke_status"] = "PASS_BOUNDED_REAL_DATA_SMOKE"
        receipt["training_started"] = False
        write_json(receipt_path, receipt)

        selected_alpha = None
        for alpha in protocol["selection"]["candidate_retain_nll_weights_ascending"]:
            alpha_tag = f"{alpha:g}"
            train_dir = root / f"dev_alpha_{alpha_tag}"
            train_cmd = [
                sys.executable, str(HERE / "npo_lora_train.py"), "--model", "3b", "--seed", "42",
                "--protocol", str(args.protocol), "--objective-check", str(args.objective_check),
                "--cohort", "dev", "--alpha-retain", alpha_tag, "--output-dir", str(train_dir),
            ]
            receipt["training_started"] = True
            receipt["candidate_alpha_in_progress"] = alpha
            write_json(receipt_path, receipt)
            if run_stage(f"dev-training-alpha-{alpha_tag}", train_cmd, root):
                raise RuntimeError(f"Development NPO-LoRA training failed for alpha={alpha_tag}")
            train_receipt = json.loads((train_dir / "training_receipt.json").read_text(encoding="utf-8"))
            if train_receipt.get("status") != "TRAINING_COMPLETE" or train_receipt.get("global_steps") != protocol["training"]["expected_steps_per_cohort"]:
                raise RuntimeError(f"Development training receipt failed for alpha={alpha_tag}")
            eval_dir = root / "dev_eval"
            eval_cmd = [
                sys.executable, str(HERE / "evaluate_npo_profile_cohort.py"), "--protocol", str(args.protocol),
                "--mode", "dev", "--alpha-retain", alpha_tag, "--adapter-dir", str(train_dir / "adapter"), "--output-dir", str(eval_dir),
            ]
            if run_stage(f"dev-evaluation-alpha-{alpha_tag}", eval_cmd, root):
                raise RuntimeError(f"Development evaluation failed for alpha={alpha_tag}")
            comparison_path = eval_dir / f"dev_alpha_{alpha_tag}_comparison.json"
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            attempt = {"alpha_retain": alpha, "training_dir": str(train_dir), "comparison": str(comparison_path), "gate_status": comparison["status"]}
            receipt["candidate_attempts"].append(attempt)
            write_json(receipt_path, receipt)
            if comparison.get("all_gates_pass") is True:
                selected_alpha = float(alpha)
                receipt["selected_alpha_retain"] = selected_alpha
                receipt["selection_rule"] = protocol["selection"]["rule"]
                write_json(receipt_path, receipt)
                break

        if selected_alpha is None:
            receipt["status"] = "NO_DEV_CANDIDATE_PASSES"
            receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            receipt["final_training_started"] = False
            write_json(receipt_path, receipt)
            return

        final_train_dir = root / f"final_alpha_{selected_alpha:g}"
        final_cmd = [
            sys.executable, str(HERE / "npo_lora_train.py"), "--model", "3b", "--seed", "42",
            "--protocol", str(args.protocol), "--objective-check", str(args.objective_check),
            "--cohort", "final", "--alpha-retain", f"{selected_alpha:g}", "--output-dir", str(final_train_dir),
        ]
        receipt["final_training_started"] = True
        write_json(receipt_path, receipt)
        if run_stage(f"final-training-alpha-{selected_alpha:g}", final_cmd, root):
            raise RuntimeError("Final cohort training failed")
        final_train_receipt = json.loads((final_train_dir / "training_receipt.json").read_text(encoding="utf-8"))
        if final_train_receipt.get("status") != "TRAINING_COMPLETE" or final_train_receipt.get("global_steps") != protocol["training"]["expected_steps_per_cohort"]:
            raise RuntimeError("Final training receipt failed")
        final_eval_dir = root / "final_eval"
        final_eval_cmd = [
            sys.executable, str(HERE / "evaluate_npo_profile_cohort.py"), "--protocol", str(args.protocol),
            "--mode", "final", "--alpha-retain", f"{selected_alpha:g}", "--adapter-dir", str(final_train_dir / "adapter"), "--output-dir", str(final_eval_dir),
        ]
        if run_stage("final-cohort-evaluation", final_eval_cmd, root):
            raise RuntimeError("Final cohort evaluation failed")
        final_comparison_path = final_eval_dir / f"final_alpha_{selected_alpha:g}_comparison.json"
        final_comparison = json.loads(final_comparison_path.read_text(encoding="utf-8"))
        receipt["final_comparison"] = str(final_comparison_path)
        receipt["status"] = final_comparison["status"]
        receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
    except Exception as error:
        receipt["status"] = "ERROR"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
        raise
    finally:
        if root.exists():
            artifact = archive(root, args.artifact_zip)
            receipt["artifact_archive"] = artifact
            write_json(receipt_path, receipt)
            print(json.dumps({"run_receipt": receipt, "artifact_archive": artifact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
