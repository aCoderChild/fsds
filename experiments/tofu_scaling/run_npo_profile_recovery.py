#!/usr/bin/env python3
"""Run the locked TOFU NPO-LoRA alpha-selection and held-out profile follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def verify_reference_support(support_dir: Path, root: Path, protocol: dict) -> Path:
    """Copy a previously verified frozen-checkpoint reference after strict provenance checks."""
    locked = protocol.get("reference_support")
    if not isinstance(locked, dict):
        raise RuntimeError("Protocol does not define a locked reference-support artifact")
    manifest_path = support_dir / "reference_support_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = {
        "source_archive_sha256": "source_archive_sha256",
        "source_protocol_sha256": "source_protocol_sha256",
        "source_independent_verification_status": "source_independent_verification_status",
        "source_independent_verification_sha256": "source_independent_verification_sha256",
    }
    for manifest_key, protocol_key in fields.items():
        if manifest.get(manifest_key) != locked.get(protocol_key):
            raise RuntimeError(f"Reference support provenance mismatch: {manifest_key}")
    if manifest.get("status") != "PASS_SOURCE_ARCHIVE_AND_PRIOR_VERIFICATION":
        raise RuntimeError("Reference support does not record a clean prior independent verification")
    copied = root / "reference_eval"
    copied.mkdir(parents=True, exist_ok=True)
    expected_files = locked.get("files", {})
    if not expected_files:
        raise RuntimeError("Protocol does not pin any reference-support files")
    for relative, expected_hash in expected_files.items():
        manifest_record = manifest.get("files", {}).get(relative, {})
        dataset_path = Path(manifest_record.get("dataset_path", relative))
        if dataset_path.is_absolute() or ".." in dataset_path.parts:
            raise RuntimeError(f"Unsafe reference-support dataset path: {dataset_path}")
        source = support_dir / dataset_path
        if not source.is_file():
            raise FileNotFoundError(f"Reference support file is missing: {source}")
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != expected_hash or manifest_record.get("sha256") != expected_hash:
            raise RuntimeError(f"Reference support file hash mismatch: {relative}")
        if manifest_record.get("size_bytes") != source.stat().st_size:
            raise RuntimeError(f"Reference support file size mismatch: {relative}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    prior_verification_path = locked.get("prior_verification_path")
    if not prior_verification_path:
        raise RuntimeError("Protocol does not pin the prior verification receipt path")
    prior_verification = json.loads((root / prior_verification_path).read_text(encoding="utf-8"))
    if (
        prior_verification.get("status") != locked.get("source_independent_verification_status")
        or prior_verification.get("problems") != []
        or prior_verification.get("protocol_sha256") != locked.get("source_protocol_sha256")
    ):
        raise RuntimeError("Prior alpha=20 independent verification receipt is not clean or does not match the pinned protocol")

    reference_path = copied / "full_reference_profile_eval.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    model = protocol["model"]
    data = protocol["data"]
    evaluation = reference.get("evaluation", {})
    runtime = reference.get("runtime", {})
    if reference.get("status") != "PASS_PROFILE_COHORT_EVAL":
        raise RuntimeError("Cached full-checkpoint reference evaluation did not pass")
    if (reference.get("model", {}).get("repo_id"), reference.get("model", {}).get("revision")) != (model["repo_id"], model["revision"]):
        raise RuntimeError("Cached reference evaluation model/revision differs from the protocol")
    if evaluation.get("dataset_revision") != data["revision"]:
        raise RuntimeError("Cached reference evaluation dataset revision differs from the protocol")
    if any(evaluation.get("dataset_sha256_by_file", {}).get(name) != digest for name, digest in data["sha256"].items()):
        raise RuntimeError("Cached reference evaluation source data hashes differ from the protocol")
    if evaluation.get("profile_cohort_mode") is not True or evaluation.get("input_truncation") is not False:
        raise RuntimeError("Cached reference evaluation has invalid cohort/truncation metadata")
    if runtime.get("cpu_offload_used") is not False:
        raise RuntimeError("Cached reference evaluation used CPU or disk offload")
    generation = evaluation.get("generation", {})
    if generation.get("do_sample") is not False or generation.get("max_new_tokens") != protocol["evaluation"]["generations_max_new_tokens"]:
        raise RuntimeError("Cached reference evaluation generation settings differ from the protocol")
    expected = {
        "forget10": (400, set(range(20))),
        "retain90": (800, set(range(60, 100))),
        "holdout10": (400, set(range(20))),
    }
    profile_size = int(data["profile_block_size"])
    for split, (count, expected_blocks) in expected.items():
        rows = reference.get("per_example", {}).get(split, [])
        if len(rows) != count or {int(row["profile_block"]) for row in rows} != expected_blocks:
            raise RuntimeError(f"Cached reference evaluation has wrong {split} row/profile coverage")
        indices = {int(row["source_row_index"]) for row in rows}
        expected_indices = {block * profile_size + offset for block in expected_blocks for offset in range(profile_size)}
        if indices != expected_indices:
            raise RuntimeError(f"Cached reference evaluation has wrong {split} source-row assignment")
        for row in rows:
            for key in ("answer_probability", "answer_token_loss", "rougeL_recall"):
                if key not in row or not math.isfinite(float(row[key])):
                    raise RuntimeError(f"Cached reference evaluation has non-finite/missing {split}.{key}")
    return reference_path


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
    parser.add_argument("--reference-support-dir", type=Path)
    parser.add_argument("--artifact-zip", type=Path, default=Path("/kaggle/working/tofu_npo_profile_recovery_artifacts.zip"))
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    for path in (
        HERE / "validate_npo_recovery_protocol.py",
        HERE / "npo_lora_train.py",
        HERE / "npo_objective.py",
        HERE / "verify_npo_objective.py",
        HERE / "tofu_official_eval.py",
        HERE / "evaluate_npo_profile_cohort.py",
        HERE / "verify_npo_profile_recovery_run.py",
        HERE / "requirements-npo-pilot.txt",
    ):
        shutil.copy2(path, root / path.name)
    shutil.copy2(args.protocol, root / "npo_lora_recovery_protocol.json")
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
        "reference_evaluation_reused": bool(args.reference_support_dir),
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
        if args.reference_support_dir:
            reference_path = verify_reference_support(args.reference_support_dir, root, protocol)
            (root / "reference_support_manifest.json").write_bytes(
                (args.reference_support_dir / "reference_support_manifest.json").read_bytes()
            )
            print(json.dumps({"reference_evaluation": "REUSED_AND_HASH_VERIFIED", "path": str(reference_path)}), flush=True)
        else:
            command = [sys.executable, str(HERE / "evaluate_npo_profile_cohort.py"), "--protocol", str(args.protocol), "--mode", "reference", "--output-dir", str(reference_dir)]
            if run_stage("full-checkpoint-reference-evaluation", command, root):
                raise RuntimeError("Full-checkpoint profile reference evaluation failed")
            reference_path = reference_dir / "full_reference_profile_eval.json"
        if not reference_path.is_file():
            raise RuntimeError("Reference evaluation receipt is missing")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference.get("status") != "PASS_PROFILE_COHORT_EVAL":
            raise RuntimeError("Full-checkpoint profile reference did not pass")

        first_alpha = float(protocol["selection"]["candidate_retain_nll_weights_ascending"][0])
        first_alpha_tag = f"{first_alpha:g}"
        smoke_dir = root / f"smoke_alpha_{first_alpha_tag}"
        command = [
            sys.executable, str(HERE / "npo_lora_train.py"), "--model", "3b", "--seed", "42",
            "--protocol", str(args.protocol), "--objective-check", str(args.objective_check),
            "--cohort", "dev", "--alpha-retain", first_alpha_tag, "--output-dir", str(smoke_dir), "--smoke",
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
                "--mode", "dev", "--alpha-retain", alpha_tag, "--adapter-dir", str(train_dir / "adapter"),
                "--reference-eval", str(reference_path), "--output-dir", str(eval_dir),
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
            "--mode", "final", "--alpha-retain", f"{selected_alpha:g}", "--adapter-dir", str(final_train_dir / "adapter"),
            "--reference-eval", str(reference_path), "--output-dir", str(final_eval_dir),
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
