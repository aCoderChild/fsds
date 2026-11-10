#!/usr/bin/env python3
"""Wait for the guarded Kaggle v2 download, then independently verify it."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath


# This file is <repo>/experiments/tofu_scaling/diagnostics/<run>/finalize_*.py;
# parents[4] is the repository root.
REPO = Path(__file__).resolve().parents[4]
ROOT = Path(__file__).resolve().parent
KAGGLE_DIR = ROOT / "kaggle_v2"
TRACKER_STATUS = KAGGLE_DIR / "tracker_v2_status.json"
TRACKER_HISTORY = ROOT / "live_status.jsonl"
PROTOCOL = REPO / "experiments/tofu_scaling/npo_lora_8b_pilot_protocol.json"
OBJECTIVE_CHECK = ROOT / "objective_equivalence.json"
RUN_VERIFIER = REPO / "experiments/tofu_scaling/verify_npo_run.py"
METRIC_VERIFIER = REPO / "experiments/tofu_scaling/verify_npo_adapter_metrics.py"
PYTHON = Path("/tmp/tofu-npo-verify-venv/bin/python")
OBJECTIVE_TOLERANCE = 1e-6


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failed for {bad_member}")
        roots = set()
        for name in archive.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe ZIP member path: {name}")
            if member.parts:
                roots.add(member.parts[0])
        if len(roots) != 1:
            raise RuntimeError(f"Expected exactly one run root, found {sorted(roots)}")
        run_root_name = next(iter(roots))
        archive.extractall(destination)
    return run_root_name


def run(command: list[str], log_path: Path) -> int:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode


def compare_objective_checks(archived: dict, local: dict) -> dict:
    """Compare objective evidence across runtimes without equating version metadata."""
    exact_fields = (
        "status",
        "upstream",
        "upstream_utils_sha256",
        "candidate_objective_sha256",
        "protocol_sha256",
        "npo_beta",
        "examples",
        "loss_reduction",
    )
    numeric_fields = (
        "upstream_loss",
        "candidate_loss",
        "absolute_loss_difference",
        "maximum_parameter_gradient_difference",
        "retain_nll_loss_difference",
        "combined_vs_sequential_backward_max_gradient_difference",
    )
    mismatches = [name for name in exact_fields if archived.get(name) != local.get(name)]
    numeric_deltas = {}
    for name in numeric_fields:
        try:
            delta = abs(float(archived[name]) - float(local[name]))
        except (KeyError, TypeError, ValueError):
            mismatches.append(name)
            continue
        numeric_deltas[name] = delta
        if delta > OBJECTIVE_TOLERANCE:
            mismatches.append(name)
    # Both executions must independently satisfy the preregistered tolerance.
    for label, check in (("archived", archived), ("local", local)):
        for name in numeric_fields[2:]:
            try:
                value = abs(float(check[name]))
            except (KeyError, TypeError, ValueError):
                mismatches.append(f"{label}.{name}")
                continue
            if value > OBJECTIVE_TOLERANCE:
                mismatches.append(f"{label}.{name}")
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "tolerance": OBJECTIVE_TOLERANCE,
        "exact_fields": list(exact_fields),
        "numeric_fields": list(numeric_fields),
        "numeric_absolute_deltas": numeric_deltas,
        "archived_runtime": {
            "python_version": archived.get("python_version"),
            "torch_version": archived.get("torch_version"),
        },
        "local_runtime": {
            "python_version": local.get("python_version"),
            "torch_version": local.get("torch_version"),
        },
        "runtime_metadata_difference_is_expected": (
            archived.get("python_version") != local.get("python_version")
            or archived.get("torch_version") != local.get("torch_version")
        ),
        "mismatches": sorted(set(mismatches)),
    }


def last_verified_tracker_status(kernel: str) -> dict | None:
    if not TRACKER_HISTORY.is_file():
        return None
    accepted = None
    for line in TRACKER_HISTORY.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            event.get("kernel") == kernel
            and event.get("status") == "COMPLETE"
            and event.get("artifact_downloaded") is True
            and event.get("artifact_crc") == "PASS"
            and event.get("artifact_path")
            and event.get("artifact_sha256")
        ):
            accepted = event
    return accepted


def audit_training_protocol_alignment(run_root: Path, receipt: dict, protocol: dict) -> dict:
    source = run_root / "npo_lora_train.py"
    schedule = receipt.get("training_schedule", {})
    method = receipt.get("method", {})
    actual_batching = {
        "forget_examples_per_microbatch": schedule.get("forget_examples_per_microbatch"),
        "retain_examples_per_microbatch": schedule.get("retain_examples_per_microbatch"),
        "gradient_accumulation_steps": schedule.get("gradient_accumulation_steps"),
    }
    expected_batching_description = (
        f"{actual_batching['forget_examples_per_microbatch']} forget + "
        f"{actual_batching['retain_examples_per_microbatch']} retain examples per microbatch; "
        f"gradient accumulation={actual_batching['gradient_accumulation_steps']}; "
        f"effective paired batch={schedule.get('forget_examples_per_optimizer_step')} per side"
    )
    if not source.is_file():
        return {"status": "FAIL_SOURCE_MISSING", "actual_batching": actual_batching}
    source_text = source.read_text(encoding="utf-8")
    source_hash_matches_receipt = digest(source) == receipt.get("training_script_sha256")
    uses_per_microbatch_token_mean = (
        "retain_loss = (r_nll.sum() / r_counts.sum().to(r_nll.dtype))" in source_text
        and '(METHOD["alpha_retain"] * retain_loss / accumulation_steps).backward()' in source_text
    )
    uses_update_token_weighting = (
        "retain_token_mean_contribution(r_nll.sum(), retain_token_count_in_update)" in source_text
    )
    batching_metadata_matches = method.get("batching") == expected_batching_description
    protocol_mentions_token_mean = "token_mean_nll" in protocol.get("objective", {}).get("formula", "")
    if uses_per_microbatch_token_mean and actual_batching["gradient_accumulation_steps"] > 1:
        weighting = "PER_MICROBATCH_TOKEN_MEAN_EQUAL_WEIGHTED_ACROSS_ACCUMULATION"
        weighting_matches = False
    elif uses_update_token_weighting:
        weighting = "TOKEN_WEIGHTED_ACROSS_OPTIMIZER_UPDATE"
        weighting_matches = True
    else:
        weighting = "UNRECOGNIZED_SOURCE_PATTERN"
        weighting_matches = False
    source_consistent = source_hash_matches_receipt
    status = "PASS" if source_consistent and weighting_matches and batching_metadata_matches else "FAIL_PROTOCOL_ALIGNMENT"
    return {
        "status": status,
        "training_script_sha256": digest(source),
        "training_script_hash_matches_receipt": source_hash_matches_receipt,
        "protocol_expected_retain_reduction": "token mean over all scored retain tokens in one optimizer update",
        "observed_retain_reduction": weighting,
        "retain_weighting_matches_protocol": weighting_matches and protocol_mentions_token_mean,
        "actual_batching": actual_batching,
        "receipt_batching_description": method.get("batching"),
        "receipt_batching_description_matches_schedule": batching_metadata_matches,
    }


def verify(status: dict) -> int:
    tracker_source = str(TRACKER_STATUS)
    if not status.get("artifact_downloaded"):
        prior = last_verified_tracker_status(status.get("kernel", ""))
        if prior is not None:
            status = prior
            tracker_source = f"prior accepted receipt in {TRACKER_HISTORY}"
    if status.get("status") not in {"COMPLETE", "ERROR", "CANCELLED"}:
        raise RuntimeError(f"Tracker has no terminal Kaggle status: {status.get('status')}")
    if not status.get("artifact_downloaded"):
        raise RuntimeError("Tracker reached a terminal state but did not download the archive")
    if status.get("artifact_crc") != "PASS":
        raise RuntimeError(f"Tracker ZIP CRC result is {status.get('artifact_crc')}")

    archive_path = Path(status["artifact_path"]).resolve()
    if not archive_path.is_file() or digest(archive_path) != status.get("artifact_sha256"):
        raise RuntimeError("Downloaded archive is missing or its SHA-256 differs from tracker receipt")

    # Keep the downloaded ZIP immutable and extract a fresh verification copy.
    extract_dir = KAGGLE_DIR / "extracted_archive"
    extract_dir.mkdir(parents=True, exist_ok=True)
    run_root = extract_dir / safe_extract(archive_path, extract_dir)
    training_dir = run_root / "npo_train"
    eval_dir = run_root / "tofu_official_eval"
    protocol_in_run = run_root / "npo_lora_8b_pilot_protocol.json"
    baseline = run_root / "baseline_inputs/llama_3_1_8b_full_official_eval.json"
    adapter = eval_dir / "llama_3_1_8b_npo_lora_seed42_official_eval.json"

    protocol_hash = digest(PROTOCOL)
    if digest(protocol_in_run) != protocol_hash:
        raise RuntimeError("Archived protocol differs from the locked local protocol")
    if not (training_dir / "training_receipt.json").is_file():
        raise RuntimeError("Archive does not contain the full-training receipt")

    receipt = json.loads((training_dir / "training_receipt.json").read_text(encoding="utf-8"))
    protocol = json.loads(protocol_in_run.read_text(encoding="utf-8"))
    semantics = audit_training_protocol_alignment(run_root, receipt, protocol)
    result = {
        "kaggle_status": status["status"],
        "tracker_receipt_source": tracker_source,
        "tracker_checked_at_utc": status.get("checked_at_utc", status.get("observed_at_utc")),
        "artifact_sha256": status["artifact_sha256"],
        "artifact_crc": status["artifact_crc"],
        "run_root": str(run_root),
        "protocol_sha256": protocol_hash,
        "training_receipt_status": receipt.get("status"),
        "training_global_steps": receipt.get("global_steps"),
        "training_protocol_alignment": semantics,
        "training_verifier": "NOT_RUN",
        "independent_metric_verifier": "NOT_RUN",
    }

    if receipt.get("status") != "TRAINING_COMPLETE" or receipt.get("global_steps") != 500:
        raise RuntimeError(f"Training receipt does not meet the locked 500-step requirement: {receipt.get('status')} / {receipt.get('global_steps')}")
    if not adapter.is_file():
        result["overall_status"] = "TRAINING_COMPLETE_EVALUATION_MISSING"
        (KAGGLE_DIR / "postrun_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 2

    if not PYTHON.is_file():
        raise RuntimeError(f"Pinned local verifier environment not found: {PYTHON}")
    if digest(run_root / "verify_npo_run.py") != digest(RUN_VERIFIER):
        raise RuntimeError("Archived NPO verifier source differs from the current source")

    archived_objective = receipt.get("objective_equivalence")
    local_objective = json.loads(OBJECTIVE_CHECK.read_text(encoding="utf-8"))
    objective_comparison = compare_objective_checks(archived_objective or {}, local_objective)
    (KAGGLE_DIR / "objective_cross_runtime_comparison.json").write_text(
        json.dumps(objective_comparison, indent=2) + "\n", encoding="utf-8"
    )
    result["objective_cross_runtime"] = objective_comparison["status"]
    if objective_comparison["status"] != "PASS":
        result["training_verifier"] = "NOT_RUN_OBJECTIVE_MISMATCH"
        result["overall_status"] = "FAIL_VERIFICATION"
        (KAGGLE_DIR / "postrun_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 1

    # The verifier's receipt-consistency check is exact by design. Supply the
    # archived check there only after independently matching it to the local
    # rerun above; run on a temporary copy so the original evaluation record
    # extracted from the ZIP remains untouched.
    with tempfile.TemporaryDirectory(prefix="npo8b-verify-", dir=KAGGLE_DIR) as temp_dir:
        temp_root = Path(temp_dir) / run_root.name
        shutil.copytree(run_root, temp_root)
        archived_objective_path = Path(temp_dir) / "objective_equivalence_archived.json"
        archived_objective_path.write_text(json.dumps(archived_objective, indent=2) + "\n", encoding="utf-8")
        rc = run([
            str(PYTHON), str(RUN_VERIFIER), "--model", "8b",
            "--training-dir", str(temp_root / "npo_train"),
            "--eval-dir", str(temp_root / "tofu_official_eval"),
            "--protocol", str(temp_root / "npo_lora_8b_pilot_protocol.json"),
            "--objective-check", str(archived_objective_path),
        ], KAGGLE_DIR / "local_npo_verifier.log")
    result["training_verifier"] = "PASS" if rc == 0 else f"FAIL_EXIT_{rc}"

    rc = run([
        str(PYTHON), str(METRIC_VERIFIER), "--baseline", str(baseline),
        "--candidate", str(adapter), "--protocol", str(protocol_in_run),
        "--output", str(KAGGLE_DIR / "independent_adapter_metric_verification.json"),
    ], KAGGLE_DIR / "local_metric_verifier.log")
    result["independent_metric_verifier"] = "PASS" if rc == 0 else f"FAIL_EXIT_{rc}"

    verifier_path = eval_dir / "npo_verification.json"
    if verifier_path.is_file():
        verdict = json.loads(verifier_path.read_text(encoding="utf-8"))
        result["research_diagnostic_gate_status"] = verdict.get("research_diagnostic_gate_status")
        result["primary_result"] = verdict.get("primary_result")
        result["utility_gates"] = verdict.get("utility_gates")
    artifact_integrity_pass = (
        rc == 0
        and result["training_verifier"] == "PASS"
        and result["objective_cross_runtime"] == "PASS"
    )
    result["artifact_integrity_status"] = "PASS" if artifact_integrity_pass else "FAIL"
    if artifact_integrity_pass and semantics.get("status") != "PASS":
        result["overall_status"] = "PASS_ARTIFACT_INTEGRITY_ONLY_PROTOCOL_DEVIATION"
    else:
        result["overall_status"] = "PASS" if artifact_integrity_pass else "FAIL_VERIFICATION"
    (KAGGLE_DIR / "postrun_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if artifact_integrity_pass else 1


def main() -> int:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Waiting for guarded tracker receipt: {TRACKER_STATUS}", flush=True)
    while not TRACKER_STATUS.is_file():
        time.sleep(60)
    status = json.loads(TRACKER_STATUS.read_text(encoding="utf-8"))
    return verify(status)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "overall_status": "POSTRUN_VERIFICATION_ERROR",
            "error": f"{type(error).__name__}: {error}",
        }
        (KAGGLE_DIR / "postrun_verification.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2), flush=True)
        raise
