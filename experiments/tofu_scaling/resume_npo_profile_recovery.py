#!/usr/bin/env python3
"""Resume the preregistered TOFU profile study from its verified alpha=5 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
EXPECTED_INPUT_SHA256 = "74a8a86c7c1f67f0c9625df2805293679a9bacf808204736a8579b839b65d370"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_stage(name: str, command: list[str], root: Path) -> None:
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
        code = process.wait()
    print(f"STAGE {name} exit={code}; elapsed_seconds={time.monotonic() - start:.1f}", flush=True)
    if code:
        raise RuntimeError(f"Stage failed: {name} (exit={code})")


def unpack_locked_artifacts(archive: Path, expected_sha256: str, root: Path) -> None:
    actual = sha256(archive)
    if expected_sha256 != EXPECTED_INPUT_SHA256 or actual != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"Resume archive SHA-256 mismatch: expected {EXPECTED_INPUT_SHA256}; got {actual}")
    if root.exists():
        raise RuntimeError(f"Resume destination already exists; refusing to overwrite: {root}")
    with zipfile.ZipFile(archive) as source:
        bad = source.testzip()
        if bad is not None:
            raise RuntimeError(f"Resume archive CRC failure: {bad}")
        members = source.infolist()
        for member in members:
            path = PurePosixPath(member.filename)
            if not path.parts or path.parts[0] != root.name or ".." in path.parts or path.is_absolute():
                raise RuntimeError(f"Unexpected path in resume archive: {member.filename}")
        source.extractall(root.parent)


def copy_locked_artifact_tree(source: Path, manifest_path: Path, expected_tree_sha256: str, root: Path) -> str:
    """Verify a Kaggle-unpacked source tree against the original ZIP inventory, then copy it."""
    manifest = read_json(manifest_path)
    if manifest.get("root") != root.name:
        raise RuntimeError(f"Unexpected resume tree root in manifest: {manifest.get('root')!r}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Resume file manifest is empty or malformed")
    expected: dict[str, str] = {}
    expected_sizes: dict[str, int] = {}
    for item in files:
        rel = PurePosixPath(item.get("path", ""))
        digest = item.get("sha256")
        if not rel.parts or rel.is_absolute() or ".." in rel.parts or len(rel.parts) == 0:
            raise RuntimeError(f"Unsafe path in resume file manifest: {item.get('path')!r}")
        if len(rel.parts) > 0 and rel.parts[0] == root.name:
            rel = PurePosixPath(*rel.parts[1:])
        if not rel.parts or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"Malformed resume file manifest entry: {item!r}")
        key = rel.as_posix()
        if key in expected:
            raise RuntimeError(f"Duplicate path in resume file manifest: {key}")
        expected[key] = digest
        size_bytes = item.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise RuntimeError(f"Malformed size in resume file manifest entry: {item!r}")
        expected_sizes[key] = size_bytes
    if manifest.get("tree_sha256") != expected_tree_sha256:
        raise RuntimeError("Notebook resume tree hash differs from embedded file manifest")

    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"Resume input is not a regular directory: {source}")
    actual_paths: set[str] = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Symlink in resume input tree is not allowed: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(source).as_posix())
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise RuntimeError(f"Resume input file inventory mismatch; missing={missing[:5]}, extra={extra[:5]}")

    lines: list[str] = []
    for rel in sorted(expected):
        path = source / Path(*PurePosixPath(rel).parts)
        actual_hash = sha256(path)
        if actual_hash != expected[rel]:
            raise RuntimeError(f"Resume input file hash mismatch: {rel}")
        size = path.stat().st_size
        if size != expected_sizes[rel]:
            raise RuntimeError(f"Resume input file size mismatch: {rel}")
        lines.append(f"{rel}\t{actual_hash}\n")
    actual_tree_hash = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    if actual_tree_hash != expected_tree_sha256:
        raise RuntimeError(f"Resume input tree hash mismatch: expected {expected_tree_sha256}; got {actual_tree_hash}")
    if root.exists():
        raise RuntimeError(f"Resume destination already exists; refusing to overwrite: {root}")
    shutil.copytree(source, root)
    return actual_tree_hash


def verify_resume_inputs(root: Path, protocol: dict, protocol_sha: str, verifier, evaluator) -> tuple[dict, dict, dict]:
    old_receipt_path = root / "recovery_run_receipt.json"
    old_receipt = read_json(old_receipt_path)
    recoverable_errors = {
        "RuntimeError: Development evaluation failed for alpha=5",
        "NameError: name 'reference_hashes' is not defined",
    }
    if old_receipt.get("status") != "ERROR" or old_receipt.get("error") not in recoverable_errors:
        raise RuntimeError("Expected a known, recoverable pre-finalization failure receipt")
    if old_receipt.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Failed attempt used a different preregistered protocol")
    if read_json(root / "profile_cohort_data_audit.json").get("status") != "PASS_PROFILE_COHORT_DATA_AUDIT":
        raise RuntimeError("Saved pinned-data audit did not pass")
    if read_json(root / "objective_equivalence.json").get("status") != "PASS":
        raise RuntimeError("Saved objective-equivalence check did not pass")

    reference_path = root / "reference_eval" / "full_reference_profile_eval.json"
    reference = read_json(reference_path)
    if reference.get("status") != "PASS_PROFILE_COHORT_EVAL":
        raise RuntimeError("Saved full-checkpoint reference evaluation is incomplete")
    if reference.get("model", {}).get("revision") != protocol["model"]["revision"]:
        raise RuntimeError("Reference evaluation model revision differs from protocol")
    if reference.get("evaluation", {}).get("dataset_revision") != protocol["data"]["revision"]:
        raise RuntimeError("Reference evaluation dataset revision differs from protocol")
    if reference.get("evaluation", {}).get("input_truncation") is not False:
        raise RuntimeError("Reference evaluation reports input truncation")
    reference_hashes = reference.get("evaluation", {}).get("dataset_sha256_by_file", {})
    if any(reference_hashes.get(name) != digest for name, digest in protocol["data"]["sha256"].items()):
        raise RuntimeError("Reference evaluation does not match one or more preregistered dataset hashes")
    if reference.get("runtime", {}).get("cpu_offload_used") is not False:
        raise RuntimeError("Reference evaluation used CPU or disk offload")
    reference_counts = {
        "forget10": (400, list(range(20))),
        "retain90": (800, list(range(60, 100))),
        "holdout10": (400, list(range(20))),
    }
    for split, (count, blocks) in reference_counts.items():
        issues = verifier.verify_metric_rows(reference.get("per_example", {}).get(split, []), split, count, blocks, 20)
        if issues:
            raise RuntimeError(f"Invalid reference {split} evaluation: {issues[:3]}")

    smoke = read_json(root / "smoke_alpha_5" / "training_receipt.json")
    if smoke.get("status") != "SMOKE_PASS" or smoke.get("global_steps") != 1 or smoke.get("smoke_max_abs_lora_parameter_update", 0) <= 0:
        raise RuntimeError("Saved bounded real-data smoke did not pass")
    if smoke.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Smoke used a different protocol")

    train_dir = root / "dev_alpha_5"
    train_script_hash = sha256(root / "npo_lora_train.py")
    issues = verifier.verify_training_dir(train_dir, "dev", 5.0, protocol, train_script_hash, protocol_sha)
    if issues:
        raise RuntimeError(f"Invalid alpha=5 training artifact: {issues[:3]}")
    candidate_path = root / "dev_eval" / "dev_alpha_5_profile_eval.json"
    candidate = read_json(candidate_path)
    if candidate.get("status") != "PASS_PROFILE_COHORT_EVAL":
        raise RuntimeError("Saved alpha=5 evaluation is incomplete")
    if candidate.get("model", {}).get("revision") != reference["model"]["revision"]:
        raise RuntimeError("Alpha=5 evaluation model revision differs from reference")
    if candidate.get("model", {}).get("adapter", {}).get("files_sha256") != read_json(train_dir / "training_receipt.json").get("adapter_files_sha256"):
        raise RuntimeError("Alpha=5 evaluation adapter hashes differ from the verified training receipt")
    if candidate.get("evaluation", {}).get("input_truncation") is not False:
        raise RuntimeError("Alpha=5 evaluation reports input truncation")
    candidate_hashes = candidate.get("evaluation", {}).get("dataset_sha256_by_file", {})
    if any(candidate_hashes.get(name) != digest for name, digest in protocol["data"]["sha256"].items()):
        raise RuntimeError("Alpha=5 evaluation does not match one or more preregistered dataset hashes")
    if candidate_hashes != reference_hashes:
        raise RuntimeError("Alpha=5 and reference evaluations used different full source-hash sets")
    if candidate.get("runtime", {}).get("cpu_offload_used") is not False:
        raise RuntimeError("Alpha=5 evaluation used CPU or disk offload")
    candidate_counts = {
        "forget10": (200, list(range(10))),
        "retain90": (400, list(range(60, 80))),
        "holdout10": (200, list(range(10))),
    }
    for split, (count, blocks) in candidate_counts.items():
        issues = verifier.verify_metric_rows(candidate.get("per_example", {}).get(split, []), split, count, blocks, 20)
        if issues:
            raise RuntimeError(f"Invalid alpha=5 {split} evaluation: {issues[:3]}")

    comparison = evaluator.gate_comparison(reference, candidate, protocol)
    independent = verifier.compare_gates(reference, candidate, protocol)
    if comparison["status"] != independent["status"] or comparison["all_gates_pass"] != independent["all_gates_pass"]:
        raise RuntimeError("Two independent gate implementations disagree on the saved alpha=5 evaluation")
    comparison_path = root / "dev_eval" / "dev_alpha_5_comparison.json"
    comparison.update({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "model_revision": reference["model"]["revision"],
        "dataset_revision": reference["evaluation"]["dataset_revision"],
        "cohort": "dev",
        "alpha_retain": 5.0,
        "training_receipt_sha256": sha256(train_dir / "training_receipt.json"),
        "evaluation_receipt": candidate_path.name,
        "reconciled_from_saved_evaluations": True,
        "failed_attempt_receipt_sha256": sha256(old_receipt_path),
    })
    write_json(comparison_path, comparison)
    return old_receipt, reference, comparison


def archive(root: Path, destination: Path) -> dict:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != destination:
                output.write(path, path.relative_to(root.parent))
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def existing_candidate_artifacts(root: Path, alpha: float) -> bool:
    """Return true for a complete saved candidate; reject partial artifacts to avoid silent reruns."""
    tag = f"{alpha:g}"
    train_dir = root / f"dev_alpha_{tag}"
    eval_dir = root / "dev_eval"
    required = [
        train_dir / "training_receipt.json",
        train_dir / "loss_trace.jsonl",
        train_dir / "adapter",
        eval_dir / f"dev_alpha_{tag}_profile_eval.json",
        eval_dir / f"dev_alpha_{tag}_comparison.json",
    ]
    present = [path.exists() for path in required]
    if not any(present):
        return False
    if any(present) and not all(present):
        missing = [str(path.relative_to(root)) for path, exists in zip(required, present) if not exists]
        raise RuntimeError(f"Partial saved artifacts for alpha={tag}; refusing to repeat GPU work; missing={missing}")
    zero_files = [path for path in required if path.is_file() and path.stat().st_size == 0]
    adapter_files = [path for path in (train_dir / "adapter").rglob("*") if path.is_file()]
    if zero_files or not adapter_files or any(path.stat().st_size == 0 for path in adapter_files):
        raise RuntimeError(f"Saved alpha={tag} artifacts contain empty or missing files")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--resume-artifacts", type=Path)
    source_group.add_argument("--resume-artifacts-dir", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--resume-manifest-json", type=Path)
    parser.add_argument("--resume-tree-sha256")
    parser.add_argument("--root", type=Path, default=Path("/kaggle/working/tofu_npo_profile_recovery"))
    parser.add_argument("--protocol", type=Path, default=HERE / "npo_lora_recovery_protocol.json")
    parser.add_argument("--upstream-src", type=Path, default=Path("/kaggle/working/open-unlearning/src"))
    parser.add_argument("--artifact-zip", type=Path, default=Path("/kaggle/working/tofu_npo_profile_recovery_artifacts_resumed.zip"))
    args = parser.parse_args()

    if args.resume_artifacts is not None:
        if args.resume_sha256 is None:
            raise RuntimeError("--resume-sha256 is required with --resume-artifacts")
        unpack_locked_artifacts(args.resume_artifacts, args.resume_sha256, args.root)
        resume_input_sha256 = sha256(args.resume_artifacts)
    else:
        if args.resume_manifest_json is None or args.resume_tree_sha256 is None:
            raise RuntimeError("Directory resume requires --resume-manifest-json and --resume-tree-sha256")
        resume_input_sha256 = copy_locked_artifact_tree(
            args.resume_artifacts_dir, args.resume_manifest_json, args.resume_tree_sha256, args.root
        )
    original_eval_source_hash = sha256(args.root / "evaluate_npo_profile_cohort.py")
    previous_receipt_path = args.root / "recovery_run_receipt.json"
    shutil.copy2(previous_receipt_path, args.root / "prior_failed_attempt_receipt.json")
    prior_source = args.root / "prior_failed_attempt_source"
    prior_source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.root / "evaluate_npo_profile_cohort.py", prior_source / "evaluate_npo_profile_cohort.py")

    protocol = read_json(args.protocol)
    protocol_sha = sha256(args.protocol)
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        raise RuntimeError("Protocol is not preregistered")
    if read_json(args.root / "npo_lora_recovery_protocol.json") != protocol:
        raise RuntimeError("Resume archive protocol differs from the current locked protocol")

    # Replace only executable project sources with this corrected continuation version.
    for name in (
        "npo_lora_recovery_protocol.json", "validate_npo_recovery_protocol.py", "npo_lora_train.py",
        "npo_objective.py", "verify_npo_objective.py", "tofu_official_eval.py",
        "evaluate_npo_profile_cohort.py", "verify_npo_profile_recovery_run.py", "requirements-npo-pilot.txt",
    ):
        source = HERE / name
        if source.is_file():
            shutil.copy2(source, args.root / name)

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Resume requires Kaggle GPU T4 x2")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if gpu_names.count("Tesla T4") != 2:
        raise RuntimeError(f"Expected two Tesla T4 GPUs, found {gpu_names}")

    import evaluate_npo_profile_cohort as evaluator
    import verify_npo_profile_recovery_run as verifier

    original_receipt, reference, alpha5_comparison = verify_resume_inputs(args.root, protocol, protocol_sha, verifier, evaluator)
    reference_hashes = reference.get("evaluation", {}).get("dataset_sha256_by_file", {})
    if not reference_hashes:
        raise RuntimeError("Saved full-checkpoint reference evaluation has no dataset hash inventory")
    expected_attempts = [float(a) for a in protocol["selection"]["candidate_retain_nll_weights_ascending"]]
    if not expected_attempts or expected_attempts[0] != 5.0 or alpha5_comparison["alpha_retain"] != 5.0:
        raise RuntimeError("The saved candidate is not the first preregistered alpha")

    receipt = {
        "status": "RUNNING",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_names": gpu_names,
        "training_started": True,
        "candidate_attempts": [{
            "alpha_retain": 5.0,
            "training_dir": "dev_alpha_5",
            "comparison": "dev_eval/dev_alpha_5_comparison.json",
            "gate_status": alpha5_comparison["status"],
            "recovered_from_saved_evaluation": True,
        }],
        "selected_alpha_retain": None,
        "final_training_started": False,
        "resume_input_sha256": resume_input_sha256,
        "prior_attempt_status": original_receipt["status"],
        "prior_attempt_error": original_receipt.get("error"),
        "prior_attempt_evaluator_sha256": original_eval_source_hash,
        "alpha5_reconciliation": "PASS_INDEPENDENT_GATE_RECOMPUTATION",
        "smoke_status": original_receipt.get("smoke_status"),
    }
    receipt_path = args.root / "recovery_run_receipt.json"
    write_json(receipt_path, receipt)
    try:
        selected_alpha = 5.0 if alpha5_comparison.get("all_gates_pass") else None
        for alpha in expected_attempts[1:]:
            if selected_alpha is not None:
                break
            tag = f"{alpha:g}"
            train_dir = args.root / f"dev_alpha_{tag}"
            eval_dir = args.root / "dev_eval"
            candidate_path = eval_dir / f"dev_alpha_{tag}_profile_eval.json"
            comparison_path = eval_dir / f"dev_alpha_{tag}_comparison.json"
            reuse_saved_candidate = existing_candidate_artifacts(args.root, alpha)
            receipt["candidate_alpha_in_progress"] = alpha
            receipt["reusing_saved_candidate_artifacts"] = reuse_saved_candidate
            write_json(receipt_path, receipt)
            if not reuse_saved_candidate:
                command = [
                    sys.executable, str(HERE / "npo_lora_train.py"), "--model", "3b", "--seed", "42",
                    "--protocol", str(args.protocol), "--objective-check", str(args.root / "objective_equivalence.json"),
                    "--cohort", "dev", "--alpha-retain", tag, "--output-dir", str(train_dir),
                ]
                run_stage(f"dev-training-alpha-{tag}", command, args.root)
            train_receipt = read_json(train_dir / "training_receipt.json")
            if train_receipt.get("status") != "TRAINING_COMPLETE" or train_receipt.get("global_steps") != protocol["training"]["expected_steps_per_cohort"]:
                raise RuntimeError(f"Development training receipt failed for alpha={tag}")
            issues = verifier.verify_training_dir(
                train_dir, "dev", float(alpha), protocol, sha256(HERE / "npo_lora_train.py"), protocol_sha
            )
            if issues:
                raise RuntimeError(f"Development alpha={tag} training artifact failed audit: {issues[:3]}")

            if not reuse_saved_candidate:
                command = [
                    sys.executable, str(HERE / "evaluate_npo_profile_cohort.py"), "--protocol", str(args.protocol),
                    "--mode", "dev", "--alpha-retain", tag, "--adapter-dir", str(train_dir / "adapter"),
                    "--reference-eval", str(args.root / "reference_eval" / "full_reference_profile_eval.json"),
                    "--output-dir", str(eval_dir),
                ]
                run_stage(f"dev-evaluation-alpha-{tag}", command, args.root)
            candidate = read_json(candidate_path)
            if candidate.get("status") != "PASS_PROFILE_COHORT_EVAL":
                raise RuntimeError(f"Development alpha={tag} evaluation is incomplete")
            if candidate.get("model", {}).get("revision") != reference["model"]["revision"]:
                raise RuntimeError(f"Development alpha={tag} model revision differs from reference")
            if candidate.get("model", {}).get("adapter", {}).get("files_sha256") != train_receipt.get("adapter_files_sha256"):
                raise RuntimeError(f"Development alpha={tag} adapter hashes differ from its training receipt")
            if candidate.get("evaluation", {}).get("input_truncation") is not False:
                raise RuntimeError(f"Development alpha={tag} evaluation reports input truncation")
            candidate_hashes = candidate.get("evaluation", {}).get("dataset_sha256_by_file", {})
            if any(candidate_hashes.get(name) != digest for name, digest in protocol["data"]["sha256"].items()):
                raise RuntimeError(f"Development alpha={tag} does not match one or more locked data hashes")
            if candidate_hashes != reference_hashes:
                raise RuntimeError(f"Development alpha={tag} and baseline used different full source-hash sets")
            if candidate.get("runtime", {}).get("cpu_offload_used") is not False:
                raise RuntimeError(f"Development alpha={tag} used CPU/disk offload")
            expected_counts = {"forget10": (200, list(range(10))), "retain90": (400, list(range(60, 80))), "holdout10": (200, list(range(10)))}
            for split, (count, blocks) in expected_counts.items():
                issues = verifier.verify_metric_rows(candidate.get("per_example", {}).get(split, []), split, count, blocks, 20)
                if issues:
                    raise RuntimeError(f"Development alpha={tag} {split} metrics failed audit: {issues[:3]}")
            comparison = read_json(comparison_path)
            independent = verifier.compare_gates(reference, candidate, protocol)
            if comparison.get("status") != independent["status"] or comparison.get("all_gates_pass") != independent["all_gates_pass"]:
                raise RuntimeError(f"Independent gate recomputation disagrees for alpha={tag}")
            receipt["candidate_attempts"].append({
                "alpha_retain": alpha,
                "training_dir": f"dev_alpha_{tag}",
                "comparison": f"dev_eval/dev_alpha_{tag}_comparison.json",
                "gate_status": comparison["status"],
                "reused_saved_artifacts": reuse_saved_candidate,
            })
            if comparison.get("all_gates_pass"):
                selected_alpha = alpha
                receipt["selected_alpha_retain"] = alpha
                receipt["selection_rule"] = protocol["selection"]["rule"]
            write_json(receipt_path, receipt)

        if selected_alpha is None:
            receipt["status"] = "NO_DEV_CANDIDATE_PASSES"
            receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(receipt_path, receipt)
            return

        receipt["selected_alpha_retain"] = selected_alpha
        final_train_dir = args.root / f"final_alpha_{selected_alpha:g}"
        command = [
            sys.executable, str(HERE / "npo_lora_train.py"), "--model", "3b", "--seed", "42",
            "--protocol", str(args.protocol), "--objective-check", str(args.root / "objective_equivalence.json"),
            "--cohort", "final", "--alpha-retain", f"{selected_alpha:g}", "--output-dir", str(final_train_dir),
        ]
        receipt["final_training_started"] = True
        write_json(receipt_path, receipt)
        run_stage(f"final-training-alpha-{selected_alpha:g}", command, args.root)
        final_receipt = read_json(final_train_dir / "training_receipt.json")
        if final_receipt.get("status") != "TRAINING_COMPLETE" or final_receipt.get("global_steps") != protocol["training"]["expected_steps_per_cohort"]:
            raise RuntimeError("Final training receipt failed")
        issues = verifier.verify_training_dir(
            final_train_dir, "final", float(selected_alpha), protocol, sha256(HERE / "npo_lora_train.py"), protocol_sha
        )
        if issues:
            raise RuntimeError(f"Final training artifact failed audit: {issues[:3]}")
        final_eval_dir = args.root / "final_eval"
        command = [
            sys.executable, str(HERE / "evaluate_npo_profile_cohort.py"), "--protocol", str(args.protocol),
            "--mode", "final", "--alpha-retain", f"{selected_alpha:g}", "--adapter-dir", str(final_train_dir / "adapter"),
            "--reference-eval", str(args.root / "reference_eval" / "full_reference_profile_eval.json"),
            "--output-dir", str(final_eval_dir),
        ]
        run_stage("final-cohort-evaluation", command, args.root)
        final_comparison_path = final_eval_dir / f"final_alpha_{selected_alpha:g}_comparison.json"
        final_comparison = read_json(final_comparison_path)
        candidate = read_json(final_eval_dir / f"final_alpha_{selected_alpha:g}_profile_eval.json")
        if candidate.get("status") != "PASS_PROFILE_COHORT_EVAL":
            raise RuntimeError("Final profile-cohort evaluation is incomplete")
        if candidate.get("model", {}).get("revision") != reference["model"]["revision"]:
            raise RuntimeError("Final model revision differs from reference")
        if candidate.get("model", {}).get("adapter", {}).get("files_sha256") != final_receipt.get("adapter_files_sha256"):
            raise RuntimeError("Final adapter hashes differ from its training receipt")
        candidate_hashes = candidate.get("evaluation", {}).get("dataset_sha256_by_file", {})
        if any(candidate_hashes.get(name) != digest for name, digest in protocol["data"]["sha256"].items()):
            raise RuntimeError("Final evaluation does not match one or more locked data hashes")
        if candidate_hashes != reference_hashes:
            raise RuntimeError("Final and reference evaluations used different full source-hash sets")
        if candidate.get("evaluation", {}).get("input_truncation") is not False:
            raise RuntimeError("Final evaluation reports input truncation")
        independent = verifier.compare_gates(reference, candidate, protocol)
        if final_comparison.get("status") != independent["status"] or final_comparison.get("all_gates_pass") != independent["all_gates_pass"]:
            raise RuntimeError("Independent final gate recomputation disagrees")
        receipt["final_comparison"] = f"final_eval/{final_comparison_path.name}"
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
        artifact = archive(args.root, args.artifact_zip)
        receipt["artifact_archive"] = artifact
        write_json(receipt_path, receipt)
        print(json.dumps({"run_receipt": receipt, "artifact_archive": artifact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
