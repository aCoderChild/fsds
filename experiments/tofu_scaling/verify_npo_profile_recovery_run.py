#!/usr/bin/env python3
"""Independently verify the saved profile-cohort NPO-LoRA run and gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import zipfile
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def verify_metric_rows(rows: list[dict], split: str, expected_rows: int, expected_blocks: list[int], profile_size: int) -> list[str]:
    problems = []
    if len(rows) != expected_rows:
        return [f"{split}: expected {expected_rows} rows, found {len(rows)}"]
    if len({int(row["source_row_index"]) for row in rows}) != len(rows):
        problems.append(f"{split}: duplicate source row index")
    groups = sorted({int(row["profile_block"]) for row in rows})
    if groups != sorted(expected_blocks):
        problems.append(f"{split}: profile blocks mismatch: {groups}")
    expected_indices = {block * profile_size + offset for block in expected_blocks for offset in range(profile_size)}
    actual_indices = {int(row["source_row_index"]) for row in rows}
    if actual_indices != expected_indices:
        problems.append(f"{split}: source row indices differ from the locked profile assignment")
    for index, row in enumerate(rows):
        if int(row["source_row_index"]) // profile_size != int(row["profile_block"]):
            problems.append(f"{split}[{index}]: source row index does not match profile block")
        for field in ("answer_probability", "answer_token_loss", "rougeL_recall"):
            if field not in row or not math.isfinite(float(row[field])):
                problems.append(f"{split}[{index}]: invalid or missing {field}")
                continue
        if not close(float(row["answer_probability"]), math.exp(-float(row["answer_token_loss"])), 2e-8):
            problems.append(f"{split}[{index}]: answer_probability does not match answer_token_loss")
        if not 0.0 <= float(row["answer_probability"]) <= 1.0:
            problems.append(f"{split}[{index}]: answer_probability outside [0,1]")
        if not 0.0 <= float(row["rougeL_recall"]) <= 1.0:
            problems.append(f"{split}[{index}]: ROUGE-L outside [0,1]")
        if not isinstance(row.get("generation"), str) or not isinstance(row.get("reference_answer"), str):
            problems.append(f"{split}[{index}]: generation/reference text missing")
        if split == "forget10":
            wrong_losses = row.get("perturbed_answer_token_losses")
            if not wrong_losses:
                problems.append(f"forget10[{index}]: perturbed-answer losses missing")
                continue
            if any(not math.isfinite(float(value)) for value in wrong_losses):
                problems.append(f"forget10[{index}]: non-finite perturbed-answer loss")
                continue
            wrong_probability = math.exp(-sum(map(float, wrong_losses)) / len(wrong_losses))
            if "paraphrased_answer_token_loss" in row:
                correct_probability = math.exp(-float(row["paraphrased_answer_token_loss"]))
                ratio = wrong_probability / (correct_probability + 1e-10)
                expected_score = min(ratio, 1.0 / (ratio + 1e-10))
            else:
                ratio = wrong_probability / (float(row["answer_probability"]) + 1e-10)
                expected_score = max(0.0, 1.0 - ratio)
            if not close(float(row.get("truth_ratio", math.nan)), ratio, 2e-8):
                problems.append(f"forget10[{index}]: Truth Ratio mismatch")
            if not close(float(row.get("truth_ratio_score", math.nan)), expected_score, 2e-8):
                problems.append(f"forget10[{index}]: Truth Ratio score mismatch")
    return problems


def paired_delta(base_rows: list[dict], candidate_rows: list[dict], field: str, draws: int, seed: int) -> dict:
    base = {int(row["source_row_index"]): row for row in base_rows}
    groups: dict[int, list[float]] = {}
    for row in candidate_rows:
        source_index = int(row["source_row_index"])
        if source_index not in base:
            raise ValueError(f"Candidate source row {source_index} is not in the reference set")
        groups.setdefault(int(row["profile_block"]), []).append(float(row[field]) - float(base[source_index][field]))
    if not groups:
        raise ValueError(f"No paired rows for {field}")
    block_ids = sorted(groups)
    point = sum(sum(values) for values in groups.values()) / sum(len(values) for values in groups.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        chosen = [rng.choice(block_ids) for _ in block_ids]
        values = [value for block in chosen for value in groups[block]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return {
        "delta": point,
        "ci95": [estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]],
        "n_questions": sum(len(values) for values in groups.values()),
        "n_profile_blocks": len(groups),
    }


def compare_gates(base: dict, candidate: dict, protocol: dict) -> dict:
    margin = float(protocol["selection"]["noninferiority_margin"])
    draws = int(protocol["selection"]["bootstrap"]["draws"])
    seed = int(protocol["selection"]["bootstrap"]["seed"])
    spec = {
        "forget10.truth_ratio_score": ("forget10", "truth_ratio_score", "lower_zero"),
        "forget10.answer_probability": ("forget10", "answer_probability", "upper_zero"),
        "retain90.answer_probability": ("retain90", "answer_probability", "lower_margin"),
        "retain90.rougeL_recall": ("retain90", "rougeL_recall", "lower_margin"),
        "holdout10.answer_probability": ("holdout10", "answer_probability", "lower_margin"),
    }
    checks = {}
    for name, (split, field, rule) in spec.items():
        delta = paired_delta(base["per_example"][split], candidate["per_example"][split], field, draws, seed)
        low, high = delta["ci95"]
        passed = low > 0 if rule == "lower_zero" else high < 0 if rule == "upper_zero" else low > -margin
        checks[name] = {**delta, "rule": rule, "threshold": 0.0 if rule.endswith("zero") else -margin, "pass": passed}
    return {"status": "PASS_DIAGNOSTIC_GATES" if all(item["pass"] for item in checks.values()) else "FAIL_DIAGNOSTIC_GATES", "all_gates_pass": all(item["pass"] for item in checks.values()), "checks": checks}


def verify_training_dir(path: Path, cohort: str, alpha: float, protocol: dict, source_hash: str, protocol_hash: str) -> list[str]:
    problems = []
    receipt_path = path / "training_receipt.json"
    trace_path = path / "loss_trace.jsonl"
    if not receipt_path.is_file() or not trace_path.is_file():
        return [f"{path.name}: missing training receipt or loss trace"]
    receipt = read_json(receipt_path)
    expected_steps = int(protocol["training"]["expected_steps_per_cohort"])
    expected_trace = int(protocol["training"]["expected_loss_trace_rows_per_cohort"])
    if receipt.get("status") != "TRAINING_COMPLETE" or receipt.get("global_steps") != expected_steps:
        problems.append(f"{path.name}: incomplete training receipt/optimizer steps")
    if receipt.get("training_script_sha256") != source_hash:
        problems.append(f"{path.name}: training source hash mismatch")
    if receipt.get("protocol_sha256") != protocol_hash:
        problems.append(f"{path.name}: training receipt protocol hash mismatch")
    followup = receipt.get("followup", {})
    if followup.get("cohort") != cohort or float(followup.get("selected_alpha_retain", -1)) != alpha:
        problems.append(f"{path.name}: cohort or alpha mismatch")
    if receipt.get("method", {}).get("alpha_retain") != alpha:
        problems.append(f"{path.name}: method alpha mismatch")
    if receipt.get("seed") != int(protocol["model"]["seed"]):
        problems.append(f"{path.name}: seed mismatch")
    if receipt.get("runtime", {}).get("cpu_or_disk_offload") is not False:
        problems.append(f"{path.name}: runtime offload is not explicitly false")
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(trace) != expected_trace:
        problems.append(f"{path.name}: expected {expected_trace} trace rows, found {len(trace)}")
    step_counts: dict[int, int] = {}
    for row in trace:
        step = int(row["global_step"])
        step_counts[step] = step_counts.get(step, 0) + 1
        for field in ("forget_npo_loss", "retain_token_nll", "combined_loss", "gradient_norm", "learning_rate"):
            if not math.isfinite(float(row[field])):
                problems.append(f"{path.name}: non-finite {field}")
    if sorted(step_counts) != list(range(1, expected_steps + 1)) or set(step_counts.values()) != {4}:
        problems.append(f"{path.name}: loss trace step accounting is invalid")
    if any(
        int(row.get("forget_examples_in_microbatch", -1)) != 2
        or int(row.get("retain_examples_in_microbatch", -1)) != 2
        for row in trace
    ):
        problems.append(f"{path.name}: paired microbatch sizes differ from the locked schedule")
    adapter = path / "adapter"
    files = {p.name: sha256(p) for p in sorted(adapter.iterdir()) if p.is_file()} if adapter.is_dir() else {}
    if receipt.get("adapter_files_sha256") != files or not files:
        problems.append(f"{path.name}: adapter hash verification failed")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--artifact-zip", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root
    protocol_path = args.protocol or root / "npo_lora_recovery_protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = sha256(protocol_path)
    problems = []
    receipt = read_json(root / "recovery_run_receipt.json")
    if receipt.get("protocol_sha256") != protocol_hash:
        problems.append("run receipt protocol hash mismatch")
    if read_json(root / "profile_cohort_data_audit.json").get("status") != "PASS_PROFILE_COHORT_DATA_AUDIT":
        problems.append("profile cohort data audit did not pass")
    if read_json(root / "objective_equivalence.json").get("status") != "PASS":
        problems.append("pinned objective equivalence did not pass")
    model_revision = "24f31ca19f6966dcb6f6b29abc511cce71222d4a"
    reference_path = root / "reference_eval" / "full_reference_profile_eval.json"
    if not reference_path.is_file():
        problems.append("full-checkpoint reference output is missing")
        reference = None
    else:
        reference = read_json(reference_path)
        if reference.get("status") != "PASS_PROFILE_COHORT_EVAL" or reference.get("model", {}).get("revision") != model_revision:
            problems.append("reference evaluation status/model revision mismatch")
        if reference.get("evaluation", {}).get("input_truncation") is not False:
            problems.append("reference evaluation indicates input truncation")
        observed_hashes = reference.get("evaluation", {}).get("dataset_sha256_by_file", {})
        if any(observed_hashes.get(name) != digest for name, digest in protocol["data"]["sha256"].items()):
            problems.append("reference evaluation does not match one or more locked dataset hashes")
        if reference.get("runtime", {}).get("cpu_offload_used") is not False:
            problems.append("reference evaluation used CPU or disk offload")
        ref_counts = {"forget10": (400, list(range(20))), "retain90": (800, list(range(60, 100))), "holdout10": (400, list(range(20)))}
        for split, (count, blocks) in ref_counts.items():
            problems.extend(verify_metric_rows(reference.get("per_example", {}).get(split, []), split, count, blocks, int(protocol["data"]["profile_block_size"])))
        if reference.get("model_utility") is not None:
            problems.append("profile reference unexpectedly reports full model-utility composite")

    first_alpha = float(protocol["selection"]["candidate_retain_nll_weights_ascending"][0])
    smoke_path = root / f"smoke_alpha_{first_alpha:g}" / "training_receipt.json"
    smoke = None
    if not smoke_path.is_file():
        problems.append("bounded real-data smoke receipt is missing")
    else:
        try:
            smoke = read_json(smoke_path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"bounded real-data smoke receipt is unreadable: {error}")
        if smoke is not None and (
            smoke.get("status") != "SMOKE_PASS"
            or smoke.get("global_steps") != 1
            or smoke.get("smoke_max_abs_lora_parameter_update", 0) <= 0
        ):
            problems.append("bounded real-data smoke did not pass its update criteria")
    if smoke is not None and smoke.get("protocol_sha256") != protocol_hash:
        problems.append("bounded smoke protocol hash mismatch")

    alpha_grid = [float(x) for x in protocol["selection"]["candidate_retain_nll_weights_ascending"]]
    attempts = receipt.get("candidate_attempts", [])
    attempted_alphas = [float(x["alpha_retain"]) for x in attempts]
    if attempted_alphas != alpha_grid[:len(attempted_alphas)]:
        problems.append("development candidates are not an ascending prefix of the locked alpha grid")
    selected = receipt.get("selected_alpha_retain")
    verified_attempts = []
    train_source = root / "npo_lora_train.py"
    source_hash = sha256(train_source) if train_source.is_file() else ""
    if not source_hash:
        problems.append("embedded training source file is missing")
    if reference is not None:
        for alpha in attempted_alphas:
            tag = f"{alpha:g}"
            train_dir = root / f"dev_alpha_{tag}"
            problems.extend(verify_training_dir(train_dir, "dev", alpha, protocol, source_hash, protocol_hash))
            eval_path = root / "dev_eval" / f"dev_alpha_{tag}_profile_eval.json"
            comparison_path = root / "dev_eval" / f"dev_alpha_{tag}_comparison.json"
            if not eval_path.is_file() or not comparison_path.is_file():
                problems.append(f"alpha={tag}: development evaluation/comparison missing")
                continue
            candidate = read_json(eval_path)
            observed_hashes = candidate.get("evaluation", {}).get("dataset_sha256_by_file", {})
            if any(observed_hashes.get(name) != digest for name, digest in protocol["data"]["sha256"].items()):
                problems.append(f"alpha={tag}: development evaluation does not match one or more locked data hashes")
            elif observed_hashes != reference.get("evaluation", {}).get("dataset_sha256_by_file", {}):
                problems.append(f"alpha={tag}: development and reference evaluations used different full source-hash sets")
            if candidate.get("runtime", {}).get("cpu_offload_used") is not False:
                problems.append(f"alpha={tag}: development evaluation used CPU or disk offload")
            expected_counts = {"forget10": (200, list(range(10))), "retain90": (400, list(range(60, 80))), "holdout10": (200, list(range(10)))}
            for split, (count, blocks) in expected_counts.items():
                problems.extend(verify_metric_rows(candidate.get("per_example", {}).get(split, []), split, count, blocks, int(protocol["data"]["profile_block_size"])))
            recomputed = compare_gates(reference, candidate, protocol)
            saved = read_json(comparison_path)
            if saved.get("status") != recomputed["status"] or saved.get("all_gates_pass") != recomputed["all_gates_pass"]:
                problems.append(f"alpha={tag}: development gate status does not reproduce")
            for name, values in recomputed["checks"].items():
                previous = saved.get("utility_and_forgetting_gates", {}).get(name, {})
                if not close(float(previous.get("delta", math.nan)), values["delta"], 1e-12):
                    problems.append(f"alpha={tag}: saved {name} point estimate does not reproduce")
                ci = previous.get("ci95", [])
                if len(ci) != 2 or not all(close(float(a), float(b), 1e-12) for a, b in zip(ci, values["ci95"])):
                    problems.append(f"alpha={tag}: saved {name} bootstrap interval does not reproduce")
            verified_attempts.append({"alpha_retain": alpha, "gate_status": recomputed["status"]})
    passing = [x["alpha_retain"] for x in verified_attempts if x["gate_status"] == "PASS_DIAGNOSTIC_GATES"]
    if selected is None:
        if passing or receipt.get("status") != "NO_DEV_CANDIDATE_PASSES" or receipt.get("final_training_started") is not False:
            problems.append("no-candidate-pass stopping decision is inconsistent")
    else:
        selected = float(selected)
        if not passing or selected != passing[0]:
            problems.append("selected alpha is not the first development candidate passing every gate")
        if receipt.get("final_training_started") is not True:
            problems.append("selected development alpha exists but final training was not marked started")
        final_train = root / f"final_alpha_{selected:g}"
        problems.extend(verify_training_dir(final_train, "final", selected, protocol, source_hash, protocol_hash))
        final_eval_path = root / "final_eval" / f"final_alpha_{selected:g}_profile_eval.json"
        final_comparison_path = root / "final_eval" / f"final_alpha_{selected:g}_comparison.json"
        if not final_eval_path.is_file() or not final_comparison_path.is_file():
            problems.append("selected-alpha final evaluation/comparison missing")
        elif reference is not None:
            candidate = read_json(final_eval_path)
            observed_hashes = candidate.get("evaluation", {}).get("dataset_sha256_by_file", {})
            if any(observed_hashes.get(name) != digest for name, digest in protocol["data"]["sha256"].items()):
                problems.append("final evaluation does not match one or more locked data hashes")
            elif observed_hashes != reference.get("evaluation", {}).get("dataset_sha256_by_file", {}):
                problems.append("final and reference evaluations used different full source-hash sets")
            if candidate.get("runtime", {}).get("cpu_offload_used") is not False:
                problems.append("final evaluation used CPU or disk offload")
            final_counts = {"forget10": (200, list(range(10, 20))), "retain90": (400, list(range(80, 100))), "holdout10": (200, list(range(10, 20)))}
            for split, (count, blocks) in final_counts.items():
                problems.extend(verify_metric_rows(candidate.get("per_example", {}).get(split, []), split, count, blocks, int(protocol["data"]["profile_block_size"])))
            recomputed = compare_gates(reference, candidate, protocol)
            saved = read_json(final_comparison_path)
            if saved.get("status") != recomputed["status"] or saved.get("all_gates_pass") != recomputed["all_gates_pass"]:
                problems.append("final gate status does not reproduce")
            for name, values in recomputed["checks"].items():
                previous = saved.get("utility_and_forgetting_gates", {}).get(name, {})
                if not close(float(previous.get("delta", math.nan)), values["delta"], 1e-12):
                    problems.append(f"final: saved {name} point estimate does not reproduce")
                ci = previous.get("ci95", [])
                if len(ci) != 2 or not all(close(float(a), float(b), 1e-12) for a, b in zip(ci, values["ci95"])):
                    problems.append(f"final: saved {name} bootstrap interval does not reproduce")
            if receipt.get("status") != recomputed["status"]:
                problems.append("run status differs from final gate status")

    archive_check = None
    if args.artifact_zip:
        with zipfile.ZipFile(args.artifact_zip) as zf:
            bad_file = zf.testzip()
            archive_check = {"crc": "PASS" if bad_file is None else "FAIL", "bad_file": bad_file}
            if bad_file is not None:
                problems.append(f"artifact ZIP CRC failed: {bad_file}")
    result = {
        "status": "PASS_ARTIFACT_AND_RESULT_RECOMPUTATION" if not problems else "FAIL_VERIFICATION",
        "problems": problems,
        "run_status": receipt.get("status"),
        "selected_alpha_retain": selected,
        "verified_development_attempts": verified_attempts,
        "artifact_zip": archive_check,
        "training_started": receipt.get("training_started"),
        "protocol_sha256": sha256(protocol_path),
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
