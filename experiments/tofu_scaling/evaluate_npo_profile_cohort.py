#!/usr/bin/env python3
"""Evaluate and gate the preregistered TOFU NPO-LoRA profile cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from tofu_official_eval import (
    BOOTSTRAP_DRAWS,
    CHECKPOINTS,
    DATASET_REVISION,
    evaluate_checkpoint,
    load_data,
)


def expand(ranges: list[list[int]]) -> list[int]:
    return [block for first, last in ranges for block in range(first, last + 1)]


def make_rows(data: dict[str, list[dict]], split: str, profile_blocks: list[int]) -> list[dict]:
    profile_size = 20
    if split == "forget10":
        source = data["forget10.json"]
        eval_source = data["forget10_perturbed.json"]
        if len(source) != len(eval_source) or any(
            (a["question"], a["answer"]) != (b["question"], b["answer"])
            for a, b in zip(source, eval_source)
        ):
            raise ValueError("forget10_perturbed does not align with forget10 source order")
    elif split == "retain90":
        source = data["retain90.json"]
        eval_source = source
    elif split == "holdout10":
        source = data["holdout10.json"]
        eval_source = source
    else:
        raise ValueError(f"Unsupported profile split: {split}")

    selected_blocks = set(profile_blocks)
    output = []
    for source_index, row in enumerate(source):
        block = source_index // profile_size
        if block not in selected_blocks:
            continue
        paired = eval_source[source_index]
        output.append({
            "split": split,
            "eval_row_index": len(output),
            "source_row_index": source_index,
            "profile_block": block,
            "question": row["question"],
            "answer": row["answer"],
            "ambiguous_question_collision": False,
            "paraphrased_answer": paired.get("paraphrased_answer") if split == "forget10" else None,
            "perturbed_answers": paired.get("perturbed_answer") if split == "forget10" else [],
        })
    expected = len(selected_blocks) * profile_size
    if len(output) != expected:
        raise ValueError(f"{split}: expected {expected} selected rows, got {len(output)}")
    if split == "forget10" and any(not row["perturbed_answers"] for row in output):
        raise ValueError("A target forget row has no perturbed answers; Truth Ratio cannot be calculated")
    return output


def cohort_sets(data: dict[str, list[dict]], protocol: dict, cohort: str) -> dict[str, list[dict]]:
    assignments = protocol["data"]["profile_assignment_ranges_inclusive"]
    return {
        split: make_rows(data, split, expand(assignments[f"{split}_{cohort}_utility_eval"] if split != "forget10" else assignments[f"forget10_{cohort}_train_and_eval"]))
        for split in ("forget10", "retain90", "holdout10")
    }


def reference_sets(data: dict[str, list[dict]], protocol: dict) -> dict[str, list[dict]]:
    assignments = protocol["data"]["profile_assignment_ranges_inclusive"]
    target = list(range(20))
    retain = expand(assignments["retain90_dev_utility_eval"]) + expand(assignments["retain90_final_utility_eval"])
    holdout = list(range(20))
    return {
        "forget10": make_rows(data, "forget10", target),
        "retain90": make_rows(data, "retain90", retain),
        "holdout10": make_rows(data, "holdout10", holdout),
    }


def paired_cluster_delta(base_rows: list[dict], candidate_rows: list[dict], field: str, seed: int = 42) -> dict:
    base_by_index = {int(row["source_row_index"]): row for row in base_rows}
    groups: dict[int, list[float]] = {}
    for row in candidate_rows:
        base = base_by_index.get(int(row["source_row_index"]))
        if base is None or field not in row or field not in base:
            continue
        groups.setdefault(int(row["profile_block"]), []).append(float(row[field]) - float(base[field]))
    if not groups:
        raise ValueError(f"No matched baseline/candidate rows for {field}")
    block_ids = sorted(groups)
    point = sum(sum(values) for values in groups.values()) / sum(len(values) for values in groups.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = [rng.choice(block_ids) for _ in block_ids]
        values = [value for block in selected for value in groups[block]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return {
        "n_questions": sum(len(values) for values in groups.values()),
        "n_profile_blocks": len(groups),
        "mean_delta_candidate_minus_full": point,
        "profile_cluster_bootstrap_95pct_ci": [
            estimates[int(0.025 * (BOOTSTRAP_DRAWS - 1))],
            estimates[int(0.975 * (BOOTSTRAP_DRAWS - 1))],
        ],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
    }


def gate_comparison(base: dict, candidate: dict, protocol: dict) -> dict:
    tests = {
        "forget10.truth_ratio_score": ("forget10", "truth_ratio_score", "lower_gt_zero"),
        "forget10.answer_probability": ("forget10", "answer_probability", "upper_lt_zero"),
        "retain90.answer_probability": ("retain90", "answer_probability", "lower_gt_minus_margin"),
        "retain90.rougeL_recall": ("retain90", "rougeL_recall", "lower_gt_minus_margin"),
        "holdout10.answer_probability": ("holdout10", "answer_probability", "lower_gt_minus_margin"),
    }
    margin = float(protocol["selection"]["noninferiority_margin"])
    checks = {}
    for name, (split, field, rule) in tests.items():
        delta = paired_cluster_delta(base["per_example"][split], candidate["per_example"][split], field)
        low, high = delta["profile_cluster_bootstrap_95pct_ci"]
        passed = low > 0 if rule == "lower_gt_zero" else high < 0 if rule == "upper_lt_zero" else low > -margin
        checks[name] = {
            "delta": delta["mean_delta_candidate_minus_full"],
            "ci95": [low, high],
            "rule": rule,
            "threshold": 0.0 if rule.endswith("zero") else -margin,
            "pass": bool(passed),
            "n_questions": delta["n_questions"],
            "n_profile_blocks": delta["n_profile_blocks"],
        }
    result = {
        "status": "PASS_DIAGNOSTIC_GATES" if all(item["pass"] for item in checks.values()) else "FAIL_DIAGNOSTIC_GATES",
        "interpretation": "One-seed exploratory profile-cohort follow-up; not confirmatory and not a standard full-split TOFU run.",
        "utility_and_forgetting_gates": checks,
        "all_gates_pass": all(item["pass"] for item in checks.values()),
        "noninferiority_margin": margin,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": 42, "unit": "profile block, paired by source row index"},
    }
    return result


def load_adapter_receipt(adapter_dir: Path, cohort: str, alpha: float) -> dict:
    receipt_path = adapter_dir.parent / "training_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Adapter training receipt missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "TRAINING_COMPLETE" or not receipt.get("adapter_saved"):
        raise ValueError("Adapter receipt is not a complete training run")
    followup = receipt.get("followup", {})
    if followup.get("cohort") != cohort or float(followup.get("selected_alpha_retain", -1)) != alpha:
        raise ValueError("Adapter cohort/alpha does not match evaluation request")
    actual_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(adapter_dir.iterdir()) if path.is_file()
    }
    if actual_hashes != receipt.get("adapter_files_sha256"):
        raise ValueError("Adapter files do not match their training receipt hashes")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("npo_lora_recovery_protocol.json"))
    parser.add_argument("--mode", choices=("reference", "dev", "final"), required=True)
    parser.add_argument("--alpha-retain", type=float)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--reference-eval", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in {
        "tofu_3b_npo_lora_profile_recovery_v1",
        "tofu_3b_npo_lora_profile_alpha40_followup_v1",
    }:
        raise ValueError("Wrong profile recovery protocol")
    if protocol["status"] != "PREREGISTERED_BEFORE_GPU_RUN":
        raise ValueError("Recovery protocol is not preregistered")
    if (args.mode == "reference") != (args.adapter_dir is None):
        raise ValueError("Reference mode requires no adapter; dev/final modes require --adapter-dir")
    if args.mode != "reference" and args.alpha_retain not in protocol["selection"]["candidate_retain_nll_weights_ascending"]:
        raise ValueError("Adapter alpha is not in the locked candidate grid")

    data, hashes = load_data()
    for filename, expected_hash in protocol["data"]["sha256"].items():
        if hashes.get(filename) != expected_hash:
            raise ValueError(f"Pinned source hash mismatch: {filename}")
    evaluation_sets = reference_sets(data, protocol) if args.mode == "reference" else cohort_sets(data, protocol, args.mode)
    if args.mode == "reference":
        adapter_path = None
        result_key = "full_reference"
    else:
        load_adapter_receipt(args.adapter_dir, args.mode, float(args.alpha_retain))
        adapter_path = args.adapter_dir
        result_key = f"{args.mode}_alpha_{args.alpha_retain:g}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_checkpoint(
        "llama_3_2_3b_full",
        args.output_dir,
        evaluation_sets,
        hashes,
        score_batch_size=1,
        generation_batch_size=1,
        max_memory_per_gpu_gib=12.5,
        adapter_path=adapter_path,
        result_key=result_key,
        adapter_seed=42 if adapter_path else None,
        profile_cohort_mode=True,
    )
    if result.get("status") != "PASS_PROFILE_COHORT_EVAL":
        raise RuntimeError("Profile cohort evaluation did not complete")
    if args.mode == "reference":
        return

    reference_path = args.reference_eval or (args.output_dir.parent / "reference_eval" / "full_reference_profile_eval.json")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Full-checkpoint profile reference evaluation must be completed first: {reference_path}")
    baseline = json.loads(reference_path.read_text(encoding="utf-8"))
    comparison = gate_comparison(baseline, result, protocol)
    comparison.update({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "model_revision": CHECKPOINTS["llama_3_2_3b_full"]["revision"],
        "dataset_revision": DATASET_REVISION,
        "cohort": args.mode,
        "alpha_retain": float(args.alpha_retain),
        "training_receipt_sha256": hashlib.sha256((args.adapter_dir.parent / "training_receipt.json").read_bytes()).hexdigest(),
        "evaluation_receipt": f"{result_key}_profile_eval.json",
    })
    comparison_path = args.output_dir / f"{args.mode}_alpha_{args.alpha_retain:g}_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
