#!/usr/bin/env python3
"""Run the pinned official TOFU evaluator on one trained NPO LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from tofu_official_eval import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    add_forget_quality,
    build_eval_sets,
    evaluate_checkpoint,
    load_data,
    summarize_examples,
)


PRIMARY_FIELD = "truth_ratio_score"
UTILITY_MARGINS = {
    "retain90.answer_probability": 0.05,
    "retain90.rougeL_recall": 0.05,
    "holdout10.answer_probability": 0.05,
}


def paired_cluster_delta(rows_a: list[dict], rows_b: list[dict], field: str, seed: int = BOOTSTRAP_SEED) -> dict:
    b_by_key = {(r["split"], r["source_row_index"]): r for r in rows_b}
    grouped = defaultdict(list)
    for row in rows_a:
        other = b_by_key.get((row["split"], row["source_row_index"]))
        if other is None or row.get(field) is None or other.get(field) is None:
            continue
        delta = float(other[field]) - float(row[field])
        grouped[row["profile_block"]].append(delta)
    if not grouped:
        raise ValueError(f"No matched cluster rows for {field}")
    cluster_ids = sorted(grouped)
    point = sum(v for values in grouped.values() for v in values) / sum(map(len, grouped.values()))
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = [rng.choice(cluster_ids) for _ in cluster_ids]
        sample = [v for group in chosen for v in grouped[group]]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return {
        "n_questions": sum(map(len, grouped.values())),
        "n_profile_blocks": len(grouped),
        "mean_delta_b_minus_a": point,
        "profile_cluster_bootstrap_95pct_ci": [
            draws[int(0.025 * (BOOTSTRAP_DRAWS - 1))],
            draws[int(0.975 * (BOOTSTRAP_DRAWS - 1))],
        ],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
    }


def compare_results(base: dict, adapter: dict, oracle: dict) -> dict:
    deltas = {}
    for split, metrics in {
        "forget10": ("answer_probability", "truth_ratio_score", "rougeL_recall"),
        "retain90": ("answer_probability", "truth_ratio_score", "rougeL_recall"),
        "holdout10": ("answer_probability", "truth_ratio_score", "rougeL_recall"),
    }.items():
        deltas[split] = {}
        for metric in metrics:
            if all(metric in row for row in base["per_example"][split]) and all(metric in row for row in adapter["per_example"][split]):
                deltas[split][f"npo_minus_full.{metric}"] = paired_cluster_delta(
                    base["per_example"][split], adapter["per_example"][split], metric
                )
            if split == "forget10" and all(metric in row for row in oracle["per_example"][split]):
                deltas[split][f"npo_minus_oracle.{metric}"] = paired_cluster_delta(
                    oracle["per_example"][split], adapter["per_example"][split], metric
                )

    utility_base = float(base["model_utility"]["harmonic_mean"])
    utility_adapter = float(adapter["model_utility"]["harmonic_mean"])
    primary = deltas["forget10"][f"npo_minus_full.{PRIMARY_FIELD}"]
    primary_pass = primary["profile_cluster_bootstrap_95pct_ci"][0] > 0.0
    utility_checks = {}
    for path, margin in UTILITY_MARGINS.items():
        split, field = path.split(".")
        delta = deltas[split][f"npo_minus_full.{field}"]
        low, high = delta["profile_cluster_bootstrap_95pct_ci"]
        utility_checks[path] = {
            "delta": delta["mean_delta_b_minus_a"],
            "ci95": [low, high],
            "noninferiority_margin": -margin,
            "pass": low > -margin,
        }
    utility_checks["model_utility.harmonic_mean"] = {
        "baseline": utility_base,
        "npo": utility_adapter,
        "delta": utility_adapter - utility_base,
        "noninferiority_margin": -0.05,
        "pass": utility_adapter - utility_base > -0.05,
        "interval": None,
        "note": "Official composite includes utility rows without profile IDs; point-difference gate only.",
    }
    return {
        "status": "PASS_DIAGNOSTIC_GATES" if primary_pass and all(x["pass"] for x in utility_checks.values()) else "FAIL_DIAGNOSTIC_GATES",
        "interpretation": "One-seed LoRA-NPO diagnostic only; not a full-parameter NPO reproduction or a confirmatory result.",
        "primary_outcome": "Forget10 Truth Ratio score (higher means perturbed and paraphrased answer likelihoods are less distinguishable; oracle proximity is assessed separately).",
        "primary_npo_minus_full": primary,
        "forget10_npo_minus_oracle": {
            metric: value for metric, value in deltas["forget10"].items() if "npo_minus_oracle" in metric
        },
        "cluster_bootstrap_deltas": deltas,
        "utility_noninferiority_gates": utility_checks,
        "primary_forget_gate_pass": primary_pass,
        "all_utility_gates_pass": all(x["pass"] for x in utility_checks.values()),
        "npo_model_utility_harmonic_mean": utility_adapter,
        "baseline_model_utility_harmonic_mean": utility_base,
        "forget10_summary": summarize_examples(adapter["per_example"]["forget10"]),
        "retain90_summary": summarize_examples(adapter["per_example"]["retain90"]),
        "holdout10_summary": summarize_examples(adapter["per_example"]["holdout10"]),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "profile block, paired by source row"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("3b", "8b"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    family_key = "llama_3_2_3b" if args.model == "3b" else "llama_3_1_8b"
    base_key = f"{family_key}_full"
    oracle_key = f"{family_key}_retain90"
    result_key = f"{family_key}_npo_lora_seed{args.seed}"
    base_path = args.output_dir / f"{base_key}_official_eval.json"
    oracle_path = args.output_dir / f"{oracle_key}_official_eval.json"
    if not base_path.is_file() or not oracle_path.is_file():
        raise FileNotFoundError("Verified full and retain90-oracle receipts are required before evaluating NPO")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if base.get("status") != "PASS_OFFICIAL_TOFU_BASELINE" or oracle.get("status") != "PASS_OFFICIAL_TOFU_BASELINE":
        raise RuntimeError("Baseline receipts did not pass")
    data, hashes = load_data()
    if base.get("evaluation", {}).get("dataset_sha256_by_file") != hashes:
        raise RuntimeError("Pinned evaluation data hashes differ from the independently verified full-checkpoint baseline")
    training_receipt_path = args.adapter_dir.parent / "training_receipt.json"
    if not training_receipt_path.is_file():
        raise FileNotFoundError("NPO adapter directory must have its training_receipt.json in the parent directory")
    training_receipt = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    if training_receipt.get("status") != "TRAINING_COMPLETE" or not training_receipt.get("adapter_saved"):
        raise RuntimeError("NPO adapter training receipt is not complete")
    adapter_files_sha256 = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.adapter_dir.iterdir()) if path.is_file()
    }
    if training_receipt.get("adapter_files_sha256") != adapter_files_sha256:
        raise RuntimeError("NPO adapter files do not match their training receipt")
    eval_sets = build_eval_sets(data)
    result = evaluate_checkpoint(
        base_key,
        args.output_dir,
        eval_sets,
        hashes,
        score_batch_size=1,
        generation_batch_size=1,
        max_memory_per_gpu_gib=12.5,
        adapter_path=args.adapter_dir,
        result_key=result_key,
        adapter_seed=args.seed,
    )
    add_forget_quality(args.output_dir, result_key, oracle_key)
    baseline = json.loads(base_path.read_text(encoding="utf-8"))
    adapter_eval = json.loads((args.output_dir / f"{result_key}_official_eval.json").read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    comparison = compare_results(baseline, adapter_eval, oracle)
    comparison.update({
        "model_family": family_key,
        "seed": args.seed,
        "base_checkpoint": base["model"],
        "dataset_revision": base["evaluation"]["dataset_revision"],
        "training_receipt": "training_receipt.json",
        "training_receipt_sha256": hashlib.sha256(training_receipt_path.read_bytes()).hexdigest(),
        "adapter_eval_receipt": f"{result_key}_official_eval.json",
    })
    (args.output_dir / "npo_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"eval_status": result["status"], "comparison_status": comparison["status"], "comparison": comparison}, indent=2), flush=True)
    if result["status"] != "PASS_OFFICIAL_TOFU_ADAPTER_EVAL":
        raise RuntimeError("Adapter evaluation did not pass")


if __name__ == "__main__":
    main()
