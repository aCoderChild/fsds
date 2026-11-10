#!/usr/bin/env python3
"""Run the full canonical TOFU evaluation for a selected recovery adapter.

This is the required second-stage check for the four original utility gates.
The profile-cohort screen alone cannot calculate the full model-utility score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from tofu_official_eval import (
    CHECKPOINTS,
    DATASET_REVISION,
    build_eval_sets,
    evaluate_checkpoint,
    load_data,
)
from verify_npo_run import DRAWS, SEED, paired_cluster_delta, validate_metrics


HERE = Path(__file__).resolve().parent
SPLIT_COUNTS = {
    "forget10": 400,
    "retain90": 400,
    "holdout10": 400,
    "real_authors": 100,
    "world_facts": 117,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"Cannot compute finite mean for {field}")
    return sum(values) / len(values)


def independent_utility(result: dict) -> dict:
    rows = result["per_example"]
    components = {
        "retain_probability": mean(rows["retain90"], "answer_probability"),
        "retain_rougeL_recall": mean(rows["retain90"], "rougeL_recall"),
        "retain_truth_ratio_score": mean(rows["retain90"], "truth_ratio_score"),
        "real_authors_normalized_option_probability": mean(rows["real_authors"], "normalized_option_probability"),
        "real_authors_rougeL_recall": mean(rows["real_authors"], "rougeL_recall"),
        "real_authors_truth_ratio_score": mean(rows["real_authors"], "truth_ratio_score"),
        "world_facts_normalized_option_probability": mean(rows["world_facts"], "normalized_option_probability"),
        "world_facts_rougeL_recall": mean(rows["world_facts"], "rougeL_recall"),
        "world_facts_truth_ratio_score": mean(rows["world_facts"], "truth_ratio_score"),
    }
    if any(value <= 0 for value in components.values()):
        harmonic = 0.0
    else:
        harmonic = len(components) / sum(1.0 / value for value in components.values())
    return {"components": components, "harmonic_mean": harmonic}


def compare_full_gates(baseline: dict, candidate: dict, gates: dict) -> dict:
    deltas = {}
    for split, field in (
        ("forget10", "truth_ratio_score"),
        ("forget10", "answer_probability"),
        ("retain90", "answer_probability"),
        ("retain90", "rougeL_recall"),
        ("holdout10", "answer_probability"),
    ):
        deltas[f"{split}.{field}"] = paired_cluster_delta(
            baseline["per_example"][split], candidate["per_example"][split], field
        )

    margin = float(gates["retain90.answer_probability"]["minimum_delta"])
    primary = deltas["forget10.truth_ratio_score"]
    forget_ap = deltas["forget10.answer_probability"]
    checks = {
        "forget10.truth_ratio_score": {
            "delta": primary["mean_delta_b_minus_a"],
            "ci95": primary["profile_cluster_bootstrap_95pct_ci"],
            "rule": "lower 95% profile-block bootstrap bound > 0",
            "pass": primary["profile_cluster_bootstrap_95pct_ci"][0] > 0,
        },
        "forget10.answer_probability": {
            "delta": forget_ap["mean_delta_b_minus_a"],
            "ci95": forget_ap["profile_cluster_bootstrap_95pct_ci"],
            "rule": "upper 95% profile-block bootstrap bound < 0",
            "pass": forget_ap["profile_cluster_bootstrap_95pct_ci"][1] < 0,
        },
    }
    for name in (
        "retain90.answer_probability",
        "retain90.rougeL_recall",
        "holdout10.answer_probability",
    ):
        delta = deltas[name]
        low = delta["profile_cluster_bootstrap_95pct_ci"][0]
        checks[name] = {
            "delta": delta["mean_delta_b_minus_a"],
            "ci95": delta["profile_cluster_bootstrap_95pct_ci"],
            "noninferiority_margin": margin,
            "rule": f"lower 95% profile-block bootstrap bound > {margin}",
            "pass": low > margin,
        }
    baseline_utility = float(baseline["model_utility"]["harmonic_mean"])
    candidate_utility = float(candidate["model_utility"]["harmonic_mean"])
    utility_delta = candidate_utility - baseline_utility
    checks["model_utility.harmonic_mean"] = {
        "baseline": baseline_utility,
        "candidate": candidate_utility,
        "delta": utility_delta,
        "noninferiority_margin": margin,
        "interval": None,
        "rule": f"point delta > {margin}; utility inputs include rows without profile IDs",
        "pass": utility_delta > margin,
    }
    all_pass = all(check["pass"] for check in checks.values())
    return {
        "status": "PASS_DIAGNOSTIC_GATES" if all_pass else "FAIL_DIAGNOSTIC_GATES",
        "all_utility_gates_pass": all(checks[name]["pass"] for name in (
            "retain90.answer_probability",
            "retain90.rougeL_recall",
            "holdout10.answer_probability",
            "model_utility.harmonic_mean",
        )),
        "primary_forget_gate_pass": checks["forget10.truth_ratio_score"]["pass"],
        "checks": checks,
        "paired_deltas": deltas,
        "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "profile block"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--recovery-protocol", type=Path, default=HERE / "npo_lora_recovery_protocol.json")
    parser.add_argument("--full-protocol", type=Path, default=HERE / "npo_lora_pilot_protocol.json")
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    recovery_protocol = read_json(args.recovery_protocol)
    full_protocol = read_json(args.full_protocol)
    train_receipt = read_json(args.training_receipt)
    recovery_receipt = read_json(args.recovery_receipt)
    baseline = read_json(args.baseline_eval)
    protocol_hash = sha256(args.recovery_protocol)
    errors = []
    if recovery_protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        errors.append("Recovery protocol was not locked before the GPU run")
    if recovery_receipt.get("protocol_sha256") != protocol_hash:
        errors.append("Recovery receipt protocol hash mismatch")
    selected_alpha = recovery_receipt.get("selected_alpha_retain")
    if selected_alpha is None or float(train_receipt.get("method", {}).get("alpha_retain", -1)) != float(selected_alpha):
        errors.append("Final adapter alpha does not match the selected alpha")
    followup = train_receipt.get("followup", {})
    if followup.get("cohort") != "final" or train_receipt.get("status") != "TRAINING_COMPLETE":
        errors.append("Adapter is not a complete final-profile recovery training run")
    adapter_hashes = {
        path.name: sha256(path)
        for path in sorted(args.adapter_dir.iterdir()) if path.is_file()
    }
    if not adapter_hashes or adapter_hashes != train_receipt.get("adapter_files_sha256"):
        errors.append("Adapter files do not match their training receipt hashes")
    checkpoint = CHECKPOINTS["llama_3_2_3b_full"]
    if baseline.get("status") != "PASS_OFFICIAL_TOFU_BASELINE":
        errors.append("Pinned full-checkpoint baseline is not a passing official evaluation")
    if baseline.get("model", {}).get("repo_id") != checkpoint["repo_id"] or baseline.get("model", {}).get("revision") != checkpoint["revision"]:
        errors.append("Baseline checkpoint identity differs from the locked 3B checkpoint")
    if baseline.get("evaluation", {}).get("dataset_revision") != DATASET_REVISION:
        errors.append("Baseline dataset revision mismatch")
    if baseline.get("evaluation", {}).get("dataset_sha256_by_file") != full_protocol["dataset"]["evaluation_file_sha256"]:
        errors.append("Baseline evaluation data hashes do not match the full protocol")
    if errors:
        raise RuntimeError("Pre-evaluation integrity gates failed: " + "; ".join(errors))

    data, data_hashes = load_data()
    if data_hashes != full_protocol["dataset"]["evaluation_file_sha256"]:
        raise RuntimeError("Fetched full evaluation source hashes differ from the locked protocol")
    eval_sets = build_eval_sets(data)
    for split, expected in SPLIT_COUNTS.items():
        if len(eval_sets[split]) != expected:
            raise RuntimeError(f"{split}: expected {expected} canonical rows, got {len(eval_sets[split])}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    result = evaluate_checkpoint(
        "llama_3_2_3b_full",
        args.output_dir,
        eval_sets,
        data_hashes,
        score_batch_size=1,
        generation_batch_size=1,
        max_memory_per_gpu_gib=12.5,
        adapter_path=args.adapter_dir,
        result_key="recovery_adapter_full_canonical",
        adapter_seed=int(recovery_protocol["model"]["seed"]),
    )
    if result.get("status") != "PASS_OFFICIAL_TOFU_ADAPTER_EVAL":
        raise RuntimeError("Full canonical adapter evaluation did not pass")
    if result.get("model", {}).get("revision") != checkpoint["revision"]:
        raise RuntimeError("Evaluated adapter base revision differs from the locked checkpoint")
    if result.get("runtime", {}).get("cpu_offload_used") is not False:
        raise RuntimeError("Full canonical evaluation used CPU/disk offload")
    if result.get("model", {}).get("adapter", {}).get("files_sha256") != adapter_hashes:
        raise RuntimeError("Evaluator loaded a different adapter than the receipt-verified adapter")

    validation_problems = []
    validate_metrics(result, baseline, full_protocol, validation_problems)
    if validation_problems:
        raise RuntimeError("Full canonical metric validation failed: " + "; ".join(validation_problems[:20]))
    saved_utility = result["model_utility"]
    computed_utility = independent_utility(result)
    if abs(float(saved_utility["harmonic_mean"]) - computed_utility["harmonic_mean"]) > 1e-10:
        raise RuntimeError("Independent nine-component utility calculation does not reproduce evaluator output")

    comparison = compare_full_gates(baseline, result, full_protocol["evaluation"]["utility_gates"])
    receipt = {
        "status": comparison["status"],
        "research_interpretation": "One-seed exploratory diagnostic. This full canonical evaluation reports all original gates but does not establish seed-level uncertainty or general effectiveness.",
        "started_at_utc": started_at,
        "training_receipt_sha256": sha256(args.training_receipt),
        "recovery_receipt_sha256": sha256(args.recovery_receipt),
        "recovery_protocol_sha256": protocol_hash,
        "full_protocol_sha256": sha256(args.full_protocol),
        "baseline_eval_sha256": sha256(args.baseline_eval),
        "adapter_files_sha256": adapter_hashes,
        "candidate_eval_file": "recovery_adapter_full_canonical_official_eval.json",
        "independently_recomputed_model_utility": computed_utility,
        "original_four_utility_gates_pass": comparison["all_utility_gates_pass"],
        "primary_forget_gate_pass": comparison["primary_forget_gate_pass"],
        "comparison": comparison,
    }
    receipt_path = args.output_dir / "full_canonical_gate_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_md = [
        "# Full canonical TOFU utility recheck",
        "",
        f"**Status:** `{comparison['status']}` (one-seed exploratory result).",
        "",
        "This evaluation uses the complete canonical forget10, retain90, holdout10, real-authors, and world-facts evaluation sets. It reports all four original utility gates and the primary forget Truth Ratio gate.",
        "",
        "| Gate | Delta | 95% profile-block CI | Pass |",
        "|---|---:|---:|:---:|",
    ]
    for name, check in comparison["checks"].items():
        interval = check.get("ci95")
        ci_text = f"[{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval else "point estimate only"
        write_md.append(f"| {name} | {float(check['delta']):+.4f} | {ci_text} | {'Yes' if check['pass'] else 'No'} |")
    write_md.extend([
        "",
        f"Nine-component model utility: baseline `{baseline['model_utility']['harmonic_mean']:.6f}`; adapter `{result['model_utility']['harmonic_mean']:.6f}`; independently recomputed adapter value `{computed_utility['harmonic_mean']:.6f}`.",
        "",
        "The harmonic-mean gate is a point-difference gate because its nine components include utility datasets without profile IDs. Other intervals resample profile blocks, not individual questions. This is not confirmatory evidence; the adapter is LoRA-based and uses one training seed.",
    ])
    (args.output_dir / "full_canonical_gate_report.md").write_text("\n".join(write_md) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
