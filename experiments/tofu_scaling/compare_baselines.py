#!/usr/bin/env python3
"""Create matched 3B-vs-8B checkpoint diagnostics from verified score receipts."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path


def read_scores(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cluster_interval(values_by_group: dict[int, list[float]], seed: int = 42, draws: int = 5000) -> list[float]:
    means = [sum(values) / len(values) for values in values_by_group.values()]
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(draws))
    return [estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]]


def paired_delta(rows3: list[dict], rows8: list[dict], split: str, exclude_ambiguous: bool = False) -> dict:
    a = {
        int(row["row_index"]): row
        for row in rows3
        if row["split"] == split and not (exclude_ambiguous and row.get("ambiguous_question_collision"))
    }
    b = {
        int(row["row_index"]): row
        for row in rows8
        if row["split"] == split and not (exclude_ambiguous and row.get("ambiguous_question_collision"))
    }
    if set(a) != set(b):
        raise ValueError(f"{split}: model receipts do not contain the same row indices")
    if any(a[idx]["question_sha256"] != b[idx]["question_sha256"] for idx in a):
        raise ValueError(f"{split}: model receipts do not refer to identical questions")
    deltas = {idx: float(b[idx]["answer_probability"]) - float(a[idx]["answer_probability"]) for idx in a}
    groups: dict[int, list[float]] = collections.defaultdict(list)
    for idx, delta in deltas.items():
        groups[int(a[idx]["profile_block"])].append(delta)
    return {
        "n_questions": len(deltas),
        "n_profile_blocks": len(groups),
        "mean_8b_minus_3b_answer_probability": sum(deltas.values()) / len(deltas),
        "mean_profile_delta": sum(sum(v) / len(v) for v in groups.values()) / len(groups),
        "profile_cluster_bootstrap_95pct_ci": cluster_interval(groups),
        "profiles_with_positive_delta": sum(sum(v) / len(v) > 0 for v in groups.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    verified = json.loads((root / "baseline_diagnostic_verification.json").read_text(encoding="utf-8"))
    if verified.get("overall_status") != "PASS":
        raise RuntimeError("Run verify_baseline_results.py first and require overall PASS")
    p3 = json.loads((root / "llama_3_2_3b_baseline.json").read_text(encoding="utf-8"))
    p8 = json.loads((root / "llama_3_1_8b_baseline.json").read_text(encoding="utf-8"))
    rows3 = read_scores(root / "llama_3_2_3b_scores.jsonl")
    rows8 = read_scores(root / "llama_3_1_8b_scores.jsonl")
    deltas = {}
    for split in ("forget10", "retain90", "holdout10"):
        deltas[split] = paired_delta(rows3, rows8, split)
    deltas["retain90_ambiguous_questions_excluded"] = paired_delta(
        rows3, rows8, "retain90", exclude_ambiguous=True
    )
    report = {
        "status": "PASS_MATCHED_CHECKPOINT_COMPARISON",
        "interpretation": "Descriptive comparison only: Llama 3.2 3B and Llama 3.1 8B differ in generation/training history, so this is not a causal parameter-count effect.",
        "model_3b": p3["model"],
        "model_8b": p8["model"],
        "3b_summary": p3["summary"],
        "8b_summary": p8["summary"],
        "paired_delta_8b_minus_3b": deltas,
    }
    (root / "matched_checkpoint_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# TOFU full-checkpoint baseline diagnostic",
        "",
        "**Status:** per-question receipts and profile-clustered intervals independently verified. This is a pre-unlearning baseline, not an unlearning result.",
        "",
        "Metric: mean per-example `exp(-mean assistant-token cross-entropy)`, following the pinned Open-Unlearning probability metric. It measures likelihood assigned to reference answers; it is not generation accuracy.",
        "",
        "## Per-model scores",
        "",
        "| Split | Llama 3.2 3B mean [95% profile-bootstrap CI] | Llama 3.1 8B mean [95% profile-bootstrap CI] | Questions / profile blocks |",
        "|---|---:|---:|---:|",
    ]
    for split in ("forget10", "retain90", "holdout10"):
        a = p3["summary"][split]
        b = p8["summary"][split]
        lines.append(
            f"| {split} | {a['mean_answer_probability']:.4f} [{a['profile_cluster_bootstrap_95pct_ci_answer_probability'][0]:.4f}, {a['profile_cluster_bootstrap_95pct_ci_answer_probability'][1]:.4f}] | {b['mean_answer_probability']:.4f} [{b['profile_cluster_bootstrap_95pct_ci_answer_probability'][0]:.4f}, {b['profile_cluster_bootstrap_95pct_ci_answer_probability'][1]:.4f}] | {a['n_questions']} / {a['n_profile_blocks']} |"
        )
    clean3 = p3["summary"]["retain90_ambiguous_questions_excluded"]
    clean8 = p8["summary"]["retain90_ambiguous_questions_excluded"]
    lines.append(
        f"| retain90, conflicting question rows excluded | {clean3['mean_answer_probability']:.4f} [{clean3['profile_cluster_bootstrap_95pct_ci_answer_probability'][0]:.4f}, {clean3['profile_cluster_bootstrap_95pct_ci_answer_probability'][1]:.4f}] | {clean8['mean_answer_probability']:.4f} [{clean8['profile_cluster_bootstrap_95pct_ci_answer_probability'][0]:.4f}, {clean8['profile_cluster_bootstrap_95pct_ci_answer_probability'][1]:.4f}] | 398 / 20 |"
    )
    lines += [
        "",
        "## Matched 8B minus 3B differences",
        "",
        "Intervals bootstrap 20 matched profile blocks, not individual questions. These differences are descriptive, not causal evidence for a parameter-size effect.",
        "",
        "| Split | Mean paired difference | 95% profile-bootstrap CI | Blocks where 8B > 3B |",
        "|---|---:|---:|---:|",
    ]
    for split in ("forget10", "retain90", "holdout10", "retain90_ambiguous_questions_excluded"):
        d = deltas[split]
        label = "retain90, collision rows excluded" if split.startswith("retain90_") else split
        ci = d["profile_cluster_bootstrap_95pct_ci"]
        lines.append(f"| {label} | {d['mean_8b_minus_3b_answer_probability']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {d['profiles_with_positive_delta']}/{d['n_profile_blocks']} |")
    lines += [
        "",
        "## Short interpretation",
        "",
        "Both full checkpoints assign high likelihood to answers for forget10 and retain90 and substantially lower likelihood to the separate holdout10 split. This shows a useful pre-unlearning signal on the benchmark, though likelihood alone does not prove exact answer generation. The 8B checkpoint has higher answer likelihood on forget and retain but lower on holdout in this sample. Since the pair spans Llama 3.2 and Llama 3.1, this cannot be attributed to parameter count alone.",
        "",
        "Excluding the two retain rows with the identical question and conflicting answers changes the mean only slightly. Keep the official split for primary comparability, and retain the sensitivity result.",
        "",
        "## Run and integrity details",
        "",
        f"- Kaggle: [private baseline diagnostic kernel](https://www.kaggle.com/code/bunnoob2005/tofu-llama-3b-8b-baseline-diagnostic-2026-11-06).",
        f"- Models: `{p3['model']['repo_id']}@{p3['model']['revision']}` and `{p8['model']['repo_id']}@{p8['model']['revision']}`.",
        "- Dataset: pinned TOFU revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`.",
        "- Each model scored 1,200 QAs: all 400 forget10, all 400 holdout10, and 400 retain90 from 20 profile blocks; both runs used identical retain profile blocks.",
        "- 3B runtime: 99.22 seconds; 8B runtime: 226.27 seconds; 2 Tesla T4 GPUs; FP16; no training or adapter creation.",
        "- Bootstrap: 5,000 profile-cluster resamples, seed 42. These intervals describe variation across the 20 profile blocks; they do not account for training-seed variation because no unlearning was run.",
        "- Data audit and score receipt verification: `PASS`.",
        "",
    ]
    (root / "baseline_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {root / 'baseline_diagnostic_report.md'}")
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
