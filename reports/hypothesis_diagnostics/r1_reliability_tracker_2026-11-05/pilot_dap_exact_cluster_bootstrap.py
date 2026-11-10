#!/usr/bin/env python3
"""Recompute the pilot's downstream-answer-persistence diagnostic exactly.

This is a post-hoc descriptive analysis of the hash-bound v4 pilot. It uses
target-language SFT-precompetent forget cases and resamples whole leakage
groups. It does not make the one-seed pilot confirmatory.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = ROOT / "reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/extracted/corrected_evaluation/case_metrics.jsonl"
OUT_JSON = HERE / "pilot_dap_exact_cluster_bootstrap.json"
OUT_MD = HERE / "pilot_dap_exact_cluster_bootstrap.md"
EXPECTED_INPUT_SHA256 = "786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9"
METHODS = ("GA", "GD", "NPO")
LANGUAGES = ("fr", "de", "hi", "th")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compositions(total: int, parts: int):
    """Yield all ordered nonnegative multiplicity vectors summing to total."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first, *rest)


def multinomial_weight(counts: tuple[int, ...]) -> int:
    value = math.factorial(sum(counts))
    for count in counts:
        value //= math.factorial(count)
    return value


def weighted_order_statistic(values: list[tuple[float, int]], numerator: int, denominator: int) -> float:
    """Exact discrete quantile at floor(p*(N-1)), with integer weights."""
    total_weight = sum(weight for _, weight in values)
    target = (numerator * (total_weight - 1)) // denominator
    cumulative = 0
    for value, weight in sorted(values):
        if cumulative + weight > target:
            return value
        cumulative += weight
    raise AssertionError("weighted quantile target was not reached")


def exact_group_bootstrap(group_counts: dict[str, tuple[int, int]]) -> dict:
    """Exact percentile CI for numerator/denominator after cluster resampling.

    Every ordered draw of G groups with replacement is represented by its
    multiplicity vector. Its multinomial weight preserves the ordered-draw
    probability. Replicates with a zero DAP denominator are reported and
    omitted from the conditional interval, since DAP is undefined there.
    """
    groups = sorted(group_counts)
    n_groups = len(groups)
    pairs = [group_counts[group] for group in groups]
    ratio_weights: Counter[float] = Counter()
    undefined_weight = 0
    vector_count = 0
    for multiplicities in compositions(n_groups, n_groups):
        vector_count += 1
        weight = multinomial_weight(multiplicities)
        numerator = sum(m * pair[0] for m, pair in zip(multiplicities, pairs))
        denominator = sum(m * pair[1] for m, pair in zip(multiplicities, pairs))
        if denominator == 0:
            undefined_weight += weight
        else:
            ratio_weights[numerator / denominator] += weight
    valid_weight = sum(ratio_weights.values())
    interval = [
        weighted_order_statistic(list(ratio_weights.items()), 1, 40),
        weighted_order_statistic(list(ratio_weights.items()), 39, 40),
    ]
    return {
        "cluster_count": n_groups,
        "ordered_resamples": n_groups**n_groups,
        "distinct_multiplicity_vectors": vector_count,
        "undefined_resample_fraction": undefined_weight / (undefined_weight + valid_weight),
        "conditional_on_defined_resamples": True,
        "exact_percentile_95ci": interval,
    }


def analyse() -> dict:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    input_sha = sha(INPUT)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise ValueError(f"Case-metric hash changed: {input_sha}")
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    keyed = {}
    groups_by_case = {}
    for row in rows:
        key = (row["model"], row["instance_id"], row["language"])
        if key in keyed:
            raise ValueError(f"Duplicate case metric {key}")
        keyed[key] = row
        case_key = (row["instance_id"], row["language"])
        group = row["bootstrap_cluster_id"]
        if case_key in groups_by_case and groups_by_case[case_key] != group:
            raise ValueError(f"Cluster assignment varies across model rows: {case_key}")
        groups_by_case[case_key] = group

    expected_rows = 6 * 5 * 129
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} case metrics; found {len(rows)}")
    results = []
    for method in METHODS:
        for language in LANGUAGES:
            competent = []
            for (model, instance_id, lang), pre in keyed.items():
                if model != "sft" or lang != language or pre["split"] != "forget":
                    continue
                if pre["direct_correct"] is True and pre["multi_correct"] is True:
                    competent.append((instance_id, pre))
            per_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            direct_failures = 0
            persistence_events = 0
            for instance_id, pre in competent:
                post = keyed.get((method, instance_id, language))
                if post is None or post["split"] != "forget":
                    raise ValueError(f"Missing {method} post row for {instance_id}/{language}")
                if post["direct_generation_failed"] or post["multi_generation_failed"]:
                    raise ValueError(f"Generation failure in verified pilot row {method}/{instance_id}/{language}")
                if post["direct_output_invalid"] or post["multi_output_invalid"]:
                    raise ValueError(f"Invalid output in verified pilot row {method}/{instance_id}/{language}")
                group = pre["bootstrap_cluster_id"]
                per_group[group][1] += int(not post["direct_correct"])
                direct_failures += int(not post["direct_correct"])
                event = (not post["direct_correct"]) and post["multi_correct"]
                per_group[group][0] += int(event)
                persistence_events += int(event)
            if not competent or direct_failures == 0:
                raise ValueError(f"DAP is undefined for {method}/{language}")
            if persistence_events > direct_failures:
                raise AssertionError("DAP numerator exceeds denominator")
            group_counts = {g: (v[0], v[1]) for g, v in per_group.items()}
            boot = exact_group_bootstrap(group_counts)
            results.append({
                "method": method,
                "language": language,
                "sft_precompetent_cases": len(competent),
                "entity_groups": len(group_counts),
                "post_direct_failures": direct_failures,
                "direct_failure_with_final_answer_correct": persistence_events,
                "dap": persistence_events / direct_failures,
                "exact_group_bootstrap": boot,
            })
    return {
        "status": "PASS_EXACT_POSTHOC_CLUSTERED_DAP_DIAGNOSTIC",
        "analysis": "DAP = P(final multi-hop remains correct | direct intermediate-fact recall fails after unlearning), among cases where SFT got both probes correct in that target language",
        "run": "Llama 3.2 1B, seed 42; exploratory v4 pilot",
        "split": "forget",
        "input_path": str(INPUT.relative_to(ROOT)),
        "input_case_metrics_sha256": input_sha,
        "bootstrap": "Exact enumeration of all ordered leakage-group draws with replacement; percentile interval uses order statistic floor(p*(N-1)); undefined zero-denominator draws are reported and excluded from the conditional interval",
        "limitations": [
            "Post-hoc diagnostic, not preregistered and not confirmatory.",
            "One 1B seed; does not estimate seed-to-seed variation.",
            "The pilot's bilingual baseline competence gates failed; this per-language DAP does not bypass those gates.",
            "AI-only translation review; no fluent-human review or human-gold controls.",
            "Exact enumeration removes Monte Carlo error, not uncertainty from the small number of entity groups.",
        ],
        "results": results,
        "script_sha256": sha(Path(__file__).resolve()),
    }


def render(report: dict) -> str:
    lines = [
        "# Pilot diagnostic: downstream-answer persistence with exact group-bootstrap intervals",
        "",
        "**Status: post-hoc exploratory diagnostic, not a confirmatory experiment.**",
        "",
        "DAP is the share of post-unlearning direct-recall failures for which the final multi-hop answer remains correct. The denominator includes only cases where the target-language full-SFT checkpoint answered both direct and multi-hop probes correctly before unlearning. Whole leakage groups are resampled; all exact ordered draws are enumerated.",
        "",
        "| Method | Language | SFT-competent cases / groups | Direct failures | Final answer persists | DAP | Exact group-bootstrap 95% CI | Undefined bootstrap draws |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        boot = row["exact_group_bootstrap"]
        low, high = boot["exact_percentile_95ci"]
        lines.append(
            f"| {row['method']} | {row['language'].upper()} | "
            f"{row['sft_precompetent_cases']} / {row['entity_groups']} | "
            f"{row['post_direct_failures']} | "
            f"{row['direct_failure_with_final_answer_correct']} | "
            f"{100*row['dap']:.1f}% | [{100*low:.1f}%, {100*high:.1f}%] | "
            f"{100*boot['undefined_resample_fraction']:.3f}%"
        )
    lines += [
        "",
        "The point estimates show final-answer persistence in 83.3–100.0% of observed direct-recall failures (8–12 failures per method-language cell). The intervals and group counts show substantial uncertainty. This pattern does not prove that the model recovered the intermediate fact or that the final answer was reached by valid reasoning.",
        "",
        "The bootstrap intervals condition on resamples with at least one direct-failure denominator event; the fraction of undefined resamples is shown. They do not account for seed variation. This analysis cannot bypass the failed bilingual competence gate or support a cross-language unlearning claim.",
        "",
        f"Reproduce with `python -B {Path(__file__).name}`. Input case-metrics SHA-256: `{report['input_case_metrics_sha256']}`. Analysis script SHA-256: `{report['script_sha256']}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = analyse()
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "cells": len(report["results"]),
        "all_rows_from_hash_bound_metrics": True,
        "min_dap": min(row["dap"] for row in report["results"]),
        "max_dap": max(row["dap"] for row in report["results"]),
        "input_case_metrics_sha256": report["input_case_metrics_sha256"],
        "json_sha256": sha(OUT_JSON),
        "markdown_sha256": sha(OUT_MD),
    }, indent=2))


if __name__ == "__main__":
    main()
