#!/usr/bin/env python3
"""Post-hoc exact sign sensitivity at the leakage-group level for the v4 pilot.

This is a conservative directional check alongside the exact cluster bootstrap.
It treats each entity group as one unit, averages case-level paired changes
within each group, excludes exactly-zero group means, then performs a two-sided
binomial sign test. It is not the primary estimand or a confirmatory analysis.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[0] / "r1_synthworlds_v4_retainfix_corrected_2026-11-03"
METRICS = PILOT / "extracted/corrected_evaluation/case_metrics.jsonl"
EXPECTED_METRICS_SHA256 = "786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9"
METHODS = ("GA", "GD", "NPO")
LANGUAGES = ("fr", "de", "hi", "th")
EXPECTED_CASES = {"fr": 10, "de": 9, "hi": 9, "th": 11}
EXPECTED_GROUPS = {"fr": 6, "de": 5, "hi": 6, "th": 7}
FAMILY_SIZE = len(METHODS) * len(LANGUAGES)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def two_sided_sign_p(positive: int, negative: int) -> float:
    """Exact two-sided binomial sign-test p-value; ties are excluded."""
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def analyse() -> dict:
    if sha(METRICS) != EXPECTED_METRICS_SHA256:
        raise ValueError("Pilot case-metrics hash does not match the archived verified input")
    rows = read_jsonl(METRICS)
    lookup = {}
    for row in rows:
        key = (row["model"], row["instance_id"], row["language"])
        if key in lookup:
            raise ValueError(f"Duplicate case metric: {key}")
        lookup[key] = row
    model_names = {row["model"] for row in rows}
    if model_names != {"pretrained", "sft", "retain_only", *METHODS}:
        raise ValueError(f"Unexpected pilot models: {sorted(model_names)}")
    case_ids = sorted({row["instance_id"] for row in rows
                       if row["model"] == "sft" and row["language"] == "en"})
    if len(case_ids) != 129:
        raise ValueError(f"Expected 129 scored cases; found {len(case_ids)}")

    all_results = {"primary": [], "secondary_did": []}
    for method in METHODS:
        for language in LANGUAGES:
            group_effects = {"primary": defaultdict(list), "secondary_did": defaultdict(list)}
            n_cases = 0
            for case_id in case_ids:
                sft_en = lookup[("sft", case_id, "en")]
                sft_target = lookup[("sft", case_id, language)]
                if sft_en["split"] != "forget":
                    continue
                if not all((sft_en["direct_correct"], sft_en["multi_correct"],
                            sft_target["direct_correct"], sft_target["multi_correct"])):
                    continue
                if sft_en["bootstrap_cluster_id"] != sft_target["bootstrap_cluster_id"]:
                    raise ValueError(f"Group mismatch for {case_id}/{language}")
                post_en = lookup[(method, case_id, "en")]
                post_target = lookup[(method, case_id, language)]
                group_id = sft_en["bootstrap_cluster_id"]
                target_change = int(post_target["multi_correct"]) - int(sft_target["multi_correct"])
                english_change = int(post_en["multi_correct"]) - int(sft_en["multi_correct"])
                group_effects["primary"][group_id].append(target_change)
                group_effects["secondary_did"][group_id].append(target_change - english_change)
                n_cases += 1

            if n_cases != EXPECTED_CASES[language]:
                raise ValueError(f"Unexpected matched-case count for {method}/{language}: {n_cases}")
            expected_groups = EXPECTED_GROUPS[language]
            if len(group_effects["primary"]) != expected_groups:
                raise ValueError(f"Unexpected group count for {method}/{language}")
            for estimand, by_group in group_effects.items():
                means = {group: sum(values) / len(values) for group, values in by_group.items()}
                positive = sum(value > 0 for value in means.values())
                negative = sum(value < 0 for value in means.values())
                tied = sum(value == 0 for value in means.values())
                p = two_sided_sign_p(positive, negative)
                record = {
                    "method": method,
                    "language": language,
                    "estimand": estimand,
                    "cases": n_cases,
                    "entity_groups": len(means),
                    "positive_groups": positive,
                    "negative_groups": negative,
                    "tied_groups": tied,
                    "non_tied_groups": positive + negative,
                    "mean_case_paired_change_pp": round(
                        100 * sum(sum(values) for values in by_group.values()) / n_cases, 10
                    ),
                    "exact_two_sided_sign_p": p,
                    "bonferroni_adjusted_p_within_12_cell_family": min(1.0, p * FAMILY_SIZE),
                }
                all_results[estimand].append(record)

    alpha = 0.05 / FAMILY_SIZE
    for family in all_results.values():
        if len(family) != FAMILY_SIZE:
            raise ValueError("Expected 12 method-language cells per estimand family")
        for record in family:
            if record["exact_two_sided_sign_p"] <= alpha:
                raise ValueError("Unexpected familywise sign-test result; inspect assumptions/data")

    # A two-sided sign test's smallest possible p-value is 2 / 2^n when all n
    # non-tied groups agree. This is a feasibility diagnostic, not a power claim.
    n_for_familywise_minimum = next(
        n for n in range(1, 100) if 2 / (2**n) <= alpha
    )
    return {
        "status": "VERIFIED_POSTHOC_EXACT_GROUP_SIGN_SENSITIVITY_EXPLORATORY_ONLY",
        "input_case_metrics_sha256": sha(METRICS),
        "primary_estimand": "Target-language post-minus-pre multi-hop exact-match change on bilingual-precompetent forget cases",
        "secondary_estimand": "Target-language-minus-English post-minus-pre multi-hop change on the same cases",
        "unit": "One sign per entity group, calculated from that group's mean paired case change; zero groups are ties and omitted from the sign-test denominator",
        "familywise_adjustment": "Bonferroni over 12 method-language cells separately for each estimand family",
        "alpha_per_cell": alpha,
        "minimum_non_tied_groups_for_any_two_sided_sign_result_to_reach_cell_alpha": n_for_familywise_minimum,
        "minimum_p_at_feasibility_floor_8_groups_if_all_non_tied_and_same_direction": 2 / (2**8),
        "results": all_results,
        "limitations": [
            "Post-hoc sensitivity analysis; it does not replace the planned paired cluster-bootstrap estimand.",
            "One seed and a failed baseline competence gate make this pilot non-confirmatory.",
            "The observed pilot has only 5-7 groups, with many zero-change groups; the sign test has very low resolution.",
            "The 15-case/8-group gate is a feasibility floor, not a power or precision target.",
        ],
    }


def render(report: dict) -> str:
    lines = [
        "# V4 pilot: exact entity-group sign sensitivity",
        "",
        "**Post-hoc exploratory diagnostic only.** The planned primary estimand remains the target-language post-minus-pre multi-hop change. This sign test is a conservative direction check at the leakage-group level, not a replacement analysis.",
        "",
        "Within each language/method cell, case-level paired changes were averaged within each independent entity group. Positive, negative, and exactly-zero group means were counted; exact-zero groups were omitted from the exact two-sided binomial sign test. Bonferroni correction is applied separately to the 12 primary cells and 12 secondary DID cells.",
        "",
        "| Family | Method | Language | Cases / groups | Positive / negative / tied groups | Exact p | Adjusted p |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for family, rows in report["results"].items():
        for row in rows:
            lines.append(
                f"| {family} | {row['method']} | {row['language'].upper()} | "
                f"{row['cases']} / {row['entity_groups']} | "
                f"{row['positive_groups']} / {row['negative_groups']} / {row['tied_groups']} | "
                f"{row['exact_two_sided_sign_p']:.4f} | "
                f"{row['bonferroni_adjusted_p_within_12_cell_family']:.4f} |"
            )
    lines += [
        "",
        f"No cell reaches the per-cell familywise threshold ({report['alpha_per_cell']:.6f}). At the design's minimum of eight groups, even a hypothetical outcome where all eight non-tied groups move in the same direction has minimum two-sided sign p = {report['minimum_p_at_feasibility_floor_8_groups_if_all_non_tied_and_same_direction']:.6f}, above that threshold. Nine non-tied groups are the mathematical minimum for this sign test to reach the threshold; that is not a sample-size or power recommendation.",
        "",
        "This reinforces that the current 15-case/8-group gate is a feasibility screen only. The 1B pilot has just 5-7 groups, many tied at zero; it cannot establish transfer or no transfer. This sensitivity remains post-hoc, one-seed, and below the competence gate.",
        "",
        f"Reproduce with `python -B {Path(__file__).name}`. Input metrics SHA-256: `{report['input_case_metrics_sha256']}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = analyse()
    (HERE / "pilot_exact_cluster_sign_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (HERE / "pilot_exact_cluster_sign_test.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "primary_cells": len(report["results"]["primary"]),
        "secondary_cells": len(report["results"]["secondary_did"]),
        "all_adjusted_p_values_one": all(
            row["bonferroni_adjusted_p_within_12_cell_family"] == 1.0
            for rows in report["results"].values() for row in rows
        ),
        "alpha_per_cell": report["alpha_per_cell"],
        "minimum_sign_test_groups_for_alpha": report["minimum_non_tied_groups_for_any_two_sided_sign_result_to_reach_cell_alpha"],
    }, indent=2))


if __name__ == "__main__":
    main()
