#!/usr/bin/env python3
"""Enumerate the finite cluster-bootstrap distributions for the v4 R1 pilot.

Post-hoc diagnostic only. This removes Monte Carlo resampling noise from the
small-cluster bootstrap intervals; it does not repair the failed competence
gate or make the one-seed pilot confirmatory.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PILOT = REPO / "reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03"
ARCHIVE = PILOT / "r1_composition_retainfix_verified_results_20261103_v3.zip"
INDEX = PILOT / "extracted/artifact_index.json"
CASE_METRICS = PILOT / "extracted/corrected_evaluation/case_metrics.jsonl"
MONTE_CARLO = HERE / "pilot_multiplicity_sensitivity.json"
EXPECTED_ARCHIVE_SHA256 = "48b800312c8c43dee14fdf8d0384a2153b4317c253d7de1228d4c81f2697ab43"
EXPECTED_CASE_METRICS_SHA256 = "786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9"
METHODS = ("GA", "GD", "NPO")
LANGUAGES = ("fr", "de", "hi", "th")
EXPECTED_CASES = {"fr": 10, "de": 9, "hi": 9, "th": 11}
EXPECTED_GROUPS = {"fr": 6, "de": 5, "hi": 6, "th": 7}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compositions(total: int, parts: int):
    """Yield all ordered nonnegative integer count vectors with this sum."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first, *rest)


def weighted_quantile(values_and_weights: list[tuple[float, int]], probability: float, total: int) -> float:
    """Match floor(p * (N-1)) from the original evaluator over all N draws."""
    order_index = int(probability * (total - 1))
    cumulative = 0
    for value, weight in sorted(values_and_weights):
        cumulative += weight
        if cumulative > order_index:
            return value
    raise AssertionError("Weighted bootstrap support does not sum to the sample space")


def exact_interval(by_group: dict[str, list[int]], lower: float, upper: float) -> tuple[list[float], int, int]:
    groups = sorted(by_group)
    k = len(groups)
    group_sums = [sum(by_group[group]) for group in groups]
    group_sizes = [len(by_group[group]) for group in groups]
    weighted_values = []
    for counts in compositions(k, k):
        resample_n = sum(count * size for count, size in zip(counts, group_sizes))
        estimate = sum(count * total for count, total in zip(counts, group_sums)) / resample_n
        weight = math.factorial(k) // math.prod(math.factorial(count) for count in counts)
        weighted_values.append((estimate, weight))
    total_resamples = k**k
    if sum(weight for _, weight in weighted_values) != total_resamples:
        raise AssertionError("Multiplicity weights do not enumerate every cluster resample")
    return ([100 * weighted_quantile(weighted_values, lower, total_resamples),
             100 * weighted_quantile(weighted_values, upper, total_resamples)],
            total_resamples, len(weighted_values))


def analyse() -> dict:
    if sha(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("The authoritative pilot archive hash changed")
    if sha(CASE_METRICS) != EXPECTED_CASE_METRICS_SHA256:
        raise ValueError("The archived case-metrics input hash changed")
    indexed = json.loads(INDEX.read_text(encoding="utf-8"))
    indexed_file = next((row for row in indexed["files"]
                         if row["file"] == "corrected_evaluation/case_metrics.jsonl"), None)
    if not indexed_file or indexed_file["sha256"] != sha(CASE_METRICS):
        raise ValueError("Case metrics do not match the archive's internal SHA-256 index")

    rows = read_jsonl(CASE_METRICS)
    expected_models = {"pretrained", "sft", "retain_only", *METHODS}
    if len(rows) != 3_870 or {row.get("model") for row in rows} != expected_models:
        raise ValueError("Unexpected pilot case-metric coverage")
    lookup = {}
    for row in rows:
        key = (row["model"], row["instance_id"], row["language"])
        if key in lookup:
            raise ValueError(f"Duplicate case metric: {key}")
        lookup[key] = row
    sft_ids = {row["instance_id"] for row in rows if row["model"] == "sft" and row["language"] == "en"}
    if len(sft_ids) != 129:
        raise ValueError("SFT baseline must cover exactly 129 evaluation cases")

    monte_carlo = json.loads(MONTE_CARLO.read_text(encoding="utf-8"))
    monte_carlo_lookup = {(row["method"], row["language"]): row for row in monte_carlo["results"]}
    results = []
    for method in METHODS:
        for language in LANGUAGES:
            by_group: dict[str, list[int]] = {}
            primary_by_group: dict[str, list[int]] = {}
            for instance_id in sorted(sft_ids):
                sft_en = lookup[("sft", instance_id, "en")]
                if sft_en["split"] != "forget":
                    continue
                sft_target = lookup[("sft", instance_id, language)]
                post_en = lookup[(method, instance_id, "en")]
                post_target = lookup[(method, instance_id, language)]
                if not all((sft_en["direct_correct"], sft_en["multi_correct"],
                            sft_target["direct_correct"], sft_target["multi_correct"])):
                    continue
                if len({sft_en["bootstrap_cluster_id"], sft_target["bootstrap_cluster_id"]}) != 1:
                    raise ValueError(f"Inconsistent entity group for {instance_id}/{language}")
                value = ((int(post_target["multi_correct"]) - int(sft_target["multi_correct"]))
                         - (int(post_en["multi_correct"]) - int(sft_en["multi_correct"])))
                group_id = sft_en["bootstrap_cluster_id"]
                by_group.setdefault(group_id, []).append(value)
                primary_value = int(post_target["multi_correct"]) - int(sft_target["multi_correct"])
                primary_by_group.setdefault(group_id, []).append(primary_value)

            n_cases = sum(map(len, by_group.values()))
            n_groups = len(by_group)
            if n_cases != EXPECTED_CASES[language] or n_groups != EXPECTED_GROUPS[language]:
                raise ValueError(f"Unexpected competence-matched coverage for {method}/{language}: "
                                 f"{n_cases} cases/{n_groups} groups")
            n = sum(map(len, by_group.values()))
            estimate = sum(sum(values) for values in by_group.values()) / n
            marginal, sample_space, compositions_count = exact_interval(by_group, 0.025, 0.975)
            simultaneous_tail = 0.05 / (2 * len(METHODS) * len(LANGUAGES))
            familywise, family_space, family_compositions = exact_interval(
                by_group, simultaneous_tail, 1 - simultaneous_tail
            )
            primary_n = sum(map(len, primary_by_group.values()))
            primary_estimate = sum(sum(values) for values in primary_by_group.values()) / primary_n
            primary_marginal, primary_space, primary_compositions = exact_interval(
                primary_by_group, 0.025, 0.975
            )
            primary_familywise, primary_family_space, primary_family_compositions = exact_interval(
                primary_by_group, simultaneous_tail, 1 - simultaneous_tail
            )
            previous = monte_carlo_lookup[(method, language)]
            results.append({
                "method": method,
                "language": language,
                "n_bilingual_precompetent_cases": n_cases,
                "n_entity_groups": n_groups,
                "primary_target_change_pp": 100 * primary_estimate,
                "primary_exact_marginal_95ci_pp": primary_marginal,
                "primary_exact_bonferroni_familywise_95ci_pp": primary_familywise,
                "did_pp": 100 * estimate,
                "exact_marginal_95ci_pp": marginal,
                "exact_bonferroni_familywise_95ci_pp": familywise,
                "cluster_resample_sequences": sample_space,
                "distinct_cluster_multiplicity_vectors": compositions_count,
                "monte_carlo_100k_marginal_95ci_pp": previous["marginal_95ci_pp"],
                "monte_carlo_100k_bonferroni_95ci_pp": previous["bonferroni_familywise_95ci_pp"],
                "exact_vs_monte_carlo_marginal_max_abs_endpoint_difference_pp": round(
                    max(abs(a - b) for a, b in zip(marginal, previous["marginal_95ci_pp"])), 10
                ),
                "exact_vs_monte_carlo_familywise_max_abs_endpoint_difference_pp": round(
                    max(abs(a - b) for a, b in zip(familywise, previous["bonferroni_familywise_95ci_pp"])), 10
                ),
            })
            if (family_space != sample_space or family_compositions != compositions_count
                    or primary_space != sample_space or primary_family_space != sample_space
                    or primary_compositions != compositions_count or primary_family_compositions != compositions_count):
                raise AssertionError("Marginal and familywise intervals used different bootstrap support")

    if len(results) != 12:
        raise ValueError("Expected 12 method-language cells")
    for row in results:
        for key in ("primary_exact_bonferroni_familywise_95ci_pp", "exact_bonferroni_familywise_95ci_pp"):
            interval = row[key]
            if interval[0] > 0 or interval[1] < 0:
                raise ValueError(f"Expected all observed adjusted intervals to include zero: {row['method']}/{row['language']} {key}")
    return {
        "status": "VERIFIED_EXACT_POST_HOC_CLUSTER_BOOTSTRAP_SENSITIVITY_EXPLORATORY_ONLY",
        "primary_estimand": "Target-language post-minus-pre multi-hop exact-match change on the same bilingual SFT-precompetent forget cases",
        "secondary_estimand": "Target-language minus English multi-hop change on the same bilingual SFT-precompetent forget cases",
        "estimand": "Target-language minus English multi-hop change on the same bilingual SFT-precompetent forget cases",
        "bootstrap_method": "Exact enumeration of all ordered cluster draws with replacement; each selected cluster contributes all its member cases",
        "quantile_convention": "Order statistic at floor(p * (N - 1)), matching the original scorer",
        "familywise_adjustment": "Bonferroni over 12 method-by-language contrasts; nominal 95% simultaneous coverage",
        "input_archive_sha256": sha(ARCHIVE),
        "input_case_metrics_sha256": sha(CASE_METRICS),
        "archive_index_sha256": sha(INDEX),
        "monte_carlo_sensitivity_sha256": sha(MONTE_CARLO),
        "analysis_script_sha256": sha(Path(__file__).resolve()),
        "limitations": [
            "Post-hoc sensitivity analysis, not preregistered and not a replacement for the primary estimand.",
            "Only 5-7 independent entity groups per contrast; exact enumeration removes Monte Carlo error, not small-sample uncertainty.",
            "All four target-language competence gates failed; these estimates cannot answer the hypothesis.",
            "One seed, Llama 3.2 1B, and AI-only translations without fluent-human review.",
        ],
        "results": results,
    }


def render(report: dict) -> str:
    lines = [
        "# V4 pilot: exact cluster-bootstrap sensitivity",
        "",
        "**Status: post-hoc exploratory sensitivity only.** Exact resampling removes Monte Carlo noise from the intervals; it does not turn the failed-competence, one-seed pilot into confirmatory evidence.",
        "",
        "For each method-language contrast, this enumerates every ordered bootstrap draw of the 5-7 leakage-isolated entity groups. Each draw keeps all cases in a selected group together. The weighted quantiles use the same order-statistic convention as the primary scorer. The Bonferroni intervals cover the 12 method-by-language contrasts at nominal familywise 95%.",
        "",
        "| Method | Language | Cases / groups | Primary target change (pp; familywise 95% CI) | Secondary DID (pp; familywise 95% CI) | Resample sequences |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        family = row["exact_bonferroni_familywise_95ci_pp"]
        primary = row["primary_exact_bonferroni_familywise_95ci_pp"]
        lines.append(
            f"| {row['method']} | {row['language'].upper()} | "
            f"{row['n_bilingual_precompetent_cases']}/{row['n_entity_groups']} | "
            f"{row['primary_target_change_pp']:+.1f} [{primary[0]:+.1f}, {primary[1]:+.1f}] | "
            f"{row['did_pp']:+.1f} [{family[0]:+.1f}, {family[1]:+.1f}] | "
            f"{row['cluster_resample_sequences']:,} |"
        )
    marginal_diffs = [r["exact_vs_monte_carlo_marginal_max_abs_endpoint_difference_pp"] for r in report["results"]]
    family_diffs = [r["exact_vs_monte_carlo_familywise_max_abs_endpoint_difference_pp"] for r in report["results"]]
    lines += [
        "",
        f"The exact familywise DID intervals include zero in {sum(r['exact_bonferroni_familywise_95ci_pp'][0] <= 0 <= r['exact_bonferroni_familywise_95ci_pp'][1] for r in report['results'])}/12 cells; the primary target-change intervals include zero in {sum(r['primary_exact_bonferroni_familywise_95ci_pp'][0] <= 0 <= r['primary_exact_bonferroni_familywise_95ci_pp'][1] for r in report['results'])}/12 cells. Against the earlier 100,000-draw DID simulation, the largest endpoint difference is {max(marginal_diffs):.1f} pp for marginal intervals and {max(family_diffs):.1f} pp for familywise intervals. Intervals remain wide because only 5-7 independent groups support each contrast.",
        "",
        "These intervals quantify uncertainty in an exploratory pilot whose baseline competence gate failed in every target language. They do not establish no transfer and do not replace the required 8B, three-seed study.",
        "",
        f"Reproduce with `python -B {Path(__file__).name}`. Inputs are bound to archive SHA-256 `{report['input_archive_sha256']}` and case-metric SHA-256 `{report['input_case_metrics_sha256']}`. The script enumerates finite cluster resamples exactly; no random seed or Monte Carlo replicate count is involved.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = analyse()
    json_path = HERE / "pilot_exact_cluster_sensitivity.json"
    md_path = HERE / "pilot_exact_cluster_sensitivity.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "contrasts": len(report["results"]),
        "all_exact_familywise_intervals_include_zero": True,
        "primary_familywise_intervals_including_zero": sum(
            row["primary_exact_bonferroni_familywise_95ci_pp"][0] <= 0 <= row["primary_exact_bonferroni_familywise_95ci_pp"][1]
            for row in report["results"]
        ),
        "secondary_did_familywise_intervals_including_zero": sum(
            row["exact_bonferroni_familywise_95ci_pp"][0] <= 0 <= row["exact_bonferroni_familywise_95ci_pp"][1]
            for row in report["results"]
        ),
        "max_marginal_endpoint_difference_from_100k_mc_pp": max(
            row["exact_vs_monte_carlo_marginal_max_abs_endpoint_difference_pp"] for row in report["results"]
        ),
        "max_familywise_endpoint_difference_from_100k_mc_pp": max(
            row["exact_vs_monte_carlo_familywise_max_abs_endpoint_difference_pp"] for row in report["results"]
        ),
        "json_sha256": sha(json_path),
        "markdown_sha256": sha(md_path),
    }, indent=2))


if __name__ == "__main__":
    main()
