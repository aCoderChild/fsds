#!/usr/bin/env python3
"""Recompute multiplicity-adjusted cluster-bootstrap intervals for the v4 pilot.

This is a post-hoc sensitivity analysis of the archived exploratory run. It
does not change the primary preregistered analysis or turn the failed
competence-gate pilot into confirmatory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PILOT = REPO / "reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03"
ARCHIVE = PILOT / "r1_composition_retainfix_verified_results_20261103_v3.zip"
INDEX = PILOT / "extracted/artifact_index.json"
CASE_METRICS = PILOT / "extracted/corrected_evaluation/case_metrics.jsonl"
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


def quantile(sorted_values: list[float], probability: float) -> float:
    # Match the order-statistic convention used by the original evaluator.
    index = int(probability * (len(sorted_values) - 1))
    return sorted_values[index]


def analyse(bootstrap_replicates: int) -> dict:
    if bootstrap_replicates < 10_000:
        raise ValueError("Use at least 10,000 replicates for stable tail quantiles")
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

    results = []
    for method in METHODS:
        for language in LANGUAGES:
            by_group: dict[str, list[int]] = {}
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
                by_group.setdefault(sft_en["bootstrap_cluster_id"], []).append(value)

            n_cases = sum(map(len, by_group.values()))
            n_groups = len(by_group)
            if n_cases != EXPECTED_CASES[language] or n_groups != EXPECTED_GROUPS[language]:
                raise ValueError(f"Unexpected competence-matched coverage for {method}/{language}: "
                                 f"{n_cases} cases/{n_groups} groups")
            groups = sorted(by_group)
            values = [value for group in groups for value in by_group[group]]
            estimate = sum(values) / len(values)
            seed_material = f"{method}|{language}|forget|cluster-did".encode()
            seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
            rng = random.Random(seed)
            replicates = []
            for _ in range(bootstrap_replicates):
                sampled = [groups[rng.randrange(len(groups))] for _ in groups]
                sample_values = [value for group in sampled for value in by_group[group]]
                replicates.append(sum(sample_values) / len(sample_values))
            replicates.sort()
            # 12 two-sided comparisons: Bonferroni tail probability alpha/(2*m).
            tail = 0.05 / (2 * len(METHODS) * len(LANGUAGES))
            results.append({
                "method": method,
                "language": language,
                "n_bilingual_precompetent_cases": n_cases,
                "n_entity_groups": n_groups,
                "did_pp": 100 * estimate,
                "marginal_95ci_pp": [100 * quantile(replicates, 0.025),
                                     100 * quantile(replicates, 0.975)],
                "bonferroni_familywise_95ci_pp": [100 * quantile(replicates, tail),
                                                  100 * quantile(replicates, 1 - tail)],
                "bootstrap_seed": seed,
            })

    if len(results) != 12 or any(r["bonferroni_familywise_95ci_pp"][0] > 0
                                 or r["bonferroni_familywise_95ci_pp"][1] < 0 for r in results):
        raise ValueError("Expected all 12 multiplicity-adjusted intervals to include zero")

    return {
        "status": "VERIFIED_POST_HOC_MULTIPLICITY_SENSITIVITY_EXPLORATORY_ONLY",
        "estimand": "Target-language minus English multi-hop change on the same bilingual SFT-precompetent forget cases",
        "input_archive_sha256": sha(ARCHIVE),
        "input_case_metrics_sha256": sha(CASE_METRICS),
        "archive_index_sha256": sha(INDEX),
        "pilot_scorer_sha256": sha(PILOT / "extracted/scripts/evaluate_crosslingual_unlearning.py"),
        "analysis_script_sha256": sha(Path(__file__).resolve()),
        "bootstrap_method": "Percentile cluster bootstrap; resample entity groups with replacement and keep all member cases together",
        "bootstrap_replicates_per_contrast": bootstrap_replicates,
        "familywise_adjustment": "Bonferroni over 12 method-by-language contrasts; nominal 95% simultaneous coverage",
        "limitations": [
            "Post-hoc sensitivity analysis, not preregistered and not a replacement for the primary estimand.",
            "Each contrast has only 5-7 independent entity groups; percentile cluster-bootstrap coverage is approximate and unstable at this cluster count.",
            "All four language competence gates failed; these estimates cannot answer the hypothesis.",
            "One seed, Llama 3.2 1B, and AI-only translations without fluent-human review.",
        ],
        "results": results,
    }


def render(report: dict) -> str:
    lines = [
        "# V4 pilot: multiplicity sensitivity for cross-language contrasts",
        "",
        "**Status: post-hoc exploratory sensitivity only.** These results do not turn the failed-competence pilot into a confirmatory test.",
        "",
        "I recomputed the 12 method-by-language contrasts from the hash-verified archived case metrics and added a Bonferroni adjustment. Each interval resamples leakage-isolated entity groups, preserving all cases within each sampled group. The adjusted intervals target nominal 95% simultaneous coverage across 12 comparisons.",
        "",
        "| Method | Language | Matched cases / groups | DID (pp) | Marginal 95% cluster-bootstrap CI (pp) | Bonferroni familywise 95% CI (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        marginal = row["marginal_95ci_pp"]
        family = row["bonferroni_familywise_95ci_pp"]
        lines.append(
            f"| {row['method']} | {row['language'].upper()} | "
            f"{row['n_bilingual_precompetent_cases']}/{row['n_entity_groups']} | "
            f"{row['did_pp']:+.1f} | [{marginal[0]:+.1f}, {marginal[1]:+.1f}] | "
            f"[{family[0]:+.1f}, {family[1]:+.1f}] |"
        )
    lines += [
        "",
        "All 12 multiplicity-adjusted intervals include zero. The intervals are extremely wide: the pilot has only 5-7 independent entity groups per contrast, so effects of meaningful size in either direction remain compatible with the data. This is evidence of inadequate precision, not evidence that cross-language transfer is absent.",
        "",
        "The adjustment is nominal because cluster-bootstrap coverage is itself approximate with so few groups. No confirmatory p-values are reported. The original marginal intervals and competence-gate results remain in [`pilot_diagnostic_results.md`](pilot_diagnostic_results.md).",
        "",
        f"Reproduce with `python -B {Path(__file__).name}`. Bootstrap replicates per contrast: {report['bootstrap_replicates_per_contrast']:,}. Input case-metric SHA-256: `{report['input_case_metrics_sha256']}`. Analysis source SHA-256 is recorded in `pilot_multiplicity_sensitivity.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=100_000)
    args = parser.parse_args()
    report = analyse(args.bootstrap)
    json_path = HERE / "pilot_multiplicity_sensitivity.json"
    md_path = HERE / "pilot_multiplicity_sensitivity.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "contrasts": len(report["results"]),
        "all_familywise_intervals_include_zero": True,
        "bootstrap_replicates_per_contrast": report["bootstrap_replicates_per_contrast"],
        "json_sha256": sha(json_path),
        "markdown_sha256": sha(md_path),
    }, indent=2))


if __name__ == "__main__":
    main()
