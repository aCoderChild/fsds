#!/usr/bin/env python3
"""Compute a retain-adjusted cross-language difference-in-differences.

This is a secondary diagnostic for the SynthWorlds-SM v4 exploratory run.
For each target language, it computes

    D_split = (post - SFT)_language - (post - SFT)_English
    adjusted = D_forget - D_retain

within cases that were direct- and multi-hop-correct in both languages after
SFT. Entity groups are the resampling units; forget and retain groups are
resampled independently. The evaluator's original primary contrast remains
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


LANGUAGES = ("fr", "de", "hi", "th")
METHODS = ("GA", "GD", "NPO")
SPLITS = ("forget", "retain")
BASELINE = "sft"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def percentile(sorted_values: list[float], probability: float) -> float:
    # Match the linear interpolation definition used by numpy's default method.
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def cluster_mean(rows: list[dict[str, Any]], chosen_clusters: list[str]) -> float:
    members: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        members[row["cluster"]].append(row["did"])
    sampled = [value for cluster in chosen_clusters for value in members[cluster]]
    return sum(sampled) / len(sampled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Use at least 1,000 bootstrap draws for this report")

    case_rows = read_jsonl(args.case_metrics)
    primary_path = args.case_metrics.parent / "cross_language_contrasts.csv"
    if not primary_path.is_file():
        raise FileNotFoundError(f"Original evaluator contrast CSV is required: {primary_path}")
    primary_rows = read_csv(primary_path)
    primary_index = {}
    for row in primary_rows:
        key = (row["model"], row["split"], row["language"])
        if key in primary_index:
            raise ValueError(f"Duplicate original evaluator contrast key: {key}")
        primary_index[key] = row
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in case_rows:
        key = (row["model"], row["instance_id"], row["language"])
        if key in index:
            raise ValueError(f"Duplicate case metric key: {key}")
        index[key] = row
        if row.get("split") not in {"forget", "retain", "holdout"}:
            raise ValueError(f"Unexpected split for {key}: {row.get('split')}")

    models = {row["model"] for row in case_rows}
    if not {BASELINE, *METHODS}.issubset(models):
        raise ValueError(f"Missing required model rows; found {sorted(models)}")

    results: list[dict[str, Any]] = []
    for method in METHODS:
        for language in LANGUAGES:
            by_split: dict[str, list[dict[str, Any]]] = {}
            for split in SPLITS:
                paired = []
                seen_instances = set()
                for sft_en in case_rows:
                    if (sft_en["model"] != BASELINE or sft_en["language"] != "en"
                            or sft_en["split"] != split):
                        continue
                    iid = sft_en["instance_id"]
                    if iid in seen_instances:
                        raise ValueError(f"Duplicate SFT English row for {(iid, split)}")
                    seen_instances.add(iid)
                    sft_lang = index.get((BASELINE, iid, language))
                    post_en = index.get((method, iid, "en"))
                    post_lang = index.get((method, iid, language))
                    if not all((sft_lang, post_en, post_lang)):
                        continue
                    if any(r["split"] != split for r in (sft_lang, post_en, post_lang)):
                        raise ValueError(f"Split mismatch for {iid}, {language}")
                    if any(r["bootstrap_cluster_id"] != sft_en["bootstrap_cluster_id"]
                           for r in (sft_lang, post_en, post_lang)):
                        raise ValueError(f"Entity-group mismatch for {iid}, {language}")
                    # Match the evaluator's bilingual precompetence gate.
                    if not all((sft_en["direct_correct"], sft_en["multi_correct"],
                                sft_lang["direct_correct"], sft_lang["multi_correct"])):
                        continue
                    english_effect = int(post_en["multi_correct"]) - int(sft_en["multi_correct"])
                    language_effect = int(post_lang["multi_correct"]) - int(sft_lang["multi_correct"])
                    paired.append({
                        "instance_id": iid,
                        "cluster": sft_en["bootstrap_cluster_id"],
                        "did": language_effect - english_effect,
                    })
                if not paired:
                    raise ValueError(f"No bilingual-precompetent cases for {method}/{language}/{split}")
                by_split[split] = paired

            forget = by_split["forget"]
            retain = by_split["retain"]
            forget_clusters = sorted({r["cluster"] for r in forget})
            retain_clusters = sorted({r["cluster"] for r in retain})
            if set(forget_clusters) & set(retain_clusters):
                raise ValueError(f"Leakage group crosses forget and retain for {method}/{language}")
            point_forget = sum(r["did"] for r in forget) / len(forget)
            point_retain = sum(r["did"] for r in retain) / len(retain)
            point_adjusted = point_forget - point_retain

            # Cross-check both component estimates and cohort sizes against
            # the original evaluator's independently emitted contrast table.
            for split, paired, point in (
                ("forget", forget, point_forget),
                ("retain", retain, point_retain),
            ):
                primary = primary_index.get((method, split, language))
                if primary is None:
                    raise ValueError(f"Missing original evaluator contrast for {method}/{split}/{language}")
                if int(primary["n_bilingual_precompetent_cases"]) != len(paired):
                    raise ValueError(f"Precompetent case count disagrees for {method}/{split}/{language}")
                if abs(float(primary["language_minus_english_did_pp"]) - 100 * point) > 1e-9:
                    raise ValueError(f"Primary contrast disagrees for {method}/{split}/{language}")

            seed_material = f"retain-adjusted|{method}|{language}|{sha256_file(args.case_metrics)}"
            seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            estimates = []
            forget_members = {c: [r for r in forget if r["cluster"] == c] for c in forget_clusters}
            retain_members = {c: [r for r in retain if r["cluster"] == c] for c in retain_clusters}
            for _ in range(args.bootstrap):
                sampled_forget = [forget_clusters[rng.randrange(len(forget_clusters))]
                                  for _ in forget_clusters]
                sampled_retain = [retain_clusters[rng.randrange(len(retain_clusters))]
                                  for _ in retain_clusters]
                d_forget = cluster_mean(forget, sampled_forget)
                d_retain = cluster_mean(retain, sampled_retain)
                estimates.append(d_forget - d_retain)
            estimates.sort()

            results.append({
                "method": method,
                "language": language,
                "forget_cases": len(forget),
                "forget_groups": len(forget_clusters),
                "retain_cases": len(retain),
                "retain_groups": len(retain_clusters),
                "D_forget_language_minus_english_pp": 100 * point_forget,
                "D_retain_language_minus_english_pp": 100 * point_retain,
                "retain_adjusted_DDD_pp": 100 * point_adjusted,
                "cluster_bootstrap_95ci_low_pp": 100 * percentile(estimates, 0.025),
                "cluster_bootstrap_95ci_high_pp": 100 * percentile(estimates, 0.975),
                "bootstrap_draws": args.bootstrap,
                "seed": seed,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "retain_adjusted_cross_language_contrasts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    report_path = args.output_dir / "retain_adjusted_diagnostic.md"
    lines = [
        "# Retain-adjusted cross-language diagnostic",
        "",
        "This is a secondary, exploratory analysis of the corrected SynthWorlds-SM v4 pilot. It does not replace the report's preregistered primary contrast.",
        "",
        "For each split, D = (post-unlearning minus SFT multi-hop correctness in the target language) minus the same change in English. The adjusted estimate is D(forget) minus D(retain). A negative adjusted estimate means the target-language change is more negative on forget than expected from its language-specific change on retain. Cases are restricted to those SFT-precompetent in English and the target language (both direct and multi-hop probes correct). Entity groups are resampled independently within forget and retain; intervals are percentile cluster-bootstrap intervals and are not multiplicity-adjusted.",
        "",
        "| Method | Language | Forget cases/groups | Retain cases/groups | D forget (pp) | D retain (pp) | Adjusted estimate (95% CI, pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        ci = f"[{row['cluster_bootstrap_95ci_low_pp']:.1f}, {row['cluster_bootstrap_95ci_high_pp']:.1f}]"
        lines.append(
            f"| {row['method']} | {row['language']} | "
            f"{row['forget_cases']}/{row['forget_groups']} | {row['retain_cases']}/{row['retain_groups']} | "
            f"{row['D_forget_language_minus_english_pp']:+.1f} | "
            f"{row['D_retain_language_minus_english_pp']:+.1f} | "
            f"{row['retain_adjusted_DDD_pp']:+.1f} ({ci}) |"
        )
    lines += [
        "",
        "## Provenance and limits",
        "",
        f"Input: `{args.case_metrics}` (SHA-256 `{sha256_file(args.case_metrics)}`). Original primary contrasts: `{primary_path}` (SHA-256 `{sha256_file(primary_path)}`); all 24 component counts and point estimates reproduced exactly. This analysis reuses the evaluator's case-level correctness labels and bilingual precompetence rule. It is based on one seed, small eligible cohorts, AI-only-reviewed translations, and a single 1B model. The adjustment is exploratory and cannot establish causal language transfer by itself.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "status": "COMPLETED_EXPLORATORY_SECONDARY_ANALYSIS",
        "input_case_metrics": str(args.case_metrics),
        "input_sha256": sha256_file(args.case_metrics),
        "primary_evaluator_contrasts": str(primary_path),
        "primary_evaluator_contrasts_sha256": sha256_file(primary_path),
        "primary_evaluator_crosscheck": "PASS: all 24 component case counts and point estimates reproduced",
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": [csv_path.name, report_path.name],
        "output_sha256": {
            csv_path.name: sha256_file(csv_path),
            report_path.name: sha256_file(report_path),
        },
        "bootstrap_draws": args.bootstrap,
        "primary_evaluation_unchanged": True,
        "resampling": "independent entity-group bootstrap within forget and retain splits",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
