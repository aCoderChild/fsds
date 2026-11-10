#!/usr/bin/env python3
"""Independently recompute saved TOFU ROUGE-L and nine-metric utility values."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from rouge_score import rouge_scorer


SPLIT_COUNTS = {
    "forget10": 400,
    "retain90": 400,
    "holdout10": 400,
    "real_authors": 100,
    "world_facts": 117,
}
UTILITY_COMPONENTS = {
    "retain_probability": ("retain90", "answer_probability"),
    "retain_rougeL_recall": ("retain90", "rougeL_recall"),
    "retain_truth_ratio_score": ("retain90", "truth_ratio_score"),
    "real_authors_normalized_option_probability": ("real_authors", "normalized_option_probability"),
    "real_authors_rougeL_recall": ("real_authors", "rougeL_recall"),
    "real_authors_truth_ratio_score": ("real_authors", "truth_ratio_score"),
    "world_facts_normalized_option_probability": ("world_facts", "normalized_option_probability"),
    "world_facts_rougeL_recall": ("world_facts", "rougeL_recall"),
    "world_facts_truth_ratio_score": ("world_facts", "truth_ratio_score"),
}
TOLERANCE = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float) -> bool:
    return math.isfinite(actual) and math.isfinite(expected) and math.isclose(
        actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE
    )


def mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field}: missing or non-finite per-example values")
    return sum(values) / len(values)


def audit_result(label: str, result: dict, problems: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    examples = result.get("per_example", {})
    if set(examples) != set(SPLIT_COUNTS):
        problems.append(f"{label}: evaluated split keys differ from expected TOFU splits")
        return {}

    checked_rows = 0
    for split, expected_count in SPLIT_COUNTS.items():
        rows = examples[split]
        if len(rows) != expected_count:
            problems.append(f"{label}/{split}: expected {expected_count} rows, found {len(rows)}")
        indices = [row.get("source_row_index") for row in rows]
        if indices != list(range(expected_count)):
            problems.append(f"{label}/{split}: source row indices do not cover the pinned ordered rows")
        for index, row in enumerate(rows):
            generation = row.get("generation")
            reference = row.get("reference_answer")
            if not isinstance(generation, str) or not isinstance(reference, str):
                problems.append(f"{label}/{split}[{index}]: generated text/reference missing")
                continue
            expected = scorer.score(reference, generation)["rougeL"].recall
            try:
                actual = float(row["rougeL_recall"])
            except (KeyError, TypeError, ValueError):
                problems.append(f"{label}/{split}[{index}]: ROUGE-L recall missing or invalid")
                continue
            if not close(actual, expected):
                problems.append(f"{label}/{split}[{index}]: ROUGE-L recall does not match saved text")
            checked_rows += 1

    utility = result.get("model_utility")
    if not isinstance(utility, dict):
        problems.append(f"{label}: model-utility receipt missing")
        return {"rouge_rows_recomputed": checked_rows}
    components = {}
    try:
        for name, (split, field) in UTILITY_COMPONENTS.items():
            components[name] = mean(examples[split], field)
            recorded = float(utility["components"][name])
            if not close(recorded, components[name]):
                problems.append(f"{label}: model-utility component {name} does not reproduce")
        if set(utility.get("components", {})) != set(UTILITY_COMPONENTS):
            problems.append(f"{label}: model-utility component names differ from the locked nine metrics")
        values = list(components.values())
        expected_harmonic_mean = 0.0 if any(value <= 0 for value in values) else len(values) / sum(1 / value for value in values)
        if not close(float(utility["harmonic_mean"]), expected_harmonic_mean):
            problems.append(f"{label}: model-utility harmonic mean does not reproduce")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        problems.append(f"{label}: could not recompute model utility: {exc}")
        expected_harmonic_mean = None

    return {
        "rouge_rows_recomputed": checked_rows,
        "model_utility_components": components,
        "model_utility_harmonic_mean_recomputed": expected_harmonic_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Full-checkpoint official-evaluation JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="NPO adapter official-evaluation JSON")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    problems: list[str] = []

    expected_hashes = protocol["dataset"]["evaluation_file_sha256"]
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        evaluation = result.get("evaluation", {})
        if evaluation.get("dataset_revision") != protocol["dataset"]["revision"]:
            problems.append(f"{label}: dataset revision differs from protocol")
        if evaluation.get("dataset_sha256_by_file") != expected_hashes:
            problems.append(f"{label}: dataset hashes differ from protocol")

    base_rows = baseline.get("per_example", {})
    candidate_rows = candidate.get("per_example", {})
    if set(base_rows) == set(SPLIT_COUNTS) and set(candidate_rows) == set(SPLIT_COUNTS):
        for split in SPLIT_COUNTS:
            if len(base_rows[split]) != len(candidate_rows[split]):
                problems.append(f"{split}: baseline/candidate row counts differ")
                continue
            for index, (base_row, candidate_row) in enumerate(zip(base_rows[split], candidate_rows[split])):
                for field in ("source_row_index", "profile_block", "question_sha256", "source_answer"):
                    if base_row.get(field) != candidate_row.get(field):
                        problems.append(f"{split}[{index}]: baseline/candidate row mismatch in {field}")

    metrics = {
        "baseline": audit_result("baseline", baseline, problems),
        "candidate": audit_result("candidate", candidate, problems),
    }
    output = {
        "overall_status": "PASS" if not problems else "FAIL",
        "scope": "Independent ROUGE-L recomputation from saved generations and independent recomputation of nine utility components/harmonic mean.",
        "baseline_file": str(args.baseline),
        "baseline_sha256": sha256(args.baseline),
        "candidate_file": str(args.candidate),
        "candidate_sha256": sha256(args.candidate),
        "protocol_sha256": sha256(args.protocol),
        "metrics": metrics,
        "problems": problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
