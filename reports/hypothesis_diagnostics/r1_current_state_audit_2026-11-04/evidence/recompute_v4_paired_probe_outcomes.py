#!/usr/bin/env python3
"""Recompute paired intermediate-probe and final-answer outcomes from v4 scores."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO / "reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/extracted"
INPUT = EXTRACTED / "corrected_evaluation/case_metrics.jsonl"
INDEX = EXTRACTED / "artifact_index.json"
OUTPUT = AUDIT / "evidence/v4_paired_probe_outcomes_2026-11-04.json"
RESCORE_RECEIPT = AUDIT / "evidence/v4_corrected_independent_rescore_2026-11-04.json"
EXPECTED_N = {"fr": (10, 6), "de": (9, 5), "hi": (9, 6), "th": (11, 7)}
METHODS = ("GA", "GD", "NPO")
LANGUAGES = ("fr", "de", "hi", "th")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction(rows: list[dict], field: str) -> dict:
    correct = sum(row[field] is True for row in rows)
    return {"correct": correct, "total": len(rows), "percent": round(100 * correct / len(rows), 1)}


def main() -> None:
    rescore = json.loads(RESCORE_RECEIPT.read_text(encoding="utf-8"))
    if (rescore.get("status") != "PASS_INDEPENDENT_SCORE_REPRODUCTION_EXPLORATORY_ONLY"
            or rescore.get("all_seven_outputs_byte_identical") is not True):
        raise ValueError("The archived independent rescore receipt does not pass")
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    indexed = {row["file"]: row for row in index["files"]}
    relpath = "corrected_evaluation/case_metrics.jsonl"
    expected = indexed[relpath]
    if INPUT.stat().st_size != expected["bytes"] or sha(INPUT) != expected["sha256"]:
        raise ValueError("case_metrics.jsonl does not match the checked archive extraction index")

    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {(row["model"], row["instance_id"], row["language"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate model/case/language score keys")

    output_rows = []
    for method in METHODS:
        for lang in LANGUAGES:
            eligible = []
            for (model, case_id, language), en_base in by_key.items():
                if model != "sft" or language != "en" or en_base["split"] != "forget":
                    continue
                target_base = by_key.get(("sft", case_id, lang))
                if (target_base and en_base["direct_correct"] is True and en_base["multi_correct"] is True
                        and target_base["direct_correct"] is True and target_base["multi_correct"] is True):
                    eligible.append(case_id)
            eligible.sort()
            groups = {by_key[("sft", case_id, "en")]["bootstrap_cluster_id"] for case_id in eligible}
            if (len(eligible), len(groups)) != EXPECTED_N[lang]:
                raise ValueError(f"Unexpected precompetent matched cohort for {lang}: {len(eligible)}/{len(groups)}")
            english = [by_key[(method, case_id, "en")] for case_id in eligible]
            target = [by_key[(method, case_id, lang)] for case_id in eligible]
            for row in english + target:
                for failed_field, invalid_field in (("direct_generation_failed", "direct_output_invalid"),
                                                     ("multi_generation_failed", "multi_output_invalid")):
                    if row.get(failed_field) is True or row.get(invalid_field) is True:
                        raise ValueError(f"Invalid or failed prediction in paired scored subset: {method}/{lang}")
            b_fail_c_pass = sum(row["direct_correct"] is False and row["multi_correct"] is True for row in target)
            output_rows.append({
                "method": method,
                "target_language": lang,
                "matched_precompetent_cases": len(eligible),
                "entity_groups": len(groups),
                "english_intermediate_direct_accuracy": fraction(english, "direct_correct"),
                "target_intermediate_direct_accuracy": fraction(target, "direct_correct"),
                "english_final_multihop_accuracy": fraction(english, "multi_correct"),
                "target_final_multihop_accuracy": fraction(target, "multi_correct"),
                "target_intermediate_wrong_final_correct": {
                    "cases": b_fail_c_pass,
                    "total": len(target),
                    "percent": round(100 * b_fail_c_pass / len(target), 1),
                },
                "paired_case_ids": eligible,
            })

    result = {
        "status": "PASS_RECOMPUTED_EXPLORATORY_PAIRED_OUTCOMES",
        "scope": "Historical v4 one-seed Llama 3.2 1B run; descriptive diagnostic, not confirmatory R1 evidence.",
        "input": {
            "path": str(INPUT.relative_to(REPO)),
            "sha256": sha(INPUT),
            "bytes": INPUT.stat().st_size,
            "archive_index_path": str(INDEX.relative_to(REPO)),
            "archive_index_entry_sha256": expected["sha256"],
            "run_bundle": rescore["run_bundle"],
            "run_bundle_sha256": rescore["run_bundle_sha256"],
            "independent_rescore_receipt": str(RESCORE_RECEIPT.relative_to(REPO)),
        },
        "cohort_definition": "Forget cases where the archived SFT checkpoint answered both direct intermediate-fact and final multi-hop probes correctly in English and in the target language.",
        "outcomes": output_rows,
        "interpretation_limit": "Direct-probe failure with a correct final answer shows only that the measured final answer remained correct; it cannot distinguish alternate associations, cueing, or probe limitations.",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(output_rows), "sha256": sha(OUTPUT), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
