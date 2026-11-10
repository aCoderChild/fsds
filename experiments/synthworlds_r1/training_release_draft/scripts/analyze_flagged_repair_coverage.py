#!/usr/bin/env python3
"""Compute an optimistic complete-case coverage bound after repair screening.

This is a diagnostic only. It combines first-stage translation passes with
Qwen-passed repair candidates, but does not claim release eligibility: repaired
rows have not passed the frozen round-trip/14B release review.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
TRAINING = SCRIPT.parents[1]
V51 = SCRIPT.parents[2]
RUN_DIR = TRAINING / "translation_repair_diagnostics" / "translategemma4b_qwen7b_all_flagged_2026-11-03"
LANGS = ("fr", "de", "hi", "th")
FIRST_PASS = "AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS"
REPAIR_PASS = "DIAGNOSTIC_QWEN_PASS"

sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(V51 / "scripts"))
from repair_all_flagged_training_translations import (  # noqa: E402
    file_sha,
    select_flagged_rows,
    sha_text,
    source_manifest,
)
from review_training_translations_qwen import read_jsonl  # noqa: E402


def atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def analyze() -> dict:
    _, source_rows = source_manifest()
    selected = select_flagged_rows(source_rows)
    parent_report_path = RUN_DIR / "report.json"
    decisions_path = RUN_DIR / "repair_candidates.jsonl"
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    decisions = read_jsonl(decisions_path)
    if (parent_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or parent_report.get("processed_rows") != len(selected)
            or len(decisions) != len(selected)
            or file_sha(decisions_path) != parent_report.get("decisions_sha256")):
        raise ValueError("The parent repair diagnostic is incomplete or its output hash differs")
    expected = [(x["language"], x["row"]["prompt_id"], x["row"]["kind"]) for x in selected]
    actual = [(x.get("language"), x.get("prompt_id"), x.get("kind")) for x in decisions]
    if actual != expected:
        raise ValueError("The repair diagnostic rows are not the complete stable selected cohort")

    source_by_key = {
        (lang, row["prompt_id"], row["kind"]): row
        for lang, rows in source_rows.items()
        for row in rows
    }
    for decision in decisions:
        key = (decision["language"], decision["prompt_id"], decision["kind"])
        source = source_by_key[key]
        if (decision.get("source_en_sha256") != sha_text(source["text_en"])
                or decision.get("old_translation_sha256") != sha_text(source["text_translation"])):
            raise ValueError(f"Repair candidate source hash mismatch: {key}")
        if (decision.get("new_translation")
                and decision.get("new_translation_sha256") != sha_text(decision["new_translation"])):
            raise ValueError(f"Repair candidate output hash mismatch: {key}")

    repairs_by_key = {
        (row["language"], row["prompt_id"], row["kind"]): row
        for row in decisions
    }
    first_report = TRAINING / "translation_reviews_roundtrip_qwen" / "report.json"
    first_report_data = json.loads(first_report.read_text(encoding="utf-8"))
    passing_by_language: dict[str, set[tuple[str, str]]] = {lang: set() for lang in LANGS}
    row_counts = {}
    for lang in LANGS:
        path = TRAINING / "translation_reviews_roundtrip_qwen" / f"{lang}.jsonl"
        first_rows = read_jsonl(path)
        detail = first_report_data.get("languages", {}).get(lang, {})
        if detail.get("review_sha256") != file_sha(path):
            raise ValueError(f"First-stage source review hash mismatch: {lang}")
        if len(first_rows) != 488:
            raise ValueError(f"Expected 488 first-stage rows in {lang}, got {len(first_rows)}")
        for row in first_rows:
            key = (row["prompt_id"], row["kind"])
            repair = repairs_by_key.get((lang, *key))
            first_pass = row.get("review_status") == FIRST_PASS
            repair_pass = repair is not None and repair.get("status") == REPAIR_PASS
            if first_pass or repair_pass:
                passing_by_language[lang].add(key)
        row_counts[lang] = {
            "source_rows": len(first_rows),
            "first_stage_pass": detail.get("review_status_counts", {}).get(FIRST_PASS, 0),
            "repair_qwen_pass": sum(
                row["language"] == lang and row.get("status") == REPAIR_PASS for row in decisions
            ),
            "optimistic_effective_pass": len(passing_by_language[lang]),
            "optimistic_effective_pass_rate": round(len(passing_by_language[lang]) / len(first_rows), 6),
        }

    atomic = read_jsonl(TRAINING / "atomic_reuse.jsonl")
    cases: dict[str, list[dict]] = defaultdict(list)
    for row in atomic:
        cases[row["source_qa_id"]].append(row)
    atomic_complete = Counter()
    atomic_total = Counter()
    for case_rows in cases.values():
        split_values = {row.get("split") for row in case_rows}
        hop_ids = {row.get("sft_id") for row in case_rows}
        hops = {row.get("hop") for row in case_rows}
        if len(split_values) != 1 or len(hops) != 2 or {"r1", "r2"} != hops or len(hop_ids) != 2:
            raise ValueError("Atomic reuse source rows do not form two-hop, single-split cases")
        split = next(iter(split_values))
        atomic_total[split] += 1
        is_complete = all(
            (row["sft_id"], "atomic") in passing_by_language[lang]
            for row in case_rows for lang in LANGS
        )
        if is_complete:
            atomic_complete[split] += 1

    composition = read_jsonl(TRAINING / "composition_translation_review_queue.jsonl")
    chains: dict[str, list[dict]] = defaultdict(list)
    for row in composition:
        if row.get("language") not in LANGS:
            raise ValueError("Unexpected language in composition review queue")
        chains[row["source_qa_id"]].append(row)
    composition_complete = 0
    for rows in chains.values():
        if len(rows) != 16:
            raise ValueError("A composition chain must have 4 prompts in each of 4 target languages")
        if all((row["prompt_id"], "composition") in passing_by_language[row["language"]] for row in rows):
            composition_complete += 1

    result = {
        "status": "OPTIMISTIC_DIAGNOSTIC_ONLY_NOT_RELEASE_APPROVAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "combination_rule": "first-stage AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS OR full-repair DIAGNOSTIC_QWEN_PASS",
        "source_review_sha256": file_sha(first_report),
        "repair_report_sha256": file_sha(parent_report_path),
        "repair_candidates_sha256": file_sha(decisions_path),
        "repair_script_sha256": parent_report.get("script_sha256"),
        "rows_by_language": row_counts,
        "optimistic_complete_atomic_cases": {
            split: {"complete": atomic_complete.get(split, 0), "required": count}
            for split, count in (("forget", 117), ("retain", 39))
        },
        "optimistic_complete_composition_chains": {
            "complete": composition_complete,
            "required": 44,
        },
        "limits": [
            "This optimistic upper bound counts first-stage passes and repair Qwen passes only.",
            "Repaired candidates have not passed the frozen round-trip and Qwen2.5 14B release review.",
            "No candidate is merged into source translations or the training release by this analysis.",
            "AI-only review is not fluent-human certification.",
        ],
    }
    output = RUN_DIR / "optimistic_coverage_report.json"
    atomic_write_json(output, result)
    result["output"] = str(output)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
