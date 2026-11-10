#!/usr/bin/env python3
"""Measure lower-bound case coverage from the killed, verified 733-row review."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
PROJECT = HERE.parents[2]
INPUTS = HERE / "independent_review_inputs_2026-11-04"
PARTIAL = HERE / "remote_output/full799/version_5"
LANGS = ("fr", "de", "hi", "th")
PASS_STATUSES = {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}

sys.path.insert(0, str(RELEASE / "scripts"))
from review_training_translations_qwen import collect_rows, read_jsonl  # noqa: E402

sys.path.insert(0, str(HERE))
import analyze_chatgpt_repair_coverage as full_coverage  # noqa: E402


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> dict:
    input_path = INPUTS / "full799.jsonl"
    launch = read_json(INPUTS / "full799.kaggle_launch_manifest.json")
    partial_result = read_json(PARTIAL / "repair_review.json")
    receipt = read_json(PARTIAL / "verification.json")
    verified_path = PARTIAL / "partial_verified_repair_reviews.jsonl"
    verified = read_jsonl(verified_path)
    raw_rows = read_jsonl(input_path)
    compact = json.dumps(raw_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if (receipt.get("status") != "VERIFIED_KILLED_PARTIAL_AI_ONLY_REPAIR_REVIEW"
            or receipt.get("release_approved") is not False
            or receipt.get("training_eligible") is not False
            or receipt.get("input_sha256") != sha_bytes(compact)
            or receipt.get("output_sha256") != sha_bytes((PARTIAL / "repair_review.json").read_bytes())
            or receipt.get("verified_rows_sha256") != sha_bytes(verified_path.read_bytes())
            or receipt.get("semantic_review_progress") != 733
            or receipt.get("pending_rows") != 66
            or len(raw_rows) != 799 or len(verified) != 733
            or launch.get("input_jsonl_sha256") != sha_bytes(input_path.read_bytes())):
        raise ValueError("Partial-review receipt does not match its exact source/input/output evidence")
    input_by_key = {row["review_key"]: row for row in raw_rows}
    verified_by_key = {row["review_key"]: row for row in verified}
    if len(input_by_key) != 799 or len(verified_by_key) != 733 or not set(verified_by_key) <= set(input_by_key):
        raise ValueError("Partial verified rows do not form a unique subset of the frozen input")

    queue_manifest = read_json(HERE / "manifest.json")
    original_report_path = RELEASE / "translation_reviews_roundtrip_qwen14b/report.json"
    original_report = read_json(original_report_path)
    if (original_report.get("status") != "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE"
            or original_report.get("human_translation_reviewed") is not False):
        raise ValueError("Original source-bound AI-only review receipt is unavailable or changed")
    pass_by_language = {lang: set() for lang in LANGS}
    original_counts = {}
    for lang in LANGS:
        source_rows, source_hashes = collect_rows(lang)
        if (len(source_rows) != 488
                or source_hashes != queue_manifest["reconstructed_source_hashes_by_language"][lang]):
            raise ValueError(f"{lang}: current source rows differ from the frozen repair queue")
        review_path = RELEASE / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl"
        detail = original_report.get("languages", {}).get(lang, {})
        if detail.get("adjudication_sha256") != sha_bytes(review_path.read_bytes()):
            raise ValueError(f"{lang}: original review checksum differs from its report")
        original_rows = read_jsonl(review_path)
        original_by_key = {(row.get("prompt_id"), row.get("kind")): row for row in original_rows}
        if len(original_by_key) != 488:
            raise ValueError(f"{lang}: original review is not a unique 488-row set")
        for key, row in original_by_key.items():
            if row.get("final_review_status") in PASS_STATUSES:
                pass_by_language[lang].add(key)
        original_counts[lang] = sum(row.get("final_review_status") in PASS_STATUSES for row in original_rows)

    candidate_counts = {lang: Counter() for lang in LANGS}
    for key, result in verified_by_key.items():
        lang = result["language"]
        if lang not in LANGS or result.get("human_translation_reviewed") is not False:
            raise ValueError(f"{key}: unexpected language or human-review claim")
        source = input_by_key[key]
        if (source["language"] != lang or source["kind"] != result["kind"]
                or result.get("backtranslation_markers_pass") is not True
                and result.get("review_status") == "AI_ONLY_REPAIR_REVIEW_PASS"):
            raise ValueError(f"{key}: partial result does not bind to its source row")
        candidate_counts[lang][result["review_status"]] += 1
        if result["review_status"] == "AI_ONLY_REPAIR_REVIEW_PASS":
            pass_by_language[lang].add((result["prompt_id"], result["kind"]))

    coverage = full_coverage.compute_coverage(
        pass_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    report = {
        "status": "PARTIAL_REPAIR_LOWER_BOUND_DIAGNOSTIC_NOT_RELEASE_APPROVAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": 799,
        "semantic_review_progress": 733,
        "verified_decisions": 733,
        "diagnostic_passes": sum(counts["AI_ONLY_REPAIR_REVIEW_PASS"] for counts in candidate_counts.values()),
        "diagnostic_flagged": sum(counts["AI_ONLY_REPAIR_REVIEW_FLAGGED"] for counts in candidate_counts.values()),
        "diagnostic_blocked": sum(counts["AI_ONLY_REPAIR_REVIEW_BLOCKED"] for counts in candidate_counts.values()),
        "pending": 66,
        "rows_by_language": {
            lang: {"original_passes": original_counts[lang], **dict(candidate_counts[lang])}
            for lang in LANGS
        },
        "coverage": coverage,
        "review_receipt_sha256": sha_bytes((PARTIAL / "verification.json").read_bytes()),
        "verified_rows_sha256": sha_bytes(verified_path.read_bytes()),
        "optimistic_only": True,
        "release_approved": False,
        "training_eligible": False,
        "human_translation_reviewed": False,
        "limitations": [
            "Pending, blocked, and flagged rows are treated as not passing; results are a lower bound on current candidate coverage.",
            "A candidate-review pass is only a possibility for the full frozen production re-screen, never approval by itself.",
            "AI-only judgments do not certify human translation quality.",
        ],
    }
    output = HERE / "partial_coverage_after_733.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    report["output"] = str(output)
    return report


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
