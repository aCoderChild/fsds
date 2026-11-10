#!/usr/bin/env python3
"""Merge source-verified first-run and retry diagnostics without approving data."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "independent_review_inputs_2026-11-04"
PARTIAL_DIR = HERE / "remote_output/full799/version_5"
MODE = "retry_full799_unresolved_v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-dir", type=Path, required=True,
                        help="Verified retry Kaggle output directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New versioned local directory for merged diagnostic artifacts")
    args = parser.parse_args()

    source_path = INPUTS / "full799.jsonl"
    launch_path = INPUTS / "full799.kaggle_launch_manifest.json"
    source_rows = read_jsonl(source_path)
    source_launch = read_json(launch_path)
    private_map_path = INPUTS / "full799.private_map.jsonl"
    private_map = read_jsonl(private_map_path)
    source_by_key = {row["review_key"]: row for row in source_rows}
    map_by_key = {row["review_key"]: row for row in private_map}
    if (len(source_rows) != 799 or len(source_by_key) != 799 or len(map_by_key) != 799
            or source_launch.get("input_jsonl_sha256") != sha(source_path.read_bytes())
            or source_launch.get("private_map_sha256") != sha(private_map_path.read_bytes())):
        raise ValueError("Frozen 799-row source or private map failed its launch-manifest checks")

    partial_result_path = PARTIAL_DIR / "repair_review.json"
    partial_verification_path = PARTIAL_DIR / "verification.json"
    partial_rows_path = PARTIAL_DIR / "partial_verified_repair_reviews.jsonl"
    partial_result = read_json(partial_result_path)
    partial_verification = read_json(partial_verification_path)
    partial_rows = read_jsonl(partial_rows_path)
    partial_keys = [r.get("review_key") for r in partial_rows]
    if (partial_verification.get("status") != "VERIFIED_KILLED_PARTIAL_AI_ONLY_REPAIR_REVIEW"
            or partial_verification.get("output_sha256") != sha(partial_result_path.read_bytes())
            or partial_verification.get("verified_rows_sha256") != sha(partial_rows_path.read_bytes())
            or partial_result.get("input_sha256") != source_launch.get("input_compact_array_sha256")
            or len(partial_keys) != 733 or len(set(partial_keys)) != 733
            or not set(partial_keys) <= set(source_by_key)):
        raise ValueError("Original partial evidence is not the verified 733-row source-bound checkpoint")

    retry_receipt_path = args.retry_dir / "retry_verification.json"
    retry_rows_path = args.retry_dir / "retry_verified_repair_reviews.jsonl"
    retry_receipt = read_json(retry_receipt_path)
    retry_rows = read_jsonl(retry_rows_path)
    retry_keys = [r.get("review_key") for r in retry_rows]
    if (retry_receipt.get("status") != "VERIFIED_COMPLETE_RETRY_AI_ONLY_DIAGNOSTIC"
            or retry_receipt.get("training_eligible") is not False
            or retry_receipt.get("release_approved") is not False
            or retry_receipt.get("verified_retry_rows_sha256") != sha(retry_rows_path.read_bytes())
            or retry_receipt.get("output_sha256") != sha((args.retry_dir / "repair_review.json").read_bytes())
            or len(retry_keys) != 97 or len(set(retry_keys)) != 97):
        raise ValueError("Retry lacks a complete, hash-bound verification receipt")

    partial_result_rows = {row.get("review_key"): row for row in partial_result.get("records", [])}
    if len(partial_result_rows) != 799 or set(partial_result_rows) != set(source_by_key):
        raise ValueError("Original checkpoint records do not cover the exact 799 frozen source rows")
    expected_retry = set()
    for key, record in partial_result_rows.items():
        decision = record.get("decision")
        unresolved = (not isinstance(decision, dict) or "error" in decision
                      or record.get("backtranslation_error") is not None)
        if unresolved:
            expected_retry.add(key)
    if len(expected_retry) != 97 or set(retry_keys) != expected_retry:
        raise ValueError("Retry rows do not exactly replace all unresolved first-run decisions")

    first_run_rows = {r["review_key"]: r for r in partial_rows
                      if r.get("review_status") in {"AI_ONLY_REPAIR_REVIEW_PASS", "AI_ONLY_REPAIR_REVIEW_FLAGGED"}}
    first_run_blocked = {r["review_key"] for r in partial_rows
                         if r.get("review_status") == "AI_ONLY_REPAIR_REVIEW_BLOCKED"}
    if (len(first_run_rows) != 702 or len(first_run_blocked) != 31
            or set(first_run_rows) & expected_retry
            or set(first_run_rows) | expected_retry != set(source_by_key)):
        raise ValueError("Verified first-run and retry rows do not form an exact, non-overlapping 799-row partition")

    retry_by_key = {r["review_key"]: r for r in retry_rows}
    merged = []
    for source in source_rows:
        key = source["review_key"]
        row = dict(retry_by_key[key] if key in retry_by_key else first_run_rows[key])
        local_map = map_by_key[key]
        expected_source_hash = hashlib.sha256(source["source_question_en"].encode()).hexdigest()
        expected_translation_hash = hashlib.sha256(source["translated_question"].encode()).hexdigest()
        if key in retry_by_key:
            source_hash = row.get("source_sha256")
            translation_hash = row.get("translation_sha256")
        else:
            original_attempt = partial_result_rows[key]
            source_hash = original_attempt.get("source_sha256")
            translation_hash = original_attempt.get("translation_sha256")
        if (row.get("prompt_id") != local_map.get("prompt_id")
                or row.get("language") != source.get("language") or row.get("kind") != source.get("kind")
                or source_hash != expected_source_hash or translation_hash != expected_translation_hash
                or row.get("candidate_translation_sha256") != local_map.get("candidate_translation_sha256")
                or row.get("human_translation_reviewed") is not False):
            raise ValueError(f"{key}: merged diagnostic row is not bound to frozen source/map identity")
        row["source_sha256"] = expected_source_hash
        row["translation_sha256"] = expected_translation_hash
        row["review_run"] = "retry" if key in retry_by_key else "first_full_run_verified_decision"
        if key in retry_by_key:
            row["supersedes_first_run_blocked_attempt"] = key in first_run_blocked
        else:
            row["supersedes_first_run_blocked_attempt"] = False
        merged.append(row)

    counts = collections.Counter(row["review_status"] for row in merged)
    cells = collections.defaultdict(collections.Counter)
    for row in merged:
        cells[f"{row['language']}/{row['kind']}"][row["review_status"]] += 1
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    merged_path = args.output_dir / "merged_full799_diagnostic_reviews.jsonl"
    merged_path.write_text(payload, encoding="utf-8")
    receipt = {
        "status": "VERIFIED_799_ROW_AI_ONLY_DIAGNOSTIC_MERGE_NOT_RELEASE_APPROVAL",
        "source_rows": 799,
        "first_run_verified_decisions_retained": len(first_run_rows),
        "first_run_blocked_attempts_superseded_by_retry": len(first_run_blocked),
        "retry_rows": len(retry_rows),
        "unique_merged_rows": len(merged),
        "exact_source_coverage": len({row["review_key"] for row in merged}) == 799,
        "source_jsonl_sha256": sha(source_path.read_bytes()),
        "source_private_map_sha256": sha(private_map_path.read_bytes()),
        "first_run_output_sha256": sha(partial_result_path.read_bytes()),
        "first_run_verification_sha256": sha(partial_verification_path.read_bytes()),
        "first_run_verified_rows_sha256": sha(partial_rows_path.read_bytes()),
        "retry_output_sha256": retry_receipt["output_sha256"],
        "retry_verification_sha256": sha(retry_receipt_path.read_bytes()),
        "retry_verified_rows_sha256": sha(retry_rows_path.read_bytes()),
        "merged_rows_sha256": sha(merged_path.read_bytes()),
        "counts": dict(counts),
        "counts_by_language_kind": {key: dict(value) for key, value in sorted(cells.items())},
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "answers_or_splits_sent": False,
        "limitations": [
            "AI-only diagnostic findings do not certify fluent-human translation quality.",
            "Diagnostic pass rows still require the complete frozen production re-screen and source-integrity proof.",
            "The merge replaces 31 blocked attempts; it does not treat blocked or pending rows as passes.",
        ],
    }
    receipt_path = args.output_dir / "merge_verification.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
