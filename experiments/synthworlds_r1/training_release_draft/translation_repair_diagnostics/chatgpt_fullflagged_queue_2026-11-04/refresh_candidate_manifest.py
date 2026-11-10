#!/usr/bin/env python3
"""Rebuild the quarantine receipt for ChatGPT translation repair candidates.

This checks file alignment and masked-marker preservation only. It does not
judge translation quality and never approves data for training.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
RESPONSES = ROOT / "responses"
RECEIPT = RESPONSES / "candidate_draft_manifest.json"
MARKER = re.compile(r"ZXQ[A-Z0-9]+XZ")
EXPECTED_LANGUAGES = ("fr", "de", "hi", "th")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def input_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = text.split("INPUT JSONL:\n", 1)[1]
    except IndexError as exc:
        raise ValueError(f"Missing INPUT JSONL marker in {path}") from exc
    rows = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: input line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("question_en"), str):
            raise ValueError(f"{path}: input line {line_number}: malformed input row")
        rows.append(row)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_candidate(language: str, source: Path, candidate: Path, queue_manifest: dict) -> dict:
    expected = input_rows(source)
    actual = read_jsonl(candidate)
    source_hash = sha256(source)
    expected_hash = queue_manifest["batch_files_sha256_by_language"][language][source.name]
    count_matches = len(actual) == len(expected)
    source_hash_matches = source_hash == expected_hash
    ids_match = count_matches and [row.get("row_id") for row in actual] == [
        row.get("row_id") for row in expected
    ]
    ids_unique = len({row.get("row_id") for row in expected}) == len(expected) and len(
        {row.get("row_id") for row in actual}
    ) == len(actual)
    exact_schema = all(set(row) == {"row_id", "translation"} for row in actual)
    translations_nonempty = all(
        isinstance(row.get("translation"), str) and row["translation"].strip()
        for row in actual
    )
    markers_match = count_matches and all(
        Counter(MARKER.findall(source_row["question_en"]))
        == Counter(MARKER.findall(candidate_row.get("translation", "")))
        for source_row, candidate_row in zip(expected, actual)
    )
    checks = {
        "source_hash_matches_queue_manifest": source_hash_matches,
        "row_count_matches": count_matches,
        "row_ids_match_in_order": ids_match,
        "row_ids_unique": ids_unique,
        "exact_schema": exact_schema,
        "translations_nonempty": translations_nonempty,
        "masked_markers_preserved": markers_match,
    }
    valid = all(checks.values())
    return {
        "language": language,
        "batch": source.stem,
        "source_sha256": source_hash,
        "candidate_sha256": sha256(candidate),
        "rows": len(actual),
        "checks": checks,
        "status": (
            "QUARANTINED_DRAFT_NOT_TRANSLATION_APPROVAL"
            if valid
            else "INVALID_DRAFT_QUARANTINED"
        ),
        "structurally_valid": valid,
    }


def main() -> int:
    queue_manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    total_rows = 0
    total_batches = 0
    records = []
    received = 0
    structurally_valid = 0
    valid_rows = 0

    for language in EXPECTED_LANGUAGES:
        for source in sorted((ROOT / language).glob("batch_*.txt")):
            total_batches += 1
            expected = input_rows(source)
            total_rows += len(expected)
            candidate = RESPONSES / language / f"{source.stem}.jsonl"
            if not candidate.exists():
                continue
            received += 1
            record = audit_candidate(language, source, candidate, queue_manifest)
            records.append(record)
            if record["structurally_valid"]:
                structurally_valid += 1
                valid_rows += record["rows"]

    receipt = {
        "created_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "translator": "ChatGPT UI; backend model/version not exposed",
        "purpose": "Diagnostic repair candidates for flagged questions only",
        "translation_quality_review": (
            "Not established by this manifest. No fluent-human review is recorded; "
            "AI checks are not translation ground truth."
        ),
        "release_approved": False,
        "training_eligible": False,
        "integrity_scope": "JSONL parsing, IDs/order, schema, nonempty strings, and masked markers only",
        "coverage": {
            "batches_received": received,
            "batches_structurally_valid": structurally_valid,
            "batches_total": total_batches,
            "rows_received": sum(row["rows"] for row in records),
            "rows_structurally_valid": valid_rows,
            "rows_total": total_rows,
        },
        "source_queue_manifest_sha256": sha256(ROOT / "manifest.json"),
        "candidates": records,
    }
    RESPONSES.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["coverage"], indent=2))
    print(f"Receipt: {RECEIPT}")
    return 0 if received == structurally_valid else 2


if __name__ == "__main__":
    sys.exit(main())
