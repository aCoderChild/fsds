#!/usr/bin/env python3
"""Audit the targeted-gap translation drafts without approving them.

This checks row coverage, answer-blind request schema, protected-entity marker
counts, target script, and basic question punctuation. It cannot determine
semantic equivalence or fluency; outputs always remain quarantined.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LANGS = ("fr", "de", "hi", "th")
MARKER = re.compile(r"\[ENTITY_[A-Z]+\]")
FORBIDDEN = {"answer", "answers", "gold", "gold_label", "gold_answers", "split", "case_id", "assignment"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        rows.append(row)
    return rows


def has_script(lang: str, text: str) -> bool:
    if lang == "hi":
        return bool(re.search(r"[\u0900-\u097f]", text))
    if lang == "th":
        return bool(re.search(r"[\u0e00-\u0e7f]", text))
    return bool(re.search(r"[A-Za-zÀ-ÿ]", text))


def main() -> int:
    source_by_lang: dict[str, dict[str, dict]] = {lang: {} for lang in LANGS}
    request_hashes: dict[str, str] = {}
    errors: list[str] = []

    for path in sorted((ROOT / "requests").glob("*/*.jsonl")):
        lang = path.parent.name
        if lang not in source_by_lang:
            errors.append(f"unexpected request language: {lang}")
            continue
        request_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
        for row in read_jsonl(path):
            if set(row) != {"row_id", "source_english", "target_language"}:
                errors.append(f"{row.get('row_id', '<missing id>')}: unexpected request fields")
            if {str(k).lower() for k in row} & FORBIDDEN:
                errors.append(f"{row.get('row_id', '<missing id>')}: forbidden answer/split field")
            if row.get("target_language") != {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}[lang]:
                errors.append(f"{row.get('row_id', '<missing id>')}: target language mismatch")
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or row_id in source_by_lang[lang]:
                errors.append(f"{row_id}: missing or duplicate source row ID")
                continue
            source_by_lang[lang][row_id] = row

    per_language: dict[str, dict] = {}
    candidate_hashes: dict[str, str] = {}
    total = 0
    for lang in LANGS:
        request_rows = source_by_lang[lang]
        path = ROOT / "candidates" / f"{lang}.jsonl"
        if not path.is_file():
            errors.append(f"missing candidate file: {path.name}")
            continue
        candidate_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
        candidates = read_jsonl(path)
        candidate_ids = [row.get("row_id") for row in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            errors.append(f"{lang}: duplicate candidate IDs")
        candidate_by_id = {row.get("row_id"): row for row in candidates if isinstance(row.get("row_id"), str)}
        missing = sorted(set(request_rows) - set(candidate_by_id))
        extra = sorted(set(candidate_by_id) - set(request_rows))
        if missing:
            errors.append(f"{lang}: {len(missing)} missing candidate rows")
        if extra:
            errors.append(f"{lang}: {len(extra)} extra candidate rows")

        checked = 0
        for row_id, candidate in candidate_by_id.items():
            if row_id not in request_rows:
                continue
            checked += 1
            if set(candidate) != {"row_id", "translation"}:
                errors.append(f"{lang}/{row_id}: candidate must contain exactly row_id and translation")
            text = candidate.get("translation")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{lang}/{row_id}: missing/empty translation")
                continue
            source_markers = Counter(MARKER.findall(request_rows[row_id]["source_english"]))
            target_markers = Counter(MARKER.findall(text))
            if source_markers != target_markers:
                errors.append(f"{lang}/{row_id}: protected-marker multiset mismatch")
            if not has_script(lang, text):
                errors.append(f"{lang}/{row_id}: target script absent")
            if not re.search(r"\?\s*$", text):
                errors.append(f"{lang}/{row_id}: question mark missing at end")
            if lang == "de" and re.search(r"\s+\?", text):
                errors.append(f"{lang}/{row_id}: whitespace before German question mark")
        total += len(candidates)
        per_language[lang] = {
            "requested": len(request_rows),
            "candidate_rows": len(candidates),
            "unique_candidate_ids": len(set(candidate_ids)),
            "matched_rows_checked": checked,
            "missing_ids": missing,
            "extra_ids": extra,
        }

    result = {
        "schema_version": 1,
        "status": "STRUCTURAL_CHECKS_PASS_QUARANTINED" if not errors else "STRUCTURAL_CHECKS_FAIL_QUARANTINED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "candidate_author": "OpenAI ChatGPT assistant; exact served model revision not exposed",
            "independent_ai_review_completed": False,
            "human_translation_reviewed": False,
        },
        "counts": {"total_candidate_rows": total, "by_language": per_language},
        "source_request_sha256": request_hashes,
        "candidate_sha256": candidate_hashes,
        "checks": {
            "exact_row_id_coverage": not any("missing candidate rows" in e or "extra candidate rows" in e for e in errors),
            "answer_blind_request_schema": not any("request fields" in e or "forbidden" in e for e in errors),
            "candidate_schema": not any("candidate must contain" in e for e in errors),
            "protected_marker_multisets": not any("marker multiset" in e for e in errors),
            "target_scripts_and_question_marks": not any("script absent" in e or "question mark" in e for e in errors),
            "semantic_equivalence_and_fluency_reviewed": False,
            "release_approved": False,
            "training_eligible": False,
        },
        "limitations": [
            "Structural checks do not establish translation accuracy, semantic equivalence, or fluency.",
            "No independent AI review or fluent-human review has been completed for these candidates.",
            "Candidates must not be merged into canonical translations or an SFT release based on this audit.",
        ],
        "errors": errors,
    }
    output = ROOT / "candidates" / "candidate_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
