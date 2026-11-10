#!/usr/bin/env python3
"""Validate ChatGPT batch responses and emit quarantined repair candidates.

The output remains diagnostic-only. This parser never edits source translations
or the SFT release and never uses answer labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RELEASE = PROJECT / "training_release_draft"
QUEUE = RELEASE / "translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04"
LANGUAGES = ("fr", "de", "hi", "th")
MARKER = re.compile(r"ZXQ[A-Z]+QXZ")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def parse_response(path: Path, expected_ids: list[str]) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError(f"{path}: malformed Markdown code fence")
        lines = lines[1:-1]
    records = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: expected one JSON object per line") from exc
        if not isinstance(row, dict) or set(row) != {"row_id", "translation"}:
            raise ValueError(f"{path}:{line_no}: expected exactly row_id and translation")
        if not isinstance(row["row_id"], str) or not isinstance(row["translation"], str) or not row["translation"].strip():
            raise ValueError(f"{path}:{line_no}: row_id and nonempty translation are required")
        records.append(row)
    ids = [row["row_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate row IDs")
    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ValueError(f"{path}: response IDs differ from the exact requested batch")
    return {row["row_id"]: row["translation"].strip() for row in records}


def restore_markers(text: str, private: dict) -> str:
    values = private["protected_values"]
    expected = private["expected_markers"]
    found: Counter[str] = Counter()

    def replace(match: re.Match[str]) -> str:
        marker = match.group(0)
        if marker not in values:
            raise ValueError(f"{private['row_id']}: unexpected protected marker {marker}")
        found[marker] += 1
        return values[marker]

    restored = MARKER.sub(replace, text).strip()
    if dict(found) != expected:
        raise ValueError(f"{private['row_id']}: protected marker counts changed; expected {expected}, found {dict(found)}")
    if MARKER.search(restored):
        raise ValueError(f"{private['row_id']}: unreplaced marker remains")
    return restored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", type=Path, default=QUEUE / "responses")
    args = parser.parse_args()
    manifest_path = QUEUE / "manifest.json"
    validation_path = QUEUE / "queue_validation.json"
    map_path = QUEUE / "private_row_map.jsonl"
    manifest = read_json(manifest_path)
    validation = read_json(validation_path)
    if (manifest.get("status") != "TRANSLATION_REQUEST_QUEUE_ONLY_NOT_APPROVED"
            or manifest.get("queue_validation_status") != "PASS_SOURCE_BOUND_MASKED_TRANSLATION_QUEUE"
            or manifest.get("queue_validation_sha256") != sha_file(validation_path)
            or validation.get("status") != "PASS_SOURCE_BOUND_MASKED_TRANSLATION_QUEUE"
            or manifest.get("private_row_map_sha256") != sha_file(map_path)):
        raise ValueError("Translation queue manifest or integrity validation failed")
    map_rows = read_jsonl(map_path)
    private_by_id = {row["row_id"]: row for row in map_rows}
    if len(private_by_id) != len(map_rows) or len(map_rows) != manifest.get("total_flagged_rows"):
        raise ValueError("Private row map is incomplete or has duplicate IDs")

    source_by_lang: dict[str, dict[tuple[str, str], dict]] = {}
    source_hashes: dict[str, str] = {}
    for lang in LANGUAGES:
        review_path = RELEASE / "translation_reviews_roundtrip_qwen14b" / f"{lang}.jsonl"
        expected_review_hash = manifest["source_review_rows_sha256_by_language"][lang]
        if sha_file(review_path) != expected_review_hash:
            raise ValueError(f"{lang}: frozen source review changed after queue construction")
        source_hashes[lang] = expected_review_hash
        sys_path = str(RELEASE / "scripts")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from review_training_translations_qwen import collect_rows, deterministic_checks
        source_rows, reconstructed_hashes = collect_rows(lang)
        if reconstructed_hashes != manifest["reconstructed_source_hashes_by_language"][lang]:
            raise ValueError(f"{lang}: reconstructed source hashes changed after queue creation")
        source_by_lang[lang] = {(row["prompt_id"], row["kind"]): row for row in source_rows}
        if len(source_by_lang[lang]) != 488:
            raise ValueError(f"{lang}: source question reconstruction is incomplete")

    candidates = []
    response_hashes: dict[str, str] = {}
    missing_batches = []
    failed_deterministic = Counter()
    passed_deterministic = Counter()
    response_root = args.responses_dir

    for lang, batch_files in manifest["batch_files_sha256_by_language"].items():
        for batch_name, expected_input_hash in batch_files.items():
            batch_path = QUEUE / lang / batch_name
            if sha_file(batch_path) != expected_input_hash:
                raise ValueError(f"{lang}/{batch_name}: request file hash mismatch")
            request_rows = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").split("INPUT JSONL:\n", 1)[1].splitlines() if line.strip()]
            response_path = response_root / lang / batch_name.replace(".txt", ".jsonl")
            if not response_path.is_file():
                missing_batches.append(f"{lang}/{response_path.name}")
                continue
            response_hashes[f"{lang}/{response_path.name}"] = sha_file(response_path)
            translations = parse_response(response_path, [row["row_id"] for row in request_rows])
            for request in request_rows:
                row_id = request["row_id"]
                private = private_by_id[row_id]
                text_translation = restore_markers(translations[row_id], private)
                key = (private["prompt_id"], private["kind"])
                source = source_by_lang[lang].get(key)
                if source is None or source["text_en"] != private["source_question_en"]:
                    raise ValueError(f"{row_id}: restored candidate is not bound to current source question")
                check_row = {**source, "text_translation": text_translation}
                checks = deterministic_checks(lang, check_row)
                deterministic_pass = all(checks.get(name) is True for name in (
                    "nonempty", "target_script_present", "protected_names_preserved",
                    "protected_numbers_preserved", "no_unrestored_marker"))
                status = ("CHATGPT_UI_REPAIR_CANDIDATE_PENDING_INDEPENDENT_REVIEW"
                          if deterministic_pass else "CHATGPT_UI_REPAIR_CANDIDATE_FAILED_DETERMINISTIC_CHECKS")
                (passed_deterministic if deterministic_pass else failed_deterministic)[lang] += 1
                candidates.append({
                    "prompt_id": private["prompt_id"], "kind": private["kind"], "language": lang,
                    "text_en": private["source_question_en"], "text_translation": text_translation,
                    "translation_model": "ChatGPT UI; exact backend model/version not exposed",
                    "translation_status": status, "deterministic_checks": checks,
                    "human_translation_reviewed": False, "answers_sent_to_translator": False,
                    "source_text_sha256": private["source_question_en_sha256"],
                    "old_text_translation_sha256": private["rejected_translation_sha256"],
                    "masked_source_sha256": private["masked_question_en_sha256"],
                    "request_batch": f"{lang}/{batch_name}", "request_batch_sha256": expected_input_hash,
                    "response_sha256": response_hashes[f"{lang}/{response_path.name}"],
                })

    candidate_path = QUEUE / "parsed_repair_candidates.jsonl"
    write_jsonl(candidate_path, candidates)
    done = len(response_hashes)
    total = sum(len(files) for files in manifest["batch_files_sha256_by_language"].values())
    all_rows = len(candidates) == manifest["total_flagged_rows"] and not missing_batches
    has_deterministic_failures = bool(failed_deterministic)
    report = {
        "status": ("CHATGPT_REPAIR_CANDIDATES_COMPLETE_PENDING_INDEPENDENT_REVIEW"
                   if all_rows and not has_deterministic_failures else
                   "CHATGPT_REPAIR_CANDIDATES_COMPLETE_WITH_DETERMINISTIC_FAILURES"
                   if all_rows else "CHATGPT_REPAIR_CANDIDATES_INCOMPLETE_DIAGNOSTIC_ONLY"),
        "do_not_train": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue_manifest_sha256": sha_file(manifest_path),
        "private_row_map_sha256": sha_file(map_path),
        "source_review_rows_sha256_by_language": source_hashes,
        "response_files_sha256": response_hashes,
        "completed_batches": done, "expected_batches": total, "missing_batches": missing_batches,
        "candidate_rows": len(candidates),
        "deterministic_passes_by_language": dict(passed_deterministic),
        "deterministic_failures_by_language": dict(failed_deterministic),
        "candidate_file": candidate_path.name, "candidate_file_sha256": sha_file(candidate_path),
        "human_translation_reviewed": False,
        "limitations": [
            "Every candidate requires a full independent semantic/fluency re-review before release assembly.",
            "AI-only review is not fluent-human certification.",
            "This output does not modify source translations, review files, release manifests, or the SFT corpus."
        ]
    }
    write_json(QUEUE / "parsed_repair_candidates_report.json", report)
    print(json.dumps({"status": report["status"], "candidate_rows": len(candidates),
                      "completed_batches": done, "expected_batches": total,
                      "deterministic_passes_by_language": report["deterministic_passes_by_language"],
                      "deterministic_failures_by_language": report["deterministic_failures_by_language"]}, indent=2))


if __name__ == "__main__":
    main()
