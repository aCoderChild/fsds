#!/usr/bin/env python3
"""Prepare answer-blind ChatGPT inputs for final-flagged training questions.

This script only prepares translation requests. It does not edit translations,
review decisions, the release corpus, or any approval manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RELEASE = PROJECT / "training_release_draft"
REVIEW_DIR = RELEASE / "translation_reviews_roundtrip_qwen14b"
OUT = RELEASE / "translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04"
LANGUAGES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
EXPECTED_ROWS = {"fr": 488, "de": 488, "hi": 488, "th": 488}
EXPECTED_FLAGS = {"fr": 164, "de": 179, "hi": 200, "th": 256}
BATCH_SIZE = 32
MARKER_PATTERN = re.compile(r"ZXQ[A-Z]+QXZ")
sys.path.insert(0, str(RELEASE / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
from review_training_translations_qwen import collect_rows  # noqa: E402
from repair_flagged_eval_translations_translategemma import protect  # noqa: E402


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


def main() -> None:
    report_path = REVIEW_DIR / "report.json"
    report = read_json(report_path)
    if (report.get("status") != "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE"
            or report.get("reviewed_every_candidate") is not True
            or report.get("do_not_train") is not True
            or report.get("answers_sent_to_reviewer") is not False
            or report.get("human_translation_reviewed") is not False):
        raise ValueError("Expected complete, answer-blind, AI-only final translation review")

    OUT.mkdir(parents=True, exist_ok=True)
    mapping: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    request_hashes: dict[str, str] = {}
    source_review_hashes: dict[str, str] = {}
    reconstructed_source_hashes: dict[str, dict[str, str]] = {}
    batch_hashes: dict[str, dict[str, str]] = {}
    for lang, language_name in LANGUAGES.items():
        rows_path = REVIEW_DIR / f"{lang}.jsonl"
        source_review_hashes[lang] = sha_file(rows_path)
        lang_report = report.get("languages", {}).get(lang, {})
        if lang_report.get("adjudication_sha256") != sha_file(rows_path):
            raise ValueError(f"{lang}: final-review JSONL checksum mismatch")
        rows = read_jsonl(rows_path)
        reconstructed_rows, reconstructed_hashes = collect_rows(lang)
        if reconstructed_hashes.get(f"flagged_repairs_{lang}") not in (None, "ABSENT"):
            raise ValueError(f"{lang}: unexpected existing repair input changed the source reconstruction")
        source_by_key = {(row["prompt_id"], row["kind"]): row for row in reconstructed_rows}
        reconstructed_source_hashes[lang] = reconstructed_hashes
        if len(source_by_key) != len(reconstructed_rows) or len(source_by_key) != EXPECTED_ROWS[lang]:
            raise ValueError(f"{lang}: source reconstruction has missing or duplicate prompt keys")
        if len(rows) != EXPECTED_ROWS[lang] or lang_report.get("questions") != EXPECTED_ROWS[lang]:
            raise ValueError(f"{lang}: expected {EXPECTED_ROWS[lang]} reviewed rows")

        by_kind: dict[str, int] = {}
        seen: set[tuple[str, str]] = set()
        selected = []
        for row in rows:
            key = (row.get("prompt_id"), row.get("kind"))
            if (not all(isinstance(value, str) and value for value in key)
                    or key in seen
                    or row.get("language") != lang
                    or not isinstance(row.get("text_en"), str)
                    or not row["text_en"].strip()
                    or not isinstance(row.get("text_translation"), str)
                    or not row["text_translation"].strip()):
                raise ValueError(f"{lang}: malformed or duplicate source-bound row {key}")
            seen.add(key)
            source_row = source_by_key.get(key)
            if (source_row is None or source_row.get("text_en") != row["text_en"]
                    or source_row.get("text_translation") != row["text_translation"]):
                raise ValueError(f"{lang}/{key}: final review does not match source-reconstructed text")
            if any(name in row for name in ("answer", "answer_en", "gold_answer", "gold_label", "final_answer")):
                raise ValueError(f"{lang}/{key}: answer-bearing field must not enter translation queue")
            status = row.get("final_review_status")
            if status == "AI_ONLY_REVIEW_FLAGGED":
                selected.append(row)
                by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
            elif status not in {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}:
                raise ValueError(f"{lang}/{key}: unexpected final review state {status!r}")
        if len(selected) != EXPECTED_FLAGS[lang]:
            raise ValueError(f"{lang}: expected {EXPECTED_FLAGS[lang]} flagged rows, found {len(selected)}")

        requests = []
        for index, row in enumerate(selected, start=1):
            repair_id = f"{lang.upper()}{index:04d}"
            source_row = source_by_key[(row["prompt_id"], row["kind"])]
            masked_question, protected_values, expected_markers = protect(
                source_row["text_en"], source_row.get("protected_entity_names", [])
            )
            names = [name for name in source_row.get("protected_entity_names", [])
                     if name and re.search(re.escape(name), source_row["text_en"], re.IGNORECASE)]
            has_protected_content = bool(names or re.search(r"\d+", source_row["text_en"]))
            if not masked_question.strip() or (has_protected_content and not protected_values):
                raise ValueError(f"{lang}/{row['prompt_id']}: failed to protect source entities/numbers")
            requests.append({"row_id": repair_id, "question_en": masked_question})
            mapping.append({
                "row_id": repair_id,
                "language": lang,
                "kind": row["kind"],
                "prompt_id": row["prompt_id"],
                "source_question_en": row["text_en"],
                "source_question_en_sha256": sha_bytes(row["text_en"].encode("utf-8")),
                "masked_question_en": masked_question,
                "masked_question_en_sha256": sha_bytes(masked_question.encode("utf-8")),
                "protected_values": protected_values,
                "expected_markers": expected_markers,
                "rejected_translation": row["text_translation"],
                "rejected_translation_sha256": sha_bytes(row["text_translation"].encode("utf-8")),
                "final_review_status": status_for(row),
            })
        counts[lang] = {"flagged": len(requests), "atomic": by_kind.get("atomic", 0), "composition": by_kind.get("composition", 0)}
        request_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in requests).encode("utf-8")
        request_hashes[lang] = sha_bytes(request_bytes)

        batches_dir = OUT / lang
        batches_dir.mkdir(parents=True, exist_ok=True)
        batch_hashes[lang] = {}
        for batch_no, start in enumerate(range(0, len(requests), BATCH_SIZE), start=1):
            batch = requests[start:start + BATCH_SIZE]
            body = "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in batch)
            prompt = (
                f"Translate each English question into natural {language_name}. Preserve the full meaning, "
                "relation direction, scope, polarity, tense, and answer type. Copy every ZXQ marker exactly "
                "and preserve its role in the question. Do not answer the questions, add explanations, or "
                "change row_id values. Return exactly one JSON object per input row, in the same order, "
                'with only the keys row_id and translation; no markdown fences or extra text.\n\n'
                "INPUT JSONL:\n" + body + "\n"
            )
            batch_path = batches_dir / f"batch_{batch_no:02d}.txt"
            batch_path.write_text(prompt, encoding="utf-8")
            batch_hashes[lang][batch_path.name] = sha_file(batch_path)

    map_path = OUT / "private_row_map.jsonl"
    write_jsonl(map_path, mapping)
    map_by_id = {row["row_id"]: row for row in mapping}
    seen_ids = []
    batch_count = 0
    for lang, files in batch_hashes.items():
        for filename, expected_hash in files.items():
            batch_path = OUT / lang / filename
            if sha_file(batch_path) != expected_hash:
                raise ValueError(f"Batch checksum changed after writing: {lang}/{filename}")
            body = batch_path.read_text(encoding="utf-8").split("INPUT JSONL:\n", 1)[1]
            for line in body.splitlines():
                if not line.strip():
                    continue
                request = json.loads(line)
                if set(request) != {"row_id", "question_en"}:
                    raise ValueError("Translation request contains unexpected fields")
                private = map_by_id.get(request["row_id"])
                if private is None or private["language"] != lang or private["masked_question_en"] != request["question_en"]:
                    raise ValueError(f"Request does not match its source-bound private mapping: {request.get('row_id')}")
                expected_markers = private["expected_markers"]
                found_markers = MARKER_PATTERN.findall(request["question_en"])
                if set(found_markers) != set(expected_markers) or any(found_markers.count(token) != count for token, count in expected_markers.items()):
                    if expected_markers or found_markers:
                        raise ValueError(f"Protected marker counts differ in request {request['row_id']}")
                for value in private["protected_values"].values():
                    if value.casefold() in request["question_en"].casefold():
                        raise ValueError(f"Unmasked protected value in ChatGPT request {request['row_id']}")
                seen_ids.append(request["row_id"])
            batch_count += 1
    if len(seen_ids) != len(set(seen_ids)) or set(seen_ids) != set(map_by_id):
        raise ValueError("Translation request batches do not cover every mapped row exactly once")
    validation = {
        "status": "PASS_SOURCE_BOUND_MASKED_TRANSLATION_QUEUE",
        "request_rows": len(seen_ids),
        "unique_row_ids": len(set(seen_ids)),
        "batches": batch_count,
        "answers_present": False,
        "protected_names_and_numbers_masked": True,
        "source_rows_reconstructed_and_matched": True,
        "files_hashed": True,
    }
    validation_path = OUT / "queue_validation.json"
    write_json(validation_path, validation)
    manifest = {
        "status": "TRANSLATION_REQUEST_QUEUE_ONLY_NOT_APPROVED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_review_report": str(report_path.relative_to(PROJECT)),
        "source_review_report_sha256": sha_file(report_path),
        "source_review_rows_sha256_by_language": source_review_hashes,
        "reconstructed_source_hashes_by_language": reconstructed_source_hashes,
        "source_review_protocol": report.get("protocol"),
        "queue_script_sha256": sha_file(Path(__file__)),
        "translator": "ChatGPT UI; exact backend model/version not exposed",
        "review_mode": "AI_ONLY_DIAGNOSTIC_UNTIL_FULL_HASH_BOUND_RESCREEN",
        "human_translation_reviewed": False,
        "answers_sent_to_translator": False,
        "answers_sent_to_reviewer": False,
        "batch_size": BATCH_SIZE,
        "language_counts": counts,
        "total_flagged_rows": len(mapping),
        "request_rows_sha256_by_language": request_hashes,
        "batch_files_sha256_by_language": batch_hashes,
        "private_row_map_sha256": sha_file(map_path),
        "queue_validation_sha256": sha_file(validation_path),
        "queue_validation_status": validation["status"],
        "limitations": [
            "No translations are generated by this script.",
            "Request files contain only row IDs and English questions; keep the private map local.",
            "Entity names and numbers in ChatGPT requests are masked with ZXQ markers; restore only from the private map after exact marker validation.",
            "Never merge a candidate into the release without a complete independent re-screen of all 488 rows per language.",
            "AI-only checks do not certify fluent-human translation quality."
        ]
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "total_flagged_rows": len(mapping), "language_counts": counts,
                      "queue_validation": validation, "private_row_map_sha256": manifest["private_row_map_sha256"]}, indent=2))


def status_for(row: dict) -> str:
    return row["final_review_status"]


if __name__ == "__main__":
    main()
