#!/usr/bin/env python3
"""Build masked, source-bound inputs to independently review repair candidates.

Only candidate questions and their English source questions are exported; no
answers, split labels, or hidden fact records are included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
REPO = HERE.parents[4]
SCRIPTS = RELEASE / "scripts"
sys.path.insert(0, str(SCRIPTS))
from review_training_translations_qwen import collect_rows, deterministic_checks, masked_pair, read_jsonl  # noqa: E402

LANGS = ("fr", "de", "hi", "th")
EXPECTED = {"fr": 164, "de": 179, "hi": 200, "th": 256}
REPORT = HERE / "parsed_repair_candidates_report.json"
CANDIDATES = HERE / "parsed_repair_candidates.jsonl"
QUEUE_MANIFEST = HERE / "manifest.json"
REVIEW_ROOT = RELEASE / "translation_reviews_roundtrip_qwen14b"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def prepare() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    queue_manifest = json.loads(QUEUE_MANIFEST.read_text(encoding="utf-8"))
    if (report.get("status") != "CHATGPT_REPAIR_CANDIDATES_COMPLETE_PENDING_INDEPENDENT_REVIEW"
            or report.get("do_not_train") is not True
            or report.get("candidate_file_sha256") != sha_file(CANDIDATES)):
        raise ValueError("Candidate receipt is incomplete, changed, or not quarantined")
    if report.get("deterministic_failures_by_language"):
        raise ValueError("Candidate file contains deterministic failures")

    candidates = read_jsonl(CANDIDATES)
    candidate_by_lang = {lang: [] for lang in LANGS}
    candidate_keys = set()
    for row in candidates:
        if row.get("language") not in candidate_by_lang:
            raise ValueError(f"Unexpected candidate language: {row.get('language')}")
        candidate_key = (row["language"], row["prompt_id"], row["kind"])
        if candidate_key in candidate_keys:
            raise ValueError(f"Duplicate candidate key: {candidate_key}")
        candidate_keys.add(candidate_key)
        candidate_by_lang[row["language"]].append(row)
    if {lang: len(rows) for lang, rows in candidate_by_lang.items()} != EXPECTED:
        raise ValueError("Candidate counts differ from the source-bound repair queue")

    inputs = []
    private_map = []
    current_review_hashes = {}
    for lang in LANGS:
        source_rows, _ = collect_rows(lang)
        source_by_key = {(r["prompt_id"], r["kind"]): r for r in source_rows}
        if len(source_by_key) != 488:
            raise ValueError(f"{lang}: current source reconstruction is incomplete")
        review_path = REVIEW_ROOT / f"{lang}.jsonl"
        reviews = {(r["prompt_id"], r["kind"]): r for r in read_jsonl(review_path)}
        if (len(reviews) != 488 or set(reviews) != set(source_by_key)
                or sha_file(review_path) != queue_manifest["source_review_rows_sha256_by_language"][lang]):
            raise ValueError(f"{lang}: frozen original review is incomplete or changed")
        current_review_hashes[lang] = sha_file(review_path)

        for candidate in sorted(candidate_by_lang[lang], key=lambda r: (r["prompt_id"], r["kind"])):
            key = (candidate["prompt_id"], candidate["kind"])
            source = source_by_key.get(key)
            old_review = reviews.get(key)
            if not source or not old_review:
                raise ValueError(f"{lang}/{key}: candidate has no current source/review row")
            if source["text_en"] != candidate["text_en"]:
                raise ValueError(f"{lang}/{key}: English source changed")
            if old_review.get("final_review_status") != "AI_ONLY_REVIEW_FLAGGED":
                raise ValueError(f"{lang}/{key}: repair target was not originally flagged")
            if (candidate.get("source_text_sha256") != sha_bytes(source["text_en"].encode("utf-8"))
                    or candidate.get("old_text_translation_sha256") != sha_bytes(source["text_translation"].encode("utf-8"))):
                raise ValueError(f"{lang}/{key}: source or rejected-translation hash mismatch")

            replaced = {**source, "text_translation": candidate["text_translation"]}
            checks = deterministic_checks(lang, replaced)
            required = ("nonempty", "target_script_present", "protected_names_preserved",
                        "protected_numbers_preserved", "no_unrestored_marker")
            if any(checks.get(name) is not True for name in required):
                raise ValueError(f"{lang}/{key}: candidate fails recomputed deterministic checks: {checks}")
            _, pair = masked_pair(replaced)
            review_key = f"R{len(inputs) + 1:04d}"
            inputs.append({
                "review_key": review_key,
                "language": lang,
                "kind": candidate["kind"],
                "source_question_en": pair["source_english"],
                "translated_question": pair["target_translation"],
            })
            private_map.append({
                "review_key": review_key,
                "language": lang,
                "prompt_id": candidate["prompt_id"],
                "kind": candidate["kind"],
                "source_text_sha256": candidate["source_text_sha256"],
                "old_translation_sha256": candidate["old_text_translation_sha256"],
                "candidate_translation_sha256": sha_bytes(candidate["text_translation"].encode("utf-8")),
                "candidate_response_sha256": candidate["response_sha256"],
                "original_review_status": old_review["final_review_status"],
            })

    if len(inputs) != sum(EXPECTED.values()) or len({r["review_key"] for r in inputs}) != len(inputs):
        raise ValueError("Full repair review input has missing or duplicate rows")
    pilot_counts = Counter((r["language"], r["kind"]) for r in inputs)
    pilot = []
    for lang in LANGS:
        for kind in ("atomic", "composition"):
            cell = [r for r in inputs if r["language"] == lang and r["kind"] == kind]
            if len(cell) < 2:
                raise ValueError(f"Pilot needs two rows in {lang}/{kind}")
            pilot.extend(cell[:2])
    pilot_keys = {r["review_key"] for r in pilot}
    map_by_key = {r["review_key"]: r for r in private_map}

    out = HERE / "independent_review_inputs_2026-11-04"
    out.mkdir(parents=True, exist_ok=True)
    for mode, selected in (("pilot16", pilot), ("full799", inputs)):
        mapping = [map_by_key[r["review_key"]] for r in selected]
        input_path = out / f"{mode}.jsonl"
        map_path = out / f"{mode}.private_map.jsonl"
        write_jsonl(input_path, selected)
        write_jsonl(map_path, mapping)
        manifest = {
            "status": "MASKED_AI_REVIEW_INPUT_ONLY_NOT_APPROVED",
            "mode": mode,
            "rows": len(selected),
            "language_counts": dict(Counter(r["language"] for r in selected)),
            "language_kind_counts": {f"{lang}/{kind}": sum(r["language"] == lang and r["kind"] == kind for r in selected)
                                     for lang in LANGS for kind in ("atomic", "composition")},
            "candidate_report_sha256": sha_file(REPORT),
            "candidate_file_sha256": sha_file(CANDIDATES),
            "source_review_file_sha256_by_language": current_review_hashes,
            "input_sha256": sha_file(input_path),
            "private_map_sha256": sha_file(map_path),
            "answers_or_splits_in_input": False,
            "human_translation_reviewed": False,
            "purpose": "Independent AI-only semantic/fluency diagnostic of flagged ChatGPT translation repair candidates.",
            "limitations": ["AI-only screening is not fluent-human certification.",
                            "This input package does not approve translations or enable training."],
        }
        write_json(out / f"{mode}.manifest.json", manifest)

    if len(pilot_keys) != 16 or len(pilot_counts) == 0:
        raise ValueError("Pilot selection failed")
    print(json.dumps({"pilot16": str(out / "pilot16.jsonl"), "full799": str(out / "full799.jsonl"),
                      "rows": len(inputs), "language_counts": {lang: len(candidate_by_lang[lang]) for lang in LANGS},
                      "pilot_cells": 16}, indent=2))


if __name__ == "__main__":
    prepare()
