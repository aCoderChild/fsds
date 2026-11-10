#!/usr/bin/env python3
"""Stage verified diagnostic passes for a complete production translation re-screen.

This writes only the existing translation-repair overlay consumed by
collect_rows(). It never edits base translations, review receipts, or an SFT
corpus. Every staged candidate remains pending the full production review.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
PROJECT = HERE.parents[2]
INPUTS = HERE / "independent_review_inputs_2026-11-04"
REMOTE = HERE / "remote_output/full799/version_5"
MERGED = HERE / "remote_output/merged_full799/version_1"
OVERRIDE_DIR = RELEASE / "translation_flagged_repairs"
LANGS = ("fr", "de", "hi", "th")
EXPECTED_FLAGGED = {"fr": 164, "de": 179, "hi": 200, "th": 256}
PASS = "AI_ONLY_REPAIR_REVIEW_PASS"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RELEASE / "scripts"))
from analyze_chatgpt_repair_coverage import analyze, compute_coverage, read_jsonl, sha_file, unique_index  # noqa: E402
from review_training_translations_qwen import collect_rows, deterministic_checks  # noqa: E402


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_overrides(candidates: list[dict], reviews: list[dict],
                    sources_by_language: dict[str, list[dict]]) -> dict[str, list[dict]]:
    candidate_by_key = unique_index(
        candidates, lambda r: (r.get("language"), r.get("prompt_id"), r.get("kind")), "candidate"
    )
    review_by_key = unique_index(
        reviews, lambda r: (r.get("language"), r.get("prompt_id"), r.get("kind")), "review"
    )
    if set(candidate_by_key) != set(review_by_key) or len(candidates) != 799:
        raise ValueError("Candidate and verified-review keys must cover the exact 799-row queue")

    sources = {}
    for lang in LANGS:
        rows = sources_by_language.get(lang, [])
        if len(rows) != 488:
            raise ValueError(f"{lang}: expected 488 current source rows")
        sources[lang] = unique_index(rows,
            lambda r: (r.get("prompt_id"), r.get("kind")), f"{lang} source")

    overrides = {lang: [] for lang in LANGS}
    for key, candidate in candidate_by_key.items():
        lang, prompt_id, kind = key
        source = sources.get(lang, {}).get((prompt_id, kind))
        review = review_by_key[key]
        if source is None:
            raise ValueError(f"{key}: candidate does not match a current source row")
        expected = {
            "source_text_sha256": sha_text(source["text_en"]),
            "old_text_translation_sha256": sha_text(source["text_translation"]),
        }
        if (candidate.get("translation_status") != "CHATGPT_UI_REPAIR_CANDIDATE_PENDING_INDEPENDENT_REVIEW"
                or candidate.get("text_en") != source.get("text_en")
                or candidate.get("human_translation_reviewed") is not False
                or candidate.get("answers_sent_to_translator") is not False
                or any(candidate.get(field) != value for field, value in expected.items())):
            raise ValueError(f"{key}: candidate source binding or review state is invalid")
        if review.get("candidate_translation_sha256") != sha_text(candidate["text_translation"]):
            raise ValueError(f"{key}: verified review does not bind the candidate text")
        if review.get("review_status") != PASS:
            continue
        checks = deterministic_checks(lang, {**source, "text_translation": candidate["text_translation"]})
        required = ("nonempty", "target_script_present", "protected_names_preserved",
                    "protected_numbers_preserved", "no_unrestored_marker")
        if not all(checks.get(name) is True for name in required):
            raise ValueError(f"{key}: AI diagnostic pass fails deterministic source checks")
        overrides[lang].append({
            "prompt_id": prompt_id,
            "kind": kind,
            "language": lang,
            "text_translation": candidate["text_translation"],
            "translation_status": "MACHINE_DRAFT_PENDING_QWEN_RERUN",
            "translation_model": candidate["translation_model"],
            "translation_route": "ChatGPT UI candidate; NLLB back-translation and Qwen2.5-14B diagnostic pass; full production re-screen required",
            "source_text_sha256": expected["source_text_sha256"],
            "old_text_translation_sha256": expected["old_text_translation_sha256"],
            "candidate_text_translation_sha256": sha_text(candidate["text_translation"]),
            "candidate_review_status": review["review_status"],
            "human_translation_reviewed": False,
            "answers_sent_to_translator": False,
        })
    for lang in LANGS:
        if len(candidate_by_key) and sum(k[0] == lang for k in candidate_by_key) != EXPECTED_FLAGGED[lang]:
            raise ValueError(f"{lang}: candidate count differs from the frozen flagged queue")
        overrides[lang].sort(key=lambda r: (r["kind"], r["prompt_id"]))
    return overrides


def write_bundle(overrides: dict[str, list[dict]], manifest: dict,
                 output_dir: Path = OVERRIDE_DIR) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing repair overrides: {output_dir}")

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".translation_flagged_repairs.", dir=parent))
    try:
        hashes = {}
        counts = {}
        payloads = {}
        for lang in LANGS:
            payloads[lang] = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                      for row in overrides[lang])
            (staging / f"{lang}.jsonl").write_text(payloads[lang], encoding="utf-8")
            hashes[lang] = hashlib.sha256(payloads[lang].encode("utf-8")).hexdigest()
            counts[lang] = len(overrides[lang])
        complete_manifest = {
            **manifest,
            "status": "CANDIDATE_OVERRIDES_STAGED_FOR_FULL_PRODUCTION_REREVIEW_NOT_APPROVED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "overrides_by_language": counts,
            "override_sha256_by_language": hashes,
            "release_approved": False,
            "training_eligible": False,
            "human_translation_reviewed": False,
            "limitations": [
                "Only rows passing the separate diagnostic are substituted; these remain candidates.",
                "All 488 rows per language must complete the frozen TranslateGemma/Qwen7B and Qwen14B production review.",
                "This bundle does not modify canonical translation sources or create a training corpus.",
                "Review is AI-only and is not fluent-human certification.",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(complete_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_dir)
        return complete_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    coverage = analyze()  # full hash-bound verification; also writes the diagnostic coverage report
    if coverage.get("decision") != "FULL_RESCREEN_MAY_BE_WORTH_EVALUATING":
        raise ValueError("The candidate passes do not reach the optimistic complete-case coverage gate")
    candidates_path = HERE / "parsed_repair_candidates.jsonl"
    verified_path = MERGED / "merged_full799_diagnostic_reviews.jsonl"
    verification_path = MERGED / "merge_verification.json"
    queue_manifest_path = HERE / "manifest.json"
    candidate_report_path = HERE / "parsed_repair_candidates_report.json"
    sources_by_language = {}
    queue_manifest = read_json(queue_manifest_path)
    for lang in LANGS:
        rows, hashes = collect_rows(lang)
        if hashes != queue_manifest["reconstructed_source_hashes_by_language"][lang]:
            raise ValueError(f"{lang}: current reconstructed source differs from the frozen repair queue")
        sources_by_language[lang] = rows
    candidates = read_jsonl(candidates_path)
    reviews = read_jsonl(verified_path)
    overrides = build_overrides(candidates, reviews, sources_by_language)
    chatgpt_passes = {lang: set() for lang in LANGS}
    for lang in LANGS:
        original_path = RELEASE / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl"
        for row in read_jsonl(original_path):
            if row.get("final_review_status") in {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}:
                chatgpt_passes[lang].add((row["prompt_id"], row["kind"]))
        chatgpt_passes[lang].update((row["prompt_id"], row["kind"]) for row in overrides[lang])
    chatgpt_only_coverage = compute_coverage(
        chatgpt_passes,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    if not (
        all(chatgpt_only_coverage["optimistic_complete_atomic_cases"][split]["complete"] == expected
            for split, expected in {"forget": 117, "retain": 39}.items())
        and chatgpt_only_coverage["optimistic_complete_composition_chains"]["complete"] == 44
    ):
        raise ValueError("ChatGPT-only staged overrides cannot meet the complete release counts")
    report_path = HERE / "optimistic_candidate_union_coverage.json"
    manifest = {
        "coverage_report_sha256": sha_file(report_path),
        "input_receipts": {
            "queue_manifest_sha256": sha_file(queue_manifest_path),
            "candidate_report_sha256": sha_file(candidate_report_path),
            "candidate_file_sha256": sha_file(candidates_path),
            "verified_review_rows_sha256": sha_file(verified_path),
            "verifier_receipt_sha256": sha_file(verification_path),
            "source_review_file_sha256_by_language": queue_manifest["source_review_rows_sha256_by_language"],
        },
        "coverage_gate": coverage["decision"],
        "chatgpt_only_coverage": chatgpt_only_coverage,
        "candidate_review_status": coverage["candidate_review_status"],
        "staging_script_sha256": sha_file(Path(__file__).resolve()),
    }
    result = write_bundle(overrides, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
