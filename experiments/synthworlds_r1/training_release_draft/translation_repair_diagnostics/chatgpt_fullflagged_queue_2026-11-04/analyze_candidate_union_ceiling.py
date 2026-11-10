#!/usr/bin/env python3
"""Compute an optimistic, source-bound coverage ceiling for existing repair candidates.

The union treats either a ChatGPT-candidate diagnostic pass or a preliminary
TranslateGemma/Qwen7 pass as a possible replacement. It is strictly a planning
ceiling: the candidate text still requires the full frozen production review.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
PROJECT = HERE.parents[2]
INPUTS = HERE / "independent_review_inputs_2026-11-04"
PARTIAL = HERE / "remote_output/full799/version_5"
TG_REPAIR = RELEASE / "translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03"
LANGS = ("fr", "de", "hi", "th")
BASE_PASS = {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}

sys.path.insert(0, str(RELEASE / "scripts"))
from review_training_translations_qwen import collect_rows, masked_pair, read_jsonl  # noqa: E402

sys.path.insert(0, str(HERE))
from analyze_chatgpt_repair_coverage import compute_coverage, read_json  # noqa: E402


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def unique_index(rows: list[dict], key_fn, name: str) -> dict:
    result = {}
    for row in rows:
        key = key_fn(row)
        if key in result:
            raise ValueError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def main() -> dict:
    source_path = INPUTS / "full799.jsonl"
    map_path = INPUTS / "full799.private_map.jsonl"
    input_manifest_path = INPUTS / "full799.manifest.json"
    launch_path = INPUTS / "full799.kaggle_launch_manifest.json"
    source_rows = read_jsonl(source_path)
    map_rows = read_jsonl(map_path)
    source_by_key = unique_index(source_rows, lambda r: r.get("review_key"), "frozen source")
    map_by_key = unique_index(map_rows, lambda r: r.get("review_key"), "private map")
    input_manifest = read_json(input_manifest_path)
    launch = read_json(launch_path)
    if (len(source_rows) != 799 or set(source_by_key) != set(map_by_key)
            or input_manifest.get("status") != "MASKED_AI_REVIEW_INPUT_ONLY_NOT_APPROVED"
            or input_manifest.get("input_sha256") != sha_bytes(source_path.read_bytes())
            or input_manifest.get("private_map_sha256") != sha_bytes(map_path.read_bytes())
            or launch.get("mode") != "full799" or launch.get("rows") != 799
            or launch.get("input_jsonl_sha256") != sha_bytes(source_path.read_bytes())
            or launch.get("private_map_sha256") != sha_bytes(map_path.read_bytes())):
        raise ValueError("Frozen review input/map manifests do not bind the exact 799 source rows")

    # Reconstruct the candidate text and require exact source and row-map identity.
    candidate_path = HERE / "parsed_repair_candidates.jsonl"
    candidate_report_path = HERE / "parsed_repair_candidates_report.json"
    candidate_report = read_json(candidate_report_path)
    candidates = read_jsonl(candidate_path)
    candidate_by_key = unique_index(candidates,
        lambda r: (r.get("language"), r.get("prompt_id"), r.get("kind")), "ChatGPT candidate")
    if (candidate_report.get("status") != "CHATGPT_REPAIR_CANDIDATES_COMPLETE_PENDING_INDEPENDENT_REVIEW"
            or candidate_report.get("candidate_file_sha256") != sha_bytes(candidate_path.read_bytes())
            or candidate_report.get("candidate_rows") != 799 or len(candidate_by_key) != 799
            or candidate_report.get("do_not_train") is not True):
        raise ValueError("ChatGPT candidate set is not the current exact 799-row quarantined draft")

    source_by_language: dict[str, dict] = {}
    original_pass_by_language = {lang: set() for lang in LANGS}
    original_counts = {}
    review_report_path = RELEASE / "translation_reviews_roundtrip_qwen14b/report.json"
    review_report = read_json(review_report_path)
    if (review_report.get("status") != "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE"
            or review_report.get("human_translation_reviewed") is not False):
        raise ValueError("Frozen source translation review is incomplete or mislabeled")
    for lang in LANGS:
        current_rows, current_hashes = collect_rows(lang)
        current_by_key = unique_index(current_rows, lambda r: (r["prompt_id"], r["kind"]), f"{lang} source")
        review_path = RELEASE / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl"
        review_rows = read_jsonl(review_path)
        review_by_key = unique_index(review_rows,
            lambda r: (r.get("prompt_id"), r.get("kind")), f"{lang} frozen review")
        expected_review_sha = review_report.get("languages", {}).get(lang, {}).get("adjudication_sha256")
        if (len(current_by_key) != 488 or len(review_by_key) != 488
                or expected_review_sha != sha_bytes(review_path.read_bytes())):
            raise ValueError(f"{lang}: frozen source/review coverage or hash mismatch")
        for key, source in current_by_key.items():
            reviewed = review_by_key.get(key)
            if (reviewed is None or reviewed.get("text_en") != source.get("text_en")
                    or reviewed.get("text_translation") != source.get("text_translation")
                    or reviewed.get("human_translation_reviewed") is not False):
                raise ValueError(f"{lang}/{key}: frozen source review no longer matches source text")
            if reviewed.get("final_review_status") in BASE_PASS:
                original_pass_by_language[lang].add(key)
        source_by_language[lang] = current_by_key
        if current_hashes != read_json(HERE / "manifest.json")["reconstructed_source_hashes_by_language"][lang]:
            raise ValueError(f"{lang}: current source hashes differ from the frozen repair queue")
        original_counts[lang] = len(original_pass_by_language[lang])

    # Verify the partial ChatGPT review receipt and derive passes, flags, blocks and pending.
    partial_result_path = PARTIAL / "repair_review.json"
    partial_receipt_path = PARTIAL / "verification.json"
    partial_rows_path = PARTIAL / "partial_verified_repair_reviews.jsonl"
    partial_result = read_json(partial_result_path)
    partial_receipt = read_json(partial_receipt_path)
    partial_verified = read_jsonl(partial_rows_path)
    partial_records = partial_result.get("records", [])
    partial_by_key = unique_index(partial_records, lambda r: r.get("review_key"), "partial output")
    partial_verified_by_key = unique_index(partial_verified,
        lambda r: r.get("review_key"), "verified partial row")
    if (partial_receipt.get("status") != "VERIFIED_KILLED_PARTIAL_AI_ONLY_REPAIR_REVIEW"
            or partial_receipt.get("release_approved") is not False
            or partial_receipt.get("training_eligible") is not False
            or partial_receipt.get("output_sha256") != sha_bytes(partial_result_path.read_bytes())
            or partial_receipt.get("verified_rows_sha256") != sha_bytes(partial_rows_path.read_bytes())
            or partial_result.get("input_sha256") != launch.get("input_compact_array_sha256")
            or len(partial_by_key) != 799 or set(partial_by_key) != set(source_by_key)
            or len(partial_verified_by_key) != 733):
        raise ValueError("ChatGPT partial review evidence is not hash-bound to the exact 799-row source")

    merged_dir = HERE / "remote_output/merged_full799/version_1"
    merged_rows_path = merged_dir / "merged_full799_diagnostic_reviews.jsonl"
    merged_receipt_path = merged_dir / "merge_verification.json"
    merged_by_key = None
    if merged_rows_path.is_file() and merged_receipt_path.is_file():
        merged_receipt = read_json(merged_receipt_path)
        merged_rows = read_jsonl(merged_rows_path)
        merged_by_key = unique_index(merged_rows, lambda r: r.get("review_key"), "merged full review")
        merged_counts = collections.Counter(r.get("review_status") for r in merged_rows)
        if (merged_receipt.get("status") != "VERIFIED_799_ROW_AI_ONLY_DIAGNOSTIC_MERGE_NOT_RELEASE_APPROVAL"
                or merged_receipt.get("exact_source_coverage") is not True
                or merged_receipt.get("source_jsonl_sha256") != sha_bytes(source_path.read_bytes())
                or merged_receipt.get("source_private_map_sha256") != sha_bytes(map_path.read_bytes())
                or merged_receipt.get("first_run_output_sha256") != sha_bytes(partial_result_path.read_bytes())
                or merged_receipt.get("first_run_verification_sha256") != sha_bytes(partial_receipt_path.read_bytes())
                or merged_receipt.get("merged_rows_sha256") != sha_bytes(merged_rows_path.read_bytes())
                or len(merged_rows) != 799 or set(merged_by_key) != set(source_by_key)
                or dict(merged_counts) != merged_receipt.get("counts")
                or merged_receipt.get("release_approved") is not False
                or merged_receipt.get("training_eligible") is not False):
            raise ValueError("Merged full-review receipt does not bind the exact source, partial, and retry rows")

    chat_counts = collections.Counter()
    chat_pass_keys = set()
    for key, source in source_by_key.items():
        record = partial_by_key[key]
        mapped = map_by_key[key]
        decision = record.get("decision")
        if (record.get("language") != source.get("language")
                or record.get("kind") != source.get("kind")
                or record.get("source_sha256") != sha_text(source["source_question_en"])
                or record.get("translation_sha256") != sha_text(source["translated_question"])):
            raise ValueError(f"{key}: ChatGPT review output is bound to different source/translation text")
        if merged_by_key is not None:
            verified = merged_by_key[key]
            if (verified.get("prompt_id") != mapped.get("prompt_id")
                    or verified.get("candidate_translation_sha256") != mapped.get("candidate_translation_sha256")
                    or verified.get("source_sha256") != record.get("source_sha256")
                    or verified.get("translation_sha256") != record.get("translation_sha256")
                    or verified.get("review_status") not in {"AI_ONLY_REPAIR_REVIEW_PASS",
                        "AI_ONLY_REPAIR_REVIEW_FLAGGED", "AI_ONLY_REPAIR_REVIEW_BLOCKED"}):
                raise ValueError(f"{key}: merged ChatGPT review is not bound to its candidate/source")
            chat_counts[verified["review_status"]] += 1
            if verified["review_status"] == "AI_ONLY_REPAIR_REVIEW_PASS":
                chat_pass_keys.add(key)
        elif key in partial_verified_by_key:
            verified = partial_verified_by_key[key]
            if (verified.get("candidate_translation_sha256") != mapped.get("candidate_translation_sha256")
                    or verified.get("prompt_id") != mapped.get("prompt_id")):
                raise ValueError(f"{key}: verified ChatGPT decision is not bound to the candidate map")
            chat_counts[verified["review_status"]] += 1
            if verified["review_status"] == "AI_ONLY_REPAIR_REVIEW_PASS":
                chat_pass_keys.add(key)
        else:
            if isinstance(decision, dict) and "error" not in decision:
                raise ValueError(f"{key}: missing verified decision is not actually pending/blocked")
            chat_counts["PENDING_OR_UNVERIFIED"] += 1
        candidate = candidate_by_key[(source["language"], mapped["prompt_id"], source["kind"])]
        current = source_by_language[source["language"]][(mapped["prompt_id"], source["kind"])]
        _, expected_masked_pair = masked_pair({**current, "text_translation": candidate.get("text_translation", "")})
        if (candidate.get("text_en") != current["text_en"]
                or candidate.get("text_translation") is None
                or candidate.get("source_text_sha256") != sha_text(current["text_en"])
                or candidate.get("old_text_translation_sha256") != mapped.get("old_translation_sha256")
                or sha_text(candidate["text_translation"]) != mapped.get("candidate_translation_sha256")):
            raise ValueError(f"{key}: ChatGPT candidate does not match its answer-blind source-bound map")
        if (expected_masked_pair["source_english"] != source["source_question_en"]
                or expected_masked_pair["target_translation"] != source["translated_question"]):
            raise ValueError(f"{key}: masked review input does not reconstruct from the private source/candidate")

    # Independently bind the alternative TranslateGemma/Qwen7 candidate set to these rows.
    tg_rows_path = TG_REPAIR / "repair_candidates.jsonl"
    tg_report_path = TG_REPAIR / "report.json"
    tg_rows = read_jsonl(tg_rows_path)
    tg_report = read_json(tg_report_path)
    tg_by_key = unique_index(tg_rows,
        lambda r: (r.get("language"), r.get("prompt_id"), r.get("kind")), "TranslateGemma candidate")
    if (tg_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or tg_report.get("decisions_sha256") != sha_bytes(tg_rows_path.read_bytes())
            or tg_report.get("human_translation_reviewed") is not False):
        raise ValueError("TranslateGemma candidate diagnostic/report hash mismatch")

    tg_pass_keys = set()
    tg_available_key_count = 0
    for key, source in source_by_key.items():
        mapped = map_by_key[key]
        candidate = tg_by_key.get((source["language"], mapped["prompt_id"], source["kind"]))
        if candidate is None:
            continue
        if (candidate.get("source_en_sha256") != mapped.get("source_text_sha256")
                or candidate.get("old_translation_sha256") != mapped.get("old_translation_sha256")
                or candidate.get("human_translation_reviewed") is not False
                or candidate.get("answers_sent_to_reviewer") is not False):
            raise ValueError(f"{key}: TranslateGemma candidate is not source-bound/answer-blind")
        tg_available_key_count += 1
        if candidate.get("deterministic_pass") is True and candidate.get("qwen_pass") is True:
            tg_pass_keys.add(key)

    # Treat every candidate diagnostic pass as a possible replacement, never approval.
    union_keys = chat_pass_keys | tg_pass_keys
    union_by_language = {lang: set(original_pass_by_language[lang]) for lang in LANGS}
    union_kind_counts = {lang: collections.Counter() for lang in LANGS}
    for key in union_keys:
        source = source_by_key[key]
        mapped = map_by_key[key]
        union_by_language[source["language"]].add((mapped["prompt_id"], source["kind"]))
        union_kind_counts[source["language"]][source["kind"]] += 1

    coverage = compute_coverage(
        union_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    report = {
        "status": "OPTIMISTIC_CANDIDATE_UNION_CEILING_NOT_RELEASE_APPROVAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_flagged_questions": 799,
        "chatgpt_review": {
            "worker_status": ("VERIFIED_RETRY_MERGED" if merged_by_key is not None else partial_result.get("status")),
            "verified_semantic_progress": (799 if merged_by_key is not None else partial_receipt.get("semantic_review_progress")),
            "counts": dict(chat_counts),
            "diagnostic_pass_rows": len(chat_pass_keys),
            "merge_receipt_sha256": (sha_bytes(merged_receipt_path.read_bytes()) if merged_by_key is not None else None),
        },
        "translategemma_candidate_set": {
            "rows_in_diagnostic": len(tg_rows),
            "source_bound_flagged_candidates_available": tg_available_key_count,
            "qwen7_and_deterministic_pass_rows": len(tg_pass_keys),
        },
        "candidate_union": {
            "chatgpt_pass_rows": len(chat_pass_keys),
            "translategemma_pass_rows": len(tg_pass_keys),
            "overlap_rows": len(chat_pass_keys & tg_pass_keys),
            "unique_rows_with_at_least_one_diagnostic_pass": len(union_keys),
            "flagged_rows_with_no_pass_from_either_current_candidate": 799 - len(union_keys),
            "union_pass_rows_by_language_kind": {
                lang: dict(union_kind_counts[lang]) for lang in LANGS
            },
        },
        "base_source_review_pass_rows_by_language": original_counts,
        "optimistic_candidate_union_coverage": coverage,
        "source_hashes": {
            "review_input": sha_bytes(source_path.read_bytes()),
            "private_map": sha_bytes(map_path.read_bytes()),
            "partial_chatgpt_output": sha_bytes(partial_result_path.read_bytes()),
            "partial_chatgpt_verification": sha_bytes(partial_receipt_path.read_bytes()),
            "partial_chatgpt_verified_rows": sha_bytes(partial_rows_path.read_bytes()),
            "merged_chatgpt_rows": (sha_bytes(merged_rows_path.read_bytes()) if merged_by_key is not None else None),
            "merged_chatgpt_receipt": (sha_bytes(merged_receipt_path.read_bytes()) if merged_by_key is not None else None),
            "chatgpt_candidate_file": sha_bytes(candidate_path.read_bytes()),
            "translategemma_candidate_file": sha_bytes(tg_rows_path.read_bytes()),
        },
        "optimistic_only": True,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "limitations": [
            "ChatGPT and TranslateGemma candidate passes come from different AI-only criteria and candidate texts.",
            "The union only asks whether a row has any promising candidate; it is not a valid merged review.",
            "Every selected translation still requires the complete frozen production review and hash reconciliation.",
            "The candidate union is an optimistic availability ceiling; each translation still requires the frozen production screen.",
            "AI-only judgments do not certify human translation quality.",
        ],
    }
    output = HERE / "optimistic_candidate_union_coverage.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output"] = str(output)
    return report


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
