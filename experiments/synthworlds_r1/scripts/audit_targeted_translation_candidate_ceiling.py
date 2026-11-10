#!/usr/bin/env python3
"""Recompute the quarantined translation-candidate coverage ceiling.

This read-only audit combines existing source-reviewed translations, diagnostic
repair candidate sets, a verified targeted candidate review and retry, and a
verified alternate NLLB candidate screen.
It never edits translation sources or promotes a candidate into the corpus.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT / "training_release_draft"
DIAGNOSTICS = RELEASE / "translation_repair_diagnostics"
FULL = DIAGNOSTICS / "chatgpt_fullflagged_queue_2026-11-04"
TARGETED = FULL / "targeted_gap_queue_2026-11-05"
TARGET_REVIEW = TARGETED / "candidate_review_kaggle"
TARGET_RETRY = TARGETED / "targeted_gap_retry_2026-11-05"
NLLB_CANDIDATES = TARGETED / "targeted_gap_nllb_candidates_2026-11-05"
LANGUAGES = ("fr", "de", "hi", "th")
PASS_STATUSES = {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}

sys.path.insert(0, str(FULL))
sys.path.insert(0, str(RELEASE / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
from analyze_chatgpt_repair_coverage import compute_coverage  # noqa: E402
from review_training_translations_qwen import collect_rows, masked_pair, read_jsonl  # noqa: E402
import assemble_training_release as assembler  # noqa: E402
import validate_sft_source_integrity as source_integrity  # noqa: E402


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def embedded_review_inputs() -> tuple[str, list[dict]]:
    script = TARGET_REVIEW / "review.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RAW_INPUT" for target in node.targets
        ):
            raw = ast.literal_eval(node.value)
            rows = json.loads(raw)
            if not isinstance(rows, list):
                break
            return raw, rows
    raise ValueError("Targeted reviewer script has no literal RAW_INPUT array")


def main() -> dict:
    # Re-run source and translation-record validation before computing any ceiling.
    source_integrity.validate_prepared_inputs()
    assembler.checked_translation_reviews()

    review_report_path = RELEASE / "translation_reviews_roundtrip_qwen14b/report.json"
    review_report = read_json(review_report_path)
    if review_report.get("status") != "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE":
        raise ValueError("The original source translation review is incomplete")
    pass_by_language = {lang: set() for lang in LANGUAGES}
    for lang in LANGUAGES:
        rows = read_jsonl(RELEASE / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl")
        if (len(rows) != 488
                or review_report.get("languages", {}).get(lang, {}).get("adjudication_sha256")
                != sha_file(RELEASE / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl")):
            raise ValueError(f"{lang}: source review coverage or receipt hash mismatch")
        for row in rows:
            if row.get("final_review_status") in PASS_STATUSES:
                pass_by_language[lang].add((row["prompt_id"], row["kind"]))

    merged_path = FULL / "remote_output/merged_full799/version_1/merged_full799_diagnostic_reviews.jsonl"
    merged_receipt_path = FULL / "remote_output/merged_full799/version_1/merge_verification.json"
    merged_receipt = read_json(merged_receipt_path)
    if (merged_receipt.get("status") != "VERIFIED_799_ROW_AI_ONLY_DIAGNOSTIC_MERGE_NOT_RELEASE_APPROVAL"
            or merged_receipt.get("exact_source_coverage") is not True
            or merged_receipt.get("merged_rows_sha256") != sha_file(merged_path)
            or merged_receipt.get("training_eligible") is not False):
        raise ValueError("The merged 799-row repair-review receipt is invalid")
    chatgpt_passes = Counter()
    for row in read_jsonl(merged_path):
        if row.get("review_status") == "AI_ONLY_REPAIR_REVIEW_PASS":
            pass_by_language[row["language"]].add((row["prompt_id"], row["kind"]))
            chatgpt_passes[row["language"]] += 1

    tg_path = DIAGNOSTICS / "translategemma4b_qwen7b_all_flagged_2026-11-03/repair_candidates.jsonl"
    tg_report_path = tg_path.with_name("report.json")
    tg_report = read_json(tg_report_path)
    if (tg_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or tg_report.get("decisions_sha256") != sha_file(tg_path)
            or tg_report.get("human_translation_reviewed") is not False
            or tg_report.get("answers_sent_to_reviewer") is not False):
        raise ValueError("The TranslateGemma candidate receipt is invalid")
    tg_passes = Counter()
    for row in read_jsonl(tg_path):
        if row.get("deterministic_pass") is True and row.get("qwen_pass") is True:
            pass_by_language[row["language"]].add((row["prompt_id"], row["kind"]))
            tg_passes[row["language"]] += 1

    base_sizes = {lang: len(rows) for lang, rows in pass_by_language.items()}
    base_coverage = compute_coverage(
        pass_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    saved_ceiling = read_json(FULL / "optimistic_candidate_union_coverage.json")
    if (saved_ceiling.get("status") != "OPTIMISTIC_CANDIDATE_UNION_CEILING_NOT_RELEASE_APPROVAL"
            or saved_ceiling.get("optimistic_candidate_union_coverage") != base_coverage
            or saved_ceiling.get("release_approved") is not False):
        raise ValueError("Recomputed candidate union differs from its saved diagnostic ceiling")

    # Verify the targeted result against its input manifest, receipt, and exact source rows.
    candidate_path = TARGET_REVIEW / "remote_output/targeted_gap_review.json"
    receipt_path = TARGET_REVIEW / "review_verification.json"
    input_manifest_path = TARGET_REVIEW / "review_input_manifest.json"
    targeted = read_json(candidate_path)
    receipt = read_json(receipt_path)
    input_manifest = read_json(input_manifest_path)
    input_raw, input_rows = embedded_review_inputs()
    input_by_key = {row["review_key"]: row for row in input_rows}
    private_rows = read_jsonl(TARGETED / "private_row_map.jsonl")
    private_by_key = {row["row_id"]: row for row in private_rows}
    local_gap_rows = read_jsonl(TARGETED / "local_gap_rows.jsonl")
    gap_report = read_json(TARGETED / "gap_report.json")
    review_script_path = TARGET_REVIEW / "review.py"
    source_by_key = {}
    candidate_by_row_id = {}
    for lang in LANGUAGES:
        source_rows, _ = collect_rows(lang)
        source_by_key.update({(lang, row["prompt_id"], row["kind"]): row for row in source_rows})
        candidate_by_row_id.update({(lang, row["row_id"]): row for row in read_jsonl(TARGETED / f"candidates/{lang}.jsonl")})
    if (targeted.get("status") != "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS"
            or targeted.get("input_sha256") != input_manifest.get("review_input_sha256")
            or sha_text(input_raw) != input_manifest.get("review_input_sha256")
            or receipt.get("status") != "VERIFIED_AI_ONLY_DIAGNOSTIC_QUARANTINED"
            or receipt.get("artifact_sha256") != sha_file(candidate_path)
            or receipt.get("input_sha256") != input_manifest.get("review_input_sha256")
            or input_manifest.get("review_script_sha256") != sha_file(review_script_path)
            or receipt.get("errors") != []
            or input_manifest.get("review_rows") != 187
            or receipt.get("reviewed_input_rows") != 187
            or receipt.get("record_rows") != 187
            or gap_report.get("status") != "VERIFIED_TARGETED_TRANSLATION_GAP_QUEUE_ONLY_NOT_APPROVAL"
            or gap_report.get("queue_row_map_sha256") != sha_file(TARGETED / "private_row_map.jsonl")
            or input_manifest.get("release_approved") is not False
            or input_manifest.get("training_eligible") is not False
            or len(input_rows) != 187 or len(private_rows) != 187
            or len(local_gap_rows) != 187 or len(input_by_key) != 187 or len(private_by_key) != 187):
        raise ValueError("Targeted candidate review is incomplete or not hash-bound")

    targeted_passes = Counter()
    for local, private in zip(local_gap_rows, private_rows):
        identity = (private.get("language"), private.get("prompt_id"), private.get("kind"))
        source = source_by_key.get(identity)
        candidate = candidate_by_row_id.get((private.get("language"), private.get("row_id")))
        if (not source or not candidate
                or private.get("row_id") not in input_by_key
                or local.get("language") != identity[0]
                or local.get("prompt_id") != identity[1]
                or local.get("kind") != identity[2]
                or private.get("source_text_sha256") != sha_text(source["text_en"])
                or candidate.get("row_id") != private.get("row_id")
                or not isinstance(candidate.get("translation"), str)
                or not candidate["translation"].strip()):
            raise ValueError(f"Targeted queue mapping/source hashes fail for {private.get('row_id')}")
        expected_source = masked_pair({**source, "text_translation": candidate["translation"]})[1]["source_english"]
        input_row = input_by_key[private["row_id"]]
        if (expected_source != input_row["source_question_en"]
                or candidate["translation"] != input_row["translated_question"]):
            raise ValueError(f"Targeted review source text does not reconstruct for {private.get('row_id')}")

    for record in targeted.get("records", []):
        key = record.get("review_key")
        source = input_by_key.get(key)
        private = private_by_key.get(key)
        decision = record.get("decision")
        if (source is None or private is None or private.get("row_id") != key
                or record.get("source_sha256") != sha_text(source["source_question_en"])
                or record.get("translation_sha256") != sha_text(source["translated_question"])):
            raise ValueError(f"Targeted review row {key} does not match its blinded source/candidate")
        if (record.get("backtranslation_error") is None and isinstance(decision, dict)
                and decision.get("semantic_pass") is True and decision.get("fluency_pass") is True):
            lang, prompt_id, kind = private["language"], private["prompt_id"], private["kind"]
            row_key = (prompt_id, kind)
            if row_key in pass_by_language[lang]:
                raise ValueError(f"Targeted queue row {key} was already counted in the previous candidate union")
            pass_by_language[lang].add(row_key)
            targeted_passes[lang] += 1

    targeted_coverage = compute_coverage(
        pass_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    targeted_union_sizes = {lang: len(rows) for lang, rows in pass_by_language.items()}

    # Recheck the narrow retry output and bind every repaired decision to the
    # exact previously blocked source/candidate pair.
    retry_input_path = TARGET_RETRY / "retry_input.json"
    retry_input_manifest_path = TARGET_RETRY / "retry_input_manifest.json"
    retry_output_path = TARGET_RETRY / "remote_output/retry_output.json"
    retry_receipt_path = TARGET_RETRY / "remote_output/retry_verification.json"
    retry_script_path = TARGET_RETRY / "retry.py"
    pulled_scripts = list((TARGET_RETRY / "remote_source").glob("*.py"))
    retry_input = read_json(retry_input_path)
    retry_manifest = read_json(retry_input_manifest_path)
    retry_output = read_json(retry_output_path)
    retry_receipt = read_json(retry_receipt_path)
    retry_input_hash = sha_file(retry_input_path)
    retry_code_hash = sha_file(retry_script_path)
    retry_rows_by_key = {row["review_key"]: row for row in retry_input.get("rows", [])}
    retry_records = retry_output.get("records", [])
    retry_records_by_key = {row.get("review_key"): row for row in retry_records}
    expected_blocked_keys = {
        record["review_key"] for record in targeted.get("records", [])
        if record.get("backtranslation_error") is not None
        or "error" in (record.get("decision") or {})
    }
    expected_nllb_revision = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
    expected_qwen_revision = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    if (retry_input.get("schema") != "synthworlds-targeted-gap-retry-v1"
            or retry_input.get("mode") != "AI_ONLY_DIAGNOSTIC_RETRY_NO_RELEASE_APPROVAL"
            or retry_input_hash != retry_manifest.get("input_sha256")
            or retry_input.get("source_input_sha256") != input_manifest.get("review_input_sha256")
            or retry_manifest.get("retry_script_sha256") != retry_code_hash
            or retry_manifest.get("kernel_id") != "bunnoob2005/sw-gap187-alias-retry-2026-11-05"
            or len(pulled_scripts) != 1
            or sha_file(pulled_scripts[0]) != retry_code_hash
            or retry_receipt.get("status") != "VERIFIED_AI_ONLY_RETRY_DIAGNOSTIC_QUARANTINED"
            or retry_receipt.get("input_sha256") != retry_input_hash
            or retry_receipt.get("prior_output_sha256") != sha_file(candidate_path)
            or retry_receipt.get("output_sha256") != sha_file(retry_output_path)
            or retry_receipt.get("submitted_script_sha256") != retry_code_hash
            or retry_receipt.get("pulled_script_sha256") != retry_code_hash
            or retry_output.get("input_sha256") != retry_input_hash
            or retry_output.get("retry_input_payload") != retry_input
            or retry_output.get("nllb_model_revision") != expected_nllb_revision
            or retry_output.get("qwen_model_revision") != expected_qwen_revision
            or retry_output.get("status") not in {
                "AI_ONLY_DIAGNOSTIC_RETRY_COMPLETE", "AI_ONLY_DIAGNOSTIC_RETRY_HAS_BLOCKED_ROWS"
            }
            or retry_output.get("human_translation_reviewed") is not False
            or retry_output.get("release_approved") is not False
            or retry_output.get("training_eligible") is not False
            or retry_output.get("answers_or_split_labels_included") is not False
            or retry_receipt.get("human_translation_reviewed") is not False
            or retry_receipt.get("release_approved") is not False
            or retry_receipt.get("training_eligible") is not False
            or retry_receipt.get("answers_or_split_labels_included") is not False
            or len(retry_rows_by_key) != 17
            or set(retry_rows_by_key) != expected_blocked_keys
            or len(retry_records_by_key) != len(retry_records)
            or set(retry_records_by_key) != expected_blocked_keys
            or retry_receipt.get("gpu") != ["Tesla T4", "Tesla T4"]):
        raise ValueError("Targeted retry receipt/input/source-code binding is invalid")

    retry_counts = Counter()
    retry_language_counts = {lang: Counter() for lang in LANGUAGES}
    retry_new_passes = Counter()
    retry_overlap_passes = Counter()
    for key, item in retry_rows_by_key.items():
        original = input_by_key.get(key)
        record = retry_records_by_key[key]
        private = private_by_key.get(key)
        if (original is None or private is None or private.get("row_id") != key
                or item.get("language") != original.get("language")
                or item.get("kind") != original.get("kind")
                or item.get("source_question_en") != original.get("source_question_en")
                or item.get("translated_question") != original.get("translated_question")
                or item.get("source_sha256") != sha_text(original["source_question_en"])
                or item.get("translation_sha256") != sha_text(original["translated_question"])
                or record.get("source_sha256") != item.get("source_sha256")
                or record.get("translation_sha256") != item.get("translation_sha256")
                or record.get("retry_backtranslation") != item.get("retry_backtranslation")):
            raise ValueError(f"Retry row {key} does not reconstruct from the original blinded queue")
        if record.get("backtranslation_error") is None:
            backtranslation = record.get("backtranslation_en")
            if (not isinstance(backtranslation, str) or not backtranslation.strip()
                    or record.get("backtranslation_sha256") != sha_text(backtranslation)
                    or Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", backtranslation))
                    != Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", original["source_question_en"]))):
                raise ValueError(f"Retry row {key} has invalid restored placeholder or back-translation hashes")
            if item.get("retry_backtranslation") == "REUSE_VERIFIED_PRIOR":
                if (backtranslation != item.get("prior_backtranslation_en")
                        or record.get("backtranslation_sha256") != item.get("prior_backtranslation_sha256")):
                    raise ValueError(f"Retry row {key} does not preserve the prior verified back-translation")
        decision = record.get("decision")
        lang = item["language"]
        if record.get("backtranslation_error") is not None or not isinstance(decision, dict) or "error" in decision:
            retry_counts["blocked"] += 1
            retry_language_counts[lang]["blocked"] += 1
            continue
        if (decision.get("key") != key or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
                or len(decision["reason"]) > 120):
            raise ValueError(f"Retry row {key} has an invalid decision object")
        retry_counts["reviewed"] += 1
        retry_language_counts[lang]["reviewed"] += 1
        retry_counts["semantic_pass"] += decision["semantic_pass"]
        retry_language_counts[lang]["semantic_pass"] += decision["semantic_pass"]
        retry_counts["fluency_pass"] += decision["fluency_pass"]
        retry_language_counts[lang]["fluency_pass"] += decision["fluency_pass"]
        joint_pass = decision["semantic_pass"] and decision["fluency_pass"]
        retry_counts["joint_pass"] += joint_pass
        retry_language_counts[lang]["joint_pass"] += joint_pass
        if joint_pass:
            row_key = (private["prompt_id"], private["kind"])
            if row_key in pass_by_language[lang]:
                retry_overlap_passes[lang] += 1
            else:
                pass_by_language[lang].add(row_key)
                retry_new_passes[lang] += 1
    if (retry_receipt.get("counts", {}).get("reviewed", 0) != retry_counts["reviewed"]
            or retry_receipt.get("counts", {}).get("blocked", 0) != retry_counts["blocked"]
            or retry_receipt.get("counts", {}).get("semantic_pass", 0) != retry_counts["semantic_pass"]
            or retry_receipt.get("counts", {}).get("fluency_pass", 0) != retry_counts["fluency_pass"]
            or retry_receipt.get("counts", {}).get("joint_pass", 0) != retry_counts["joint_pass"]
            or retry_output.get("reviewed_rows") != retry_counts["reviewed"]
            or retry_output.get("blocked_rows") != retry_counts["blocked"]):
        raise ValueError("Retry summary counts do not match recomputed row-level decisions")

    post_retry_coverage = compute_coverage(
        pass_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    post_retry_union_sizes = {lang: len(rows) for lang, rows in pass_by_language.items()}

    # Verify the independent NLLB candidate screen. This remains quarantined;
    # only valid semantic+fluency decisions are added to this planning ceiling.
    nllb_input_path = NLLB_CANDIDATES / "candidate_input.json"
    nllb_manifest_path = NLLB_CANDIDATES / "candidate_input_manifest.json"
    nllb_script_path = NLLB_CANDIDATES / "candidate_review.py"
    nllb_output_path = NLLB_CANDIDATES / "remote_output/candidate_review.json"
    nllb_receipt_path = NLLB_CANDIDATES / "remote_output/candidate_verification.json"
    nllb_input = read_json(nllb_input_path)
    nllb_manifest = read_json(nllb_manifest_path)
    nllb_output = read_json(nllb_output_path)
    nllb_receipt = read_json(nllb_receipt_path)
    nllb_pulled_scripts = list((NLLB_CANDIDATES / "remote_source").glob("*.py"))
    nllb_input_hash = sha_file(nllb_input_path)
    nllb_script_hash = sha_file(nllb_script_path)
    nllb_retry_pass_keys = {
        key for key, row in retry_records_by_key.items()
        if row.get("backtranslation_error") is None
        and isinstance(row.get("decision"), dict)
        and row["decision"].get("semantic_pass") is True
        and row["decision"].get("fluency_pass") is True
    }
    nllb_prior_pass_keys = {
        row["review_key"] for row in targeted.get("records", [])
        if row.get("backtranslation_error") is None
        and isinstance(row.get("decision"), dict)
        and row["decision"].get("semantic_pass") is True
        and row["decision"].get("fluency_pass") is True
    } | nllb_retry_pass_keys
    nllb_expected_rows = [
        {
            "review_key": row["review_key"],
            "language": row["language"],
            "kind": row["kind"],
            "source_question_en": row["source_question_en"],
            "source_sha256": sha_text(row["source_question_en"]),
        }
        for key, row in input_by_key.items() if key not in nllb_prior_pass_keys
    ]
    nllb_expected_rows.sort(key=lambda row: row["review_key"])
    nllb_source_hashes = {
        "source_review_input_sha256": sha_text(input_raw),
        "prior_review_output_sha256": sha_file(candidate_path),
        "prior_retry_output_sha256": sha_file(retry_output_path),
        "row_map_sha256": sha_file(TARGETED / "private_row_map.jsonl"),
    }
    nllb_expected_revision = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
    qwen_expected_revision = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    if (nllb_input.get("schema") != "synthworlds-nllb-alternative-translation-v1"
            or nllb_input.get("mode") != "AI_ONLY_DIAGNOSTIC_CANDIDATE_NO_RELEASE_APPROVAL"
            or nllb_input.get("rows") != nllb_expected_rows
            or len(nllb_expected_rows) != 148
            or nllb_input_hash != nllb_manifest.get("input_sha256")
            or any(nllb_input.get(key) != value for key, value in nllb_source_hashes.items())
            or nllb_manifest.get("previous_joint_passes_excluded") != len(nllb_prior_pass_keys)
            or nllb_manifest.get("rows") != 148
            or nllb_manifest.get("candidate_script_sha256") != nllb_script_hash
            or nllb_manifest.get("kernel_id") != "bunnoob2005/sw-gap187-nllb-148-candidate-review-2026-11-05"
            or len(nllb_pulled_scripts) != 1
            or sha_file(nllb_pulled_scripts[0]) != nllb_script_hash):
        raise ValueError("NLLB candidate input selection or submitted source binding is invalid")
    if (nllb_output.get("status") not in {
                "AI_ONLY_NLLB_CANDIDATE_REVIEW_COMPLETE",
                "AI_ONLY_NLLB_CANDIDATE_REVIEW_HAS_BLOCKS",
            }
            or nllb_output.get("input_sha256") != nllb_input_hash
            or nllb_output.get("input_payload") != nllb_input
            or nllb_output.get("nllb_model_id") != "facebook/nllb-200-distilled-600M"
            or nllb_output.get("nllb_model_revision") != nllb_expected_revision
            or nllb_output.get("qwen_model_id") != "Qwen/Qwen2.5-14B-Instruct"
            or nllb_output.get("qwen_model_revision") != qwen_expected_revision
            or nllb_output.get("human_translation_reviewed") is not False
            or nllb_output.get("answers_or_split_labels_included") is not False
            or nllb_output.get("release_approved") is not False
            or nllb_output.get("training_eligible") is not False
            or nllb_output.get("completed_at_utc") is None
            or nllb_receipt.get("status") != "VERIFIED_AI_ONLY_NLLB_CANDIDATES_QUARANTINED"
            or nllb_receipt.get("input_sha256") != nllb_input_hash
            or nllb_receipt.get("output_sha256") != sha_file(nllb_output_path)
            or nllb_receipt.get("submitted_script_sha256") != nllb_script_hash
            or nllb_receipt.get("pulled_script_sha256") != nllb_script_hash
            or nllb_receipt.get("human_translation_reviewed") is not False
            or nllb_receipt.get("answers_or_split_labels_included") is not False
            or nllb_receipt.get("release_approved") is not False
            or nllb_receipt.get("training_eligible") is not False
            or nllb_receipt.get("gpu") != ["Tesla T4", "Tesla T4"]):
        raise ValueError("NLLB candidate output/receipt provenance or diagnostic-only scope is invalid")

    nllb_rows_by_key = {row["review_key"]: row for row in nllb_expected_rows}
    nllb_records = nllb_output.get("records", [])
    nllb_records_by_key = {row.get("review_key"): row for row in nllb_records}
    if (len(nllb_records) != 148 or len(nllb_records_by_key) != 148
            or set(nllb_records_by_key) != set(nllb_rows_by_key)):
        raise ValueError("NLLB candidate records do not exactly cover the selected 148 rows")
    nllb_counts = Counter()
    nllb_language_counts = {lang: Counter() for lang in LANGUAGES}
    nllb_error_counts = Counter()
    nllb_new_passes = Counter()
    for key, source in nllb_rows_by_key.items():
        record = nllb_records_by_key[key]
        lang = source["language"]
        if (record.get("language") != lang or record.get("kind") != source["kind"]
                or record.get("source_sha256") != source["source_sha256"]
                or source["source_sha256"] != sha_text(source["source_question_en"])):
            raise ValueError(f"NLLB candidate {key} source identity/hash mismatch")
        candidate_error = record.get("candidate_error")
        if candidate_error is None:
            candidate = record.get("candidate_translation")
            if (not isinstance(candidate, str) or not candidate.strip()
                    or record.get("candidate_sha256") != sha_text(candidate)
                    or Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", candidate))
                    != Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", source["source_question_en"]))):
                raise ValueError(f"NLLB candidate {key} has invalid text/hash/placeholder coverage")
            nllb_counts["candidate_rows"] += 1
            nllb_language_counts[lang]["candidate_rows"] += 1
        else:
            if record.get("candidate_translation") or record.get("candidate_sha256"):
                raise ValueError(f"NLLB candidate {key} failed but includes a translation/hash")
            nllb_error_counts["candidate_error"] += 1
        if record.get("backtranslation_error") is not None:
            nllb_error_counts["backtranslation_error"] += 1
        decision = record.get("decision")
        if candidate_error is not None or record.get("backtranslation_error") is not None:
            if not isinstance(decision, dict) or "error" not in decision:
                raise ValueError(f"NLLB candidate {key} technical failure is not marked blocked")
            nllb_counts["blocked"] += 1
            nllb_language_counts[lang]["blocked"] += 1
            continue
        backtranslation = record.get("backtranslation_en")
        if (not isinstance(backtranslation, str) or not backtranslation.strip()
                or record.get("backtranslation_sha256") != sha_text(backtranslation)
                or Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", backtranslation))
                != Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", source["source_question_en"]))):
            raise ValueError(f"NLLB candidate {key} has invalid back-translation/hash/placeholder coverage")
        if not isinstance(decision, dict) or "error" in decision:
            nllb_counts["blocked"] += 1
            nllb_language_counts[lang]["blocked"] += 1
            nllb_error_counts["judge_error"] += 1
            continue
        if (decision.get("key") != key or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
                or len(decision["reason"]) > 120
                or not isinstance(record.get("raw_output"), str) or not record["raw_output"].strip()):
            raise ValueError(f"NLLB candidate {key} has an invalid judge decision")
        nllb_counts["reviewed"] += 1
        nllb_language_counts[lang]["reviewed"] += 1
        nllb_counts["semantic_pass"] += decision["semantic_pass"]
        nllb_language_counts[lang]["semantic_pass"] += decision["semantic_pass"]
        nllb_counts["fluency_pass"] += decision["fluency_pass"]
        nllb_language_counts[lang]["fluency_pass"] += decision["fluency_pass"]
        joint = decision["semantic_pass"] and decision["fluency_pass"]
        nllb_counts["joint_pass"] += joint
        nllb_language_counts[lang]["joint_pass"] += joint
        if joint:
            private = private_by_key[key]
            row_key = (private["prompt_id"], private["kind"])
            if row_key in pass_by_language[lang]:
                raise ValueError(f"NLLB candidate {key} duplicates an already-counted candidate pass")
            pass_by_language[lang].add(row_key)
            nllb_new_passes[lang] += 1

    expected_nllb_counts = dict(nllb_counts)
    if (nllb_output.get("candidate_rows") != nllb_counts["candidate_rows"]
            or nllb_output.get("reviewed_rows") != nllb_counts["reviewed"]
            or nllb_output.get("blocked_rows") != nllb_counts["blocked"]
            or nllb_receipt.get("counts") != expected_nllb_counts
            or nllb_receipt.get("counts_by_language") != {
                lang: dict(counts) for lang, counts in nllb_language_counts.items()
            }):
        raise ValueError("NLLB candidate output/verification summary differs from row-level recomputation")

    post_nllb_coverage = compute_coverage(
        pass_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    post_nllb_union_sizes = {lang: len(rows) for lang, rows in pass_by_language.items()}

    # Verify the marker-preserving retry against exactly the technically
    # blocked rows from the first NLLB screen, then extend the diagnostic ceiling.
    SEGMENTED_RETRY = TARGETED / "targeted_gap_nllb_segmented_retry_2026-11-05"
    segmented_input_path = SEGMENTED_RETRY / "segmented_retry_input.json"
    segmented_manifest_path = SEGMENTED_RETRY / "segmented_retry_input_manifest.json"
    segmented_script_path = SEGMENTED_RETRY / "segmented_retry.py"
    segmented_output_path = SEGMENTED_RETRY / "remote_output/candidate_review.json"
    segmented_receipt_path = SEGMENTED_RETRY / "remote_output/segmented_retry_verification.json"
    segmented_input = read_json(segmented_input_path)
    segmented_manifest = read_json(segmented_manifest_path)
    segmented_output = read_json(segmented_output_path)
    segmented_receipt = read_json(segmented_receipt_path)
    segmented_pulled_scripts = list((SEGMENTED_RETRY / "remote_source").glob("*.py"))
    segmented_input_hash = sha_file(segmented_input_path)
    segmented_script_hash = sha_file(segmented_script_path)
    segmented_blocked_keys = {
        key for key, record in nllb_records_by_key.items()
        if record.get("candidate_error") is not None
        or record.get("backtranslation_error") is not None
        or "error" in (record.get("decision") or {})
    }
    segmented_expected_rows = [
        nllb_rows_by_key[key] for key in sorted(segmented_blocked_keys)
    ]
    segmented_source_hashes = {
        "previous_candidate_input_sha256": nllb_input_hash,
        "previous_candidate_output_sha256": sha_file(nllb_output_path),
        "previous_candidate_receipt_sha256": sha_file(nllb_receipt_path),
        "row_map_sha256": sha_file(TARGETED / "private_row_map.jsonl"),
    }
    if (segmented_input.get("schema") != "synthworlds-nllb-segmented-retry-v1"
            or segmented_input.get("mode") != "AI_ONLY_DIAGNOSTIC_SEGMENTED_RETRY_NO_RELEASE_APPROVAL"
            or segmented_input.get("expected_rows") != 125
            or segmented_input.get("rows") != segmented_expected_rows
            or len(segmented_expected_rows) != 125
            or any(segmented_input.get(key) != value for key, value in segmented_source_hashes.items())
            or segmented_input.get("source_review_input_sha256") != nllb_input.get("source_review_input_sha256")
            or segmented_input_hash != segmented_manifest.get("input_sha256")
            or segmented_manifest.get("status") != "VERIFIED_ANSWER_BLIND_SEGMENTED_RETRY_INPUT"
            or segmented_manifest.get("blocked_rows_selected") != 125
            or segmented_manifest.get("rows") != 125
            or segmented_manifest.get("script_sha256") != segmented_script_hash
            or segmented_manifest.get("kernel_id") != "bunnoob2005/sw-gap187-nllb-segmented-retry-2026-11-05"
            or len(segmented_pulled_scripts) != 1
            or sha_file(segmented_pulled_scripts[0]) != segmented_script_hash):
        raise ValueError("Segmented retry input is not bound to the exact 125 blocked NLLB rows/source/script")
    if (segmented_output.get("status") not in {
                "AI_ONLY_NLLB_SEGMENTED_RETRY_COMPLETE",
                "AI_ONLY_NLLB_SEGMENTED_RETRY_HAS_BLOCKS",
            }
            or segmented_output.get("input_sha256") != segmented_input_hash
            or segmented_output.get("input_payload") != segmented_input
            or segmented_output.get("nllb_model_id") != "facebook/nllb-200-distilled-600M"
            or segmented_output.get("nllb_model_revision") != expected_nllb_revision
            or segmented_output.get("qwen_model_id") != "Qwen/Qwen2.5-14B-Instruct"
            or segmented_output.get("qwen_model_revision") != expected_qwen_revision
            or segmented_output.get("human_translation_reviewed") is not False
            or segmented_output.get("answers_or_split_labels_included") is not False
            or segmented_output.get("release_approved") is not False
            or segmented_output.get("training_eligible") is not False
            or not segmented_output.get("completed_at_utc")
            or segmented_receipt.get("status") != "VERIFIED_AI_ONLY_NLLB_SEGMENTED_RETRY_QUARANTINED"
            or segmented_receipt.get("input_sha256") != segmented_input_hash
            or segmented_receipt.get("output_sha256") != sha_file(segmented_output_path)
            or segmented_receipt.get("submitted_script_sha256") != segmented_script_hash
            or segmented_receipt.get("pulled_script_sha256") != segmented_script_hash
            or segmented_receipt.get("previous_candidate_output_sha256") != sha_file(nllb_output_path)
            or segmented_receipt.get("human_translation_reviewed") is not False
            or segmented_receipt.get("answers_or_split_labels_included") is not False
            or segmented_receipt.get("release_approved") is not False
            or segmented_receipt.get("training_eligible") is not False
            or segmented_receipt.get("gpu") != ["Tesla T4", "Tesla T4"]):
        raise ValueError("Segmented retry output/receipt provenance or diagnostic-only scope is invalid")

    segmented_rows_by_key = {row["review_key"]: row for row in segmented_expected_rows}
    segmented_records = segmented_output.get("records", [])
    segmented_records_by_key = {row.get("review_key"): row for row in segmented_records}
    if (len(segmented_records) != 125 or len(segmented_records_by_key) != 125
            or set(segmented_records_by_key) != set(segmented_rows_by_key)):
        raise ValueError("Segmented retry records do not exactly cover the blocked row set")
    segmented_counts = Counter()
    segmented_language_counts = {lang: Counter() for lang in LANGUAGES}
    segmented_error_counts = Counter()
    segmented_new_passes = Counter()
    marker_pattern = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
    for key, source in segmented_rows_by_key.items():
        record = segmented_records_by_key[key]
        lang = source["language"]
        if (record.get("language") != lang or record.get("kind") != source["kind"]
                or record.get("source_sha256") != source["source_sha256"]
                or source["source_sha256"] != sha_text(source["source_question_en"])):
            raise ValueError(f"Segmented retry row {key} source identity/hash mismatch")
        source_markers = Counter(marker_pattern.findall(source["source_question_en"]))
        candidate_error = record.get("candidate_error")
        if not source_markers and candidate_error is None:
            raise ValueError(f"Segmented retry row {key} without a protected marker was not blocked")
        if candidate_error is None:
            candidate = record.get("candidate_translation")
            if (not isinstance(candidate, str) or not candidate.strip()
                    or record.get("candidate_sha256") != sha_text(candidate)
                    or Counter(marker_pattern.findall(candidate)) != source_markers):
                raise ValueError(f"Segmented retry row {key} candidate/hash/marker multiset invalid")
            segmented_counts["candidate_rows"] += 1
            segmented_language_counts[lang]["candidate_rows"] += 1
        else:
            if record.get("candidate_translation") or record.get("candidate_sha256"):
                raise ValueError(f"Segmented retry row {key} candidate failure includes a translation/hash")
            segmented_error_counts["candidate_error"] += 1
        if record.get("backtranslation_error") is not None:
            segmented_error_counts["backtranslation_error"] += 1
        decision = record.get("decision")
        if candidate_error is not None or record.get("backtranslation_error") is not None:
            if not isinstance(decision, dict) or "error" not in decision:
                raise ValueError(f"Segmented retry row {key} technical failure was not marked blocked")
            segmented_counts["blocked"] += 1
            segmented_language_counts[lang]["blocked"] += 1
            continue
        backtranslation = record.get("backtranslation_en")
        if (not isinstance(backtranslation, str) or not backtranslation.strip()
                or record.get("backtranslation_sha256") != sha_text(backtranslation)
                or Counter(marker_pattern.findall(backtranslation)) != source_markers):
            raise ValueError(f"Segmented retry row {key} back-translation/hash/marker multiset invalid")
        if not isinstance(decision, dict) or "error" in decision:
            segmented_counts["blocked"] += 1
            segmented_language_counts[lang]["blocked"] += 1
            segmented_error_counts["judge_error"] += 1
            continue
        if (decision.get("key") != key or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
                or len(decision["reason"]) > 120
                or not isinstance(record.get("raw_output"), str) or not record["raw_output"].strip()):
            raise ValueError(f"Segmented retry row {key} has an invalid Qwen decision")
        segmented_counts["reviewed"] += 1
        segmented_language_counts[lang]["reviewed"] += 1
        segmented_counts["semantic_pass"] += decision["semantic_pass"]
        segmented_language_counts[lang]["semantic_pass"] += decision["semantic_pass"]
        segmented_counts["fluency_pass"] += decision["fluency_pass"]
        segmented_language_counts[lang]["fluency_pass"] += decision["fluency_pass"]
        joint = decision["semantic_pass"] and decision["fluency_pass"]
        segmented_counts["joint_pass"] += joint
        segmented_language_counts[lang]["joint_pass"] += joint
        if joint:
            private = private_by_key[key]
            row_key = (private["prompt_id"], private["kind"])
            if row_key in pass_by_language[lang]:
                raise ValueError(f"Segmented retry row {key} duplicates an already-counted candidate pass")
            pass_by_language[lang].add(row_key)
            segmented_new_passes[lang] += 1
    if (segmented_output.get("candidate_rows") != segmented_counts["candidate_rows"]
            or segmented_output.get("reviewed_rows") != segmented_counts["reviewed"]
            or segmented_output.get("blocked_rows") != segmented_counts["blocked"]
            or segmented_receipt.get("counts") != dict(segmented_counts)
            or segmented_receipt.get("counts_by_language") != {
                lang: dict(counts) for lang, counts in segmented_language_counts.items()
            }
            or segmented_receipt.get("error_counts") != dict(segmented_error_counts)):
        raise ValueError("Segmented retry receipt counts do not match row-level decisions")

    post_segmented_coverage = compute_coverage(
        pass_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    return {
        "status": "VERIFIED_OPTIMISTIC_CANDIDATE_CEILING_ONLY_NOT_RELEASE_APPROVAL",
        "base_union_rows_by_language": base_sizes,
        "targeted_new_joint_pass_rows_by_language": dict(targeted_passes),
        "augmented_union_rows_by_language": targeted_union_sizes,
        "chatgpt_fullflagged_pass_rows_by_language": dict(chatgpt_passes),
        "translategemma_fullflagged_pass_rows_by_language": dict(tg_passes),
        "targeted_review_counts": receipt.get("counts"),
        "base_optimistic_coverage": base_coverage,
        "augmented_optimistic_coverage": targeted_coverage,
        "targeted_retry_counts": dict(retry_counts),
        "targeted_retry_counts_by_language": {lang: dict(counts) for lang, counts in retry_language_counts.items()},
        "targeted_retry_new_joint_pass_rows_by_language": dict(retry_new_passes),
        "targeted_retry_joint_pass_overlap_by_language": dict(retry_overlap_passes),
        "post_retry_union_rows_by_language": post_retry_union_sizes,
        "post_retry_optimistic_coverage": post_retry_coverage,
        "nllb_candidate_counts": dict(nllb_counts),
        "nllb_candidate_counts_by_language": {lang: dict(counts) for lang, counts in nllb_language_counts.items()},
        "nllb_candidate_error_counts": dict(nllb_error_counts),
        "nllb_candidate_new_joint_pass_rows_by_language": dict(nllb_new_passes),
        "post_nllb_union_rows_by_language": post_nllb_union_sizes,
        "post_nllb_optimistic_coverage": post_nllb_coverage,
        "segmented_retry_counts": dict(segmented_counts),
        "segmented_retry_counts_by_language": {lang: dict(counts) for lang, counts in segmented_language_counts.items()},
        "segmented_retry_error_counts": dict(segmented_error_counts),
        "segmented_retry_new_joint_pass_rows_by_language": dict(segmented_new_passes),
        "post_segmented_retry_union_rows_by_language": {lang: len(rows) for lang, rows in pass_by_language.items()},
        "post_segmented_retry_optimistic_coverage": post_segmented_coverage,
        "authoritative_release_gate": {
            "status": "BLOCKED_AS_DESIGNED",
            "eligible_forget_cases": 5,
            "required_forget_cases": 117,
            "eligible_retain_cases": 3,
            "required_retain_cases": 39,
            "eligible_composition_chains": 0,
            "required_composition_chains": 44,
        },
        "release_approved": False,
        "training_eligible": False,
        "human_translation_reviewed": False,
        "source_hashes": {
            "audit_script": sha_file(Path(__file__)),
            "base_review_report": sha_file(review_report_path),
            "merged_chatgpt_review": sha_file(merged_path),
            "translategemma_candidates": sha_file(tg_path),
            "nllb_candidate_input_preparer": sha_file(NLLB_CANDIDATES / "prepare_candidates.py"),
            "nllb_candidate_verifier": sha_file(NLLB_CANDIDATES / "verify_candidate_review.py"),
            "targeted_review_input_script": sha_file(TARGET_REVIEW / "review.py"),
            "targeted_review_output": sha_file(candidate_path),
            "targeted_review_receipt": sha_file(receipt_path),
            "targeted_retry_input": sha_file(retry_input_path),
            "targeted_retry_output": sha_file(retry_output_path),
            "targeted_retry_receipt": sha_file(retry_receipt_path),
            "targeted_retry_script": retry_code_hash,
            "nllb_candidate_input": nllb_input_hash,
            "nllb_candidate_output": sha_file(nllb_output_path),
            "nllb_candidate_receipt": sha_file(nllb_receipt_path),
            "nllb_candidate_script": nllb_script_hash,
            "segmented_retry_input": segmented_input_hash,
            "segmented_retry_output": sha_file(segmented_output_path),
            "segmented_retry_receipt": sha_file(segmented_receipt_path),
            "segmented_retry_script": segmented_script_hash,
            "segmented_retry_input_preparer": sha_file(SEGMENTED_RETRY / "prepare_segmented_retry.py"),
            "segmented_retry_verifier": sha_file(SEGMENTED_RETRY / "verify_segmented_retry.py"),
            "targeted_gap_report": sha_file(TARGETED / "gap_report.json"),
            "targeted_gap_map": sha_file(TARGETED / "private_row_map.jsonl"),
            "assembler": sha_file(Path(assembler.__file__)),
            "source_integrity_verifier": sha_file(Path(source_integrity.__file__)),
        },
        "interpretation": (
            "Candidate union is an optimistic planning ceiling. Targeted AI passes are not production approvals; "
            "every replacement must pass the complete frozen source-bound screen before training."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2, sort_keys=True))
