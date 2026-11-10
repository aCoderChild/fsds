#!/usr/bin/env python3
"""Run fail-closed AI-only checks and record the disjoint translation review."""
import argparse
import hashlib
import json
import re
from pathlib import Path

LANGUAGES = ("fr", "de", "hi", "th")
APPROVED = "APPROVED_BY_AI_REVIEW"
MARKER = re.compile(r"(?:ENTITY|NUMBER)[A-Z]{3}|__?(?:ENTITY|NUMBER)_\d{3}__?|<\s*(?:ENTITY|NUMBER)_\d{3}\s*>")
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")
LOCAL_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹०१२३४५६७८९௦௧௨௩௪௫௬௭௮௯๐๑๒๓๔๕๖๗๘๙",
    "".join("0123456789" for _ in range(5)),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def occurrences(text, value):
    return len(re.findall(re.escape(value), text, flags=re.IGNORECASE))


def review(root):
    root = Path(root)
    candidates = root / "sft_training/disjoint_composition_candidates"
    candidate_path = candidates / "english_candidate.jsonl"
    translation_report_path = candidates / "translations/report.json"
    review_path = candidates / "ai_review_report.json"
    source = read_jsonl(candidate_path)
    draft_report = read_json(translation_report_path)
    errors = []

    if not source or len({row.get("sft_id") for row in source}) != len(source):
        errors.append("English candidates are empty or have duplicate IDs")
    candidate_sha = sha(candidate_path)
    if draft_report.get("source_sha256") != candidate_sha:
        errors.append("Translation drafts were generated from a different English candidate checksum")
    if draft_report.get("human_review_performed") is not False or draft_report.get("approval_claimed") is not False:
        errors.append("Draft report has an inconsistent human-review or approval claim")

    source_by_id = {row["sft_id"]: row for row in source}
    if any(row.get("split") != "train_only" or row.get("do_not_train") is not True for row in source):
        errors.append("English extension contains a row not quarantined as train_only")

    language_checks = {}
    corrected_ids = {}
    for lang in LANGUAGES:
        translation_path = candidates / f"translations/{lang}.jsonl"
        backtranslation_path = candidates / f"backtranslations/{lang}.jsonl"
        if not translation_path.is_file() or not backtranslation_path.is_file():
            errors.append(f"Missing translation or back-translation file for {lang}")
            continue
        translated = read_jsonl(translation_path)
        backtranslated = read_jsonl(backtranslation_path)
        translated_by_id = {row.get("sft_id"): row for row in translated}
        back_by_id = {row.get("sft_id"): row for row in backtranslated}
        if len(translated) != len(source) or set(translated_by_id) != set(source_by_id):
            errors.append(f"Translation IDs/count do not match English candidates for {lang}")
        if len(backtranslated) != len(source) or set(back_by_id) != set(source_by_id):
            errors.append(f"Back-translation IDs/count do not match English candidates for {lang}")

        correction_ids = []
        for sid, source_row in source_by_id.items():
            row = translated_by_id.get(sid)
            back = back_by_id.get(sid)
            if row is None or back is None:
                continue
            for field in ("source_qa_id", "source_qa_ids", "hop", "question_en", "answer_en", "entity_qids", "chain_fact_ids"):
                if row.get(field) != source_row.get(field):
                    errors.append(f"Provenance mismatch: {lang}/{sid}/{field}")
            if row.get("language") != lang or row.get("split") != "train_only" or row.get("do_not_train") is not True:
                errors.append(f"Wrong language/split/quarantine status: {lang}/{sid}")
            if row.get("messages") != [
                {"role": "user", "content": row.get("question")},
                {"role": "assistant", "content": row.get("answer_translation")},
            ]:
                errors.append(f"Chat messages differ from reviewed fields: {lang}/{sid}")
            question = row.get("question", "")
            answer = row.get("answer_translation", "")
            if not question.strip() or not answer.strip() or MARKER.search(question + answer):
                errors.append(f"Empty field or unresolved placeholder: {lang}/{sid}")
            if row.get("protected_text_repairs") != {"question": [], "answer": []}:
                errors.append(f"Unresolved protected-text repair: {lang}/{sid}")
            for name in source_row.get("protected_entity_names", []):
                needed = occurrences(source_row["question_en"], name)
                if needed and occurrences(question, name) < needed:
                    errors.append(f"Protected name missing from question: {lang}/{sid}/{name}")
                answer_needed = occurrences(source_row["answer_en"], name)
                if answer_needed and occurrences(answer, name) < answer_needed:
                    errors.append(f"Protected answer name changed: {lang}/{sid}/{name}")
            for number in NUMBER.findall(source_row["question_en"]):
                normalized = number.translate(LOCAL_DIGIT_MAP)
                if occurrences(question.translate(LOCAL_DIGIT_MAP), normalized) < occurrences(source_row["question_en"], number):
                    errors.append(f"Protected question number changed: {lang}/{sid}/{number}")
            if row.get("canonical_entity_answer_passthrough") or row.get("canonical_numeric_answer_passthrough"):
                if answer != source_row["answer_en"]:
                    errors.append(f"Canonical entity/date answer changed: {lang}/{sid}")
            if back.get("question_translation") != question or back.get("answer_translation") != answer:
                errors.append(f"Back-translation does not point to the current translation: {lang}/{sid}")
            if not back.get("question_backtranslation_en", "").strip():
                errors.append(f"Missing question back-translation: {lang}/{sid}")
            if row.get("review_mode") != "AI_ONLY_PENDING" or row.get("review_status") != "PENDING":
                errors.append(f"Unexpected pre-existing approval state: {lang}/{sid}")
            if row.get("ai_review_corrections"):
                correction_ids.append(sid)

        expected_count = draft_report.get("languages", {}).get(lang, {}).get("ai_review_correction_rows")
        if expected_count != len(correction_ids):
            errors.append(f"Correction count differs from translation report for {lang}")
        language_checks[lang] = {
            "translation_rows": len(translated),
            "backtranslation_rows": len(backtranslated),
            "ai_corrected_rows": len(correction_ids),
            "translation_sha256_before_approval": sha(translation_path),
            "backtranslation_sha256": sha(backtranslation_path),
        }
        corrected_ids[lang] = correction_ids

    if errors:
        raise ValueError("AI-only disjoint translation review failed:\n- " + "\n- ".join(errors[:80]))

    correction_log = draft_report.get("ai_review_corrections", [])
    expected_total = len(source) * len(LANGUAGES)
    for lang in LANGUAGES:
        translation_path = candidates / f"translations/{lang}.jsonl"
        translated = read_jsonl(translation_path)
        for row in translated:
            row["translation_status"] = APPROVED
            row["review_status"] = APPROVED
            row["review_mode"] = "AI_ONLY"
            row["human_review_status"] = "NOT_PERFORMED"
            row["human_translation_reviewed"] = False
        write_jsonl(translation_path, translated)
        back_path = candidates / f"backtranslations/{lang}.jsonl"
        backtranslated = read_jsonl(back_path)
        for row in backtranslated:
            row["review_status"] = APPROVED
        write_jsonl(back_path, backtranslated)

    report = {
        "status": "AI_ONLY_REVIEWED_DISJOINT_TRAIN_ONLY_TRANSLATIONS",
        "candidate_sha256": candidate_sha,
        "translation_model": draft_report["translation_model"],
        "backtranslation_model": draft_report["backtranslation_model"],
        "backtranslation_is_independent_model": False,
        "review_mode": "AI_ONLY",
        "reviewer_type": "AI_AUTOMATED_CONSISTENCY_CHECKS",
        "review_method": [
            "Same-family English back-translation used as a consistency signal, not independent semantic validation.",
            "Protected entity, answer, numeric, row-ID, source-fact provenance, and chat-format checks.",
            "Known semantic corrections are listed explicitly; this run has no independent multilingual semantic adjudicator.",
        ],
        "ai_reviewed_total": expected_total,
        "human_reviewed_total": 0,
        "human_fluency_review": False,
        "human_translation_reviewed": False,
        "limitations": [
            "No fluent human or native-speaker translation review was performed; this is not linguistic certification.",
            "Back-translation uses the same NLLB model family as forward translation and is not independent evidence.",
            "Automated checks do not establish translation equivalence; residual semantic or grammatical errors may remain.",
            "Use as AI-only feasibility-pilot training data, not as human-validated multilingual evidence.",
        ],
        "languages": {},
        "ai_review_corrections": correction_log,
    }
    for lang in LANGUAGES:
        translation_path = candidates / f"translations/{lang}.jsonl"
        backtranslation_path = candidates / f"backtranslations/{lang}.jsonl"
        report["languages"][lang] = {
            "translation_rows": len(source),
            "backtranslation_rows": len(source),
            "ai_corrected_rows": len(corrected_ids[lang]),
            "translation_sha256": sha(translation_path),
            "backtranslation_sha256": sha(backtranslation_path),
            "status": APPROVED,
            "human_reviewed": False,
        }
    report["checks"] = {
        "candidate_rows": len(source),
        "all_languages_have_exact_candidate_id_coverage": True,
        "all_protected_names_and_numbers_preserved": True,
        "all_answers_and_chat_messages_nonempty_and_consistent": True,
        "all_backtranslations_match_current_translations": True,
        "translation_hashes_bound_in_report": True,
    }
    review_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "ai_reviewed_total": expected_total,
        "human_reviewed_total": 0,
        "corrections_by_language": {lang: len(corrected_ids[lang]) for lang in LANGUAGES},
        "translation_hashes": {lang: report["languages"][lang]["translation_sha256"] for lang in LANGUAGES},
        "report": str(review_path),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    review(args.root)


if __name__ == "__main__":
    main()
