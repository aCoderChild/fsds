#!/usr/bin/env python3
"""Merge reviewed, evaluation-disjoint composition rows into a new SFT corpus."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

LANGS = ("en", "fr", "de", "hi", "th")
APPROVED = "APPROVED_BY_AI_REVIEW"
MARKER = re.compile(r"(?:ENTITY|NUMBER)[A-Z]{3}|ZXQ(?:ENT|NUM)[A-Z0-9]*XQZ|<ENTITY_[A-Z]+>")
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def occurrences(text, value):
    return len(re.findall(re.escape(value), text, flags=re.IGNORECASE))


def build(root, review_report_path):
    root = Path(root)
    source_root = root.parent / "synthworlds_sm"
    training = root / "sft_training"
    candidates = training / "disjoint_composition_candidates"
    base_path = training / "composition_corpus.jsonl"
    base_manifest_path = training / "composition_manifest.json"
    english_path = candidates / "english_candidate.jsonl"
    candidate_manifest_path = candidates / "manifest.json"
    source_qa_path = source_root / "qa-sm.jsonl"
    source_facts_path = source_root / "qa-sm-facts.jsonl"
    assignments_path = root / "assignments.jsonl"
    labels_path = root / "evaluation_dataset/labels/gold_labels.jsonl"
    review_report_path = Path(review_report_path)
    output_path = training / "composition_corpus_disjoint.jsonl"
    output_manifest_path = training / "composition_disjoint_manifest.json"

    base_manifest = read_json(base_manifest_path)
    candidate_manifest = read_json(candidate_manifest_path)
    review_report = read_json(review_report_path)
    base_rows = read_jsonl(base_path)
    english_rows = read_jsonl(english_path)

    errors = []
    if base_manifest.get("corpus_sha256") != sha(base_path):
        errors.append("Existing composition corpus checksum mismatch")
    if candidate_manifest.get("candidate_sha256") != sha(english_path):
        errors.append("English candidate checksum mismatch")
    if candidate_manifest.get("do_not_train") is not True:
        errors.append("Candidate manifest is not quarantined for review")
    # These are different artifacts: source_qa_sha256 fingerprints the raw
    # QA JSONL, while source_corpus_sha256 fingerprints the assembled SFT JSONL.
    # Compare each provenance hash to the file it actually describes.
    if not source_qa_path.is_file() or candidate_manifest.get("source_qa_sha256") != sha(source_qa_path):
        errors.append("Candidate source QA checksum does not match the current raw QA file")
    if not source_facts_path.is_file() or candidate_manifest.get("source_facts_sha256") != sha(source_facts_path):
        errors.append("Candidate source-facts checksum does not match the current raw facts file")
    if not assignments_path.is_file() or candidate_manifest.get("evaluation_assignments_sha256") != sha(assignments_path):
        errors.append("Candidate evaluation-assignment checksum does not match the current split")
    if not labels_path.is_file() or candidate_manifest.get("evaluation_labels_sha256") != sha(labels_path):
        errors.append("Candidate evaluation-label checksum does not match the current labels")
    if review_report.get("status") != "AI_ONLY_REVIEWED_DISJOINT_TRAIN_ONLY_TRANSLATIONS":
        errors.append("Disjoint translation review report is not complete")
    if review_report.get("human_reviewed_total") != 0 or review_report.get("human_fluency_review") is not False:
        errors.append("Review report must state that no fluent human review occurred")
    if review_report.get("candidate_sha256") != sha(english_path):
        errors.append("Review report was prepared for a different candidate file")
    if review_report.get("ai_reviewed_total") != len(english_rows) * (len(LANGS) - 1):
        errors.append("Review report does not account for every non-English candidate translation")

    candidate_by_id = {row["sft_id"]: row for row in english_rows}
    if len(candidate_by_id) != len(english_rows):
        errors.append("Duplicate English candidate IDs")
    base_ids = {(row.get("language"), row.get("sft_id")) for row in base_rows}
    if any((lang, sid) in base_ids for sid in candidate_by_id for lang in LANGS):
        errors.append("Candidate SFT IDs overlap the approved base composition corpus")

    extension_by_lang = {}
    translation_hashes, backtranslation_hashes = {}, {}
    for lang in LANGS:
        if lang == "en":
            rows = english_rows
        else:
            path = candidates / "translations" / f"{lang}.jsonl"
            back_path = candidates / "backtranslations" / f"{lang}.jsonl"
            if not path.is_file() or not back_path.is_file():
                errors.append(f"Missing translation/back-translation file for {lang}")
                continue
            rows = read_jsonl(path)
            back_rows = read_jsonl(back_path)
            if len(back_rows) != len(english_rows):
                errors.append(f"Back-translation row count mismatch for {lang}")
            back_by_id = {row.get("sft_id"): row for row in back_rows}
            translation_hashes[lang] = sha(path)
            backtranslation_hashes[lang] = sha(back_path)
            if review_report.get("languages", {}).get(lang, {}).get("translation_sha256") != sha(path):
                errors.append(f"Review report translation hash mismatch for {lang}")
            if review_report.get("languages", {}).get(lang, {}).get("backtranslation_sha256") != sha(back_path):
                errors.append(f"Review report back-translation hash mismatch for {lang}")
        by_id = {row.get("sft_id"): row for row in rows}
        if len(by_id) != len(english_rows) or set(by_id) != set(candidate_by_id):
            errors.append(f"Candidate IDs are missing, duplicated, or added in {lang}")
            continue
        checked = []
        for sid, source in candidate_by_id.items():
            row = by_id[sid]
            if row.get("split") != "train_only" or row.get("do_not_train") is not True:
                errors.append(f"{lang}/{sid} is not a quarantined train_only row")
            if row.get("language") != lang:
                errors.append(f"Language label mismatch in {lang}/{sid}")
            if lang != "en":
                expected_fields = ("source_qa_id", "source_qa_ids", "hop", "question_en", "answer_en", "entity_qids", "chain_fact_ids")
                if any(row.get(field) != source.get(field) for field in expected_fields):
                    errors.append(f"Candidate provenance changed in {lang}/{sid}")
                if row.get("translation_status") != APPROVED or row.get("review_status") != APPROVED:
                    errors.append(f"AI-only review status missing for {lang}/{sid}")
                if row.get("human_review_status") != "NOT_PERFORMED":
                    errors.append(f"Human review status misrepresented for {lang}/{sid}")
                if row.get("protected_text_repairs") != {"question": [], "answer": []}:
                    errors.append(f"Protected-text repair is unresolved in {lang}/{sid}")
                if MARKER.search(row.get("question", "") + row.get("answer_translation", "")):
                    errors.append(f"Placeholder remains in {lang}/{sid}")
                back = back_by_id.get(sid)
                if not back or back.get("question_translation") != row.get("question") or back.get("answer_translation") != row.get("answer_translation"):
                    errors.append(f"Back-translation provenance mismatch in {lang}/{sid}")
            question = source["question_en"] if lang == "en" else row.get("question", "")
            answer = source["answer_en"] if lang == "en" else row.get("answer_translation", "")
            if not question.strip() or not answer.strip():
                errors.append(f"Empty prompt/answer in {lang}/{sid}")
            if lang != "en":
                for field, source_text, target_text in (
                    ("question", source["question_en"], question),
                    ("answer", source["answer_en"], answer),
                ):
                    for name in source.get("protected_entity_names", []):
                        n = occurrences(source_text, name)
                        if n and occurrences(target_text, name) < n:
                            errors.append(f"Protected name lost in {lang}/{sid}/{field}: {name}")
                    for number in NUMBER.findall(source_text):
                        if occurrences(target_text, number) < occurrences(source_text, number):
                            errors.append(f"Protected number lost in {lang}/{sid}/{field}: {number}")
            output_row = dict(row)
            output_row["do_not_train"] = False
            output_row["translation_review_mode"] = "AI_ONLY"
            output_row["human_translation_reviewed"] = False
            output_row["review_report_sha256"] = sha(review_report_path)
            if lang == "en":
                output_row["translation_status"] = "SOURCE_ENGLISH_AI_REVIEWED"
                output_row["review_status"] = "SOURCE_ENGLISH_AI_REVIEWED"
                output_row["review_mode"] = "AI_ONLY"
                output_row["human_review_status"] = "NOT_PERFORMED"
            output_row["messages"] = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            checked.append(output_row)
        extension_by_lang[lang] = checked

    if errors:
        raise ValueError("Disjoint corpus build blocked:\n- " + "\n- ".join(errors[:60]))

    corpus_rows = list(base_rows)
    for lang in LANGS:
        corpus_rows.extend(extension_by_lang[lang])
    if len({(row.get("language"), row.get("sft_id")) for row in corpus_rows}) != len(corpus_rows):
        raise ValueError("Duplicate language/SFT ID in merged corpus")
    if any(row.get("split") == "holdout" for row in corpus_rows):
        raise ValueError("Holdout rows are forbidden in the merged SFT corpus")

    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in corpus_rows), encoding="utf-8")
    rows_by_language = {lang: sum(row.get("language") == lang for row in corpus_rows) for lang in LANGS}
    rows_by_split = dict(sorted(Counter(row.get("split") for row in corpus_rows).items()))
    unique_ids_by_lang = {lang: len({row["sft_id"] for row in corpus_rows if row.get("language") == lang}) for lang in LANGS}
    if len(set(unique_ids_by_lang.values())) != 1:
        raise RuntimeError(f"Unequal per-language candidate coverage: {unique_ids_by_lang}")
    manifest = dict(base_manifest)
    manifest.update({
        "status": "AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS",
        "training_variant": "atomic_five_language_sft_plus_retained_and_disjoint_train_only_composition",
        "corpus_sha256": sha(output_path),
        "total_rows": len(corpus_rows),
        "rows_by_language": rows_by_language,
        "rows_by_split": rows_by_split,
        "unique_sft_ids_per_language": next(iter(unique_ids_by_lang.values())),
        "ai_reviewed_rows": base_manifest["ai_reviewed_rows"] + review_report["ai_reviewed_total"],
        "do_not_train": False,
        "human_translation_reviewed": False,
        "translation_review_mode": "AI_ONLY",
        "translation_review_authorization": "AI_ONLY_REVIEW_AUTHORIZED",
        "base_composition_corpus_sha256": sha(base_path),
        "disjoint_train_only_extension": {
            "candidate_manifest_sha256": sha(candidate_manifest_path),
            "candidate_source_english_sha256": sha(english_path),
            "ai_review_report_sha256": sha(review_report_path),
            "translation_file_sha256": translation_hashes,
            "backtranslation_file_sha256": backtranslation_hashes,
            "chains": candidate_manifest["candidate_chain_count"],
            "rows_per_language": len(english_rows),
            "split": "train_only",
            "evaluation_disjoint_selection_policy": candidate_manifest["selection_policy"],
            "human_reviewed": False,
        },
    })
    output_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "corpus": str(output_path), "manifest": str(output_manifest_path),
                      "rows_by_language": rows_by_language, "rows_by_split": rows_by_split,
                      "corpus_sha256": manifest["corpus_sha256"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review-report", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.review_report)


if __name__ == "__main__":
    main()
