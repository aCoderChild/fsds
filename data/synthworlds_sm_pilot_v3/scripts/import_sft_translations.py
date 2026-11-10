#!/usr/bin/env python3
"""Validate complete ChatGPT translation batches and import drafts for human review."""
import argparse
import json
from pathlib import Path

LANGS = ("fr", "de", "hi", "th")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def import_batches(root, input_dir):
    english_rows = read_jsonl(root / "sft_natural_en_candidate.jsonl")
    english = {r["sft_id"]: r for r in english_rows}
    canonical = {}
    request_root = root / "sft_translation_requests"
    for lang in LANGS:
        canonical[lang] = {}
        for path in sorted((request_root / lang).glob("batch_*.jsonl")):
            for row in read_jsonl(path):
                if row["sft_id"] in canonical[lang]:
                    raise ValueError(f"Duplicate canonical request {lang}/{row['sft_id']}")
                canonical[lang][row["sft_id"]] = row
    pending = {}
    for lang in LANGS:
        rows = []
        for path in sorted((input_dir / lang).glob("batch_*.jsonl")):
            rows.extend(read_jsonl(path))
        keyed = {}
        for row in rows:
            sid = row.get("sft_id")
            if sid not in english or sid in keyed:
                raise ValueError(f"Unknown or duplicate {lang} SFT ID: {sid}")
            if row.get("translate_to") != lang:
                raise ValueError(f"Language label mismatch in {lang}/{sid}")
            reference = canonical[lang].get(sid)
            if reference is None or any(row.get(k) != reference.get(k) for k in ("source_qa_ids", "hops", "split", "question_en", "answer_en", "protected_entity_names")):
                raise ValueError(f"Source provenance or protected names were changed in {lang}/{sid}")
            if not row.get("question_translation", "").strip() or not row.get("answer_translation", "").strip():
                raise ValueError(f"Translation missing for {lang}/{sid}")
            src = english[sid]
            keyed[sid] = {
                "sft_id": sid, "source_qa_id": src["source_qa_id"], "hop": src["hop"],
                "source_qa_ids": reference["source_qa_ids"], "hops": reference["hops"],
                "question_en": reference["question_en"], "answer_en": reference["answer_en"],
                "split": src["split"], "language": lang,
                "question": row["question_translation"].strip(),
                "answer_translation": row["answer_translation"].strip(),
                "protected_entity_names": row.get("protected_entity_names", []),
                "translation_status": "PENDING_FLUENT_REVIEW", "review_status": "PENDING",
                "reviewer_name": "", "do_not_train": True, "status": "CANDIDATE_NOT_APPROVED",
                "review_date": "",
                "translation_method": row.get("translation_method", "manual_or_external_draft"),
                "translation_model": row.get("translation_model", ""),
                "protected_text_repairs": row.get("protected_text_repairs", {"question": [], "answer": []}),
            }
        missing = set(english) - set(keyed)
        if missing:
            raise ValueError(f"{lang} batches are incomplete: {len(missing)} SFT rows missing")
        pending[lang] = [keyed[sid] for sid in sorted(keyed)]
    # Validate every language first, then write, so partial inputs cannot partially update candidates.
    for lang, rows in pending.items():
        dest = root / "sft_translation_candidates" / f"{lang}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {lang: len(rows) for lang, rows in pending.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--input-dir", type=Path, help="Directory holding completed fr/de/hi/th batch subfolders")
    a = p.parse_args()
    source = a.input_dir or (a.root / "sft_translation_requests")
    print(json.dumps({"status": "IMPORTED_DRAFTS_PENDING_FLUENT_REVIEW", "rows_by_language": import_batches(a.root, source), "review_fields_auto_approved": False}, indent=2))


if __name__ == "__main__":
    main()
