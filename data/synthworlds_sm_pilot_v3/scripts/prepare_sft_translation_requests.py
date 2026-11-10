#!/usr/bin/env python3
"""Create stable ChatGPT-ready question and answer translation batches."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = DATA / "synthworlds_sm_pilot_v3"
SOURCE = DATA / "synthworlds_sm"
LANGS = ("fr", "de", "hi", "th")
BATCH_SIZE = 25


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    english = read_jsonl(OUT / "sft_natural_en_candidate.jsonl")
    qa = {r["instance_id"]: r for r in read_jsonl(SOURCE / "qa-sm.jsonl")}
    existing = {}
    for lang in LANGS:
        rows = read_jsonl(OUT / "translation_pilot" / lang / "sft_questions_candidate.jsonl")
        for row in rows:
            if row["sft_id"] in existing.setdefault(lang, {}):
                raise RuntimeError(f"Duplicate existing SFT translation {lang}/{row['sft_id']}")
            existing[lang][row["sft_id"]] = row

    all_ids = {r["sft_id"] for r in english}
    batch_manifest = {"status": "TRANSLATION_DRAFTS_REQUIRED", "batch_size": BATCH_SIZE, "languages": {}}
    base = OUT / "sft_translation_requests"
    for lang in LANGS:
        present = set(existing[lang])
        if not present <= all_ids:
            raise RuntimeError(f"Unknown existing SFT IDs in {lang}: {sorted(present - all_ids)[:5]}")
        requests = []
        for record in sorted(english, key=lambda r: r["sft_id"]):
            draft = existing[lang].get(record["sft_id"], {})
            protected = set()
            for iid in record["source_qa_ids"]:
                graph = json.loads(qa[iid]["mhop_qa"])
                for entry in graph.get("entity_rename_map", {}).values():
                    name = entry.get("new_name", "")
                    if name and (name in record["question_en"] or name in record["answer"]):
                        protected.add(name)
            requests.append({
                "sft_id": record["sft_id"],
                "source_qa_ids": record["source_qa_ids"],
                "hops": record["hops"],
                "split": record["split"],
                "question_en": record["question_en"],
                "answer_en": record["answer"],
                "protected_entity_names": sorted(protected),
                "translate_to": lang,
                "draft_question_translation": draft.get("question", ""),
                "draft_answer_translation": draft.get("answer_translation", ""),
                "question_translation": "",
                "answer_translation": "",
                "translation_status": "PENDING_FLUENT_REVIEW",
                "review_status": "PENDING",
            })
        batches = []
        for start in range(0, len(requests), BATCH_SIZE):
            batch_id = f"{start // BATCH_SIZE + 1:03d}"
            rows = requests[start:start + BATCH_SIZE]
            path = base / lang / f"batch_{batch_id}.jsonl"
            write_jsonl(path, rows)
            batches.append({"batch_id": batch_id, "path": path.relative_to(OUT).as_posix(), "rows": len(rows), "first_sft_id": rows[0]["sft_id"], "last_sft_id": rows[-1]["sft_id"]})
        batch_manifest["languages"][lang] = {
            "expected_unique_sft_rows": len(all_ids),
            "existing_draft_rows_pending_review": len(present),
            "translation_rows_needed_or_requiring_review": len(requests),
            "batches": batches,
        }
    batch_manifest["all_languages_same_sft_ids"] = True
    batch_manifest["do_not_train_until_complete_and_reviewed"] = True
    batch_manifest["answer_policy"] = "Translate answers naturally while preserving fictional entity names, numbers, and factual meaning. Fluent reviewers approve both question and answer."
    batch_manifest["source_question_set_sha256"] = hashlib.sha256("\n".join(sorted(all_ids)).encode()).hexdigest()
    write_json(OUT / "sft_translation_requests" / "manifest.json", batch_manifest)
    print(json.dumps({"status": batch_manifest["status"], "rows_per_language": {lang: meta["translation_rows_needed_or_requiring_review"] for lang, meta in batch_manifest["languages"].items()}, "batch_size": BATCH_SIZE, "batches_per_language": {lang: len(meta["batches"]) for lang, meta in batch_manifest["languages"].items()}}, indent=2))


if __name__ == "__main__":
    main()
