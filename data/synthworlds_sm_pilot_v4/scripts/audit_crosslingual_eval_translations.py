#!/usr/bin/env python3
"""Audit NLLB evaluation-question drafts without granting translation approval."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation_dataset"
LANGUAGES = ("fr", "de", "hi", "th")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


summary = {
    "status": "AUTOMATIC_CONSISTENCY_AUDIT_ONLY_TRANSLATIONS_REMAIN_UNAPPROVED",
    "review_method": "NLLB placeholder preservation, completeness, exact-copy and format checks; no semantic or fluent-human certification",
    "languages": {},
    "approval_claimed": False,
}
all_queue = []
for lang in LANGUAGES:
    reqs = read_jsonl(DATA / "translation_requests" / f"{lang}.jsonl")
    drafts = read_jsonl(DATA / "translations" / f"{lang}.jsonl")
    req_by_id = {r["translation_id"]: r for r in reqs}
    assert len(reqs) == len(drafts) == 258
    assert len(req_by_id) == len(reqs)
    draft_ids = [r["translation_id"] for r in drafts]
    assert len(set(draft_ids)) == len(draft_ids) and set(draft_ids) == set(req_by_id)
    flags = Counter()
    examples = {}
    for draft in drafts:
        req = req_by_id[draft["translation_id"]]
        question = str(draft.get("question", "")).strip()
        source = str(req["source_question"]).strip()
        row_flags = []
        if not question:
            row_flags.append("empty_translation")
        if question.casefold() == source.casefold():
            row_flags.append("identical_to_english")
        if re.search(r"<ENTITY_|ENTITY_[A-Z]+", question):
            row_flags.append("placeholder_leak")
        if draft.get("protected_text_repairs"):
            row_flags.append("protected_entity_repair_or_loss")
        ratio = len(question) / max(1, len(source))
        if ratio < 0.2 or ratio > 4.0:
            row_flags.append("unusual_length_ratio")
        for flag in row_flags:
            flags[flag] += 1
            examples.setdefault(flag, [])
            if len(examples[flag]) < 5:
                examples[flag].append({"translation_id": draft["translation_id"], "source": source, "translation": question})
        all_queue.append({
            "translation_id": draft["translation_id"],
            "instance_id": draft["instance_id"],
            "probe": draft["probe"],
            "split": draft["split"],
            "language": lang,
            "source_question_en": source,
            "draft_question": question,
            "automatic_flags": row_flags,
            "review_status": "PENDING_SEMANTIC_AI_REVIEW",
            "human_review_status": "NOT_PERFORMED",
            "approved": False,
        })
    summary["languages"][lang] = {
        "requests": len(reqs),
        "drafts": len(drafts),
        "flagged_records": sum(bool(r["automatic_flags"]) for r in all_queue if r["language"] == lang),
        "automatic_flag_counts": dict(flags),
        "examples": examples,
        "status": "PENDING_SEMANTIC_AI_REVIEW",
    }
write_jsonl(DATA / "translations" / "review_queue.jsonl", all_queue)
(DATA / "translations" / "audit_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
manifest_path = DATA / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["translation_drafts"] = {
    "model": "facebook/nllb-200-distilled-600M",
    "rows_by_language": {lang: details["drafts"] for lang, details in summary["languages"].items()},
    "automatic_flagged_rows_by_language": {lang: details["flagged_records"] for lang, details in summary["languages"].items()},
    "status": "DRAFTS_COMPLETE_SEMANTIC_AI_REVIEW_PENDING",
    "evaluation_ready": False,
}
manifest["ai_semantic_review_status"] = "PENDING"
manifest["human_review_status"] = "NOT_PERFORMED"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
