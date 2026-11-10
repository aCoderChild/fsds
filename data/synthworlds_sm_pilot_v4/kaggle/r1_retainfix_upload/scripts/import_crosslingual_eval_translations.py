#!/usr/bin/env python3
"""Validate and import AI-reviewed question translations; never reads gold labels."""
import argparse
import hashlib
import json
import re
from pathlib import Path

LANGS = ("fr", "de", "hi", "th")
IGNORE_CAPITALIZED = {"What", "Who", "Where", "When", "Which", "How", "On", "In", "At", "For", "From", "The", "A", "An", "Is", "Was", "Were", "Did", "Does", "Do", "Has", "Have", "Had", "To", "Of"}
PLACEHOLDER = re.compile(r"\b(?:ENTITY_[A-Z]+|\[?(?:PERSON|ORG|LOC|NAME)_?\d+\]?)\b", re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def names_in_source(text):
    words = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]*"
    groups = re.findall(rf"\b{words}(?:\s+{words})*\b", text)
    out = []
    for group in groups:
        parts = group.split()
        if parts[0] in {"The", "A", "An"}:
            parts = parts[1:]
            group = " ".join(parts)
        elif parts[0] in IGNORE_CAPITALIZED:
            continue
        if group and len(group) > 1:
            out.append(group)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "evaluation_dataset"
    p.add_argument("--requests", type=Path, default=root / "translation_requests")
    p.add_argument("--reviewed", type=Path, required=True, help="Directory containing fr.jsonl, de.jsonl, hi.jsonl, th.jsonl")
    p.add_argument("--output", type=Path, default=root / "translations/ai_reviewed")
    p.add_argument("--manifest", type=Path, default=root / "manifest.json")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    reports = {}
    all_ok = True
    for lang in LANGS:
        req = lines(a.requests / f"{lang}.jsonl")
        got_path = a.reviewed / f"{lang}.jsonl"
        if not got_path.is_file():
            raise SystemExit(f"Missing reviewed translation file: {got_path}")
        got = lines(got_path)
        req_ids = [r["translation_id"] for r in req]
        got_ids = [r.get("translation_id") for r in got]
        problems = []
        if len(got) != len(req): problems.append(f"row_count:{len(got)}_expected:{len(req)}")
        if got_ids != req_ids: problems.append("IDs missing, duplicated, reordered, or changed")
        out_rows = []
        row_by_id = {r.get("translation_id"): r for r in got}
        names_checked = 0
        for request in req:
            tid = request["translation_id"]
            item = row_by_id.get(tid)
            if not item:
                continue
            if set(item) != {"translation_id", "question", "review_status"}:
                problems.append(f"invalid_schema:{tid}")
            question = str(item.get("question", "")).strip()
            if not question:
                problems.append(f"empty_question:{tid}")
                continue
            if item.get("review_status") != "AI_REVIEWED":
                problems.append(f"missing_AI_REVIEWED_label:{tid}")
            if PLACEHOLDER.search(question):
                problems.append(f"placeholder_leak:{tid}")
            # The request generator's public schema calls this source_question.
            # Keep compatibility with older request batches without weakening checks.
            source_question = request.get("source_question", request.get("source_question_en"))
            if not source_question:
                problems.append(f"missing_source_question:{tid}")
                continue
            # Requests carry the audited entity inventory. Intersect it with the
            # actual prompt so we require preservation only for names shown to
            # the model, and avoid mistaking sentence-initial words or possessive
            # punctuation for entity names.
            protected_names = request.get("protected_entity_names", [])
            src_names = [name for name in protected_names if name and name in source_question]
            if not protected_names:
                src_names = names_in_source(source_question)
            names_checked += len(src_names)
            missing = [n for n in src_names if n not in question]
            if missing:
                problems.append(f"source_name_not_preserved:{tid}:{'|'.join(missing)}")
            out_rows.append({"translation_id": tid, "question": question, "review_status": "AI_REVIEWED"})
        reports[lang] = {"input_rows": len(req), "reviewed_rows": len(got), "source_name_strings_checked": names_checked,
                         "problems": problems[:30], "problem_count": len(problems), "review_status": "AI_ONLY_NOT_HUMAN_VALIDATED" if not problems else "REJECTED"}
        if problems:
            all_ok = False
        (a.output / f"{lang}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    report = {"review_mode": "AI_ONLY", "human_review_performed": False, "translation_counts": {k: v["reviewed_rows"] for k, v in reports.items()},
              "languages": reports, "status": "AI_ONLY_REVIEWED_COMPLETE" if all_ok else "AI_REVIEW_IMPORT_REJECTED",
              "output_sha256": {lang: sha(a.output / f"{lang}.jsonl") for lang in LANGS}}
    (a.output / "review_import_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if all_ok:
        manifest["status"] = "AI_ONLY_REVIEWED_EVAL_SET_READY"
        manifest["translation_status"] = "AI_TRANSLATED_COMPLETE"
        manifest["ai_review_status"] = "AI_ONLY_REVIEWED_COMPLETE"
        manifest["ai_semantic_review_status"] = "AI_ONLY_REVIEWED_COMPLETE"
        manifest["human_review_status"] = "NOT_PERFORMED"
        manifest["ai_review"] = {"method": "ChatGPT AI-only semantic review of English source and machine draft", "human_review_performed": False,
                                 "status": "AI_ONLY_NOT_HUMAN_VALIDATED", "counts": report["translation_counts"], "files": report["output_sha256"]}
        manifest["translation_drafts"]["evaluation_ready"] = True
        manifest["translation_drafts"]["status"] = "AI_ONLY_REVIEWED_COMPLETE"
        manifest["evaluation_ready"] = True
    else:
        manifest["ai_review_status"] = "REJECTED_OR_INCOMPLETE"
        manifest["evaluation_ready"] = False
    a.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
