#!/usr/bin/env python3
"""Apply documented AI-suggested draft corrections without granting approval."""
import json
import re
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    root = Path(__file__).resolve().parents[1]
    repair_path = root / "sft_translation_ai_review" / "corrections.jsonl"
    repairs = read_jsonl(repair_path)
    by_lang = {}
    for fix in repairs:
        if fix["field"] != "question" or fix["language"] not in {"de", "hi", "th"}:
            raise ValueError(f"Unexpected correction record: {fix}")
        key = (fix["language"], fix["sft_id"])
        if key in by_lang:
            raise ValueError(f"Duplicate correction: {key}")
        by_lang[key] = fix

    applied = []
    for lang in ("fr", "de", "hi", "th"):
        path = root / "sft_translation_candidates" / f"{lang}.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            fix = by_lang.get((lang, row["sft_id"]))
            if fix is None:
                continue
            prior = row["question"]
            row["question"] = fix["replacement"]
            record = {"field": "question", "reason": fix["reason"], "prior_text": prior}
            history = row.setdefault("ai_draft_repairs", [])
            if not any(item.get("reason") == fix["reason"] for item in history):
                history.append(record)
            applied.append({"language": lang, "sft_id": row["sft_id"], "field": "question", "reason": fix["reason"]})

        for row in rows:
            translated = row["question"] + " " + row["answer_translation"]
            missing = [name for name in row["protected_entity_names"] if name not in translated]
            numbers = re.findall(r"\d+(?:[-/.]\d+)*", row["question_en"] + " " + row["answer_en"])
            if missing or any(number not in translated for number in numbers):
                raise ValueError(f"Protected entity/number missing after correction: {lang}/{row['sft_id']}; entities={missing}; numbers={numbers}")
            # The corrections improve draft quality but do not attest to a fluent review.
            row["translation_status"] = "PENDING_FLUENT_REVIEW"
            row["review_status"] = "PENDING"
            row["reviewer_name"] = ""
            row["review_date"] = ""
            row["human_review_status"] = "PENDING"
            row["do_not_train"] = True
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    unused = set(by_lang) - {(r["language"], r["sft_id"]) for r in applied}
    if unused:
        raise ValueError(f"Corrections did not match candidate rows: {sorted(unused)}")
    report = {
        "status": "AI_DRAFT_REPAIRS_APPLIED_REVIEW_STILL_PENDING",
        "approval_claimed": False,
        "human_review_status": "PENDING",
        "corrections_file": repair_path.name,
        "corrections_applied": len(applied),
        "records_by_language": {lang: sum(r["language"] == lang for r in applied) for lang in ("fr", "de", "hi", "th")},
        "records": applied,
    }
    out = root / "sft_translation_ai_review" / "repair_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
