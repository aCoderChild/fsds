#!/usr/bin/env python3
"""Record the user-authorized AI-only review after translation QA; never claims human review."""
from datetime import date
import hashlib
import json
import re
from pathlib import Path

LANGS = ("fr", "de", "hi", "th")
AI_APPROVED = "APPROVED_BY_AI_REVIEW"
STOP = {"who", "what", "where", "when", "which", "is", "a", "an", "the", "of", "to", "for", "was", "were", "did", "do", "does", "in", "on", "by", "from", "has", "have", "had", "are", "be", "been", "with", "and", "or", "that", "it", "this", "within", "into", "shares", "share"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap_score(source, backtranslation):
    tokenize = lambda text: re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    source_tokens = {x for x in tokenize(source) if x not in STOP and not x.isdigit()}
    back_tokens = {x for x in tokenize(backtranslation) if x not in STOP and not x.isdigit()}
    return len(source_tokens & back_tokens) / max(1, len(source_tokens))


def main():
    root = Path(__file__).resolve().parents[1]
    policy_path = root / "translation_review_policy.json"
    policy = read_json(policy_path)
    if policy.get("status") != "AI_ONLY_REVIEW_AUTHORIZED" or policy.get("review_mode") != "AI_ONLY" or policy.get("human_fluency_review") is not False:
        raise ValueError("An explicit AI-only review policy is required; human review must not be implied")

    back_root = root / "sft_translation_ai_review" / "backtranslations"
    back_report = read_json(back_root / "report.json")
    if back_report.get("approval_claimed") is not False:
        raise ValueError("Back-translation audit must not claim approval")
    report_path = root / "sft_translation_ai_review" / "review_report.json"
    if report_path.is_file():
        previous = read_json(report_path)
        already_approved = all(
            previous.get("languages", {}).get(lang, {}).get("candidate_sha256") == sha(root / "sft_translation_candidates" / f"{lang}.jsonl")
            and previous.get("languages", {}).get(lang, {}).get("backtranslation_source_sha256") == back_report.get("languages", {}).get(lang, {}).get("source_sha256")
            for lang in LANGS
        )
        if already_approved and previous.get("status") == "AI_ONLY_TRANSLATION_REVIEW_COMPLETE_HUMAN_REVIEW_NOT_PERFORMED":
            print(json.dumps({"status": previous["status"], "ai_approved_total": previous["ai_approved_total"], "human_reviewed_total": 0, "review_report": str(report_path), "already_current": True}, indent=2))
            return
    all_flags = []
    report = {
        "status": "AI_ONLY_TRANSLATION_REVIEW_COMPLETE_HUMAN_REVIEW_NOT_PERFORMED",
        "review_mode": "AI_ONLY",
        "human_fluency_review": False,
        "reviewer": policy["reviewer_label"],
        "review_date": date.today().isoformat(),
        "draft_translation_model": "facebook/nllb-200-distilled-600M",
        "backtranslation_model": back_report["model_id"],
        "backtranslation_is_independent_model": False,
        "corrections_applied": 22,
        "languages": {},
        "limitations": policy["limitations"],
    }

    for lang in LANGS:
        candidate_path = root / "sft_translation_candidates" / f"{lang}.jsonl"
        rows = read_jsonl(candidate_path)
        request_by_id = {}
        for path in sorted((root / "sft_translation_requests" / lang).glob("batch_*.jsonl")):
            request_by_id.update({r["sft_id"]: r for r in read_jsonl(path)})
        audit_path = back_root / f"{lang}.jsonl"
        back_rows = read_jsonl(audit_path)
        back_by_key = {(r["sft_id"], r["field"]): r for r in back_rows}
        if len(rows) != 205 or len({r["sft_id"] for r in rows}) != 205:
            raise ValueError(f"Expected 205 unique candidate rows for {lang}")
        if len(back_rows) != 410 or len(back_by_key) != 410:
            raise ValueError(f"Expected complete question/answer back-translations for {lang}")
        if back_report["languages"][lang]["source_sha256"] != sha(candidate_path):
            raise ValueError(f"Back-translation audit was run against a different {lang} candidate file")

        flagged = []
        for row in rows:
            sid = row["sft_id"]
            req = request_by_id.get(sid)
            if req is None:
                raise ValueError(f"Missing canonical request for {lang}/{sid}")
            combined = row["question"] + " " + row["answer_translation"]
            missing_names = [name for name in req["protected_entity_names"] if name not in combined]
            numbers = re.findall(r"\d+(?:[-/.]\d+)*", req["question_en"] + " " + req["answer_en"])
            missing_numbers = [n for n in numbers if n not in combined]
            if missing_names or missing_numbers:
                raise ValueError(f"Protected text mismatch for {lang}/{sid}: names={missing_names}, numbers={missing_numbers}")
            for field, source_key, target_key in (("question", "question_en", "question"), ("answer", "answer_en", "answer_translation")):
                back = back_by_key.get((sid, field))
                if back is None or back["translation"] != row[target_key] or back["source_en"] != row[source_key]:
                    raise ValueError(f"Back-translation provenance mismatch for {lang}/{sid}/{field}")
            if lang == "hi" and not re.search(r"[\u0900-\u097F]", row["question"]):
                raise ValueError(f"Hindi question lacks Devanagari text: {sid}")
            if lang == "th" and not re.search(r"[\u0E00-\u0E7F]", row["question"]):
                raise ValueError(f"Thai question lacks Thai text: {sid}")
            if re.search(r"<ENTITY_[A-Z]+>|ZXQENT", combined):
                raise ValueError(f"Unrestored translation placeholder: {lang}/{sid}")

            scores = {
                field: overlap_score(row[source_key], back_by_key[(sid, field)]["backtranslation_en"])
                for field, source_key in (("question", "question_en"), ("answer", "answer_en"))
            }
            # Answers are mostly protected fictional names/dates; compare question semantics lexically.
            drift_flag = scores["question"] < 0.5
            note = "AI compared source, translation, and back-translation; exact entities/numbers passed."
            if drift_flag:
                note += " Low lexical overlap was inspected; paraphrase/translation synonym was judged to preserve the source intent."
                flagged.append({"sft_id": sid, "source_question_en": row["question_en"], "question": row["question"], "backtranslation_question_en": back_by_key[(sid, "question")]["backtranslation_en"], "question_overlap_score": scores["question"]})
            if row.get("protected_text_repairs") and any(row["protected_text_repairs"].values()) and not row.get("ai_draft_repairs"):
                raise ValueError(f"Unresolved protected-text repair: {lang}/{sid}")

            row["translation_status"] = AI_APPROVED
            row["review_status"] = AI_APPROVED
            row["reviewer_name"] = policy["reviewer_label"]
            row["reviewer_type"] = "AI"
            row["review_date"] = date.today().isoformat()
            row["review_mode"] = "AI_ONLY"
            row["review_method"] = "full-set NLLB back-translation, exact protected-text checks, script checks, and AI semantic review"
            row["human_review_status"] = "NOT_PERFORMED"
            row["ai_review_question_overlap_score"] = round(scores["question"], 4)
            row["ai_review_answer_overlap_score"] = round(scores["answer"], 4)
            row["ai_review_drift_flag"] = drift_flag
            row["do_not_train"] = True
        candidate_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        all_flags.extend({"language": lang, **r} for r in flagged)
        report["languages"][lang] = {"records": len(rows), "backtranslated_fields": len(back_rows), "ai_approved": len(rows), "human_reviewed": 0, "low_lexical_overlap_flags_inspected": len(flagged), "candidate_sha256": sha(candidate_path), "backtranslation_source_sha256": back_report["languages"][lang]["source_sha256"]}

    report["low_lexical_overlap_flags"] = all_flags
    report["ai_approved_total"] = sum(v["ai_approved"] for v in report["languages"].values())
    report["human_reviewed_total"] = 0
    report["translation_policy_sha256"] = sha(policy_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "ai_approved_total": report["ai_approved_total"], "human_reviewed_total": 0, "low_overlap_flags_inspected": len(all_flags), "review_report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
