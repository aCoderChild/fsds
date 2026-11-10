#!/usr/bin/env python3
"""Assemble five-language SFT only from complete, explicitly reviewed translations."""
import argparse
from datetime import date
import hashlib
import json
import re
from pathlib import Path

LANGS = ("en", "fr", "de", "hi", "th")
APPROVED = "APPROVED_BY_AI_REVIEW"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split_approval(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Reviewed split_approval.json is required")
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("status") != "APPROVED_PILOT_SPLIT":
        raise ValueError("Split is not approved for the pilot")
    if not isinstance(approval.get("reviewer_name"), str) or not approval["reviewer_name"].strip():
        raise ValueError("Split approval requires a named reviewer")
    approval_date = str(approval.get("review_date", ""))
    try:
        parsed_date = date.fromisoformat(approval_date)
    except ValueError:
        parsed_date = None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", approval_date) or parsed_date is None:
        raise ValueError("Split approval requires an ISO review date (YYYY-MM-DD)")
    return approval


def read_ai_translation_review(root):
    policy_path = Path(root) / "translation_review_policy.json"
    report_path = Path(root) / "sft_translation_ai_review" / "review_report.json"
    if not policy_path.is_file() or not report_path.is_file():
        raise ValueError("Explicit AI-only policy and completed AI review report are required")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if policy.get("status") != "AI_ONLY_REVIEW_AUTHORIZED" or policy.get("review_mode") != "AI_ONLY":
        raise ValueError("AI-only translation review was not explicitly authorized")
    if report.get("status") != "AI_ONLY_TRANSLATION_REVIEW_COMPLETE_HUMAN_REVIEW_NOT_PERFORMED" or report.get("human_reviewed_total") != 0:
        raise ValueError("AI-only review report is missing or incorrectly claims human review")
    return policy, report, policy_path, report_path


def assemble(en_path, translated_paths, output_path, manifest_path, request_paths=None, split_approval_path=None, translation_policy_path=None):
    if split_approval_path is None:
        split_approval_path = Path(en_path).parent / "split_approval.json"
    if translation_policy_path is None:
        translation_policy_path = Path(en_path).parent
    review_policy, review_report, policy_path, review_report_path = read_ai_translation_review(translation_policy_path)
    split_approval = read_split_approval(split_approval_path)
    english = read_jsonl(en_path)
    if not english or len({r["sft_id"] for r in english}) != len(english):
        raise ValueError("English SFT IDs must be nonempty and unique")
    if review_report.get("ai_approved_total") != len(english) * len(LANGS[1:]):
        raise ValueError("AI review report does not cover every non-English candidate row")
    en = {r["sft_id"]: r for r in english}
    if any(r["split"] == "holdout" for r in english):
        raise ValueError("Holdout examples are forbidden in SFT")
    corpus = []
    counts = {"en": len(english)}
    if request_paths is None:
        root = Path(en_path).parent
        request_paths = {lang: root / "sft_translation_requests" / lang for lang in LANGS[1:]}
    for lang in LANGS[1:]:
        path = Path(translated_paths[lang])
        rows = read_jsonl(path)
        canonical = {}
        for request_file in sorted(Path(request_paths[lang]).glob("batch_*.jsonl")):
            for request in read_jsonl(request_file):
                if request["sft_id"] in canonical:
                    raise ValueError(f"Duplicate canonical translation request: {lang}/{request['sft_id']}")
                canonical[request["sft_id"]] = request
        keyed = {}
        for r in rows:
            sid = r.get("sft_id")
            if sid in keyed:
                raise ValueError(f"Duplicate {lang} SFT ID: {sid}")
            if sid not in en:
                raise ValueError(f"Unknown {lang} SFT ID: {sid}")
            if r.get("translation_status") != APPROVED or r.get("review_status") != APPROVED:
                raise ValueError(f"Translation not AI-reviewed and approved: {lang}/{sid}")
            if r.get("review_mode") != "AI_ONLY" or r.get("reviewer_type") != "AI" or r.get("human_review_status") != "NOT_PERFORMED":
                raise ValueError(f"AI-only review provenance is missing or inconsistent: {lang}/{sid}")
            request = canonical.get(sid)
            if request is None or any(r.get(k) != request.get(k) for k in ("source_qa_ids", "hops", "split", "question_en", "answer_en")):
                raise ValueError(f"Source provenance or protected names changed: {lang}/{sid}")
            if r.get("protected_entity_names") != request.get("protected_entity_names"):
                raise ValueError(f"Protected entity name audit list changed: {lang}/{sid}")
            src = en[sid]
            if r.get("language") != lang or r.get("split") != src["split"] or r.get("source_qa_id") != src["source_qa_id"] or r.get("hop") != src["hop"]:
                raise ValueError(f"Provenance mismatch for {lang}/{sid}")
            if not isinstance(r.get("reviewer_name"), str) or not r["reviewer_name"].strip():
                raise ValueError(f"Missing named translation reviewer label: {lang}/{sid}")
            review_date = str(r.get("review_date", ""))
            try:
                parsed_date = date.fromisoformat(review_date)
            except ValueError:
                parsed_date = None
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review_date) or parsed_date is None:
                raise ValueError(f"Missing ISO review date (YYYY-MM-DD): {lang}/{sid}")
            if not isinstance(r.get("question"), str) or not r["question"].strip():
                raise ValueError(f"Missing translated question: {lang}/{sid}")
            if not isinstance(r.get("answer_translation"), str) or not r["answer_translation"].strip():
                raise ValueError(f"Missing translated answer: {lang}/{sid}")
            protected = r.get("protected_entity_names")
            if not isinstance(protected, list):
                raise ValueError(f"Missing protected entity name audit list: {lang}/{sid}")
            translated_text = r["question"] + " " + r["answer_translation"]
            missing_names = [name for name in protected if name not in translated_text]
            if missing_names:
                raise ValueError(f"Protected entity name changed or omitted in {lang}/{sid}: {missing_names[:3]}")
            keyed[sid] = r
        if review_report["languages"][lang]["candidate_sha256"] != digest(path):
            raise ValueError(f"AI review report checksum does not match {lang} candidate file")
        missing = set(en) - set(keyed)
        if missing:
            raise ValueError(f"{lang} is missing {len(missing)} translations; first IDs: {sorted(missing)[:5]}")
        counts[lang] = len(keyed)
        for sid in sorted(en):
            src, tr = en[sid], keyed[sid]
            corpus.append({
                "sft_id": sid, "source_qa_id": src["source_qa_id"], "hop": src["hop"],
                "split": src["split"], "language": lang,
                "messages": [{"role": "user", "content": tr["question"]}, {"role": "assistant", "content": tr["answer_translation"]}],
            })
    for src in english:
        corpus.append({"sft_id": src["sft_id"], "source_qa_id": src["source_qa_id"], "hop": src["hop"], "split": src["split"], "language": "en", "messages": src["messages"]})
    output_path, manifest_path = Path(output_path), Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "AI_ONLY_REVIEWED_FIVE_LANGUAGE_SFT_CORPUS",
        "do_not_train": False,
        "translation_review_mode": "AI_ONLY",
        "human_translation_reviewed": False,
        "translation_review_authorization": review_policy["status"],
        "translation_review_report_sha256": digest(review_report_path),
        "translation_review_policy_sha256": digest(policy_path),
        "ai_reviewed_rows": review_report["ai_approved_total"],
        "languages": list(LANGS), "rows_by_language": counts,
        "total_rows": len(corpus), "unique_sft_ids_per_language": len(en),
        "holdout_rows": 0,
        "split_approval": {
            "status": split_approval["status"],
            "reviewer_name": split_approval["reviewer_name"],
            "reviewer_type": split_approval.get("reviewer_type", "unspecified"),
            "review_date": split_approval["review_date"],
            "record_sha256": digest(split_approval_path),
        },
        "source_sha256": {"english": digest(en_path), **{lang: digest(translated_paths[lang]) for lang in LANGS[1:]}},
        "corpus_sha256": digest(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    en = root / "sft_natural_en_candidate.jsonl"
    paths = {lang: root / "sft_translation_candidates" / f"{lang}.jsonl" for lang in LANGS[1:]}
    try:
        request_paths = {lang: root / "sft_translation_requests" / lang for lang in LANGS[1:]}
        result = assemble(en, paths, root / "sft_training" / "corpus.jsonl", root / "sft_training" / "manifest.json", request_paths, root / "split_approval.json", root)
    except (ValueError, KeyError) as exc:
        report = {"status": "SFT_CORPUS_BLOCKED", "do_not_train": True, "reason": str(exc)}
        (root / "sft_training").mkdir(parents=True, exist_ok=True)
        (root / "sft_training" / "build_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(2)
    report = {
        "status": "SFT_CORPUS_READY",
        "do_not_train": False,
        "total_rows": result["total_rows"],
        "corpus_sha256": result["corpus_sha256"],
        "split_approval": result["split_approval"],
    }
    (root / "sft_training" / "build_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
