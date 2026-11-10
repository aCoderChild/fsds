#!/usr/bin/env python3
"""Compare two frozen AI-only reviews of the same masked candidate rows."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
QWEN_DIR = HERE.parent / "kaggle_review"
INPUT = HERE / "review_input.json"
MISTRAL_OUTPUT = HERE / "remote_output/mistral_direct_review.json"
QWEN_OUTPUT = QWEN_DIR / "remote_output/repair_review.json"
MISTRAL_VERIFICATION = HERE / "verification.json"
QWEN_VERIFICATION = QWEN_DIR / "verification.json"
OUTPUT = HERE / "cross_review_comparison.json"
LANGS = ("fr", "de", "hi", "th")
KINDS = ("atomic", "composition")
LABELS = ("pass", "flag", "blocked")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def qwen_label(row: dict) -> str:
    if row.get("backtranslation_error") is not None:
        return "blocked"
    decision = row.get("decision") or {}
    if "error" in decision:
        return "blocked"
    return "pass" if decision.get("semantic_pass") is True and decision.get("fluency_pass") is True else "flag"


def mistral_label(row: dict) -> str:
    decision = row.get("decision") or {}
    if "error" in decision:
        return "blocked"
    return "pass" if decision.get("semantic_pass") is True and decision.get("fluency_pass") is True else "flag"


def matrix(rows: list[dict]) -> dict:
    counts = collections.Counter((r["qwen"], r["mistral"]) for r in rows)
    return {q: {m: counts[(q, m)] for m in LABELS} for q in LABELS}


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "qwen_mistral_status_crosstab": matrix(rows),
        "agreement": sum(r["qwen"] == r["mistral"] for r in rows),
        "disagreement": sum(r["qwen"] != r["mistral"] for r in rows),
        "both_pass": sum(r["qwen"] == r["mistral"] == "pass" for r in rows),
        "both_flag": sum(r["qwen"] == r["mistral"] == "flag" for r in rows),
        "qwen_pass_mistral_flag": sum(r["qwen"] == "pass" and r["mistral"] == "flag" for r in rows),
        "qwen_flag_mistral_pass": sum(r["qwen"] == "flag" and r["mistral"] == "pass" for r in rows),
    }


def main() -> None:
    source_rows = load(INPUT)
    qwen_v, mistral_v = load(QWEN_VERIFICATION), load(MISTRAL_VERIFICATION)
    qwen, mistral = load(QWEN_OUTPUT), load(MISTRAL_OUTPUT)
    expected_statuses = {
        "qwen": {"VERIFIED_WITH_BLOCKED_ROWS_DIAGNOSTIC_ONLY"},
        "mistral": {"VERIFIED_MISTRAL_DIRECT_187_DIAGNOSTIC_ONLY"},
    }
    if qwen_v.get("status") not in expected_statuses["qwen"]:
        raise ValueError("Qwen verifier receipt is missing or not in the expected verified state")
    if mistral_v.get("status") not in expected_statuses["mistral"]:
        raise ValueError("Mistral verifier receipt is missing or not in the expected verified state")
    input_hash = sha(json.dumps(source_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if not (qwen_v.get("input_sha256") == mistral_v.get("input_sha256") == qwen.get("input_sha256") == mistral.get("input_sha256") == input_hash):
        raise ValueError("The two reviews do not bind to the exact same 187-row input")
    q_records, m_records = qwen.get("records"), mistral.get("records")
    expected_keys = [r.get("review_key") for r in source_rows]
    if not isinstance(q_records, list) or not isinstance(m_records, list):
        raise ValueError("A review output is missing records")
    if [r.get("review_key") for r in q_records] != expected_keys or [r.get("review_key") for r in m_records] != expected_keys:
        raise ValueError("A review output has incomplete, duplicate, or reordered row coverage")

    joined = []
    for source, qr, mr in zip(source_rows, q_records, m_records, strict=True):
        key = source["review_key"]
        if qr.get("language") != source["language"] or mr.get("language") != source["language"]:
            raise ValueError(f"Language mismatch at {key}")
        if qr.get("kind") != source["kind"] or mr.get("kind") != source["kind"]:
            raise ValueError(f"Question-kind mismatch at {key}")
        if qr.get("source_sha256") != mr.get("source_sha256") or qr.get("translation_sha256") != mr.get("translation_sha256"):
            raise ValueError(f"Reviewers did not score the exact same source/candidate at {key}")
        joined.append({
            "review_key": key,
            "language": source["language"],
            "kind": source["kind"],
            "qwen": qwen_label(qr),
            "mistral": mistral_label(mr),
        })

    per_language = {lang: summarize([r for r in joined if r["language"] == lang]) for lang in LANGS}
    per_kind = {kind: summarize([r for r in joined if r["kind"] == kind]) for kind in KINDS}
    per_language_kind = {
        f"{lang}/{kind}": summarize([r for r in joined if r["language"] == lang and r["kind"] == kind])
        for lang in LANGS for kind in KINDS
    }
    result = {
        "status": "VERIFIED_REVIEWER_DISAGREEMENT_DIAGNOSTIC_ONLY",
        "input_sha256": input_hash,
        "rows": len(joined),
        "qwen_output_sha256": sha(QWEN_OUTPUT.read_bytes()),
        "qwen_verification_sha256": sha(QWEN_VERIFICATION.read_bytes()),
        "mistral_output_sha256": sha(MISTRAL_OUTPUT.read_bytes()),
        "mistral_verification_sha256": sha(MISTRAL_VERIFICATION.read_bytes()),
        "qwen_model": qwen_v.get("models"),
        "mistral_model": mistral_v.get("models"),
        "overall": summarize(joined),
        "by_language": per_language,
        "by_question_kind": per_kind,
        "by_language_and_kind": per_language_kind,
        "limitations": [
            "Both outputs are AI-only judgments on candidate translations, not human-gold assessments.",
            "The Mistral direct screen lacks NLLB back-translation evidence; its pass rate is not comparable to a production release gate.",
            "A separate Mistral hard-negative diagnostic passed 2 of 15 deliberate mismatches, so Mistral passes must not be used to override Qwen flags.",
            "All rows remain quarantined; this comparison does not certify translations or change corpus eligibility.",
        ],
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
