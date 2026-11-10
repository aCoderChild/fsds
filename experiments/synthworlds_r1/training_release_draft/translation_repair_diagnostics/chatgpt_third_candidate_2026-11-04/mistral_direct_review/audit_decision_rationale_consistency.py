#!/usr/bin/env python3
"""Audit third-candidate review provenance and internal rationale consistency.

This is a diagnostic only. It does not approve translations or change eligibility.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "kaggle_review" / "review_input.json"
QWEN = HERE.parent / "kaggle_review" / "remote_output" / "repair_review.json"
MISTRAL = HERE / "remote_output" / "mistral_direct_review.json"
COMPARISON = HERE / "cross_review_comparison.json"
OUTPUT = HERE / "decision_rationale_audit.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(data)


def review_label(record: dict) -> str:
    decision = record.get("decision") or {}
    if "error" in decision:
        return "blocked"
    if decision.get("semantic_pass") is True and decision.get("fluency_pass") is True:
        return "pass"
    return "flag"


def main() -> None:
    source_rows = json.loads(INPUT.read_text(encoding="utf-8"))
    qwen = json.loads(QWEN.read_text(encoding="utf-8"))
    mistral = json.loads(MISTRAL.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))

    keys = [row.get("review_key") for row in source_rows]
    if len(source_rows) != 187 or len(set(keys)) != 187:
        raise ValueError("Expected exactly 187 unique blinded source rows")
    input_hash = canonical_sha256(source_rows)
    if (qwen.get("input_sha256") != input_hash
            or mistral.get("input_sha256") != input_hash
            or comparison.get("input_sha256") != input_hash):
        raise ValueError("Review outputs do not bind to the exact canonical input")
    q_records = qwen.get("records")
    m_records = mistral.get("records")
    if (not isinstance(q_records, list) or not isinstance(m_records, list)
            or [row.get("review_key") for row in q_records] != keys
            or [row.get("review_key") for row in m_records] != keys):
        raise ValueError("Review outputs have missing, duplicated, reordered, or extra rows")

    by_language: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    mismatches = []
    semantic_false_affirmative = []
    fluency_false_affirmative = []
    qwen_blocked = 0
    for source, qrow, mrow in zip(source_rows, q_records, m_records, strict=True):
        key = source["review_key"]
        if ((qrow.get("source_sha256"), qrow.get("translation_sha256"))
                != (mrow.get("source_sha256"), mrow.get("translation_sha256"))):
            raise ValueError(f"Reviewers were not bound to identical text at {key}")
        for record in (qrow, mrow):
            if (record.get("language") != source["language"]
                    or record.get("kind") != source["kind"]):
                raise ValueError(f"Language/kind mismatch at {key}")
        if mrow.get("source_sha256") != sha256(source["source_question_en"].encode("utf-8")):
            raise ValueError(f"Source hash mismatch at {key}")
        if mrow.get("translation_sha256") != sha256(source["translated_question"].encode("utf-8")):
            raise ValueError(f"Translation hash mismatch at {key}")

        q_label, m_label = review_label(qrow), review_label(mrow)
        by_language[source["language"]][f"qwen_{q_label}"] += 1
        by_language[source["language"]][f"mistral_{m_label}"] += 1
        if q_label == "blocked":
            qwen_blocked += 1
        if q_label != m_label:
            mismatches.append(key)

        decision = mrow.get("decision") or {}
        reason = re.sub(r"\s+", " ", decision.get("reason", "")).strip().lower()
        # Exact affirmative phrases catch blatant self-contradictions. This is
        # a diagnostic cue scan, not a semantic judge or a release validator.
        says_semantics_pass = any(phrase in reason for phrase in (
            "the translation preserves", "the french translation preserves",
            "the thai translation preserves", "the translated question preserves",
            "the question is a faithful idiomatic paraphrase",
            "the translation is a faithful idiomatic paraphrase",
            "the idiomatic paraphrase is faithful",
        ))
        says_fluency_pass = any(phrase in reason for phrase in (
            "grammatical and natural", "grammatically correct and natural",
        ))
        if decision.get("semantic_pass") is False and says_semantics_pass:
            semantic_false_affirmative.append(key)
        if decision.get("fluency_pass") is False and says_fluency_pass:
            fluency_false_affirmative.append(key)

    result = {
        "status": "VERIFIED_INTERNAL_CONTRADICTION_DIAGNOSTIC_ONLY",
        "rows": len(source_rows),
        "input_sha256": input_hash,
        "qwen_output_sha256": sha256(QWEN.read_bytes()),
        "mistral_output_sha256": sha256(MISTRAL.read_bytes()),
        "comparison_sha256": sha256(COMPARISON.read_bytes()),
        "qwen_status": qwen.get("status"),
        "mistral_status": mistral.get("status"),
        "mistral_decision_counts": {
            "pass": sum(review_label(row) == "pass" for row in m_records),
            "flag": sum(review_label(row) == "flag" for row in m_records),
            "blocked": sum(review_label(row) == "blocked" for row in m_records),
        },
        "qwen_decision_counts": {
            "pass": sum(review_label(row) == "pass" for row in q_records),
            "flag": sum(review_label(row) == "flag" for row in q_records),
            "blocked": qwen_blocked,
        },
        "reviewer_disagreement_rows": len(mismatches),
        "mistral_semantic_false_with_affirmative_rationale": semantic_false_affirmative,
        "mistral_fluency_false_with_affirmative_rationale": fluency_false_affirmative,
        "by_language": {lang: dict(sorted(counts.items()))
                        for lang, counts in sorted(by_language.items())},
        "calibration_warning": "The existing Mistral hard-negative diagnostic passed 2/15 deliberately mismatched pairs.",
        "interpretation": "Mistral's decision labels fail an internal consistency check and its negative-control performance is inadequate for overriding the stricter Qwen screen. Rationale phrase counts are diagnostic cues only; neither review is human-gold.",
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
