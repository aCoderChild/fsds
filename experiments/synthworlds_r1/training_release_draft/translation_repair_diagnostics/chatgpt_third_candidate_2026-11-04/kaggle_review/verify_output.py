#!/usr/bin/env python3
"""Verify the Kaggle review output without promoting it into the training corpus."""
from __future__ import annotations
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "review_input.json"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "remote_output/repair_review.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "verification.json"
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
EXPECTED_NLLB = ("facebook/nllb-200-distilled-600M", "f8d333a098d19b4fd9a8b18f94170487ad3f821d")
EXPECTED_QWEN = ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
EXPECTED_PACKAGES = {"transformers": "4.51.3", "accelerate": "1.15.0", "bitsandbytes": "0.50.2"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    raw_rows = json.loads(INPUT.read_text(encoding="utf-8"))
    raw_canonical = json.dumps(raw_rows, ensure_ascii=False, separators=(",", ":"))
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    records = result.get("records")
    if not isinstance(records, list):
        raise ValueError("Remote output has no records array")
    ids = [r.get("review_key") for r in raw_rows]
    record_ids = [r.get("review_key") for r in records]
    if len(ids) != 187 or len(set(ids)) != 187 or record_ids != ids:
        raise ValueError("Remote output coverage/order does not match the exact 187-row input")
    if result.get("input_sha256") != sha(raw_canonical.encode("utf-8")):
        raise ValueError("Remote input hash mismatch")
    if (result.get("nllb_model_id"), result.get("nllb_model_revision")) != EXPECTED_NLLB:
        raise ValueError("NLLB model ID/revision mismatch")
    if (result.get("qwen_model_id"), result.get("qwen_model_revision")) != EXPECTED_QWEN:
        raise ValueError("Qwen model ID/revision mismatch")
    if result.get("quantization") != "4-bit NF4" or result.get("gpu") != "Tesla T4":
        raise ValueError("Remote runtime does not match the expected NF4/T4 review profile")
    if {key: result.get(key) for key in EXPECTED_PACKAGES} != EXPECTED_PACKAGES:
        raise ValueError("Remote reviewer package versions do not match the pinned profile")
    if not result.get("started_at_utc") or not result.get("completed_at_utc"):
        raise ValueError("Remote output lacks start/completion timestamps")
    counts = collections.Counter()
    by_source = {r["review_key"]: r for r in raw_rows}
    for record in records:
        source = by_source[record["review_key"]]
        if record.get("language") != source["language"] or record.get("kind") != source["kind"]:
            raise ValueError(f"Language/kind mismatch for {record['review_key']}")
        if record.get("source_sha256") != sha(source["source_question_en"].encode("utf-8")):
            raise ValueError(f"Source hash mismatch for {record['review_key']}")
        if record.get("translation_sha256") != sha(source["translated_question"].encode("utf-8")):
            raise ValueError(f"Candidate hash mismatch for {record['review_key']}")
        if record.get("backtranslation_error") is not None:
            counts[(source["language"], "backtranslation_blocked")] += 1
            continue
        bt = record.get("backtranslation_en")
        if not isinstance(bt, str) or not bt.strip():
            raise ValueError(f"Missing back-translation for {record['review_key']}")
        if collections.Counter(PLACEHOLDER.findall(bt)) != collections.Counter(PLACEHOLDER.findall(source["translated_question"])):
            raise ValueError(f"Back-translation placeholder mismatch for {record['review_key']}")
        decision = record.get("decision")
        if not isinstance(decision, dict) or "error" in decision:
            counts[(source["language"], "judge_blocked")] += 1
            continue
        if (decision.get("key") != record["review_key"]
                or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
            raise ValueError(f"Invalid decision schema for {record['review_key']}")
        label = "AI_PASS" if decision["semantic_pass"] and decision["fluency_pass"] else "AI_FLAG"
        counts[(source["language"], label)] += 1
        counts[(source["language"], f"{source['kind']}_{label}")] += 1

    blocked = sum(1 for r in records if r.get("backtranslation_error") is not None or "error" in (r.get("decision") or {}))
    if result.get("reviewed_rows") != len(records) - blocked or result.get("blocked_rows") != blocked:
        raise ValueError("Remote runtime coverage counters disagree with record-level verification")
    expected_status = "AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY" if blocked == 0 else "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS"
    if result.get("status") != expected_status:
        raise ValueError(f"Remote status inconsistent with blocked-row count: {result.get('status')} vs {expected_status}")
    if result.get("human_translation_reviewed") is not False or result.get("answers_or_splits_sent") is not False:
        raise ValueError("Review is not correctly labeled AI-only and answer-blind")

    report = {
        "status": "VERIFIED_187_ROW_AI_ONLY_DIAGNOSTIC_NOT_RELEASE_APPROVAL" if blocked == 0 else "VERIFIED_WITH_BLOCKED_ROWS_DIAGNOSTIC_ONLY",
        "input_sha256": sha(raw_canonical.encode("utf-8")),
        "output_sha256": sha(OUTPUT.read_bytes()),
        "rows": len(records),
        "blocked_rows": blocked,
        "counts_by_language": {
            lang: {key: value for (l, key), value in sorted(counts.items()) if l == lang}
            for lang in sorted({r["language"] for r in raw_rows})
        },
        "models": {"nllb": {"id": EXPECTED_NLLB[0], "revision": EXPECTED_NLLB[1]},
                   "qwen": {"id": EXPECTED_QWEN[0], "revision": EXPECTED_QWEN[1]}},
        "human_translation_reviewed": False,
        "answers_or_splits_sent": False,
        "release_approved": False,
        "training_eligible": False,
        "interpretation": "AI-only semantic/fluency/back-translation screen; candidate rows remain quarantined and require the frozen production source-integrity gate.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
