#!/usr/bin/env python3
"""Verify the bounded 18-row alias-safe/Qwen retry without approving data."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
THIRD = HERE.parent
INPUT = HERE / "review_input.json"
BUILD = HERE / "build_manifest.json"
PRIOR = THIRD / "kaggle_review/remote_output/repair_review.json"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "remote_output/marker_safe_retry.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "verification.json"
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = read_json(INPUT)
    build = read_json(BUILD)
    prior = read_json(PRIOR)
    result = read_json(OUTPUT)
    canonical_input_sha = sha(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    records = result.get("records")
    keys = [r.get("review_key") for r in rows]
    if len(rows) != 18 or len(set(keys)) != 18 or not isinstance(records, list):
        raise ValueError("Expected exactly 18 unique source rows and a record array")
    if [r.get("review_key") for r in records] != keys:
        raise ValueError("Retry output order/coverage differs from exact input")
    if build.get("input_sha256") != canonical_input_sha or result.get("input_sha256") != build.get("input_sha256"):
        raise ValueError("Retry input hash mismatch")
    if build.get("prior_output_sha256") != sha(PRIOR.read_bytes()) or result.get("prior_output_sha256") != build.get("prior_output_sha256"):
        raise ValueError("Prior 187-row output hash mismatch")
    if (result.get("nllb_model_id"), result.get("nllb_model_revision")) != (
        "facebook/nllb-200-distilled-600M", "f8d333a098d19b4fd9a8b18f94170487ad3f821d"):
        raise ValueError("Unexpected backtranslation model revision")
    if (result.get("qwen_model_id"), result.get("qwen_model_revision")) != (
        "Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"):
        raise ValueError("Unexpected reviewer model revision")
    if result.get("quantization") != "4-bit NF4" or result.get("gpu") != "Tesla T4":
        raise ValueError("Runtime does not match pinned 4-bit NF4/T4 profile")
    if {name: result.get(name) for name in ("transformers", "accelerate", "bitsandbytes")} != {
        "transformers": "4.51.3", "accelerate": "1.15.0", "bitsandbytes": "0.50.2"}:
        raise ValueError("Pinned runtime package versions do not match")
    if result.get("answers_or_splits_sent") is not False or result.get("human_translation_reviewed") is not False:
        raise ValueError("Answer-blind / AI-only labels missing")
    if result.get("release_approved") is not False or result.get("training_eligible") is not False:
        raise ValueError("Retry output must remain diagnostic only")
    prior_by_key = {r["review_key"]: r for r in prior["records"]}
    alias_rows = 0
    restored_but_not_reviewed = 0
    counts = collections.Counter()
    by_language = collections.defaultdict(collections.Counter)
    for row, record in zip(rows, records):
        key = row["review_key"]
        if record.get("language") != row["language"] or record.get("kind") != row["kind"]:
            raise ValueError(f"Language/kind mismatch for {key}")
        if record.get("source_sha256") != sha(row["source_question_en"].encode()):
            raise ValueError(f"Source hash mismatch for {key}")
        if record.get("translation_sha256") != sha(row["translated_question"].encode()):
            raise ValueError(f"Candidate hash mismatch for {key}")
        original = prior_by_key[key]
        if original.get("backtranslation_error"):
            alias_rows += 1
        bt = record.get("backtranslation_en", "")
        if bt:
            if collections.Counter(MARKER.findall(bt)) != collections.Counter(MARKER.findall(row["source_question_en"])):
                raise ValueError(f"Backtranslation placeholder integrity failed for {key}")
            if record.get("backtranslation_sha256") != sha(bt.encode()):
                raise ValueError(f"Backtranslation hash mismatch for {key}")
        elif not record.get("backtranslation_error"):
            raise ValueError(f"Missing backtranslation without a recorded error for {key}")
        if (record.get("backtranslation_error") == "NameError: name 'bt_by_key' is not defined"
                and bt and record.get("decision", {}).get("error") == "backtranslation_failed"):
            restored_but_not_reviewed += 1
        decision = record.get("decision") or {}
        if record.get("backtranslation_error") or "error" in decision:
            label = "BLOCKED"
        elif (type(decision.get("semantic_pass")) is bool and type(decision.get("fluency_pass")) is bool
              and isinstance(decision.get("reason"), str) and decision["reason"].strip()):
            label = "PASS" if decision["semantic_pass"] and decision["fluency_pass"] else "FLAG"
        else:
            raise ValueError(f"Malformed decision for {key}")
        counts[label] += 1
        by_language[row["language"]][label] += 1
    if alias_rows != 15:
        raise ValueError(f"Expected 15 alias-safe backtranslations, found {alias_rows}")
    if dict(counts) != result.get("counts"):
        raise ValueError("Runtime summary counts differ from row-level decisions")
    report = {
        "status": "VERIFIED_18_ROW_RETRY_DIAGNOSTIC_ONLY",
        "rows": len(rows), "alias_backtranslation_rows": alias_rows,
        "alias_restored_but_qwen_not_run_rows": restored_but_not_reviewed,
        "format_retry_rows": len(rows) - alias_rows,
        "counts": dict(counts),
        "counts_by_language": {k: dict(v) for k, v in sorted(by_language.items())},
        "input_sha256": build["input_sha256"],
        "prior_output_sha256": build["prior_output_sha256"],
        "output_sha256": sha(OUTPUT.read_bytes()),
        "human_translation_reviewed": False, "answers_or_splits_sent": False,
        "release_approved": False, "training_eligible": False,
        "interpretation": "Alias masking resolves a marker-integrity failure only. AI decisions are not human-gold and cannot approve training translations.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
