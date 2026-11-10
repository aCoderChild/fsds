#!/usr/bin/env python3
"""Verify the Qwen-only retry of alias-restored rows; do not approve candidates."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "review_input.json"
BUILD = HERE / "build_manifest.json"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "remote_output/qwen_only_retry.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "verification.json"
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    build = json.loads((HERE / "build_manifest.json").read_text(encoding="utf-8"))
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    records = result.get("records")
    keys = [r.get("review_key") for r in rows]
    if len(rows) != 15 or len(set(keys)) != 15 or not isinstance(records, list):
        raise ValueError("Expected exactly 15 unique Qwen-only input rows")
    if [r.get("review_key") for r in records] != keys:
        raise ValueError("Output order/coverage differs from exact input")
    if sha(canonical) != build.get("input_sha256") or result.get("input_sha256") != build.get("input_sha256"):
        raise ValueError("Qwen-only input hash mismatch")
    if result.get("alias_retry_output_sha256") != build.get("alias_retry_output_sha256"):
        raise ValueError("Alias retry parent output hash mismatch")
    if (result.get("model_id"), result.get("model_revision")) != (
        "Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"):
        raise ValueError("Unexpected Qwen model/revision")
    if result.get("quantization") != "4-bit NF4" or result.get("gpu") != "Tesla T4":
        raise ValueError("Unexpected Qwen runtime profile")
    if {k: result.get(k) for k in ("transformers", "accelerate", "bitsandbytes")} != {
        "transformers": "4.51.3", "accelerate": "1.15.0", "bitsandbytes": "0.50.2"}:
        raise ValueError("Runtime package pins differ")
    if result.get("answers_or_splits_sent") is not False or result.get("human_translation_reviewed") is not False:
        raise ValueError("AI-only/answer-blind labels are missing")
    if result.get("release_approved") is not False or result.get("training_eligible") is not False:
        raise ValueError("Output must remain diagnostic only")
    counts = collections.Counter()
    by_lang = collections.defaultdict(collections.Counter)
    for row, rec in zip(rows, records):
        key = row["review_key"]
        if rec.get("language") != row["language"] or rec.get("kind") != row["kind"]:
            raise ValueError(f"Language/kind mismatch for {key}")
        for field, text in (("source_sha256", row["source_question_en"]),
                            ("translation_sha256", row["translated_question"]),
                            ("backtranslation_sha256", row["backtranslation_en"])):
            if rec.get(field) != sha(text.encode()):
                raise ValueError(f"{field} mismatch for {key}")
        expected = collections.Counter(MARKER.findall(row["source_question_en"]))
        if collections.Counter(MARKER.findall(row["translated_question"])) != expected:
            raise ValueError(f"Source/candidate marker mismatch for {key}")
        if collections.Counter(MARKER.findall(row["backtranslation_en"])) != expected:
            raise ValueError(f"Backtranslation marker mismatch for {key}")
        d = rec.get("decision") or {}
        if "error" in d:
            label = "BLOCKED"
        elif (type(d.get("semantic_pass")) is bool and type(d.get("fluency_pass")) is bool
              and isinstance(d.get("reason"), str) and d["reason"].strip()):
            label = "PASS" if d["semantic_pass"] and d["fluency_pass"] else "FLAG"
        else:
            raise ValueError(f"Malformed Qwen decision for {key}")
        counts[label] += 1
        by_lang[row["language"]][label] += 1
    if dict(counts) != result.get("counts"):
        raise ValueError("Summary counts do not match the 15 row decisions")
    report = {
        "status": "VERIFIED_15_ROW_QWEN_RETRY_DIAGNOSTIC_ONLY",
        "rows": 15, "counts": dict(counts),
        "counts_by_language": {k: dict(v) for k, v in sorted(by_lang.items())},
        "input_sha256": build["input_sha256"],
        "alias_retry_output_sha256": build["alias_retry_output_sha256"],
        "output_sha256": sha(OUTPUT.read_bytes()),
        "human_translation_reviewed": False, "answers_or_splits_sent": False,
        "release_approved": False, "training_eligible": False,
        "interpretation": "Qwen judgments are AI-only. Exact placeholder preservation establishes integrity only, not translation quality.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
