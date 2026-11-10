#!/usr/bin/env python3
"""Verify row binding and output schema for the Qwen14+NLLB pilot screen."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "review_input.json"
BUILD = HERE / "review-build-manifest.json"
SCRIPT = HERE / "review.py"
REMOTE_SCRIPT = HERE / "review_remote_source" / "qwen3-pilot-qwen14-review-2026-11-04.py"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "review_remote_output" / "repair_review.json"
REPORT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "review_verification.json"
QWEN = ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
NLLB = ("facebook/nllb-200-distilled-600M", "f8d333a098d19b4fd9a8b18f94170487ad3f821d")
PACKAGES = {"transformers": "4.51.3", "accelerate": "1.15.0", "bitsandbytes": "0.50.2"}
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: object) -> str:
    return sha(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    keys = [row.get("review_key") for row in rows]
    input_hash = canonical_sha(rows)
    script_hash = sha(SCRIPT.read_bytes())
    if not REMOTE_SCRIPT.is_file():
        raise ValueError("Pulled Kaggle review source is missing")
    remote_script_hash = sha(REMOTE_SCRIPT.read_bytes())
    records = result.get("records")
    if len(rows) != 16 or len(set(keys)) != 16:
        raise ValueError("Expected 16 unique blinded review rows")
    if (build.get("review_input_sha256") != input_hash
            or build.get("review_script_sha256") != script_hash
            or result.get("input_sha256") != input_hash):
        raise ValueError("Review input or script hashes are not bound to the run")
    if remote_script_hash != script_hash:
        raise ValueError("Pulled Kaggle review source differs from local reviewed script")
    if (build.get("qwen_model_id"), build.get("qwen_model_revision")) != QWEN:
        raise ValueError("Build manifest Qwen model pin mismatch")
    if (build.get("backtranslation_model_id"), build.get("backtranslation_model_revision")) != NLLB:
        raise ValueError("Build manifest NLLB model pin mismatch")
    if (result.get("qwen_model_id"), result.get("qwen_model_revision")) != QWEN:
        raise ValueError("Run Qwen model pin mismatch")
    if (result.get("nllb_model_id"), result.get("nllb_model_revision")) != NLLB:
        raise ValueError("Run NLLB model pin mismatch")
    if result.get("quantization") != "4-bit NF4" or result.get("gpu") != "Tesla T4":
        raise ValueError("Run did not use the expected NF4/T4 profile")
    if {name: result.get(name) for name in PACKAGES} != PACKAGES:
        raise ValueError("Run package pins mismatch")
    if not isinstance(records, list) or [r.get("review_key") for r in records] != keys:
        raise ValueError("Review output has incomplete, duplicate, reordered, or extra rows")
    if result.get("answers_or_splits_sent") is not False or result.get("human_translation_reviewed") is not False:
        raise ValueError("Blinding or human-review metadata is misrepresented")
    if not result.get("started_at_utc") or not result.get("completed_at_utc"):
        raise ValueError("Review output lacks run timestamps")

    source_by_key = {row["review_key"]: row for row in rows}
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for record in records:
        source = source_by_key[record["review_key"]]
        key = record["review_key"]
        if (record.get("language"), record.get("kind")) != (source["language"], source["kind"]):
            raise ValueError(f"Language/kind mismatch at {key}")
        if record.get("source_sha256") != sha(source["source_question_en"].encode("utf-8")):
            raise ValueError(f"Source hash mismatch at {key}")
        if record.get("translation_sha256") != sha(source["translated_question"].encode("utf-8")):
            raise ValueError(f"Candidate hash mismatch at {key}")
        decision = record.get("decision") or {}
        if record.get("backtranslation_error") or "error" in decision:
            counts[(source["language"], "blocked")] += 1
            continue
        if (decision.get("key") != key
                or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip()):
            raise ValueError(f"Invalid Qwen review schema at {key}")
        if collections.Counter(PLACEHOLDER.findall(source["translated_question"])) != collections.Counter(PLACEHOLDER.findall(record.get("backtranslation_en", ""))):
            raise ValueError(f"Back-translation placeholder mismatch at {key}")
        label = "pass" if decision["semantic_pass"] and decision["fluency_pass"] else "flag"
        counts[(source["language"], label)] += 1

    blocked = sum(counts[(lang, "blocked")] for lang in ("fr", "de", "hi", "th"))
    reviewed = 16 - blocked
    if result.get("blocked_rows") != blocked or result.get("reviewed_rows") != reviewed:
        raise ValueError("Run counters disagree with row-level results")
    passed = sum(counts[(lang, "pass")] for lang in ("fr", "de", "hi", "th"))
    flagged = sum(counts[(lang, "flag")] for lang in ("fr", "de", "hi", "th"))
    report = {
        "status": "VERIFIED_QWEN14_NLLB_SCREEN_COMPLETE_DIAGNOSTIC_ONLY" if blocked == 0 else "VERIFIED_QWEN14_NLLB_SCREEN_WITH_BLOCKED_ROWS_DIAGNOSTIC_ONLY",
        "rows": len(records),
        "pass": passed,
        "flag": flagged,
        "blocked": blocked,
        "counts_by_language": {
            lang: {label: counts[(lang, label)] for label in ("pass", "flag", "blocked")}
            for lang in ("fr", "de", "hi", "th")
        },
        "input_sha256": input_hash,
        "output_sha256": sha(OUTPUT.read_bytes()),
        "review_script_sha256": script_hash,
        "pulled_kaggle_review_script_sha256": remote_script_hash,
        "pulled_kaggle_review_script_matches": True,
        "models": {"qwen": {"id": QWEN[0], "revision": QWEN[1]},
                   "nllb": {"id": NLLB[0], "revision": NLLB[1]}},
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "note": "This is one AI-only semantic/fluency review with NLLB back-translation. It is diagnostic evidence, not human certification or corpus approval.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
