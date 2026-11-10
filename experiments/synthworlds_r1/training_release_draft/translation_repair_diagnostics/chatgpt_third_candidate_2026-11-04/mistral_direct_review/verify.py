#!/usr/bin/env python3
"""Verify the 187-row Mistral cross-review and its source provenance."""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "review_input.json"
BUILD = HERE / "build_manifest.json"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "remote_output/mistral_direct_review.json"
REPORT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "verification.json"
LOCAL_SCRIPT = HERE / "review.py"
REMOTE_SCRIPT = HERE / "remote_kernel_source.py"
EXPECTED_MODEL = ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71")
EXPECTED_PACKAGES = {"transformers": "4.51.3", "accelerate": "1.15.0", "bitsandbytes": "0.50.2"}
LANGUAGES = ("fr", "de", "hi", "th")
KINDS = ("atomic", "composition")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return sha_bytes(canonical.encode("utf-8"))


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    records = result.get("records")
    input_hash = sha_json(rows)
    script_hash = sha_bytes(LOCAL_SCRIPT.read_bytes())
    remote_script_hash = sha_bytes(REMOTE_SCRIPT.read_bytes())

    if build.get("rows") != 187 or len(rows) != 187:
        raise ValueError("Expected the frozen 187-row review input")
    keys = [row.get("review_key") for row in rows]
    if len(set(keys)) != 187 or not isinstance(records, list) or len(records) != 187:
        raise ValueError("Input or output does not contain 187 unique rows")
    if [record.get("review_key") for record in records] != keys:
        raise ValueError("Output order/coverage differs from the exact input")
    if build.get("input_sha256") != input_hash or result.get("input_sha256") != input_hash:
        raise ValueError("Input hash does not match build manifest and run output")
    if build.get("review_script_sha256") != script_hash or remote_script_hash != script_hash:
        raise ValueError("Kaggle-pulled source does not match the hash-bound local review script")
    if (build.get("model_id"), build.get("model_revision")) != EXPECTED_MODEL:
        raise ValueError("Build manifest model/revision mismatch")
    if (result.get("model_id"), result.get("model_revision")) != EXPECTED_MODEL:
        raise ValueError("Run output model/revision mismatch")
    if result.get("quantization") != "4-bit NF4" or result.get("gpu") != "Tesla T4":
        raise ValueError("Run output does not match the expected NF4/T4 profile")
    if {name: result.get(name) for name in EXPECTED_PACKAGES} != EXPECTED_PACKAGES:
        raise ValueError("Run output package pins do not match the expected profile")
    if not result.get("started_at_utc") or not result.get("completed_at_utc"):
        raise ValueError("Run output is missing runtime timestamps")

    source_by_key = {row["review_key"]: row for row in rows}
    counts: collections.Counter[tuple] = collections.Counter()
    for record in records:
        source = source_by_key[record["review_key"]]
        language, kind = source.get("language"), source.get("kind")
        if language not in LANGUAGES or kind not in KINDS:
            raise ValueError(f"Unexpected language/kind for {record['review_key']}")
        if record.get("language") != language or record.get("kind") != kind:
            raise ValueError(f"Language/kind mismatch for {record['review_key']}")
        if record.get("source_sha256") != sha_bytes(source["source_question_en"].encode("utf-8")):
            raise ValueError(f"Source hash mismatch for {record['review_key']}")
        if record.get("translation_sha256") != sha_bytes(source["translated_question"].encode("utf-8")):
            raise ValueError(f"Candidate hash mismatch for {record['review_key']}")

        decision = record.get("decision") or {}
        if "error" in decision:
            counts[(language, "blocked")] += 1
            counts[(language, kind, "blocked")] += 1
            continue
        if (decision.get("key") != record["review_key"]
                or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip()):
            raise ValueError(f"Decision schema/key mismatch for {record['review_key']}")
        label = "pass" if decision["semantic_pass"] and decision["fluency_pass"] else "flag"
        counts[(language, label)] += 1
        counts[(language, kind, label)] += 1

    blocked = sum("error" in (record.get("decision") or {}) for record in records)
    if result.get("reviewed_rows") != 187 - blocked or result.get("blocked_rows") != blocked:
        raise ValueError("Run output counters disagree with row-level decisions")
    if (result.get("answers_or_splits_sent") is not False
            or result.get("qwen_decisions_or_backtranslations_sent") is not False
            or result.get("human_translation_reviewed") is not False):
        raise ValueError("Blinding or human-review metadata is misrepresented")

    counts_by_language = {
        lang: {
            "pass": counts[(lang, "pass")],
            "flag": counts[(lang, "flag")],
            "blocked": counts[(lang, "blocked")],
            "by_kind": {
                kind: {label: counts[(lang, kind, label)] for label in ("pass", "flag", "blocked")}
                for kind in KINDS
            },
        }
        for lang in LANGUAGES
    }
    report = {
        "status": "VERIFIED_MISTRAL_DIRECT_187_DIAGNOSTIC_ONLY" if blocked == 0 else "VERIFIED_MISTRAL_DIRECT_WITH_BLOCKED_ROWS_DIAGNOSTIC_ONLY",
        "rows": 187,
        "blocked_rows": blocked,
        "counts_by_language": counts_by_language,
        "input_sha256": input_hash,
        "output_sha256": sha_bytes(OUTPUT.read_bytes()),
        "local_review_script_sha256": script_hash,
        "kaggle_pulled_review_script_sha256": remote_script_hash,
        "models": {"id": EXPECTED_MODEL[0], "revision": EXPECTED_MODEL[1]},
        "runtime": {
            "gpu": result["gpu"],
            "quantization": result["quantization"],
            "packages": {name: result[name] for name in EXPECTED_PACKAGES},
            "started_at_utc": result["started_at_utc"],
            "completed_at_utc": result["completed_at_utc"],
        },
        "answers_or_splits_sent": False,
        "qwen_decisions_or_backtranslations_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "note": "Direct source-to-translation Mistral review is an independent AI diagnostic, not translation certification.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
