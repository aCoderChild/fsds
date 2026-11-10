#!/usr/bin/env python3
"""Verify the blinded Qwen3 translation pilot without approving training data."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "generation_input.json"
BUILD = HERE / "build_manifest.json"
SCRIPT = HERE / "generation.py"
REMOTE_SCRIPT = HERE / "generation_remote_source" / "synthworlds-qwen3-translation-pilot-2026-11-04.py"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "remote_output" / "qwen3_translation_pilot.json"
REPORT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "verification.json"
MODEL_ID = "Qwen/Qwen3-14B"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
PACKAGES = {"transformers": "4.51.3", "accelerate": "1.15.0", "bitsandbytes": "0.50.2"}
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: object) -> str:
    return sha(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    build = json.loads((HERE / "build_manifest.json").read_text(encoding="utf-8"))
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    records = result.get("records")
    keys = [row.get("review_key") for row in rows]
    input_hash = canonical_sha(rows)
    script_hash = sha(SCRIPT.read_bytes())
    if not REMOTE_SCRIPT.is_file():
        raise ValueError("Pulled Kaggle generation source is missing")
    remote_script_hash = sha(REMOTE_SCRIPT.read_bytes())
    if len(rows) != 16 or len(set(keys)) != 16:
        raise ValueError("Expected 16 unique blinded input rows")
    if build.get("rows") != 16 or build.get("input_sha256") != input_hash:
        raise ValueError("Build manifest does not bind to the exact pilot input")
    if build.get("generation_script_sha256") != script_hash:
        raise ValueError("Build manifest does not bind to the current generation script")
    if remote_script_hash != script_hash:
        raise ValueError("Pulled Kaggle generation source differs from local reviewed script")
    if result.get("input_sha256") != input_hash:
        raise ValueError("Remote output does not bind to the exact pilot input")
    if result.get("model_id") != MODEL_ID or result.get("model_revision") != MODEL_REVISION:
        raise ValueError("Remote output model revision mismatch")
    if {name: result.get(name) for name in PACKAGES} != PACKAGES:
        raise ValueError("Remote output package pins mismatch")
    if result.get("quantization") != "4-bit NF4" or result.get("gpu") != "Tesla T4":
        raise ValueError("Remote output is not the expected NF4/T4 profile")
    if result.get("status") not in {"AI_ONLY_TRANSLATION_PILOT_COMPLETE", "AI_ONLY_TRANSLATION_PILOT_BLOCKED"}:
        raise ValueError("Unexpected translation pilot run status")
    if not isinstance(records, list) or [r.get("review_key") for r in records] != keys:
        raise ValueError("Remote output has incomplete, duplicate, reordered, or extra rows")
    if (result.get("answers_or_splits_sent") is not False
            or result.get("human_translation_reviewed") is not False):
        raise ValueError("Blinding or human-review metadata is misrepresented")
    if not result.get("started_at_utc") or not result.get("completed_at_utc"):
        raise ValueError("Remote result lacks runtime timestamps")

    source_by_key = {row["review_key"]: row for row in rows}
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    translated = blocked = 0
    for record in records:
        source = source_by_key[record["review_key"]]
        if record.get("language") != source["language"] or record.get("kind") != source["kind"]:
            raise ValueError(f"Language/kind mismatch at {record['review_key']}")
        if record.get("source_sha256") != sha(source["source_question_en"].encode("utf-8")):
            raise ValueError(f"Source hash mismatch at {record['review_key']}")
        counts[(source["language"], source["kind"])] += 1
        translation = record.get("translation")
        if record.get("error") is not None:
            if translation is not None:
                raise ValueError(f"Blocked row unexpectedly has a translation at {record['review_key']}")
            blocked += 1
            errors[record["error"].split(":", 1)[0]] += 1
            continue
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError(f"Missing translation without a recorded error at {record['review_key']}")
        if record.get("translation_sha256") != sha(translation.encode("utf-8")):
            raise ValueError(f"Translation hash mismatch at {record['review_key']}")
        if collections.Counter(PLACEHOLDER.findall(source["source_question_en"])) != collections.Counter(PLACEHOLDER.findall(translation)):
            raise ValueError(f"Placeholder mismatch at {record['review_key']}")
        if source["language"] == "hi" and not any("\u0900" <= char <= "\u097f" for char in translation):
            raise ValueError(f"Hindi script check failed at {record['review_key']}")
        if source["language"] == "th" and not any("\u0e00" <= char <= "\u0e7f" for char in translation):
            raise ValueError(f"Thai script check failed at {record['review_key']}")
        translated += 1

    expected_counts = {(lang, kind): 2 for lang in ("fr", "de", "hi", "th") for kind in ("atomic", "composition")}
    if dict(counts) != expected_counts:
        raise ValueError("Pilot is not balanced two-by-two across languages and question kinds")
    if result.get("translated_rows") != translated or result.get("blocked_rows") != blocked:
        raise ValueError("Remote completion counters disagree with row-level results")
    expected_status = "AI_ONLY_TRANSLATION_PILOT_COMPLETE" if blocked == 0 else "AI_ONLY_TRANSLATION_PILOT_BLOCKED"
    if result.get("status") != expected_status:
        raise ValueError("Remote status does not match its row-level outcome")
    report = {
        "status": "VERIFIED_QWEN3_TRANSLATION_PILOT_DIAGNOSTIC_ONLY" if blocked == 0 else "VERIFIED_QWEN3_TRANSLATION_PILOT_WITH_BLOCKED_ROWS_DIAGNOSTIC_ONLY",
        "rows": len(records),
        "translated_rows": translated,
        "blocked_rows": blocked,
        "generation_error_types": dict(sorted(errors.items())),
        "counts_by_language_kind": {f"{lang}/{kind}": counts[(lang, kind)]
                                    for lang in ("fr", "de", "hi", "th")
                                    for kind in ("atomic", "composition")},
        "input_sha256": input_hash,
        "output_sha256": sha(OUTPUT.read_bytes()),
        "generation_script_sha256": script_hash,
        "pulled_kaggle_generation_script_sha256": remote_script_hash,
        "pulled_kaggle_generation_script_matches": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime": {
            "gpu": result["gpu"],
            "quantization": result["quantization"],
            "packages": {name: result[name] for name in PACKAGES},
            "started_at_utc": result["started_at_utc"],
            "completed_at_utc": result["completed_at_utc"],
        },
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "note": "This verifies provenance, structure, and placeholder preservation only; translation quality still requires the separate blinded review screen.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
