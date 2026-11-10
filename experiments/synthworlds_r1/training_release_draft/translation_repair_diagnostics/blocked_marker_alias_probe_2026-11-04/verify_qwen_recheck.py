#!/usr/bin/env python3
"""Verify the blinded 31-row Qwen recheck output and input binding."""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "qwen_recheck_manifest.json"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_decision(raw: str, expected_key: str) -> dict:
    text = raw.strip()
    if not text.startswith("{"):
        raise ValueError("model response is not a JSON object from its first character")
    parsed, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(text)
    if text[end:].strip():
        raise ValueError("non-whitespace trailing text after JSON")
    if not isinstance(parsed, dict):
        raise ValueError("top-level response is not an object")
    if set(parsed) == {"decisions"}:
        decisions = parsed["decisions"]
        if not isinstance(decisions, list) or len(decisions) != 1:
            raise ValueError("expected exactly one decision in decisions array")
        decision = decisions[0]
    elif set(parsed) == {expected_key} and isinstance(parsed[expected_key], dict):
        decision = {"key": expected_key, **parsed[expected_key]}
    else:
        decision = parsed
    if (not isinstance(decision, dict)
            or set(decision) != {"key", "semantic_pass", "fluency_pass", "reason"}
            or decision.get("key") != expected_key
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str)
            or not decision["reason"].strip()):
        raise ValueError("single decision failed exact key/schema checks")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--pulled-script", type=Path, required=True)
    args = parser.parse_args()
    local_script = (HERE / "qwen_kernel/review.py").read_bytes()
    pulled_script = args.pulled_script.read_bytes()
    if local_script != pulled_script:
        raise ValueError("Kaggle-pulled recheck script differs from submitted local source")
    source = local_script.decode("utf-8")
    b64 = re.search(r'^RAW_INPUT_B64 = "([^"]+)"$', source, re.M)
    input_hash = re.search(r'^INPUT_SHA256 = "([^"]+)"$', source, re.M)
    if not b64 or not input_hash:
        raise ValueError("embedded input fields missing")
    raw_input = base64.b64decode(b64.group(1))
    if sha(raw_input) != input_hash.group(1):
        raise ValueError("embedded input hash mismatch")
    inputs = json.loads(raw_input)
    manifest_path = MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(inputs) != 31 or manifest.get("input_sha256") != input_hash.group(1):
        raise ValueError("31-row input differs from its build manifest")
    output_path = args.remote_dir / "qwen_recheck.json"
    output = json.loads(output_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if (output.get("status") != "DIAGNOSTIC_COMPLETE"
            or output.get("model_id") != MODEL or output.get("model_revision") != REVISION
            or output.get("input_sha256") != input_hash.group(1)
            or output.get("quantization") != "4-bit NF4"
            or output.get("lm_format_enforcer") != "0.11.1"
            or output.get("output_schema") != "single-item decisions array"
            or output.get("human_translation_reviewed") is not False
            or output.get("answers_or_splits_sent") is not False):
        raise ValueError("remote model, input, or scope receipt is invalid")
    expected = {row["review_key"]: row for row in inputs}
    records = output.get("records", [])
    if not isinstance(records, list) or len(records) != 31:
        raise ValueError("remote output does not contain exactly 31 records")
    by_key = {}
    for row in records:
        key = row.get("review_key") if isinstance(row, dict) else None
        if key in by_key:
            raise ValueError(f"duplicate remote review key: {key}")
        by_key[key] = row
    if len(by_key) != 31 or set(by_key) != set(expected):
        raise ValueError("remote row coverage differs from exact 31-row input")
    counts = collections.Counter()
    for key, record in by_key.items():
        source = expected[key]
        if (record.get("language") != source["language"] or record.get("kind") != source["kind"]
                or record.get("source_sha256") != sha(source["source_question_en"].encode())
                or record.get("translation_sha256") != sha(source["translated_question"].encode())
                or record.get("back_translation_sha256") != sha(source["back_translation_en"].encode())):
            raise ValueError(f"{key}: record does not bind to exact source/target/back-translation")
        marker_counts = [collections.Counter(MARKER.findall(source[field]))
                         for field in ("source_question_en", "translated_question", "back_translation_en")]
        if not (marker_counts[0] == marker_counts[1] == marker_counts[2]):
            raise ValueError(f"{key}: input placeholder multisets differ")
        raw = record.get("raw_output")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{key}: raw model output was not preserved")
        decision = record.get("decision")
        try:
            parsed = parse_decision(raw, key)
        except Exception as exc:
            expected_error = {"error": type(exc).__name__, "detail": str(exc)}
            if decision != expected_error:
                raise ValueError(f"{key}: malformed output was not retained as blocked") from exc
            counts["MALFORMED_BLOCKED"] += 1
        else:
            if decision != parsed:
                raise ValueError(f"{key}: normalized decision differs from raw model output")
            counts["PASS" if decision["semantic_pass"] and decision["fluency_pass"] else "FLAG"] += 1
    if output.get("counts") != dict(counts):
        raise ValueError("reported outcome counts differ from per-row verified decisions")
    receipt = {
        "status": "VERIFIED_BLINDED_QWEN_RECHECK_DIAGNOSTIC_ONLY",
        "release_approved": False, "training_eligible": False,
        "rows": len(records), "counts": dict(counts),
        "model_id": MODEL, "model_revision": REVISION,
        "input_sha256": input_hash.group(1),
        "submitted_script_sha256": sha(local_script), "pulled_script_sha256": sha(pulled_script),
        "output_sha256": sha(output_path.read_bytes()),
        "human_translation_reviewed": False, "answers_or_splits_sent": False,
        "interpretation": "Hash-bound AI-only semantic/fluency judgments; no translation release approval.",
    }
    (args.remote_dir / "verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
