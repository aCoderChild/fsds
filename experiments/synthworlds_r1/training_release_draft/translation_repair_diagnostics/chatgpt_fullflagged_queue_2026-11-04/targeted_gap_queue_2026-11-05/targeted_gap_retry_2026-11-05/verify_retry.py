#!/usr/bin/env python3
"""Verify a Kaggle retry receipt without converting AI review into approval."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--pulled-script", type=Path, required=True)
    args = parser.parse_args()
    input_bytes = (HERE / "retry_input.json").read_bytes()
    input_payload = json.loads(input_bytes, object_pairs_hook=unique_object)
    input_sha = sha(input_bytes)
    submitted_script = (HERE / "retry.py").read_bytes()
    input_manifest = json.loads((HERE / "retry_input_manifest.json").read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if sha(submitted_script) != input_manifest.get("retry_script_sha256"):
        raise ValueError("submitted retry script differs from its prepared manifest")
    pulled_script = args.pulled_script.read_bytes()
    if pulled_script != submitted_script:
        raise ValueError("pulled Kaggle script differs from the locally submitted source")
    prior_output_path = HERE.parent / "candidate_review_kaggle" / "remote_output" / "targeted_gap_review.json"
    prior_output_bytes = prior_output_path.read_bytes()
    if sha(prior_output_bytes) != input_payload.get("prior_output_sha256"):
        raise ValueError("retry input does not reference the verified prior output")

    output_path = args.remote_dir / "retry_output.json"
    output_bytes = output_path.read_bytes()
    output = json.loads(output_bytes, object_pairs_hook=unique_object)
    if (output.get("status") not in {"AI_ONLY_DIAGNOSTIC_RETRY_COMPLETE", "AI_ONLY_DIAGNOSTIC_RETRY_HAS_BLOCKED_ROWS"}
            or not output.get("completed_at_utc")
            or output.get("input_sha256") != input_sha
            or output.get("retry_input_payload") != input_payload
            or output.get("nllb_model_id") != NLLB_ID
            or output.get("nllb_model_revision") != NLLB_REVISION
            or output.get("qwen_model_id") != QWEN_ID
            or output.get("qwen_model_revision") != QWEN_REVISION):
        raise ValueError("output input binding or model provenance mismatch")
    if (output.get("human_translation_reviewed") is not False
            or output.get("release_approved") is not False
            or output.get("training_eligible") is not False
            or output.get("answers_or_split_labels_included") is not False):
        raise ValueError("AI-only diagnostic scope flags are invalid")
    gpu = output.get("gpu")
    if not isinstance(gpu, list) or len(gpu) < 2 or not all("T4" in name for name in gpu):
        raise ValueError(f"expected a T4 x2 runtime, observed {gpu!r}")

    expected = {row["review_key"]: row for row in input_payload["rows"]}
    records = output.get("records")
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("output does not contain the exact retry row count")
    by_key = {record.get("review_key"): record for record in records}
    if len(by_key) != len(records) or set(by_key) != set(expected):
        raise ValueError("output keys do not exactly cover the blinded retry input")

    old_output = json.loads(prior_output_bytes, object_pairs_hook=unique_object)
    old_by_key = {record["review_key"]: record for record in old_output["records"]}
    counters = collections.Counter()
    for key, row in expected.items():
        record = by_key[key]
        for field in ("source_sha256", "translation_sha256", "retry_backtranslation"):
            if record.get(field) != row[field]:
                raise ValueError(f"{key}: output {field} does not match retry input")
        if sha(row["source_question_en"].encode()) != row["source_sha256"]:
            raise ValueError(f"{key}: source hash mismatch")
        if sha(row["translated_question"].encode()) != row["translation_sha256"]:
            raise ValueError(f"{key}: candidate hash mismatch")

        backtranslation = record.get("backtranslation_en")
        if record.get("backtranslation_error") is None:
            if not isinstance(backtranslation, str) or not backtranslation.strip():
                raise ValueError(f"{key}: missing successful backtranslation")
            if sha(backtranslation.encode()) != record.get("backtranslation_sha256"):
                raise ValueError(f"{key}: backtranslation hash mismatch")
            if collections.Counter(MARKER.findall(backtranslation)) != collections.Counter(MARKER.findall(row["source_question_en"])):
                raise ValueError(f"{key}: backtranslation placeholder multiset differs from source")
            if row["retry_backtranslation"] == "REUSE_VERIFIED_PRIOR":
                if (backtranslation != row.get("prior_backtranslation_en")
                        or sha(backtranslation.encode()) != row.get("prior_backtranslation_sha256")):
                    raise ValueError(f"{key}: reused backtranslation does not match prior verified text")
                if (any(field in record for field in (
                        "backtranslation_raw", "alias_counts_expected", "alias_counts_observed"))
                        and (record.get("backtranslation_raw", "") != ""
                             or record.get("alias_counts_expected", {}) != {}
                             or record.get("alias_counts_observed", {}) != {})):
                    raise ValueError(f"{key}: reused backtranslation has unexpected alias diagnostics")
            else:
                marker_to_alias = {"[ENTITY_A]": "Alice", "[ENTITY_B]": "Bob"}
                expected_aliases = collections.Counter(
                    marker_to_alias[marker] for marker in MARKER.findall(row["translated_question"])
                )
                raw = record.get("backtranslation_raw")
                expected_aliases = dict(expected_aliases)
                observed_aliases = {alias: raw.count(alias) for alias in expected_aliases} if isinstance(raw, str) else None
                has_alias_trace = any(field in record for field in (
                    "backtranslation_raw", "alias_counts_expected", "alias_counts_observed"))
                if has_alias_trace:
                    if (not isinstance(raw, str) or not raw.strip()
                            or record.get("alias_counts_expected") != expected_aliases
                            or record.get("alias_counts_observed") != expected_aliases
                            or observed_aliases != expected_aliases):
                        raise ValueError(f"{key}: alias counts do not verify against raw NLLB back-translation")
                elif not isinstance(record.get("backtranslation_en"), str):
                    raise ValueError(f"{key}: source-pinned retry did not record a restored back-translation")

        decision = record.get("decision")
        if not isinstance(decision, dict) or "error" in decision:
            if not record.get("backtranslation_error") and not record.get("judge_attempts"):
                raise ValueError(f"{key}: blocked output has no recorded failure")
            counters["blocked"] += 1
            continue
        if (decision.get("key") != key or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip() or len(decision["reason"]) > 120):
            raise ValueError(f"{key}: invalid final judge decision")
        if not isinstance(record.get("raw_output"), str) or not record["raw_output"].strip():
            raise ValueError(f"{key}: successful decision lacks raw output")
        counters["reviewed"] += 1
        counters["semantic_pass"] += decision["semantic_pass"]
        counters["fluency_pass"] += decision["fluency_pass"]
        counters["joint_pass"] += decision["semantic_pass"] and decision["fluency_pass"]
        old = old_by_key[key]
        counters["recovered_from_block"] += "error" in (old.get("decision") or {}) or old.get("backtranslation_error") is not None

    receipt = {
        "status": "VERIFIED_AI_ONLY_RETRY_DIAGNOSTIC_QUARANTINED",
        "input_sha256": input_sha,
        "prior_output_sha256": input_payload["prior_output_sha256"],
        "output_sha256": sha(output_bytes),
        "submitted_script_sha256": sha(submitted_script),
        "pulled_script_sha256": sha(pulled_script),
        "rows": len(expected),
        "counts": dict(counters),
        "gpu": gpu,
        "nllb_model_revision": NLLB_REVISION,
        "qwen_model_revision": QWEN_REVISION,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "answers_or_split_labels_included": False,
        "interpretation": "AI-only retry diagnostics; no translation is certified or admitted to the release by this receipt.",
    }
    if (output.get("reviewed_rows") != counters["reviewed"]
            or output.get("blocked_rows") != counters["blocked"]
            or ((counters["blocked"] == 0) != (output["status"] == "AI_ONLY_DIAGNOSTIC_RETRY_COMPLETE"))):
        raise ValueError("runtime summary counts/status differ from row-level verification")
    (args.remote_dir / "retry_verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
