#!/usr/bin/env python3
"""Verify source coverage, exact code, markers, decisions, and quarantine scope."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETED = HERE.parent
PREVIOUS = TARGETED / "targeted_gap_nllb_candidates_2026-11-05"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha(text.encode("utf-8"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--pulled-script", type=Path, required=True)
    args = parser.parse_args()

    input_path = HERE / "segmented_retry_input.json"
    manifest_path = HERE / "segmented_retry_input_manifest.json"
    script_path = HERE / "segmented_retry.py"
    output_path = args.remote_dir / "candidate_review.json"
    input_bytes = input_path.read_bytes()
    previous_input_path = PREVIOUS / "candidate_input.json"
    previous_output_path = PREVIOUS / "remote_output/candidate_review.json"
    previous_receipt_path = PREVIOUS / "remote_output/candidate_verification.json"
    row_map_path = TARGETED / "private_row_map.jsonl"
    previous_input = read_json(previous_input_path)
    previous_output = read_json(previous_output_path)
    manifest = read_json(manifest_path)
    payload = json.loads(input_bytes, object_pairs_hook=unique_object)
    output_bytes = output_path.read_bytes()
    output = json.loads(output_bytes, object_pairs_hook=unique_object)
    local_script = script_path.read_bytes()
    pulled_script = args.pulled_script.read_bytes()
    private_rows = [
        json.loads(line, object_pairs_hook=unique_object)
        for line in row_map_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    private_by_key = {row["row_id"]: row for row in private_rows}
    previous_input_by_key = {row["review_key"]: row for row in previous_input["rows"]}
    previous_records_by_key = {row["review_key"]: row for row in previous_output["records"]}
    blocked_keys = {
        key for key, record in previous_records_by_key.items()
        if record.get("candidate_error") is not None
        or record.get("backtranslation_error") is not None
        or "error" in (record.get("decision") or {})
    }
    expected_rows = [previous_input_by_key[key] for key in sorted(blocked_keys)]
    previous_sha = sha(previous_output_path.read_bytes())
    previous_receipt_sha = sha(previous_receipt_path.read_bytes())

    if (payload.get("schema") != "synthworlds-nllb-segmented-retry-v1"
            or payload.get("mode") != "AI_ONLY_DIAGNOSTIC_SEGMENTED_RETRY_NO_RELEASE_APPROVAL"
            or payload.get("expected_rows") != 125
            or payload.get("rows") != expected_rows
            or len(expected_rows) != 125
            or len(private_by_key) != 187
            or manifest.get("status") != "VERIFIED_ANSWER_BLIND_SEGMENTED_RETRY_INPUT"
            or sha(input_bytes) != manifest.get("input_sha256")
            or payload.get("source_review_input_sha256") != previous_input.get("source_review_input_sha256")
            or payload.get("previous_candidate_input_sha256") != sha(previous_input_path.read_bytes())
            or payload.get("previous_candidate_output_sha256") != previous_sha
            or payload.get("previous_candidate_receipt_sha256") != previous_receipt_sha
            or payload.get("row_map_sha256") != sha(row_map_path.read_bytes())
            or manifest.get("previous_candidate_output_sha256") != previous_sha
            or manifest.get("previous_candidate_receipt_sha256") != previous_receipt_sha
            or manifest.get("blocked_rows_selected") != len(blocked_keys)
            or manifest.get("script_sha256") != sha(local_script)
            or pulled_script != local_script
            or manifest.get("kernel_id") != "bunnoob2005/sw-gap187-nllb-segmented-retry-2026-11-05"):
        raise ValueError("Segmented retry input, blocked-row selection, or submitted source binding is invalid")
    if (payload.get("answers_or_split_labels_included") is not False
            or payload.get("human_translation_reviewed") is not False
            or payload.get("release_approved") is not False
            or payload.get("training_eligible") is not False):
        raise ValueError("Segmented retry input improperly claims approval or includes disallowed fields")

    if (output.get("status") not in {
                "AI_ONLY_NLLB_SEGMENTED_RETRY_COMPLETE",
                "AI_ONLY_NLLB_SEGMENTED_RETRY_HAS_BLOCKS",
            }
            or output.get("input_sha256") != sha(input_bytes)
            or output.get("input_payload") != payload
            or output.get("nllb_model_id") != "facebook/nllb-200-distilled-600M"
            or output.get("nllb_model_revision") != NLLB_REVISION
            or output.get("qwen_model_id") != "Qwen/Qwen2.5-14B-Instruct"
            or output.get("qwen_model_revision") != QWEN_REVISION
            or output.get("human_translation_reviewed") is not False
            or output.get("answers_or_split_labels_included") is not False
            or output.get("release_approved") is not False
            or output.get("training_eligible") is not False
            or not output.get("completed_at_utc")):
        raise ValueError("Remote segmented retry provenance or diagnostic-only scope is invalid")
    gpu = output.get("gpu")
    if not isinstance(gpu, list) or len(gpu) != 2 or not all("T4" in name for name in gpu):
        raise ValueError(f"Expected exactly two Tesla T4 GPUs; got {gpu!r}")

    expected_by_key = {row["review_key"]: row for row in expected_rows}
    records = output.get("records")
    by_key = {row.get("review_key"): row for row in records or []}
    if len(records or []) != 125 or len(by_key) != 125 or set(by_key) != set(expected_by_key):
        raise ValueError("Segmented retry output does not exactly cover the 125 blocked rows")

    counts = collections.Counter()
    by_language = {language: collections.Counter() for language in ("fr", "de", "hi", "th")}
    error_counts = collections.Counter()
    for key, source in expected_by_key.items():
        record = by_key[key]
        lang = source["language"]
        private = private_by_key.get(key)
        if (private is None or private.get("row_id") != key
                or private.get("language") != lang or private.get("kind") != source["kind"]
                or source.get("source_sha256") != sha_text(source["source_question_en"])
                or record.get("language") != lang or record.get("kind") != source["kind"]
                or record.get("source_sha256") != source["source_sha256"]):
            raise ValueError(f"{key}: source row/map/hash mismatch")
        if record.get("candidate_error") is None:
            candidate = record.get("candidate_translation")
            if (not isinstance(candidate, str) or not candidate.strip()
                    or record.get("candidate_sha256") != sha_text(candidate)
                    or collections.Counter(MARKER.findall(candidate))
                    != collections.Counter(MARKER.findall(source["source_question_en"]))):
                raise ValueError(f"{key}: candidate hash or placeholder multiset mismatch")
            counts["candidate_rows"] += 1
            by_language[lang]["candidate_rows"] += 1
        else:
            if record.get("candidate_translation") or record.get("candidate_sha256"):
                raise ValueError(f"{key}: failed candidate has unexpected text/hash")
            error_counts["candidate_error"] += 1

        if record.get("candidate_error") is not None or record.get("backtranslation_error") is not None:
            if not isinstance(record.get("decision"), dict) or "error" not in record["decision"]:
                raise ValueError(f"{key}: translation failure is not recorded as blocked")
            if record.get("backtranslation_error") is not None:
                error_counts["backtranslation_error"] += 1
            counts["blocked"] += 1
            by_language[lang]["blocked"] += 1
            continue

        backtranslation = record.get("backtranslation_en")
        if (not isinstance(backtranslation, str) or not backtranslation.strip()
                or record.get("backtranslation_sha256") != sha_text(backtranslation)
                or collections.Counter(MARKER.findall(backtranslation))
                != collections.Counter(MARKER.findall(source["source_question_en"]))):
            raise ValueError(f"{key}: back-translation hash or placeholder multiset mismatch")
        decision = record.get("decision")
        if not isinstance(decision, dict) or "error" in decision:
            counts["blocked"] += 1
            by_language[lang]["blocked"] += 1
            error_counts["judge_error"] += 1
            continue
        if (decision.get("key") != key or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip() or len(decision["reason"]) > 120
                or not isinstance(record.get("raw_output"), str) or not record["raw_output"].strip()):
            raise ValueError(f"{key}: Qwen decision schema/key invalid")
        counts["reviewed"] += 1
        by_language[lang]["reviewed"] += 1
        counts["semantic_pass"] += decision["semantic_pass"]
        by_language[lang]["semantic_pass"] += decision["semantic_pass"]
        counts["fluency_pass"] += decision["fluency_pass"]
        by_language[lang]["fluency_pass"] += decision["fluency_pass"]
        joint = decision["semantic_pass"] and decision["fluency_pass"]
        counts["joint_pass"] += joint
        by_language[lang]["joint_pass"] += joint

    if (output.get("candidate_rows") != counts["candidate_rows"]
            or output.get("reviewed_rows") != counts["reviewed"]
            or output.get("blocked_rows") != counts["blocked"]
            or output.get("status") != (
                "AI_ONLY_NLLB_SEGMENTED_RETRY_COMPLETE" if counts["blocked"] == 0
                else "AI_ONLY_NLLB_SEGMENTED_RETRY_HAS_BLOCKS"
            )):
        raise ValueError("Remote segmented retry summary differs from row-level recomputation")

    receipt = {
        "status": "VERIFIED_AI_ONLY_NLLB_SEGMENTED_RETRY_QUARANTINED",
        "input_sha256": sha(input_bytes),
        "output_sha256": sha(output_bytes),
        "submitted_script_sha256": sha(local_script),
        "pulled_script_sha256": sha(pulled_script),
        "previous_candidate_output_sha256": previous_sha,
        "rows": len(expected_rows),
        "counts": dict(counts),
        "counts_by_language": {lang: dict(values) for lang, values in by_language.items()},
        "error_counts": dict(error_counts),
        "gpu": gpu,
        "nllb_model_revision": NLLB_REVISION,
        "qwen_model_revision": QWEN_REVISION,
        "human_translation_reviewed": False,
        "answers_or_split_labels_included": False,
        "release_approved": False,
        "training_eligible": False,
        "interpretation": (
            "AI-only marker-preserving candidate retry; outputs are quarantined and cannot approve a translation."
        ),
    }
    receipt_path = args.remote_dir / "segmented_retry_verification.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
