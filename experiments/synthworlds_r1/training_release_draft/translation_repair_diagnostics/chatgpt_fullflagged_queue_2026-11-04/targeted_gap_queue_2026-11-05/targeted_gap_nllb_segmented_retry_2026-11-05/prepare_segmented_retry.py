#!/usr/bin/env python3
"""Prepare an answer-blind retry for rows blocked by alias-based NLLB screening."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETED = HERE.parent
PREVIOUS = TARGETED / "targeted_gap_nllb_candidates_2026-11-05"
EXPECTED_ROWS = 125
KERNEL_ID = "bunnoob2005/sw-gap187-nllb-segmented-retry-2026-11-05"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha(text.encode("utf-8"))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> None:
    previous_input_bytes = (PREVIOUS / "candidate_input.json").read_bytes()
    previous_output_bytes = (PREVIOUS / "remote_output/candidate_review.json").read_bytes()
    previous_manifest = json.loads(
        (PREVIOUS / "candidate_input_manifest.json").read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
    )
    previous_input = json.loads(previous_input_bytes, object_pairs_hook=unique_object)
    previous_output = json.loads(previous_output_bytes, object_pairs_hook=unique_object)
    previous_receipt = json.loads(
        (PREVIOUS / "remote_output/candidate_verification.json").read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
    )
    row_map_path = TARGETED / "private_row_map.jsonl"
    row_map_bytes = row_map_path.read_bytes()
    private_rows = [
        json.loads(line, object_pairs_hook=unique_object)
        for line in row_map_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    private_by_key = {row["row_id"]: row for row in private_rows}
    if (previous_input.get("schema") != "synthworlds-nllb-alternative-translation-v1"
            or len(previous_input.get("rows", [])) != 148
            or len(previous_output.get("records", [])) != 148
            or previous_output.get("status") != "AI_ONLY_NLLB_CANDIDATE_REVIEW_HAS_BLOCKS"
            or previous_receipt.get("status") != "VERIFIED_AI_ONLY_NLLB_CANDIDATES_QUARANTINED"
            or previous_receipt.get("output_sha256") != sha(previous_output_bytes)
            or previous_manifest.get("input_sha256") != sha(previous_input_bytes)
            or previous_input.get("release_approved") is not False
            or previous_input.get("training_eligible") is not False
            or len(private_by_key) != 187):
        raise ValueError("The previous NLLB screen or source map is incomplete/unverified")

    input_by_key = {row["review_key"]: row for row in previous_input["rows"]}
    record_by_key = {row["review_key"]: row for row in previous_output["records"]}
    if len(input_by_key) != 148 or len(record_by_key) != 148 or set(input_by_key) != set(record_by_key):
        raise ValueError("Previous input and candidate output do not have exact unique row coverage")

    blocked_keys = set()
    for key, record in record_by_key.items():
        source = input_by_key[key]
        private = private_by_key.get(key)
        decision = record.get("decision") or {}
        if (private is None or private.get("row_id") != key
                or private.get("language") != source.get("language")
                or private.get("kind") != source.get("kind")
                or source.get("source_sha256") != sha_text(source["source_question_en"])
                or record.get("source_sha256") != source.get("source_sha256")):
            raise ValueError(f"Previous row {key} does not match the source-bound map")
        if (record.get("candidate_error") is not None
                or record.get("backtranslation_error") is not None
                or not isinstance(decision, dict) or "error" in decision):
            blocked_keys.add(key)

    selected = [input_by_key[key] for key in sorted(blocked_keys)]
    if len(selected) != EXPECTED_ROWS or len(blocked_keys) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} blocked rows from the previous screen; got {len(selected)}")
    for row in selected:
        if set(row) != {"review_key", "language", "kind", "source_question_en", "source_sha256"}:
            raise ValueError(f"Unexpected input fields for {row['review_key']}")
        if row["language"] not in {"fr", "de", "hi", "th"}:
            raise ValueError(f"Unsupported target language for {row['review_key']}")

    payload = {
        "schema": "synthworlds-nllb-segmented-retry-v1",
        "mode": "AI_ONLY_DIAGNOSTIC_SEGMENTED_RETRY_NO_RELEASE_APPROVAL",
        "expected_rows": EXPECTED_ROWS,
        "source_review_input_sha256": previous_input["source_review_input_sha256"],
        "previous_candidate_input_sha256": sha(previous_input_bytes),
        "previous_candidate_output_sha256": sha(previous_output_bytes),
        "previous_candidate_receipt_sha256": sha((PREVIOUS / "remote_output/candidate_verification.json").read_bytes()),
        "row_map_sha256": sha(row_map_bytes),
        "nllb_model_id": "facebook/nllb-200-distilled-600M",
        "nllb_model_revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "qwen_model_id": "Qwen/Qwen2.5-14B-Instruct",
        "qwen_model_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "rows": selected,
        "answers_or_split_labels_included": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (HERE / "segmented_retry_input.json").write_bytes(encoded)
    language_rows = {lang: sum(row["language"] == lang for row in selected) for lang in ("fr", "de", "hi", "th")}
    manifest = {
        "status": "VERIFIED_ANSWER_BLIND_SEGMENTED_RETRY_INPUT",
        "input_sha256": sha(encoded),
        "previous_candidate_input_sha256": sha(previous_input_bytes),
        "previous_candidate_output_sha256": sha(previous_output_bytes),
        "previous_candidate_receipt_sha256": payload["previous_candidate_receipt_sha256"],
        "row_map_sha256": sha(row_map_bytes),
        "rows": len(selected),
        "blocked_rows_selected": len(blocked_keys),
        "language_rows": language_rows,
        "answers_or_split_labels_included": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "kernel_id": KERNEL_ID,
    }
    template = (HERE / "segmented_retry_template.py").read_text(encoding="utf-8")
    if template.count("__INPUT_B64__") != 1 or template.count("__INPUT_SHA256__") != 1:
        raise ValueError("segmented retry template placeholders are missing or duplicated")
    source = template.replace("__INPUT_B64__", base64.b64encode(encoded).decode())
    source = source.replace("__INPUT_SHA256__", sha(encoded))
    (HERE / "segmented_retry.py").write_text(source, encoding="utf-8")
    manifest["script_sha256"] = sha(source.encode())
    (HERE / "segmented_retry_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": "SW Gap187 NLLB Segmented Retry 2026-11-05",
        "code_file": "segmented_retry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
