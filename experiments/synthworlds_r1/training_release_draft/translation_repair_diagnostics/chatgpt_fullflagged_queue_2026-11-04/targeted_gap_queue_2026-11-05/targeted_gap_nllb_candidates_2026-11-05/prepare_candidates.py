#!/usr/bin/env python3
"""Prepare only still-unresolved rows for a pinned NLLB candidate translation pass."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETED = HERE.parent
REVIEW = TARGETED / "candidate_review_kaggle"
RETRY = TARGETED / "targeted_gap_retry_2026-11-05"
LANGUAGES = {"fr", "de", "hi", "th"}
EXPECTED_ROWS = 148


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


def embedded_source_rows() -> tuple[str, list[dict]]:
    tree = ast.parse((REVIEW / "review.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RAW_INPUT" for target in node.targets
        ):
            raw = ast.literal_eval(node.value)
            return raw, json.loads(raw, object_pairs_hook=unique_object)
    raise ValueError("source review does not contain the expected literal input")


def main() -> None:
    raw, source_rows = embedded_source_rows()
    source_sha = sha(raw.encode("utf-8"))
    prior_review_path = REVIEW / "remote_output/targeted_gap_review.json"
    retry_output_path = RETRY / "remote_output/retry_output.json"
    prior_review_bytes = prior_review_path.read_bytes()
    retry_output_bytes = retry_output_path.read_bytes()
    prior_review = json.loads(prior_review_bytes, object_pairs_hook=unique_object)
    retry_output = json.loads(retry_output_bytes, object_pairs_hook=unique_object)
    prior_by_key = {row["review_key"]: row for row in prior_review["records"]}
    retry_by_key = {row["review_key"]: row for row in retry_output["records"]}
    if (len(source_rows) != 187 or len(prior_by_key) != 187
            or retry_output.get("status") != "AI_ONLY_DIAGNOSTIC_RETRY_COMPLETE"
            or retry_output.get("training_eligible") is not False
            or len(retry_by_key) != 17):
        raise ValueError("prior 187-row review or verified 17-row retry is incomplete")

    map_path = TARGETED / "private_row_map.jsonl"
    private_rows = [json.loads(line, object_pairs_hook=unique_object)
                    for line in map_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    private_by_key = {row["row_id"]: row for row in private_rows}
    if len(private_by_key) != 187:
        raise ValueError("private row map does not cover the 187-row source queue")

    passed = set()
    for row in prior_review["records"]:
        decision = row.get("decision") or {}
        if (row.get("backtranslation_error") is None and isinstance(decision, dict)
                and decision.get("semantic_pass") is True and decision.get("fluency_pass") is True):
            passed.add(row["review_key"])
    for row in retry_output["records"]:
        decision = row.get("decision") or {}
        if (row.get("backtranslation_error") is None and isinstance(decision, dict)
                and "error" not in decision and decision.get("semantic_pass") is True
                and decision.get("fluency_pass") is True):
            passed.add(row["review_key"])

    selected = []
    for row in source_rows:
        key = row["review_key"]
        private = private_by_key.get(key)
        if private is None:
            raise ValueError(f"missing private mapping for {key}")
        if row["language"] not in LANGUAGES or row["kind"] not in {"atomic", "composition"}:
            raise ValueError(f"invalid language/kind for {key}")
        if (private["language"] != row["language"] or private["kind"] != row["kind"]
                or private["row_id"] != key
                or private["source_text_sha256"] == ""):
            raise ValueError(f"row-map mismatch for {key}")
        if sha_text(row["source_question_en"]) != prior_by_key[key]["source_sha256"]:
            raise ValueError(f"source hash mismatch in previous screen for {key}")
        if key in passed:
            continue
        selected.append({
            "review_key": key,
            "language": row["language"],
            "kind": row["kind"],
            "source_question_en": row["source_question_en"],
            "source_sha256": sha_text(row["source_question_en"]),
        })

    if len(selected) != EXPECTED_ROWS or len({row["review_key"] for row in selected}) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} unresolved unique rows; got {len(selected)}")
    selected.sort(key=lambda row: row["review_key"])
    payload = {
        "schema": "synthworlds-nllb-alternative-translation-v1",
        "mode": "AI_ONLY_DIAGNOSTIC_CANDIDATE_NO_RELEASE_APPROVAL",
        "source_review_input_sha256": source_sha,
        "prior_review_output_sha256": sha(prior_review_bytes),
        "prior_retry_output_sha256": sha(retry_output_bytes),
        "row_map_sha256": sha(map_path.read_bytes()),
        "nllb_model_id": "facebook/nllb-200-distilled-600M",
        "nllb_model_revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "rows": selected,
        "answers_or_split_labels_included": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (HERE / "candidate_input.json").write_bytes(encoded)
    manifest = {
        "status": "VERIFIED_ANSWER_BLIND_NLLB_CANDIDATE_INPUT",
        "input_sha256": sha(encoded),
        "source_review_input_sha256": source_sha,
        "prior_review_output_sha256": sha(prior_review_bytes),
        "prior_retry_output_sha256": sha(retry_output_bytes),
        "row_map_sha256": sha(map_path.read_bytes()),
        "rows": len(selected),
        "previous_joint_passes_excluded": len(passed),
        "language_rows": {lang: sum(row["language"] == lang for row in selected)
                           for lang in sorted(LANGUAGES)},
        "answers_or_split_labels_included": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    (HERE / "candidate_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    template = (HERE / "candidate_review_template.py").read_text(encoding="utf-8")
    if template.count("__INPUT_B64__") != 1 or template.count("__INPUT_SHA256__") != 1:
        raise ValueError("candidate review template placeholders missing or duplicated")
    source = template.replace("__INPUT_B64__", base64.b64encode(encoded).decode())
    source = source.replace("__INPUT_SHA256__", sha(encoded))
    (HERE / "candidate_review.py").write_text(source, encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/sw-gap187-nllb-148-candidate-review-2026-11-05",
        "title": "SW Gap187 NLLB 148 Candidate Review 2026-11-05",
        "code_file": "candidate_review.py",
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
    manifest["candidate_script_sha256"] = sha(source.encode())
    manifest["kernel_id"] = metadata["id"]
    (HERE / "candidate_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
