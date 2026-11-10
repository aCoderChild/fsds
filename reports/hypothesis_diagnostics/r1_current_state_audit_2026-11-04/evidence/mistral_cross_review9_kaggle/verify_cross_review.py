#!/usr/bin/env python3
"""Verify provenance and exact row alignment for the diagnostic Mistral review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
QWEN_DIR = HERE.parent / "chatgpt_translation_calibration32_kaggle"
EXPECTED_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
EXPECTED_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = load_json(HERE / "BUILD_MANIFEST.json")
    qwen = load_json(QWEN_DIR / "review.json")
    result = load_json(HERE / "review.json")
    input_path = HERE / "input.jsonl"
    input_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    qwen_flagged = [
        {key: row[key] for key in ("prompt_id", "language", "kind", "source", "translation")}
        for row in qwen["records"]
        if not (row["decision"]["semantic_pass"] and row["decision"]["fluency_pass"])
    ]
    records = result.get("records", [])
    input_sha = sha(input_path)
    local_script_sha = sha(HERE / "review.py")
    remote_script_sha = sha(HERE / "remote_kernel_source.py")

    assert manifest["status"] == "DIAGNOSTIC_ONLY_NOT_RELEASE"
    assert input_sha == manifest["input_sha256"] == result["input_sha256"]
    assert sha(QWEN_DIR / "review.json") == manifest["source_review_sha256"]
    assert local_script_sha == remote_script_sha == manifest["review_script_sha256"]
    assert result["status"] == "DIAGNOSTIC_CROSS_REVIEW_COMPLETE"
    assert result["model_id"] == EXPECTED_MODEL == manifest["model_id"]
    assert result["model_revision"] == EXPECTED_REVISION == manifest["model_revision"]
    assert result["schema_errors"] == 0
    assert len(input_rows) == manifest["rows"] == len(records) == 9
    assert len({row["prompt_id"] for row in input_rows}) == 9
    assert input_rows == qwen_flagged
    assert not any("answer" in key.lower() or "label" in key.lower() for row in input_rows for key in row)

    for expected, observed in zip(input_rows, records, strict=True):
        assert {key: observed[key] for key in expected} == expected
        decision = observed.get("decision", {})
        assert type(decision.get("semantic_pass")) is bool
        assert type(decision.get("fluency_pass")) is bool
        assert isinstance(decision.get("reason"), str)
        assert not ("answer" in decision or "label" in decision)

    all_pass = sum(r["decision"]["semantic_pass"] and r["decision"]["fluency_pass"] for r in records)
    proof = {
        "status": "VERIFIED_DIAGNOSTIC_ALIGNMENT_ONLY_NOT_TRANSLATION_APPROVAL",
        "rows": len(records),
        "unique_ids": len({r["prompt_id"] for r in records}),
        "input_sha256": input_sha,
        "source_qwen_review_sha256": sha(QWEN_DIR / "review.json"),
        "mistral_output_sha256": sha(HERE / "review.json"),
        "mistral_log_sha256": sha(HERE / "synthworlds-chatgpt-cal32-mistral-cross-review-9.log"),
        "review_script_sha256": local_script_sha,
        "pulled_remote_script_sha256": remote_script_sha,
        "model_id": result["model_id"],
        "model_revision": result["model_revision"],
        "schema_errors": result["schema_errors"],
        "answer_or_label_fields_included": False,
        "qwen_flagged_rows_passed_by_mistral": all_pass,
        "qwen_flagged_rows_total": len(records),
        "interpretation": "Exact input/output provenance and schema verified. Mistral's judgments are diagnostic and do not establish translation correctness or authorize corpus release.",
    }
    (HERE / "cross_review_validation.json").write_text(
        json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
