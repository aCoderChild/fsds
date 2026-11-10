#!/usr/bin/env python3
"""Normalize valid one-key Qwen decisions while preserving the raw Kaggle result."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "remote_output/version_2"
RAW_PATH = RUN_DIR / "review.json"
INPUT = json.loads((HERE / "input_array.json").read_text(encoding="utf-8"))
raw_bytes = RAW_PATH.read_bytes()
raw = json.loads(raw_bytes)
assert raw["status"] == "DIAGNOSTIC_SCHEMA_ERRORS"
assert raw["schema_errors"] == len(INPUT) == 16
assert len(raw["records"]) == len(INPUT)
assert raw["input_sha256"] == json.loads((HERE / "control_manifest.json").read_text())["input_sha256_compact_array"]

by_id = {row["control_id"]: row for row in INPUT}
records = []
counts = collections.Counter()
for remote_record in raw["records"]:
    control_id = remote_record["control_id"]
    source = by_id[control_id]
    assert remote_record["language"] == source["language"]
    assert remote_record["kind"] == source["kind"]
    assert remote_record["source"] == source["source"]
    assert remote_record["translation_sha256"] == hashlib.sha256(source["translation"].encode()).hexdigest()
    assert remote_record["back_translation_sha256"] == hashlib.sha256(source["back_translation_en"].encode()).hexdigest()

    parsed = json.loads(remote_record["raw_output"])
    assert isinstance(parsed, dict) and set(parsed) == {control_id}
    decision = parsed[control_id]
    assert isinstance(decision, dict)
    assert set(decision) == {"semantic_pass", "fluency_pass", "reason"}
    assert type(decision["semantic_pass"]) is bool
    assert type(decision["fluency_pass"]) is bool
    assert isinstance(decision["reason"], str) and decision["reason"].strip()
    counts["semantic_reject" if decision["semantic_pass"] is False else "semantic_pass"] += 1
    counts["fluency_pass" if decision["fluency_pass"] else "fluency_reject"] += 1
    records.append({**remote_record, "decision": decision})

assert len({r["control_id"] for r in records}) == len(INPUT)
normalized = {
    "status": "LOCALLY_NORMALIZED_DIAGNOSTIC_COMPLETE",
    "release_status": "NOT_A_RELEASE_GATE_OR_TRANSLATION_APPROVAL",
    "normalization": "Parsed each raw JSON object keyed by its sole expected control_id; no decision values were changed.",
    "raw_remote_status": raw["status"],
    "raw_remote_schema_errors": raw["schema_errors"],
    "raw_remote_review_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    "normalizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "schema_errors_after_normalization": 0,
    "model_id": raw["model_id"],
    "model_revision": raw["model_revision"],
    "input_sha256": raw["input_sha256"],
    "torch": raw["torch"],
    "transformers": raw["transformers"],
    "accelerate": raw["accelerate"],
    "gpu": raw["gpu"],
    "records": records,
    "semantic_rejects_on_known_mismatches": counts["semantic_reject"],
    "semantic_passes_on_known_mismatches": counts["semantic_pass"],
    "fluency_passes": counts["fluency_pass"],
}
out = RUN_DIR / "review_normalized.json"
out.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: normalized[k] for k in (
    "status", "raw_remote_schema_errors", "schema_errors_after_normalization",
    "semantic_rejects_on_known_mismatches", "semantic_passes_on_known_mismatches", "fluency_passes",
    "raw_remote_review_sha256", "normalizer_sha256")}, indent=2))
