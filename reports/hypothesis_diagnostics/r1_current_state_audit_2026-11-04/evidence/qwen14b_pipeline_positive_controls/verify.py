#!/usr/bin/env python3
"""Verify schema-constrained positive controls against source rows and artifacts."""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TRAINING = REPO / "experiments/synthworlds_r1/training_release_draft"
sys.path.insert(0, str(TRAINING / "scripts"))
from review_training_translations_qwen import NUMBER, alpha_label, collect_rows, masked_pair  # noqa: E402

RUN = HERE / "remote_output/version_2"
INPUT_RAW = (HERE / "input_array.json").read_text(encoding="utf-8").strip()
INPUT = json.loads(INPUT_RAW)
MANIFEST = json.loads((HERE / "control_manifest.json").read_text(encoding="utf-8"))
RESULT_PATH = RUN / "review.json"
SOURCE_PATH = RUN / "source/synthworlds-qwen14b-positive-controls-2026-11-04.py"
LOG_PATH = RUN / "synthworlds-qwen14b-positive-controls-2026-11-04.log"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def masked_backtranslation(source: dict, text: str) -> str:
    original = source["text_en"]
    names = [name for name in source.get("protected_entity_names", []) if name and name in original]
    names = sorted(set(names), key=lambda name: original.index(name))
    replacements = {name: f"[ENTITY_{alpha_label(i)}]" for i, name in enumerate(names)}
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(name, replacements[name])
    for i, number in enumerate(NUMBER.findall(original)):
        text = text.replace(number, f"[NUMBER_{alpha_label(i)}]", 1)
    return text


assert MANIFEST["status"] == "AI_ONLY_POSITIVE_REVIEW_CONSISTENCY_DIAGNOSTIC_ONLY"
assert len(INPUT) == 16 and len({r["control_id"] for r in INPUT}) == 16
assert sha(INPUT_RAW.encode()) == MANIFEST["input_sha256_compact_array"]
assert len(MANIFEST["controls"]) == 16
assert sha((HERE.parent / "qwen14b_pipeline_hard_controls/control_manifest.json").read_bytes()) == MANIFEST["source_negative_control_manifest_sha256"]
assert json.loads(INPUT_RAW) == [json.loads(line) for line in (HERE / "input.jsonl").read_text().splitlines() if line]
assert sha((HERE / "input.jsonl").read_bytes()) == MANIFEST["input_jsonl_sha256"]

control_by_id = {r["control_id"]: r for r in MANIFEST["controls"]}
source_rows = {lang: {(r["prompt_id"], r["kind"]): r for r in collect_rows(lang)[0]}
               for lang in ("fr", "de", "hi", "th")}
for row in INPUT:
    assert set(row) == {"control_id", "language", "kind", "source", "translation", "back_translation_en"}
    control = control_by_id[row["control_id"]]
    assert control["language"] == row["language"] and control["kind"] == row["kind"]
    review_path = TRAINING / "translation_reviews_roundtrip_qwen14b" / f"{row['language']}.jsonl"
    assert sha(review_path.read_bytes()) == control["base_review_file_sha256"]
    review_rows = [json.loads(line) for line in review_path.read_text().splitlines() if line.strip()]
    review = next(r for r in review_rows if (r["prompt_id"], r["kind"]) == (control["prompt_id"], control["kind"]))
    assert review["final_review_status"] == control["base_review_status"]
    source = source_rows[row["language"]][(control["prompt_id"], control["kind"])]
    pair = masked_pair(source)[1]
    assert row["source"] == pair["source_english"]
    assert row["translation"] == pair["target_translation"]
    assert row["back_translation_en"] == masked_backtranslation(source, review["backtranslation_en"])
    assert sha(row["translation"].encode()) == control["translation_sha256"]
    assert sha(row["back_translation_en"].encode()) == control["backtranslation_sha256"]

assert SOURCE_PATH.exists() and SOURCE_PATH.read_bytes() == (HERE / "review.py").read_bytes()
result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
assert result["status"] == "DIAGNOSTIC_COMPLETE" and result["schema_errors"] == 0
assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
assert result.get("quantization") == "4-bit NF4" and result.get("bitsandbytes") == "0.50.2"
assert result["input_sha256"] == MANIFEST["input_sha256_compact_array"]
records = result["records"]
assert len(records) == 16 and len({r["control_id"] for r in records}) == 16
expected = {r["control_id"]: r for r in INPUT}
assert set(expected) == {r["control_id"] for r in records}
counts = collections.Counter()
cells = collections.defaultdict(collections.Counter)
for record in records:
    row = expected[record["control_id"]]
    assert record["language"] == row["language"] and record["kind"] == row["kind"]
    assert record["source"] == row["source"]
    assert record["translation_sha256"] == sha(row["translation"].encode())
    assert record["back_translation_sha256"] == sha(row["back_translation_en"].encode())
    parsed = json.loads(record["raw_output"], object_pairs_hook=unique_object)
    assert set(parsed) == {"decisions"} and len(parsed["decisions"]) == 1
    decision = record["decision"]
    assert parsed["decisions"][0] == decision
    assert set(decision) == {"key", "semantic_pass", "fluency_pass", "reason"}
    assert decision["key"] == record["control_id"]
    assert type(decision["semantic_pass"]) is bool and type(decision["fluency_pass"]) is bool
    assert isinstance(decision["reason"], str) and decision["reason"].strip()
    counts["semantic_pass" if decision["semantic_pass"] else "semantic_reject"] += 1
    counts["fluency_pass" if decision["fluency_pass"] else "fluency_reject"] += 1
    counts["strict_review_accept"] += bool(decision["semantic_pass"] and decision["fluency_pass"])
    cell = cells[f"{record['language']}/{record['kind']}"]
    cell["semantic_pass" if decision["semantic_pass"] else "semantic_reject"] += 1
    cell["strict_review_accept"] += bool(decision["semantic_pass"] and decision["fluency_pass"])

log = LOG_PATH.read_text(encoding="utf-8", errors="replace")
assert len(re.findall(r"Reviewed \d+/16 controls", log)) == 16
summary = {
    "status": "VERIFIED_AI_ONLY_POSITIVE_CONSISTENCY_DIAGNOSTIC",
    "release_status": "NOT_A_RELEASE_GATE_OR_TRANSLATION_APPROVAL",
    "rows": len(records), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "input_sha256": MANIFEST["input_sha256_compact_array"],
    "review_script_sha256": sha((HERE / "review.py").read_bytes()),
    "submitted_script_sha256": sha(SOURCE_PATH.read_bytes()),
    "submitted_script_matches_local": SOURCE_PATH.read_bytes() == (HERE / "review.py").read_bytes(),
    "output_sha256": sha(RESULT_PATH.read_bytes()), "run_log_sha256": sha(LOG_PATH.read_bytes()),
    "gpu": result.get("gpu"), "torch": result.get("torch"),
    "transformers": result.get("transformers"), "accelerate": result.get("accelerate"),
    "bitsandbytes": result.get("bitsandbytes"), "quantization": result.get("quantization"),
    "counts": dict(counts),
    "results_by_language_and_kind": {key: dict(value) for key, value in sorted(cells.items())},
    "interpretation": "Schema-constrained Transformers generation used the exact production system prompt and JSON schema. The positive cases were selected from prior AI-only passes, not human gold, so the pass rate measures consistency only.",
    "limitations": MANIFEST["limitations"],
}
(RUN / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
