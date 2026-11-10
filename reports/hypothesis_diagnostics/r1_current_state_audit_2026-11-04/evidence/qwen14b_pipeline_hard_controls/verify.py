#!/usr/bin/env python3
"""Verify Qwen14B decisions on hard controls with production back-translation inputs."""
import ast
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
MANIFEST = json.loads((HERE / "control_manifest.json").read_text(encoding="utf-8"))
INPUT_RAW = (HERE / "input_array.json").read_text(encoding="utf-8").strip()
INPUT = json.loads(INPUT_RAW)
SCRIPT = HERE / "review.py"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "remote_output/version_2/review_normalized.json"
RUN_DIR = OUTPUT.parent
RAW_OUTPUT = RUN_DIR / "review.json"
REMOTE_SCRIPT = RUN_DIR / "source/synthworlds-qwen14b-pipeline-controls-2026-11-04.py"
LOG = RUN_DIR / "synthworlds-qwen14b-pipeline-controls-2026-11-04.log"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

assert MANIFEST["status"] == "BLINDED_FULL_REVIEW_PIPELINE_CONTROLS_DIAGNOSTIC_ONLY"
assert len(INPUT) == 16 and len({r["control_id"] for r in INPUT}) == 16
assert hashlib.sha256(INPUT_RAW.encode()).hexdigest() == MANIFEST["input_sha256_compact_array"]
assert json.loads(INPUT_RAW) == [json.loads(line) for line in (HERE / "input.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert hashlib.sha256((HERE / "input.jsonl").read_bytes()).hexdigest() == MANIFEST["input_jsonl_sha256"]
tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
embedded = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RAW_INPUT" for t in node.targets))
assert hashlib.sha256(embedded.encode()).hexdigest() == MANIFEST["input_sha256_compact_array"]
assert json.loads(embedded) == INPUT

relations = {"kinship": ("father", "mother"), "event": ("officially open", "officially close"),
             "family_relation": ("spouse", "sibling"), "profession": ("director", "editor")}
controls = {r["control_id"]: r for r in MANIFEST["controls"]}
source_rows_by_language = {lang: {(r["prompt_id"], r["kind"]): r for r in collect_rows(lang)[0]}
                           for lang in ("fr", "de", "hi", "th")}
for row in INPUT:
    control = controls[row["control_id"]]
    review_path = TRAINING / "translation_reviews_roundtrip_qwen14b" / f"{control['language']}.jsonl"
    assert hashlib.sha256(review_path.read_bytes()).hexdigest() == control["source_review_file_sha256"]
    assert hashlib.sha256(review_path.read_bytes()).hexdigest() == control["base_review_file_sha256"]
    review_rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    base_review = next(r for r in review_rows if r.get("prompt_id") == control["prompt_id"] and r.get("kind") == control["kind"])
    assert base_review["final_review_status"] == control["base_review_status"]
    assert base_review["backtranslation_model"] == control["backtranslation_model"]
    assert hashlib.sha256(base_review["backtranslation_en"].encode()).hexdigest() == control["backtranslation_sha256"]
    source = source_rows_by_language[control["language"]][(control["prompt_id"], control["kind"])]
    _, base_pair = masked_pair(source)
    assert base_pair["source_english"] == control["source_original"]
    assert base_pair["target_translation"] == control["translation"]
    original = source["text_en"]
    names = [name for name in source.get("protected_entity_names", []) if name and name in original]
    names = sorted(set(names), key=lambda name: original.index(name))
    replacements = {name: f"[ENTITY_{alpha_label(i)}]" for i, name in enumerate(names)}
    back = base_review["backtranslation_en"]
    for name in sorted(names, key=len, reverse=True):
        back = back.replace(name, replacements[name])
    for i, number in enumerate(NUMBER.findall(original)):
        back = back.replace(number, f"[NUMBER_{alpha_label(i)}]", 1)
    assert back == control["back_translation_en"]
    left, right = relations[control["family"]]
    pattern = re.compile(rf"\b(?:{re.escape(left)}|{re.escape(right)})\b", re.I)
    orig = pattern.findall(control["source_original"])
    changed = pattern.findall(control["source_counterfactual"])
    assert len(orig) == len(changed) == 1 and orig[0].casefold() != changed[0].casefold()
    assert pattern.sub("<REL>", control["source_original"]) == pattern.sub("<REL>", control["source_counterfactual"])
    assert row["source"] == control["source_counterfactual"]
    assert row["translation"] == control["translation"]
    assert row["back_translation_en"] == control["back_translation_en"]

result = json.loads(OUTPUT.read_text(encoding="utf-8"))
raw_result = json.loads(RAW_OUTPUT.read_text(encoding="utf-8"))
assert result["status"] == "LOCALLY_NORMALIZED_DIAGNOSTIC_COMPLETE"
assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
assert result["input_sha256"] == MANIFEST["input_sha256_compact_array"]
assert raw_result["status"] == "DIAGNOSTIC_SCHEMA_ERRORS"
assert raw_result["schema_errors"] == 16
assert result["raw_remote_status"] == raw_result["status"]
assert result["raw_remote_schema_errors"] == raw_result["schema_errors"]
assert result["raw_remote_review_sha256"] == hashlib.sha256(RAW_OUTPUT.read_bytes()).hexdigest()
assert result["schema_errors_after_normalization"] == 0
assert REMOTE_SCRIPT.exists() and REMOTE_SCRIPT.read_bytes() == SCRIPT.read_bytes()
assert result["normalizer_sha256"] == hashlib.sha256((HERE / "normalize_review_output.py").read_bytes()).hexdigest()
records = result["records"]
assert len(records) == 16 and len({r["control_id"] for r in records}) == 16
expected = {r["control_id"]: r for r in INPUT}
assert set(expected) == {r["control_id"] for r in records}
counts = collections.Counter()
by_cell = collections.defaultdict(collections.Counter)
accepted = []
for record in records:
    row = expected[record["control_id"]]
    assert record["language"] == row["language"] and record["kind"] == row["kind"]
    assert record["source"] == row["source"]
    assert record["translation_sha256"] == hashlib.sha256(row["translation"].encode()).hexdigest()
    assert record["back_translation_sha256"] == hashlib.sha256(row["back_translation_en"].encode()).hexdigest()
    d = record.get("decision", {})
    assert (type(d.get("semantic_pass")) is bool and type(d.get("fluency_pass")) is bool
            and isinstance(d.get("reason"), str) and bool(d["reason"].strip())), record["control_id"]
    cell = f"{record['language']}/{record['kind']}"
    counts["semantic_reject" if not d["semantic_pass"] else "semantic_pass"] += 1
    counts["fluency_pass" if d["fluency_pass"] else "fluency_reject"] += 1
    by_cell[cell]["semantic_reject" if not d["semantic_pass"] else "semantic_pass"] += 1
    if d["semantic_pass"]:
        accepted.append({"control_id":record["control_id"], "language":record["language"],
                         "kind":record["kind"], "source":record["source"], "reason":d["reason"]})
screen_pass = counts["semantic_reject"] == 16
summary = {
    "status": "VERIFIED_PIPELINE_INPUT_DIAGNOSTIC_PASSED" if screen_pass else "VERIFIED_PIPELINE_INPUT_DIAGNOSTIC_FAILED",
    "release_status": "NOT_A_RELEASE_GATE_OR_TRANSLATION_APPROVAL",
    "interpretation": "The remote run used the production system prompt and input fields, but unconstrained generation returned a keyed JSON object rather than the production schema's decisions array. The local normalizer accepted only an exact one-key object with the expected ID and decision fields; raw output is preserved and hash-bound.",
    "all_known_mismatches_rejected": screen_pass,
    "rows": len(records), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "input_sha256": result["input_sha256"],
    "review_script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    "submitted_script_sha256": hashlib.sha256(REMOTE_SCRIPT.read_bytes()).hexdigest(),
    "submitted_script_matches_local": REMOTE_SCRIPT.read_bytes() == SCRIPT.read_bytes(),
    "normalizer_sha256": result["normalizer_sha256"],
    "raw_remote_review_sha256": result["raw_remote_review_sha256"],
    "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "run_log_sha256": hashlib.sha256(LOG.read_bytes()).hexdigest() if LOG.exists() else None,
    "gpu": result.get("gpu"), "torch": result.get("torch"),
    "transformers": result.get("transformers"), "accelerate": result.get("accelerate"),
    "counts": dict(counts),
    "semantic_results_by_language_and_kind": {k:dict(v) for k,v in sorted(by_cell.items())},
    "accepted_known_mismatches": accepted,
    "limitations": MANIFEST["limitations"],
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
