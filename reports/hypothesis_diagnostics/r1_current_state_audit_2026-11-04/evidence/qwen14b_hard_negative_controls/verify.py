#!/usr/bin/env python3
"""Verify provenance and summarize the Qwen14B hard-negative diagnostic."""
import ast
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "control_manifest.json").read_text(encoding="utf-8"))
INPUT = [json.loads(line) for line in (HERE / "input.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
INPUT_RAW = (HERE / "input_array.json").read_text(encoding="utf-8").strip()
SCRIPT = HERE / "review.py"
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "remote_output/review.json"
REMOTE_SCRIPT = HERE / "remote_output/submitted_source.py"
LOG = HERE / "remote_output/synthworlds-qwen14b-hard-negative-2026-11-04.log"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

assert MANIFEST["status"] == "BLINDED_HARD_NEGATIVE_CONTROLS_DIAGNOSTIC_ONLY"
assert len(INPUT) == 16 and len({r["control_id"] for r in INPUT}) == 16
assert hashlib.sha256(INPUT_RAW.encode()).hexdigest() == MANIFEST["input_sha256_compact_array"]
assert json.loads(INPUT_RAW) == INPUT
assert hashlib.sha256((HERE / "input.jsonl").read_bytes()).hexdigest() == MANIFEST["input_jsonl_sha256"]
relations = {
    "kinship": ("father", "mother"),
    "event": ("officially open", "officially close"),
    "family_relation": ("spouse", "sibling"),
    "profession": ("director", "editor"),
}
controls = {r["control_id"]: r for r in MANIFEST["controls"]}
for row in INPUT:
    control = controls[row["control_id"]]
    left, right = relations[control["family"]]
    pattern = re.compile(rf"\b(?:{re.escape(left)}|{re.escape(right)})\b", re.I)
    original_matches = pattern.findall(control["source_original"])
    counterfactual_matches = pattern.findall(control["source_counterfactual"])
    assert len(original_matches) == len(counterfactual_matches) == 1
    assert original_matches[0].casefold() != counterfactual_matches[0].casefold()
    assert pattern.sub("<REL>", control["source_original"]) == pattern.sub("<REL>", control["source_counterfactual"])
    assert row["source"] == control["source_counterfactual"]
    assert row["translation"] == control["translation"]
tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
raw = next(ast.literal_eval(n.value) for n in tree.body
           if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RAW_INPUT" for t in n.targets))
assert hashlib.sha256(raw.encode()).hexdigest() == MANIFEST["input_sha256_compact_array"]
assert json.loads(raw) == INPUT

result = json.loads(OUTPUT.read_text(encoding="utf-8"))
assert result["status"] == "DIAGNOSTIC_COMPLETE"
assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
assert result["input_sha256"] == MANIFEST["input_sha256_compact_array"]
records = result["records"]
assert len(records) == 16 and len({r["control_id"] for r in records}) == 16
expected = {r["control_id"]: r for r in INPUT}
assert set(expected) == {r["control_id"] for r in records}
errors = []
accepted_mismatches = []
counts = collections.Counter()
by_cell = collections.defaultdict(collections.Counter)
for record in records:
    base = expected[record["control_id"]]
    assert record["language"] == base["language"] and record["kind"] == base["kind"]
    assert record["source"] == base["source"]
    assert record["translation_sha256"] == hashlib.sha256(base["translation"].encode()).hexdigest()
    decision = record.get("decision", {})
    if (type(decision.get("semantic_pass")) is not bool or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str)):
        errors.append(record["control_id"])
        continue
    cell = f"{record['language']}/{record['kind']}"
    counts["semantic_reject" if not decision["semantic_pass"] else "semantic_pass"] += 1
    counts["fluency_pass" if decision["fluency_pass"] else "fluency_reject"] += 1
    counts["both_pass" if decision["semantic_pass"] and decision["fluency_pass"] else "not_both_pass"] += 1
    if decision["semantic_pass"]:
        accepted_mismatches.append({"control_id":record["control_id"], "language":record["language"],
            "kind":record["kind"], "source":record["source"], "reason":decision["reason"]})
    by_cell[cell]["semantic_reject" if not decision["semantic_pass"] else "semantic_pass"] += 1
assert not errors, f"Malformed decisions: {errors}"
screen_pass = counts["semantic_reject"] == 16 and not errors
summary = {
    "status": "VERIFIED_DIAGNOSTIC_SCREEN_PASSED" if screen_pass else "VERIFIED_DIAGNOSTIC_SCREEN_FAILED",
    "release_status": "NOT_A_RELEASE_GATE_OR_TRANSLATION_APPROVAL",
    "predeclared_rule_passed": screen_pass,
    "next_step_allowed": "larger_reviewer_calibration_only" if screen_pass else "reviewer_requires_revision_and_fresh_controls",
    "rows": len(records),
    "model_id": result["model_id"],
    "model_revision": result["model_revision"],
    "input_sha256": result["input_sha256"],
    "review_script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    "submitted_script_sha256": hashlib.sha256(REMOTE_SCRIPT.read_bytes()).hexdigest() if REMOTE_SCRIPT.exists() else None,
    "submitted_script_matches_local": REMOTE_SCRIPT.read_bytes() == SCRIPT.read_bytes() if REMOTE_SCRIPT.exists() else None,
    "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "run_log_sha256": hashlib.sha256(LOG.read_bytes()).hexdigest() if LOG.exists() else None,
    "gpu": result.get("gpu"),
    "torch": result.get("torch"),
    "transformers": result.get("transformers"),
    "accelerate": result.get("accelerate"),
    "counts": dict(counts),
    "accepted_known_mismatches": accepted_mismatches,
    "semantic_rejections_by_language_and_kind": {k: dict(v) for k, v in sorted(by_cell.items())},
    "limitations": MANIFEST["limitations"],
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
