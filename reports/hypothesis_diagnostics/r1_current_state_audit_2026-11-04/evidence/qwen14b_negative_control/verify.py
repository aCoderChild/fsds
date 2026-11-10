#!/usr/bin/env python3
"""Verify Qwen14B negative-control provenance and summarize diagnostic decisions."""
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "review.py"
OUTPUT = Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else HERE / "review.json"
INPUT_HASH = "8e4a157d862439d71150cf341728826e880ca66dd590237ebb0b27f2df0ba7e0"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

def embedded_rows(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    raw = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RAW_INPUT" for t in n.targets))
    return raw, json.loads(raw)

raw, rows = embedded_rows(SCRIPT)
assert hashlib.sha256(raw.encode("utf-8")).hexdigest() == INPUT_HASH
assert len(rows) == 15 and len({(r["language"], r["prompt_id"]) for r in rows}) == 15
assert all(r["source"] == "What color is [ENTITY_A]?" for r in rows)
local_rows = [json.loads(line) for line in (HERE / "input.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert local_rows == rows
result = json.loads(OUTPUT.read_text(encoding="utf-8"))
assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
assert result["input_sha256"] == INPUT_HASH
records = result["records"]
assert len(records) == 15 and len({(r["language"], r["prompt_id"]) for r in records}) == 15
by_key = {(r["language"], r["prompt_id"]): r for r in rows}
assert set(by_key) == {(r["language"], r["prompt_id"]) for r in records}
errors = []
counts = collections.Counter()
for record in records:
    decision = record.get("decision", {})
    if (type(decision.get("semantic_pass")) is not bool or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str)):
        errors.append(record["prompt_id"])
    else:
        counts["semantic_pass" if decision["semantic_pass"] else "semantic_reject"] += 1
        counts["fluency_pass" if decision["fluency_pass"] else "fluency_reject"] += 1
        counts["both_pass" if decision["semantic_pass"] and decision["fluency_pass"] else "not_both_pass"] += 1
assert not errors, f"Malformed decisions: {errors}"
summary = {
    "status": "VERIFIED_DIAGNOSTIC_ONLY_NOT_RELEASE",
    "rows": len(rows),
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "input_sha256": INPUT_HASH,
    "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "counts": dict(counts),
    "limitation": "All 15 rows intentionally pair an unrelated fixed English color question with a target-language question from another source. This measures rejection of obvious mismatches only; it does not validate subtle errors or fluency.",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
