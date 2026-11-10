#!/usr/bin/env python3
"""Verify saved diagnostic review structure and binding; not translation correctness."""
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
script = HERE / "kaggle_assistant_review/review.py"
result = HERE / "evidence/assistant_kaggle_review_v2/review.json"
tree = ast.parse(script.read_text(encoding="utf-8"))
raw = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "raw" for t in n.targets))
inputs = json.loads(raw)
data = json.loads(result.read_text(encoding="utf-8"))
assert data["status"] == "DIAGNOSTIC_AI_REVIEW_ONLY"
assert data["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
assert data["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
assert data["input_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
assert len(inputs) == len(data["records"]) == 15
counts = collections.Counter()
languages = collections.defaultdict(collections.Counter)
for inp, record in zip(inputs, data["records"]):
    assert all(inp[k] == record[k] for k in ("language", "kind", "prompt_id"))
    match = re.search(r"\{.*\}", record["raw_output"], re.S)
    assert match and json.loads(match.group()) == record["decision"]
    decision = record["decision"]
    assert type(decision.get("semantic_pass")) is bool and type(decision.get("fluency_pass")) is bool
    assert isinstance(decision.get("reason"), str)
    label = "pass" if decision["semantic_pass"] and decision["fluency_pass"] else "flag"
    counts[label] += 1
    languages[record["language"]][label] += 1
print(json.dumps({"status": "STRUCTURE_AND_INPUT_BINDING_ONLY", "review_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                  "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "counts": dict(counts),
                  "languages": {k: dict(v) for k,v in languages.items()},
                  "limitations": ["Model judgments and explanations are not independently validated.", "This is not release review."]}, indent=2))
