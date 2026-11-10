#!/usr/bin/env python3
"""Recompute source binding, restoration, and marker counts for surrogate run."""
import ast
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
script = HERE / 'kaggle_assistant_backtranslation/backtranslate.py'
result = HERE / 'evidence/assistant_nllb_backtranslation_surrogates/backtranslations.json'
tree = ast.parse(script.read_text(encoding='utf-8'))
raw = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'raw' for t in n.targets))
rows = json.loads(raw)
data = json.loads(result.read_text(encoding='utf-8'))
assert data['status'] == 'DIAGNOSTIC_BACKTRANSLATION_ONLY'
assert data['input_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
assert data['model_id'] == 'facebook/nllb-200-distilled-1.3B'
assert data['model_revision'] == '7be3e24664b38ce1cac29b8aeed6911aa0cf0576'
assert len(rows) == len(data['records']) == 15
pattern = re.compile(r'\[(?:ENTITY|NUMBER)_[A-Z]+\]')
fail = []
for row, record in zip(rows, data['records']):
    assert all(row[k] == record[k] for k in ('language','kind','prompt_id'))
    subs = record['substitutions']
    assert set(subs) == set(pattern.findall(row['translation'])) and len(set(subs.values())) == len(subs)
    assert all(value not in row['translation'] for value in subs.values())
    restored = record['raw_backtranslation']
    for marker, surrogate in subs.items():
        restored = restored.replace(surrogate, marker)
    assert restored == record['backtranslation']
    ok = Counter(pattern.findall(restored)) == Counter(pattern.findall(row['translation']))
    assert record['marker_integrity'] is ok
    if not ok: fail.append(row['prompt_id'])
print(json.dumps({'status':'SOURCE_AND_MARKERS_ONLY','rows':len(rows),'marker_failures':fail,
                  'output_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
                  'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
                  'limitation':'Preserved markers do not establish semantic equivalence.'},indent=2))
