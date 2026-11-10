#!/usr/bin/env python3
"""Validate source and marker integrity of the diagnostic back-translation."""
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
script = HERE / 'evidence/assistant_nllb_backtranslation/submitted_v1.py'
result = HERE / 'evidence/assistant_nllb_backtranslation/backtranslations.json'
tree = ast.parse(script.read_text(encoding='utf-8'))
raw = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'raw' for t in n.targets))
inputs = json.loads(raw)
data = json.loads(result.read_text(encoding='utf-8'))
assert data['status'] == 'DIAGNOSTIC_BACKTRANSLATION_ONLY'
assert data['model_id'] == 'facebook/nllb-200-distilled-1.3B'
assert data['model_revision'] == '7be3e24664b38ce1cac29b8aeed6911aa0cf0576'
assert data['input_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
assert len(inputs) == len(data['records']) == 15
marker = re.compile(r'\[(?:ENTITY|NUMBER)_[A-Z]+\]')
bad = []
for row, record in zip(inputs, data['records']):
    assert all(row[k] == record[k] for k in ('language', 'kind', 'prompt_id'))
    ok = collections.Counter(marker.findall(row['translation'])) == collections.Counter(marker.findall(record['backtranslation']))
    assert record['marker_integrity'] is ok
    if not ok:
        bad.append({'language': row['language'], 'prompt_id': row['prompt_id']})
print(json.dumps({'status':'STRUCTURE_AND_MARKERS_ONLY', 'rows':len(inputs), 'marker_failures':bad,
                  'output_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
                  'limitation':'Marker integrity and back-translation do not establish translation correctness.'},indent=2))
