#!/usr/bin/env python3
"""Check blinded probe provenance and paired model decisions, not linguistic truth."""
import ast,collections,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
script=HERE/'kaggle_mistral_review_probe/review.py'
tree=ast.parse(script.read_text(encoding='utf-8'))
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
inputs=json.loads(raw)
pa=HERE/'evidence/mistral_review_probe/review.json';pb=HERE/'evidence/assistant_kaggle_review_v2/review.json'
a=json.loads(pa.read_text());b=json.loads(pb.read_text())
assert a['status']==b['status']=='DIAGNOSTIC_AI_REVIEW_ONLY'
assert a['model_id']=='mistralai/Mistral-7B-Instruct-v0.3' and a['model_revision']=='c170c708c41dac9275d15a8fff4eca08d52bab71'
assert b['model_id']=='Qwen/Qwen2.5-7B-Instruct' and b['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
digest=hashlib.sha256(raw.encode()).hexdigest();assert a['input_sha256']==b['input_sha256']==digest
assert len(inputs)==len(a['records'])==len(b['records'])==15
counts=collections.Counter()
for row,x,y in zip(inputs,a['records'],b['records']):
 for r in (x,y):
  assert all(r[k]==row[k] for k in ('language','kind','prompt_id'))
  match=re.search(r'\{.*\}',r['raw_output'],re.S);assert match
  decision=json.loads(match.group());assert decision==r['decision']
  assert type(decision.get('semantic_pass')) is bool and type(decision.get('fluency_pass')) is bool and isinstance(decision.get('reason'),str)
 def label(r):return 'pass' if r['decision']['semantic_pass'] and r['decision']['fluency_pass'] else 'flag'
 counts[f"mistral_{label(x)}_qwen_{label(y)}"]+=1
print(json.dumps({'status':'PAIRED_DIAGNOSTIC_ONLY','rows':len(inputs),'counts':dict(counts),'input_sha256':digest,'mistral_sha256':hashlib.sha256(pa.read_bytes()).hexdigest(),'qwen_sha256':hashlib.sha256(pb.read_bytes()).hexdigest(),'limitation':'No linguistic gold labels; agreement is not evidence of correctness.'},indent=2))
