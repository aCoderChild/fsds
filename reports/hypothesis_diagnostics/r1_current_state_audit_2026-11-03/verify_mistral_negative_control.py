#!/usr/bin/env python3
"""Verify fixed mismatches and model records; not an estimate of review accuracy."""
import ast,collections,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
def inputs(path):
 tree=ast.parse(path.read_text(encoding='utf-8'))
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
raw=inputs(HERE/'kaggle_mistral_negative_control/review.py')
original=inputs(HERE/'kaggle_mistral_review_probe/review.py')
rows=json.loads(raw);prev=json.loads(original)
path=HERE/'evidence/mistral_negative_control/review.json';data=json.loads(path.read_text())
assert data['status']=='DIAGNOSTIC_AI_REVIEW_ONLY'
assert data['model_id']=='mistralai/Mistral-7B-Instruct-v0.3' and data['model_revision']=='c170c708c41dac9275d15a8fff4eca08d52bab71'
assert data['input_sha256']==hashlib.sha256(raw.encode()).hexdigest()
assert len(rows)==len(prev)==len(data['records'])==15
counts=collections.Counter()
for row,old,result in zip(rows,prev,data['records']):
 assert row['source']=='What color is [ENTITY_A]?' and row['source']!=old['source']
 assert all(row[k]==old[k] for k in ('language','kind','prompt_id','translation'))
 assert all(row[k]==result[k] for k in ('language','kind','prompt_id'))
 match=re.search(r'\{.*\}',result['raw_output'],re.S);assert match
 d=json.loads(match.group());assert d==result['decision'] and type(d['semantic_pass']) is bool and type(d['fluency_pass']) is bool
 counts['semantic_pass' if d['semantic_pass'] else 'semantic_flag']+=1
 counts['both_pass' if d['semantic_pass'] and d['fluency_pass'] else 'at_least_one_flag']+=1
print(json.dumps({'status':'CONTROLLED_MISMATCH_DIAGNOSTIC_ONLY','rows':len(rows),'counts':dict(counts),'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'limitation':'Obvious synthetic mismatches are not representative of real translation errors.'},indent=2))
