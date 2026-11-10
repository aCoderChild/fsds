#!/usr/bin/env python3
"""Verify same-input Qwen negative control; not a language-quality guarantee."""
import ast,collections,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read_raw(path):
 tree=ast.parse(path.read_text(encoding='utf-8'))
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
raw=read_raw(HERE/'kaggle_qwen_negative_control/review.py')
assert raw==read_raw(HERE/'kaggle_mistral_negative_control/review.py')
rows=json.loads(raw);path=HERE/'evidence/qwen_negative_control/review.json';data=json.loads(path.read_text())
assert data['status']=='DIAGNOSTIC_AI_REVIEW_ONLY' and data['input_sha256']==hashlib.sha256(raw.encode()).hexdigest()
assert data['model_id']=='Qwen/Qwen2.5-7B-Instruct' and data['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
assert len(rows)==len(data['records'])==15
counts=collections.Counter()
for row,record in zip(rows,data['records']):
 assert row['source']=='What color is [ENTITY_A]?' and all(row[k]==record[k] for k in ('language','kind','prompt_id'))
 match=re.search(r'\{.*\}',record['raw_output'],re.S);assert match
 d=json.loads(match.group());assert d==record['decision'] and type(d['semantic_pass']) is bool and type(d['fluency_pass']) is bool
 counts['semantic_pass' if d['semantic_pass'] else 'semantic_flag']+=1
 counts['both_pass' if d['semantic_pass'] and d['fluency_pass'] else 'at_least_one_flag']+=1
print(json.dumps({'status':'CONTROLLED_MISMATCH_DIAGNOSTIC_ONLY','rows':len(rows),'counts':dict(counts),'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'limitation':'Obvious fixed-question mismatches cannot measure subtle translation errors.'},indent=2))
