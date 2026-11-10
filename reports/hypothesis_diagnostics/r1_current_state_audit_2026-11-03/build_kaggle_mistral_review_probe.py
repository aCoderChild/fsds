#!/usr/bin/env python3
"""Build independent-family direct review of the fixed 15 diagnostic pairs."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
prior=HERE/'kaggle_assistant_review/review.py'
code=prior.read_text(encoding='utf-8')
tree=ast.parse(code)
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
assert len(json.loads(raw))==15
old_id='Qwen/Qwen2.5-7B-Instruct';old_revision='a09a35458c702b33eeacc393d103063234e8bc28'
new_id='mistralai/Mistral-7B-Instruct-v0.3';new_revision='c170c708c41dac9275d15a8fff4eca08d52bab71'
assert code.count(old_id)==1 and code.count(old_revision)==1
code=code.replace(old_id,new_id).replace(old_revision,new_revision)
compile(code,'review.py','exec')
out=HERE/'kaggle_mistral_review_probe';out.mkdir(exist_ok=True)
(out/'review.py').write_text(code,encoding='utf-8')
meta={'id':'bunnoob2005/synthworlds-mistral-review-probe','title':'SynthWorlds Mistral review probe','code_file':'review.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'rows':15,'masked_input_sha256':hashlib.sha256(raw.encode()).hexdigest(),'script_sha256':hashlib.sha256(code.encode()).hexdigest(),'status':'DIAGNOSTIC_NOT_RELEASE_REVIEW'},indent=2))
