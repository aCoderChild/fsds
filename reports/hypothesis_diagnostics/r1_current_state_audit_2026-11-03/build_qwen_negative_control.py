#!/usr/bin/env python3
"""Use the same controlled mismatches to assess Qwen screening behavior."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
code=(HERE/'kaggle_mistral_negative_control/review.py').read_text(encoding='utf-8')
old='mistralai/Mistral-7B-Instruct-v0.3';old_rev='c170c708c41dac9275d15a8fff4eca08d52bab71'
assert code.count(old)==code.count(old_rev)==1
code=code.replace(old,'Qwen/Qwen2.5-7B-Instruct').replace(old_rev,'a09a35458c702b33eeacc393d103063234e8bc28')
tree=ast.parse(code)
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
assert len(json.loads(raw))==15
out=HERE/'kaggle_qwen_negative_control';out.mkdir(exist_ok=True)
(out/'review.py').write_text(code,encoding='utf-8')
meta={'id':'bunnoob2005/synthworlds-qwen-negative-control','title':'SynthWorlds Qwen negative control','code_file':'review.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'rows':15,'input_sha256':hashlib.sha256(raw.encode()).hexdigest(),'script_sha256':hashlib.sha256(code.encode()).hexdigest()},indent=2))
