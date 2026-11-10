#!/usr/bin/env python3
"""Make a controlled mismatch probe for the Mistral reviewer; no gold answers."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
prior=HERE/'kaggle_mistral_review_probe/review.py'
code=prior.read_text(encoding='utf-8');tree=ast.parse(code)
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
rows=json.loads(raw);assert len(rows)==15
for row in rows:
 assert '[ENTITY_A]' in row['source'] and '[ENTITY_B]' not in row['source']
 row['source']='What color is [ENTITY_A]?'
new=json.dumps(rows,ensure_ascii=False,separators=(',',':'))
old_hash=hashlib.sha256(raw.encode()).hexdigest();new_hash=hashlib.sha256(new.encode()).hexdigest()
assert code.count('raw = '+repr(raw))==1 and code.count(old_hash)>=2
code=code.replace('raw = '+repr(raw),'raw = '+repr(new)).replace(old_hash,new_hash)
compile(code,'review.py','exec')
out=HERE/'kaggle_mistral_negative_control';out.mkdir(exist_ok=True)
(out/'review.py').write_text(code,encoding='utf-8')
meta={'id':'bunnoob2005/synthworlds-mistral-negative-control','title':'SynthWorlds Mistral negative control','code_file':'review.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'input_sha256':new_hash,'script_sha256':hashlib.sha256(code.encode()).hexdigest(),'status':'CONTROLLED_MISMATCH_DIAGNOSTIC_ONLY'},indent=2))
