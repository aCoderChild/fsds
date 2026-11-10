#!/usr/bin/env python3
"""Build independent-size diagnostic review for the 147 unresolved pairs."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-B',str(HERE/'build_candidate_review_ledger.py')],check=True,stdout=subprocess.DEVNULL)
ledger=HERE/'evidence/candidate_review_ledger.json';data=json.loads(ledger.read_text())
keys={(x['language'],x['kind'],x['prompt_id']) for x in data['records'] if x['diagnostic_selection']=='unresolved'}
assert len(keys)==147
source=HERE/'kaggle_full_candidate_review/review.py';code=source.read_text(encoding='utf-8');tree=ast.parse(code)
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
rows=[x for x in json.loads(raw) if (x['language'],x['kind'],x['prompt_id']) in keys]
assert len(rows)==len(keys)
new=json.dumps(rows,ensure_ascii=False,separators=(',',':'))
oldhash=hashlib.sha256(raw.encode()).hexdigest();newhash=hashlib.sha256(new.encode()).hexdigest()
old='Qwen/Qwen2.5-7B-Instruct';new_model='Qwen/Qwen2.5-14B-Instruct'
old_rev='a09a35458c702b33eeacc393d103063234e8bc28';new_rev='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
assert code.count('raw = '+repr(raw))==1 and code.count(old)==code.count(old_rev)==1
code=code.replace('raw = '+repr(raw),'raw = '+repr(new)).replace(oldhash,newhash).replace(old,new_model).replace(old_rev,new_rev)
compile(code,'review.py','exec')
out=HERE/'kaggle_qwen14_unresolved_review';out.mkdir(exist_ok=True)
(out/'review.py').write_text(code,encoding='utf-8')
(out/'selection_manifest.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY_NOT_RELEASE','ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),'input_sha256':newhash,'script_sha256':hashlib.sha256(code.encode()).hexdigest(),'rows':len(rows),'limitation':'Qwen reviewers share a model family; adjudication is adaptive and not release review.'},indent=2)+'\n')
meta={'id':'bunnoob2005/synthworlds-qwen14-unresolved-review','title':'SynthWorlds Qwen14 unresolved review','code_file':'review.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print((out/'selection_manifest.json').read_text())
