#!/usr/bin/env python3
"""Generate alternative NLLB checkpoint drafts for all full-screen flags."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-B',str(HERE/'verify_full_candidate_review.py')],check=True,stdout=subprocess.DEVNULL)
screen=HERE/'evidence/full_candidate_review/review.json'
review=json.loads(screen.read_text())['records']
flags={(r['language'],r['kind'],r['prompt_id']) for r in review if r['decision'].get('semantic_pass') is not True or r['decision'].get('fluency_pass') is not True}
assert len(flags)==180
original=HERE/'kaggle_full_nllb_candidates/translate.py';code=original.read_text(encoding='utf-8');tree=ast.parse(code)
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
rows=[r for r in json.loads(raw) if (r['language'],r['kind'],r['prompt_id']) in flags]
assert len(rows)==len(flags) and len({(r['language'],r['kind'],r['prompt_id']) for r in rows})==len(flags)
new=json.dumps(rows,ensure_ascii=False,separators=(',',':'))
old_hash=hashlib.sha256(raw.encode()).hexdigest();new_hash=hashlib.sha256(new.encode()).hexdigest()
old_model='facebook/nllb-200-distilled-1.3B';new_model='facebook/nllb-200-distilled-600M'
old_rev='7be3e24664b38ce1cac29b8aeed6911aa0cf0576';new_rev='f8d333a098d19b4fd9a8b18f94170487ad3f821d'
assert code.count('raw='+repr(raw))==1 and code.count(old_model)==code.count(old_rev)==1
code=code.replace('raw='+repr(raw),'raw='+repr(new)).replace(old_hash,new_hash).replace(old_model,new_model).replace(old_rev,new_rev)
compile(code,'translate.py','exec')
out=HERE/'kaggle_nllb600_flagged_repairs';out.mkdir(exist_ok=True)
(out/'translate.py').write_text(code,encoding='utf-8')
meta={'id':'bunnoob2005/synthworlds-nllb600-flagged-repairs','title':'SynthWorlds NLLB600 flagged repairs','code_file':'translate.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(out/'selection_manifest.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY_NOT_RELEASE','screen_sha256':hashlib.sha256(screen.read_bytes()).hexdigest(),'input_sha256':new_hash,'script_sha256':hashlib.sha256(code.encode()).hexdigest(),'rows':len(rows),'limitation':'NLLB checkpoints share a model family; outputs require independent review.'},indent=2)+'\n')
print((out/'selection_manifest.json').read_text())
