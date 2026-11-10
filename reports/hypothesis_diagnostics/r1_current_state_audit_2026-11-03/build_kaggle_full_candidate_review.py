#!/usr/bin/env python3
"""Prepare answer-blind Qwen screening of all structurally usable NLLB drafts."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
TRAINING=HERE.parents[2]/'experiments/synthworlds_r1/training_release_draft/scripts'
sys.path.insert(0,str(TRAINING))
import repair_all_flagged_training_translations_v2 as repair
from assemble_training_release import checked_translation_reviews
from review_masking_v2 import mask_triplet

def sha(b):return hashlib.sha256(b).hexdigest()
for name in ('verify_full_nllb_candidates.py','diagnose_transliterated_surrogates.py'):
 subprocess.run([sys.executable,'-B',str(HERE/name)],check=True,stdout=subprocess.DEVNULL)
base=HERE/'evidence/full_nllb_candidates'
direct=json.loads((base/'restored_diagnostics.json').read_text())
heuristic=json.loads((base/'transliteration_restoration_diagnostic.json').read_text())
assert direct['status']=='UNREVIEWED_DIAGNOSTIC_ONLY' and heuristic['status']=='UNREVIEWED_PROVISIONAL_DIAGNOSTIC_ONLY'
h={(x['language'],x['kind'],x['prompt_id']):x for x in heuristic['records']}
_,by_lang=repair.source_manifest(); selected=repair.select_flagged_rows(by_lang,checked_translation_reviews())
assert len(selected)==len(direct['records'])==799
inputs=[];bound=[]
for item,r in zip(selected,direct['records']):
 row=item['row'];key=(item['language'],row['kind'],row['prompt_id'])
 assert (r['language'],r['kind'],r['prompt_id'])==key
 if r['structural_pass']:
  candidate=r['restored_translation'];method='exact_surrogate'
 else:
  v=h[key];assert v['status']=='provisional_structural_pass';candidate=v['restored_translation'];method='heuristic_transliteration'
 source,target,_=mask_triplet({**row,'text_translation':candidate,'backtranslation_en':row['text_en']})
 inputs.append({'language':key[0],'kind':key[1],'prompt_id':key[2],'source':source,'translation':target})
 bound.append({'language':key[0],'kind':key[1],'prompt_id':key[2],'restoration_method':method,'candidate_sha256':sha(candidate.encode())})
assert len(h)==427 and len(inputs)==799
payload=json.dumps(inputs,ensure_ascii=False,separators=(',',':'))
prior=HERE/'kaggle_assistant_review/review.py'
code=prior.read_text(encoding='utf-8');tree=ast.parse(code)
old=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
old_hash=sha(old.encode());new_hash=sha(payload.encode())
assert code.count('raw = '+repr(old))==1 and code.count(old_hash)>=2
code=code.replace('raw = '+repr(old),'raw = '+repr(payload)).replace(old_hash,new_hash)
code=code.replace('"generator_revision":"not recorded"','"generator_revision":"facebook/nllb-200-distilled-1.3B@7be3e24664b38ce1cac29b8aeed6911aa0cf0576"')
compile(code,'review.py','exec')
out=HERE/'kaggle_full_candidate_review';out.mkdir(exist_ok=True)
(out/'review.py').write_text(code,encoding='utf-8')
(out/'candidate_bindings.json').write_text(json.dumps({'status':'LOCAL_DIAGNOSTIC_BINDINGS_ONLY','direct_sha256':sha((base/'restored_diagnostics.json').read_bytes()),'heuristic_sha256':sha((base/'transliteration_restoration_diagnostic.json').read_bytes()),'bindings':bound},ensure_ascii=False,indent=2)+'\n')
metadata={'id':'bunnoob2005/synthworlds-full-candidate-review','title':'SynthWorlds full candidate review','code_file':'review.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'rows':len(inputs),'heuristic_restorations':len(h),'masked_input_sha256':new_hash,'script_sha256':sha(code.encode()),'status':'DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW'},indent=2))
