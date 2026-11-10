#!/usr/bin/env python3
"""Prepare a blinded screen for changed, source-checked alternative drafts."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'experiments/synthworlds_r1/training_release_draft/scripts'
sys.path.insert(0,str(ROOT))
import repair_all_flagged_training_translations_v2 as repair
from assemble_training_release import checked_translation_reviews
from review_masking_v2 import mask_triplet
def sha(b):return hashlib.sha256(b).hexdigest()
for f in ('verify_full_candidate_review.py','verify_nllb600_flagged_repairs.py'):
 subprocess.run([sys.executable,'-B',str(HERE/f)],check=True,stdout=subprocess.DEVNULL)
base=HERE/'evidence/full_nllb_candidates';alternative=HERE/'evidence/nllb600_flagged_repairs/restored_diagnostics.json'
direct=json.loads((base/'restored_diagnostics.json').read_text())['records'];heuristic=json.loads((base/'transliteration_restoration_diagnostic.json').read_text())['records']
old={(x['language'],x['kind'],x['prompt_id']):x['restored_translation'] for x in direct if x['structural_pass']}
old.update({(x['language'],x['kind'],x['prompt_id']):x['restored_translation'] for x in heuristic if x['status']=='provisional_structural_pass'})
_,by_lang=repair.source_manifest();selected=repair.select_flagged_rows(by_lang,checked_translation_reviews())
source={(x['language'],x['row']['kind'],x['row']['prompt_id']):x['row'] for x in selected}
alt=json.loads(alternative.read_text());assert alt['status']=='UNREVIEWED_DIAGNOSTIC_ONLY'
rows=[];bindings=[];same=0;unresolved=0
for x in alt['records']:
 key=(x['language'],x['kind'],x['prompt_id']);assert key in source and key in old
 if x['status']=='unresolved':unresolved+=1;continue
 candidate=x['restored_translation']
 if candidate==old[key]:same+=1;continue
 row=source[key];en,target,_=mask_triplet({**row,'text_translation':candidate,'backtranslation_en':row['text_en']})
 rows.append({'language':key[0],'kind':key[1],'prompt_id':key[2],'source':en,'translation':target})
 bindings.append({'language':key[0],'kind':key[1],'prompt_id':key[2],'restoration_method':x['status'],'candidate_sha256':sha(candidate.encode())})
assert len(rows)==100 and same==41 and unresolved==39
payload=json.dumps(rows,ensure_ascii=False,separators=(',',':'))
prior=HERE/'kaggle_full_candidate_review/review.py';code=prior.read_text();tree=ast.parse(code)
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
oldhash=sha(raw.encode());newhash=sha(payload.encode());assert code.count('raw = '+repr(raw))==1
code=code.replace('raw = '+repr(raw),'raw = '+repr(payload)).replace(oldhash,newhash)
code=code.replace('facebook/nllb-200-distilled-1.3B@7be3e24664b38ce1cac29b8aeed6911aa0cf0576','facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d')
compile(code,'review.py','exec')
out=HERE/'kaggle_nllb600_repair_review';out.mkdir(exist_ok=True)
(out/'review.py').write_text(code,encoding='utf-8')
(out/'bindings.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY','alternative_sha256':sha(alternative.read_bytes()),'input_sha256':newhash,'bindings':bindings,'unchanged':same,'unresolved':unresolved},indent=2)+'\n')
meta={'id':'bunnoob2005/synthworlds-nllb600-repair-review','title':'SynthWorlds NLLB600 repair review','code_file':'review.py','language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'screen_rows':len(rows),'unchanged':same,'unresolved':unresolved,'input_sha256':newhash,'script_sha256':sha(code.encode()),'status':'NOT_RELEASE_REVIEW'},indent=2))
