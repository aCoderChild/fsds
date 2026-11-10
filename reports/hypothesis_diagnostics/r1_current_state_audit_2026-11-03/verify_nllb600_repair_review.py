#!/usr/bin/env python3
"""Validate source binding and parse decisions for the 100 draft screen."""
import ast,collections,hashlib,json,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'experiments/synthworlds_r1/training_release_draft/scripts';sys.path.insert(0,str(ROOT))
import repair_all_flagged_training_translations_v2 as repair
from assemble_training_release import checked_translation_reviews
from review_masking_v2 import mask_triplet
def sha(b):return hashlib.sha256(b).hexdigest()
job=HERE/'kaggle_nllb600_repair_review';result=HERE/'evidence/nllb600_repair_review/review.json'
tree=ast.parse((job/'review.py').read_text());raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
inputs=json.loads(raw);data=json.loads(result.read_text());bindings=json.loads((job/'bindings.json').read_text())
altpath=HERE/'evidence/nllb600_flagged_repairs/restored_diagnostics.json';assert bindings['alternative_sha256']==sha(altpath.read_bytes()) and bindings['input_sha256']==sha(raw.encode())
alt={(r['language'],r['kind'],r['prompt_id']):r for r in json.loads(altpath.read_text())['records']}
_,by=repair.source_manifest();selected=repair.select_flagged_rows(by,checked_translation_reviews());source={(x['language'],x['row']['kind'],x['row']['prompt_id']):x['row'] for x in selected}
assert data['status']=='DIAGNOSTIC_AI_REVIEW_ONLY' and data['model_id']=='Qwen/Qwen2.5-7B-Instruct' and data['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28' and data['input_sha256']==sha(raw.encode())
assert len(inputs)==len(bindings['bindings'])==len(data['records'])==100
counts=collections.Counter();langs=collections.defaultdict(collections.Counter);seen=set()
for inp,b,record in zip(inputs,bindings['bindings'],data['records']):
 key=(inp['language'],inp['kind'],inp['prompt_id']);assert key not in seen;seen.add(key)
 assert all((x['language'],x['kind'],x['prompt_id'])==key for x in (b,record))
 candidate=alt[key]['restored_translation'];assert candidate and b['candidate_sha256']==sha(candidate.encode()) and b['restoration_method']==alt[key]['status']
 row=source[key];en,target,_=mask_triplet({**row,'text_translation':candidate,'backtranslation_en':row['text_en']});assert (inp['source'],inp['translation'])==(en,target)
 match=re.search(r'\{.*\}',record['raw_output'],re.S)
 try:
  d=json.loads(match.group()) if match else None
  assert isinstance(d,dict) and type(d.get('semantic_pass')) is bool and type(d.get('fluency_pass')) is bool and isinstance(d.get('reason'),str) and d==record['decision']
  label='pass' if d['semantic_pass'] and d['fluency_pass'] else 'flag'
 except (ValueError,AssertionError):
  assert 'error' in record['decision'];label='error'
 counts[label]+=1;langs[key[0]][label]+=1
print(json.dumps({'status':'DIAGNOSTIC_ONLY_NOT_RELEASE','rows':len(inputs),'counts':dict(counts),'by_language':{k:dict(v) for k,v in langs.items()},'review_sha256':sha(result.read_bytes()),'input_sha256':sha(raw.encode()),'limitation':'Adaptive same-reviewer judgments are not release approval.'},indent=2))
