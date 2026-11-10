#!/usr/bin/env python3
"""Recompute input bindings and review counts, without approving candidates."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import re
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
TRAINING=HERE.parents[2]/'experiments/synthworlds_r1/training_release_draft/scripts'
sys.path.insert(0,str(TRAINING))
import repair_all_flagged_training_translations_v2 as repair
from assemble_training_release import checked_translation_reviews
from review_masking_v2 import mask_triplet

def sha(b):return hashlib.sha256(b).hexdigest()
job=HERE/'kaggle_full_candidate_review'
result=HERE/'evidence/full_candidate_review/review.json'
script=(job/'review.py').read_text(encoding='utf-8')
tree=ast.parse(script)
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
inputs=json.loads(raw); data=json.loads(result.read_text(encoding='utf-8'))
bindings=json.loads((job/'candidate_bindings.json').read_text(encoding='utf-8'))
base=HERE/'evidence/full_nllb_candidates'
direct_path=base/'restored_diagnostics.json';heuristic_path=base/'transliteration_restoration_diagnostic.json'
assert bindings['direct_sha256']==sha(direct_path.read_bytes()) and bindings['heuristic_sha256']==sha(heuristic_path.read_bytes())
direct=json.loads(direct_path.read_text())['records']
heuristic={(r['language'],r['kind'],r['prompt_id']):r for r in json.loads(heuristic_path.read_text())['records']}
_,by_lang=repair.source_manifest();selected=repair.select_flagged_rows(by_lang,checked_translation_reviews())
assert data['status']=='DIAGNOSTIC_AI_REVIEW_ONLY'
assert data['model_id']=='Qwen/Qwen2.5-7B-Instruct' and data['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
assert data['input_sha256']==sha(raw.encode())
assert len(selected)==len(direct)==len(inputs)==len(bindings['bindings'])==len(data['records'])==799
counts=Counter();groups=defaultdict(Counter);seen=set()
for item,d,inp,b,rec in zip(selected,direct,inputs,bindings['bindings'],data['records']):
 row=item['row'];key=(item['language'],row['kind'],row['prompt_id']);assert key not in seen;seen.add(key)
 for x in (d,inp,b,rec):assert (x['language'],x['kind'],x['prompt_id'])==key
 if d['structural_pass']:
  candidate=d['restored_translation'];method='exact_surrogate'
 else:
  v=heuristic[key];assert v['status']=='provisional_structural_pass';candidate=v['restored_translation'];method='heuristic_transliteration'
 assert b['restoration_method']==method and b['candidate_sha256']==sha(candidate.encode())
 source,target,_=mask_triplet({**row,'text_translation':candidate,'backtranslation_en':row['text_en']})
 assert inp['source']==source and inp['translation']==target
 match=re.search(r'\{.*\}',rec['raw_output'],re.S)
 try:
  decision=json.loads(match.group()) if match else None
  assert isinstance(decision,dict) and type(decision.get('semantic_pass')) is bool and type(decision.get('fluency_pass')) is bool and isinstance(decision.get('reason'),str)
  assert rec['decision']==decision
  label='pass' if decision['semantic_pass'] and decision['fluency_pass'] else 'flag'
 except (ValueError, AssertionError):
  assert 'error' in rec['decision'];label='error'
 counts[label]+=1;groups[f'{key[0]}:{method}'][label]+=1
print(json.dumps({'status':'DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW','rows':len(inputs),'counts':dict(counts),'by_language_and_restoration':{k:dict(v) for k,v in groups.items()},'review_sha256':sha(result.read_bytes()),'input_sha256':sha(raw.encode()),'limitations':['Model decisions are not proof of semantic or fluency accuracy.','Heuristic name restoration remains unvalidated.']},indent=2))
