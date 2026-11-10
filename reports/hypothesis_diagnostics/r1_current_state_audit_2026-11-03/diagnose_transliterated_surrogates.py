#!/usr/bin/env python3
"""Try exact, conservative surrogate restoration; never release these drafts."""
from collections import Counter
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'experiments/synthworlds_r1/training_release_draft/scripts'
sys.path.insert(0,str(ROOT))
import repair_all_flagged_training_translations_v2 as repair
from assemble_training_release import checked_translation_reviews, norm

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def nfc(s): return unicodedata.normalize('NFC',s)

variants={
 'hi':{'Zorvani':['ज़ोरवानी','ज़ोर्वानी','जोरवानी','जोर्वानी'], 'Kelvoria':['केल्वोरिया','केलवोरिया']},
 'th':{'Zorvani':['โซร์วานี่','โซร์วานี','ซอร์วานี่','ซอร์วานี','ซอร์วานิ','จอร์วานี่'], 'Kelvoria':['เคลโวเรีย']},
}
variants={lang:{key:sorted(set(map(nfc,vals)),key=len,reverse=True) for key,vals in d.items()} for lang,d in variants.items()}
candidate_path=HERE/'evidence/full_nllb_candidates/candidates.json'
prior_path=HERE/'evidence/full_nllb_candidates/restored_diagnostics.json'
bindings_path=HERE/'kaggle_full_nllb_candidates/source_bindings.json'
candidates=json.loads(candidate_path.read_text())['records']
prior=json.loads(prior_path.read_text());assert prior['candidate_sha256']==sha(candidate_path)
bindings=json.loads(bindings_path.read_text())['bindings']
_, by_lang=repair.source_manifest()
selected=repair.select_flagged_rows(by_lang,checked_translation_reviews())
assert len(candidates)==len(prior['records'])==len(bindings)==len(selected)==799
answers={}
for filename in ('atomic_reuse.jsonl','atomic_translation_requests.jsonl','composition_reuse.jsonl'):
 for line in (ROOT.parent/filename).read_text().splitlines():
  r=json.loads(line);value=r.get('answer_en') or r.get('answer')
  if value:
   assert r['sft_id'] not in answers or answers[r['sft_id']]==value
   answers[r['sft_id']]=value
required=('nonempty','target_script_present','protected_names_preserved','protected_numbers_preserved','no_unrestored_marker')
counts=Counter();output=[]
for c,b,old,item in zip(candidates,bindings,prior['records'],selected):
 row=item['row'];key=(item['language'],row['kind'],row['prompt_id'])
 for x in (c,b,old):assert (x['language'],x['kind'],x['prompt_id'])==key
 if old['structural_pass'] or key[0] not in variants:continue
 names=sorted({n for n in row['protected_entity_names'] if n in row['text_en']},key=lambda n:(-len(n),n))
 mapping=dict(zip(names,['Zorvani','Kelvoria']));assert b['names_to_surrogates']==mapping
 text=nfc(c['masked_translation']); matched=[]; failure=False
 for name,surrogate in mapping.items():
  choices=variants[key[0]][surrogate]+[surrogate]
  pattern=re.compile('|'.join(map(re.escape,sorted(set(choices),key=len,reverse=True))))
  matches=list(pattern.finditer(text));expected=row['text_en'].count(name)
  if len(matches)!=expected:failure=True;break
  text=pattern.sub(lambda _:name,text);matched.append(surrogate)
 if not failure:
  checks=repair.deterministic_checks(key[0],{**row,'text_translation':text})
  answer=answers.get(key[2]);assert answer
  leak=bool(norm(answer) and norm(answer) in norm(text))
  failure=not all(checks.get(k) is True for k in required) or leak
 else:checks={};leak=False
 label='provisional_structural_pass' if not failure else 'unresolved'
 counts[(key[0],label)]+=1
 output.append({'language':key[0],'kind':key[1],'prompt_id':key[2],'status':label,'restored_translation':text if not failure else None,'checks':checks,'exact_answer_mention':leak})
out=HERE/'evidence/full_nllb_candidates/transliteration_restoration_diagnostic.json'
out.write_text(json.dumps({'status':'UNREVIEWED_PROVISIONAL_DIAGNOSTIC_ONLY','inputs':{'candidates_sha256':sha(candidate_path),'prior_sha256':sha(prior_path),'bindings_sha256':sha(bindings_path)},'variants':variants,'counts':{f'{k[0]}:{k[1]}':v for k,v in counts.items()},'records':output,'limitations':['Curated exact transliterations are heuristic and are not independently reviewed.','Back-translation and semantic/fluency review are required.']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'UNREVIEWED_PROVISIONAL_DIAGNOSTIC_ONLY','counts':{f'{k[0]}:{k[1]}':v for k,v in counts.items()},'output_sha256':sha(out)},indent=2))
