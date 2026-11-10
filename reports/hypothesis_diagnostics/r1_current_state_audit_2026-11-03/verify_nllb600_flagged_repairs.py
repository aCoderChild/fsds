#!/usr/bin/env python3
"""Source-check alternative drafts without admitting them into training."""
import ast,collections,hashlib,json,re,sys,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'experiments/synthworlds_r1/training_release_draft/scripts'
sys.path.insert(0,str(ROOT))
import repair_all_flagged_training_translations_v2 as repair
from assemble_training_release import checked_translation_reviews,norm
def sha(b):return hashlib.sha256(b).hexdigest()
job=HERE/'kaggle_nllb600_flagged_repairs';result=HERE/'evidence/nllb600_flagged_repairs/candidates.json'
script=(job/'translate.py').read_text();tree=ast.parse(script)
raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw' for t in n.targets))
inputs=json.loads(raw); data=json.loads(result.read_text());manifest=json.loads((job/'selection_manifest.json').read_text())
assert manifest['script_sha256']==sha(script.encode()) and manifest['input_sha256']==sha(raw.encode())
screen=HERE/'evidence/full_candidate_review/review.json';assert manifest['screen_sha256']==sha(screen.read_bytes())
flags={(r['language'],r['kind'],r['prompt_id']) for r in json.loads(screen.read_text())['records'] if r['decision'].get('semantic_pass') is not True or r['decision'].get('fluency_pass') is not True}
assert data['status']=='UNREVIEWED_DIAGNOSTIC_ONLY' and data['model_id']=='facebook/nllb-200-distilled-600M' and data['model_revision']=='f8d333a098d19b4fd9a8b18f94170487ad3f821d' and data['input_sha256']==sha(raw.encode())
_,by_lang=repair.source_manifest();source=repair.select_flagged_rows(by_lang,checked_translation_reviews())
source={(x['language'],x['row']['kind'],x['row']['prompt_id']):x['row'] for x in source}
variants=json.loads((HERE/'evidence/full_nllb_candidates/transliteration_restoration_diagnostic.json').read_text())['variants']
answers={}
for name in ('atomic_reuse.jsonl','atomic_translation_requests.jsonl','composition_reuse.jsonl'):
 for line in (ROOT.parent/name).read_text().splitlines():
  x=json.loads(line);v=x.get('answer_en') or x.get('answer')
  if v:
   assert x['sft_id'] not in answers or answers[x['sft_id']]==v
   answers[x['sft_id']]=v
assert len(inputs)==len(data['records'])==len(flags)==180
required=('nonempty','target_script_present','protected_names_preserved','protected_numbers_preserved','no_unrestored_marker')
counts=collections.Counter();by=collections.defaultdict(collections.Counter);seen=set();out=[]
for inp,rec in zip(inputs,data['records']):
 key=(inp['language'],inp['kind'],inp['prompt_id']);assert key in flags and key not in seen;seen.add(key)
 assert (rec['language'],rec['kind'],rec['prompt_id'])==key
 row=source[key];names=sorted({n for n in row['protected_entity_names'] if n in row['text_en']},key=lambda n:(-len(n),n));mapping=dict(zip(names,['Zorvani','Kelvoria']))
 masked=row['text_en']
 for name,s in mapping.items():masked=masked.replace(name,s)
 assert masked==inp['masked_source'] and inp['surrogate_counts']=={s:row['text_en'].count(n) for n,s in mapping.items()}
 target=rec['masked_translation'];basic=bool(target.strip()) and all(target.count(k)==v for k,v in inp['surrogate_counts'].items());assert rec['surrogate_integrity'] is basic
 text=unicodedata.normalize('NFC',target);method='exact_surrogate';failed=False
 for name,s in mapping.items():
  choices=[s]+variants.get(key[0],{}).get(s,[]);pat=re.compile('|'.join(re.escape(x) for x in sorted(set(choices),key=len,reverse=True)))
  matches=list(pat.finditer(text))
  if len(matches)!=row['text_en'].count(name):failed=True;break
  if any(m.group()!=s for m in matches):method='heuristic_transliteration'
  text=pat.sub(lambda _:name,text)
 if not failed:
  checks=repair.deterministic_checks(key[0],{**row,'text_translation':text});answer=answers.get(key[2]);assert answer
  failed=not all(checks.get(k) is True for k in required) or bool(norm(answer) and norm(answer) in norm(text))
 else:checks={}
 label='unresolved' if failed else method;counts[label]+=1;by[key[0]][label]+=1
 out.append({'language':key[0],'kind':key[1],'prompt_id':key[2],'status':label,'restored_translation':None if failed else text,'checks':checks})
assert seen==flags
dest=result.parent/'restored_diagnostics.json';dest.write_text(json.dumps({'status':'UNREVIEWED_DIAGNOSTIC_ONLY','input_sha256':sha(result.read_bytes()),'counts':dict(counts),'by_language':{k:dict(v) for k,v in by.items()},'records':out},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'UNREVIEWED_DIAGNOSTIC_ONLY','counts':dict(counts),'by_language':{k:dict(v) for k,v in by.items()},'output_sha256':sha(dest.read_bytes()),'limitation':'Heuristic restoration and semantic quality are unvalidated.'},indent=2))
