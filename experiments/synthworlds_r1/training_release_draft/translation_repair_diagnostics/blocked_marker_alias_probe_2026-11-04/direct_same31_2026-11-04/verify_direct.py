#!/usr/bin/env python3
"""Verify direct 31-row review and compare it to the matched back-translation screen."""
from __future__ import annotations
import collections, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
MODEL='Qwen/Qwen2.5-14B-Instruct'
REVISION='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
def sha(b): return hashlib.sha256(b).hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise ValueError(f'duplicate JSON key: {k}')
  d[k]=v
 return d
def parse(raw,key):
 text=raw.strip()
 if not text.startswith('{'): raise ValueError('raw response does not start with JSON')
 obj,end=json.JSONDecoder(object_pairs_hook=unique).raw_decode(text)
 if text[end:].strip(): raise ValueError('trailing text after JSON')
 if not isinstance(obj,dict) or set(obj)!={'decisions'} or not isinstance(obj['decisions'],list) or len(obj['decisions'])!=1: raise ValueError('invalid response wrapper')
 d=obj['decisions'][0]
 if not isinstance(d,dict) or set(d)!={'key','semantic_pass','fluency_pass','reason'} or d.get('key')!=key or type(d.get('semantic_pass')) is not bool or type(d.get('fluency_pass')) is not bool or not isinstance(d.get('reason'),str) or not d['reason'].strip(): raise ValueError('invalid decision schema/key')
 return d
def main():
 import argparse
 p=argparse.ArgumentParser(); p.add_argument('--remote-dir',type=Path,required=True); p.add_argument('--pulled-script',type=Path,required=True); a=p.parse_args()
 local=(HERE/'kaggle_kernel/direct_review.py').read_bytes(); pulled=a.pulled_script.read_bytes()
 if local!=pulled: raise ValueError('pulled Kaggle script differs from reviewed source')
 manifest=json.loads((HERE/'manifest.json').read_text()); raw=(HERE/'review_input.json').read_bytes().rstrip(b'\n'); inputs=json.loads(raw)
 parent_path=BASE/'final_diagnostic_overlay_2026-11-04/rechecked_blocked_31_rows.jsonl'; parent_receipt_path=BASE/'final_diagnostic_overlay_2026-11-04/verification.json'
 parent_rows=[json.loads(x) for x in parent_path.read_text().splitlines() if x.strip()]
 allowed={'review_key','language','kind','source_question_en','translated_question'}
 projected=[{k:r[k] for k in ('review_key','language','kind','source_question_en','translated_question')} for r in parent_rows]
 if (len(inputs)!=31 or len(parent_rows)!=31 or any(set(r)!=allowed for r in inputs)
     or inputs!=projected or sha(raw)!=manifest['input_sha256']
     or sha(local)!=manifest['script_sha256']
     or sha(parent_path.read_bytes())!=manifest['parent_overlay_sha256']
     or json.loads(parent_receipt_path.read_text()).get('result_rows_sha256')!=sha(parent_path.read_bytes())):
  raise ValueError('direct input/source/script or parent overlay binding failed')
 outpath=a.remote_dir/'direct_recheck.json'; out=json.loads(outpath.read_text(),object_pairs_hook=unique)
 if (out.get('status')!='DIAGNOSTIC_COMPLETE' or out.get('model_id')!=MODEL or out.get('model_revision')!=REVISION or out.get('input_sha256')!=manifest['input_sha256'] or out.get('parent_overlay_sha256')!=manifest['parent_overlay_sha256'] or out.get('quantization')!='4-bit NF4' or out.get('lm_format_enforcer')!='0.11.1' or out.get('backtranslation_provided') is not False or out.get('answers_or_splits_sent') is not False or out.get('human_translation_reviewed') is not False): raise ValueError('direct run metadata/scope mismatch')
 records=out.get('records')
 if not isinstance(records,list) or len(records)!=31: raise ValueError('expected exactly 31 output records')
 expected={r['review_key']:r for r in inputs}; observed={}
 for rec in records:
  k=rec.get('review_key')
  if k in observed: raise ValueError(f'duplicate row key {k}')
  observed[k]=rec
 if set(observed)!=set(expected): raise ValueError('direct review key set mismatch')
 counts=collections.Counter(); roundtrip_to_direct=collections.Counter(); sem=collections.Counter(); flu=collections.Counter(); result=[]
 parent_by_key={r['review_key']:r for r in parent_rows}
 for k,rec in observed.items():
  src=expected[k]; prior=parent_by_key[k]
  if (rec.get('language')!=src['language'] or rec.get('kind')!=src['kind'] or rec.get('source_sha256')!=sha(src['source_question_en'].encode()) or rec.get('translation_sha256')!=sha(src['translated_question'].encode())): raise ValueError(f'{k}: direct output not bound to source/target')
  d=parse(rec.get('raw_output',''),k)
  if rec.get('decision')!=d: raise ValueError(f'{k}: normalized decision differs from raw')
  direct='PASS' if d['semantic_pass'] and d['fluency_pass'] else 'FLAG'
  prior_status='PASS' if prior['review_status']=='AI_ONLY_REPAIR_REVIEW_PASS' else 'FLAG'
  counts[direct]+=1; roundtrip_to_direct[(prior_status,direct)]+=1
  sem['AGREE' if prior['decision']['semantic_pass']==d['semantic_pass'] else 'DISAGREE']+=1
  flu['AGREE' if prior['decision']['fluency_pass']==d['fluency_pass'] else 'DISAGREE']+=1
  result.append({'review_key':k,'language':src['language'],'kind':src['kind'],'backtranslation_screen':prior_status,'direct_screen':direct,'semantic_agreement':prior['decision']['semantic_pass']==d['semantic_pass'],'fluency_agreement':prior['decision']['fluency_pass']==d['fluency_pass'],'direct_reason':d['reason']})
 if out.get('counts')!=dict(counts): raise ValueError('summary counts mismatch')
 matrix={f'{a}->{b}':n for (a,b),n in sorted(roundtrip_to_direct.items())}
 receipt={'status':'VERIFIED_MATCHED_DIRECT_REVIEW_DIAGNOSTIC_ONLY','rows':31,'direct_counts':dict(counts),'backtranslation_to_direct_matrix':matrix,'semantic_agreement':dict(sem),'fluency_agreement':dict(flu),'overall_pass_flag_agreement':sum(n for (x,y),n in roundtrip_to_direct.items() if x==y),'formerly_flagged_direct_pass':roundtrip_to_direct[('FLAG','PASS')],'formerly_pass_direct_flag':roundtrip_to_direct[('PASS','FLAG')],'model_id':MODEL,'model_revision':REVISION,'input_sha256':manifest['input_sha256'],'parent_overlay_sha256':manifest['parent_overlay_sha256'],'submitted_script_sha256':sha(local),'pulled_script_sha256':sha(pulled),'output_sha256':sha(outpath.read_bytes()),'answers_or_splits_sent':False,'human_translation_reviewed':False,'release_approved':False,'training_eligible':False,'interpretation':'Agreement or disagreement measures sensitivity to reviewer presentation; neither screen is human-gold and this cannot establish which judgment is correct.'}
 (a.remote_dir/'paired_diagnostic.json').write_text(json.dumps({'verification':receipt,'rows':result},ensure_ascii=False,indent=2)+'\n')
 (a.remote_dir/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
