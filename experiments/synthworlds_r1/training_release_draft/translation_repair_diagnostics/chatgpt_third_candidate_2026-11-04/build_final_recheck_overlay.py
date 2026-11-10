#!/usr/bin/env python3
"""Build a diagnostic-only 187-row overlay from the frozen screen and verified retries."""
from __future__ import annotations
import collections, hashlib, json, re
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE/'kaggle_review/remote_output/repair_review.json'
INPUT=HERE/'kaggle_review/review_input.json'
FIRST=HERE/'marker_safe_retry_2026-11-05/remote_output/marker_safe_retry.json'
QWEN=HERE/'qwen_only_retry_2026-11-05/remote_output/qwen_only_retry.json'
KEY=HERE/'key_constrained_retry_2026-11-05/remote_output/key_constrained_retry.json'
OUT=HERE/'final_recheck_overlay_2026-11-05'
MARKER=re.compile(r'\[(?:ENTITY|NUMBER)_[A-Z]+\]')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def canonical_sha(rows):return sha(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode())
def checked_decision(record,key):
 d=record.get('decision') or {}
 if (d.get('key')!=key or type(d.get('semantic_pass')) is not bool or type(d.get('fluency_pass')) is not bool
   or not isinstance(d.get('reason'),str) or not d['reason'].strip()):
  raise ValueError(f'Missing or invalid decision for {key}')
 return d
def main():
 source=read(INPUT); parent=read(PARENT); first=read(FIRST); qwen=read(QWEN); key_retry=read(KEY)
 if len(source)!=187 or len(parent.get('records',[]))!=187 or len(first.get('records',[]))!=18:
  raise ValueError('Frozen source, parent screen, or 18-row retry coverage is incomplete')
 if parent.get('input_sha256')!=canonical_sha(source):raise ValueError('Parent screen input hash mismatch')
 parent_by={r['review_key']:r for r in parent['records']}
 first_by={r['review_key']:r for r in first['records']}
 qwen_by={r['review_key']:r for r in qwen['records']}
 key_record=key_retry.get('records',[None])[0]
 if len(first_by)!=18 or len(qwen_by)!=15 or not key_record or key_record.get('review_key')!='R0349':
  raise ValueError('Retry row coverage/key mismatch')
 if first.get('prior_output_sha256')!=sha(PARENT.read_bytes()):raise ValueError('Initial retry is not bound to parent')
 if qwen.get('alias_retry_output_sha256')!=sha(FIRST.read_bytes()):raise ValueError('Qwen-only retry not bound to saved alias output')
 if key_retry.get('parent_output_sha256')!=sha(QWEN.read_bytes()):raise ValueError('Key-constrained retry not bound to Qwen output')
 repaired={}
 for row in source:
  rid=row['review_key']; prior=parent_by[rid]
  if not ((prior.get('backtranslation_error') is not None) or 'error' in (prior.get('decision') or {})):
   continue
  retry=first_by[rid]
  if retry.get('source_sha256')!=sha(row['source_question_en'].encode()) or retry.get('translation_sha256')!=sha(row['translated_question'].encode()):
   raise ValueError(f'{rid}: source/target changed across retry')
  if prior.get('backtranslation_error') is not None:
   if not retry.get('backtranslation_en') or collections.Counter(MARKER.findall(retry['backtranslation_en']))!=collections.Counter(MARKER.findall(row['source_question_en'])):
    raise ValueError(f'{rid}: alias-restored backtranslation marker mismatch')
   if retry.get('backtranslation_sha256')!=sha(retry['backtranslation_en'].encode()):raise ValueError(f'{rid}: alias-restored backtranslation hash mismatch')
   qrec=qwen_by.get(rid)
   if qrec is None or qrec.get('backtranslation_sha256')!=retry.get('backtranslation_sha256'):
    raise ValueError(f'{rid}: Qwen-only input used a different backtranslation')
   if rid=='R0349':
    # The unconstrained retry emitted a wrong row key; accept only the later
    # schema-constrained decision bound to this exact Qwen retry output.
    d=checked_decision(key_record,rid);raw=key_record.get('raw_output');method='alias_safe_backtranslation_qwen_key_constrained_retry'
   else:
    d=checked_decision(qrec,rid); raw=qrec.get('raw_output')
    method='alias_safe_backtranslation_qwen_short_reason'
   final=dict(retry);final['decision']=d;final['raw_output']=raw;final['backtranslation_error']=None;final['repair_method']=method
  else:
   d=checked_decision(retry,rid)
   if not retry.get('backtranslation_en') or retry.get('backtranslation_error'):
    raise ValueError(f'{rid}: format retry did not preserve its prior backtranslation')
   final=dict(retry);final['repair_method']='existing_backtranslation_qwen_short_reason_retry'
  repaired[rid]=final
 if len(repaired)!=18:raise ValueError(f'Expected 18 repaired diagnostic decisions; got {len(repaired)}')
 records=[];counts=collections.Counter();by_lang=collections.defaultdict(collections.Counter);by_kind=collections.defaultdict(collections.Counter)
 for row in source:
  rid=row['review_key'];record=dict(repaired.get(rid,parent_by[rid]))
  decision=record.get('decision') or {}
  if record.get('backtranslation_error') or 'error' in decision:label='BLOCKED'
  else:label='PASS' if decision['semantic_pass'] and decision['fluency_pass'] else 'FLAG'
  record['diagnostic_label']=label
  counts[label]+=1;by_lang[row['language']][label]+=1;by_kind[row['kind']][label]+=1
  records.append(record)
 if len(records)!=187 or sum(counts.values())!=187:raise ValueError('Overlay does not cover exactly 187 rows')
 # Deterministic rebuilds should be safe and repeatable for this frozen input.
 OUT.mkdir(exist_ok=True)
 overlay={"status":"COMPLETE_AI_ONLY_THIRD_CANDIDATE_RECHECK_DIAGNOSTIC_ONLY",
  "mode":"original_169_decisions_plus_18_source_bound_retries",
  "input_sha256":parent['input_sha256'],"parent_output_sha256":sha(PARENT.read_bytes()),
  "initial_retry_output_sha256":sha(FIRST.read_bytes()),"qwen_only_retry_output_sha256":sha(QWEN.read_bytes()),
  "key_constrained_retry_output_sha256":sha(KEY.read_bytes()),"nllb_model_id":parent['nllb_model_id'],
  "nllb_model_revision":parent['nllb_model_revision'],"qwen_model_id":parent['qwen_model_id'],
  "qwen_model_revision":parent['qwen_model_revision'],"records":records,"rows":len(records),
  "counts":dict(counts),"counts_by_language":{k:dict(v) for k,v in sorted(by_lang.items())},
  "counts_by_kind":{k:dict(v) for k,v in sorted(by_kind.items())},
  "human_translation_reviewed":False,"answers_or_splits_sent":False,"release_approved":False,"training_eligible":False,
  "limitations":["This is an AI-only diagnostic overlay; no human translation certification.",
   "Eighteen formerly blocked rows use short-reason retries; one additionally used a key-constrained schema.",
   "This overlay does not edit the frozen 187-row review, source corpus, or training release."]}
 outpath=OUT/'final_recheck_overlay.json';outpath.write_text(json.dumps(overlay,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 receipt={"status":"BUILT_HASH_BOUND_RECHECK_OVERLAY_NOT_RELEASE_APPROVAL","rows":187,
  "repaired_rows":len(repaired),"counts":dict(counts),"input_sha256":parent['input_sha256'],
  "parent_output_sha256":sha(PARENT.read_bytes()),"initial_retry_output_sha256":sha(FIRST.read_bytes()),
  "qwen_only_retry_output_sha256":sha(QWEN.read_bytes()),"key_constrained_retry_output_sha256":sha(KEY.read_bytes()),
  "overlay_sha256":sha(outpath.read_bytes()),"release_approved":False,"training_eligible":False}
 (OUT/'build_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
