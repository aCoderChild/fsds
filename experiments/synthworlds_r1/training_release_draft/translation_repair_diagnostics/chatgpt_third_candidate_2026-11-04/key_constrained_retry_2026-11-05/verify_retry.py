#!/usr/bin/env python3
"""Verify the one-row key-constrained Qwen response."""
import collections,hashlib,json,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
INPUT=HERE/'review_input.json';BUILD=HERE/'build_manifest.json'
OUTPUT=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'remote_output/key_constrained_retry.json'
OUT=Path(sys.argv[2]) if len(sys.argv)>2 else HERE/'verification.json'
MARKER=re.compile(r'\[(?:ENTITY|NUMBER)_[A-Z]+\]')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 rows=json.loads(INPUT.read_text());build=json.loads(BUILD.read_text());result=json.loads(OUTPUT.read_text())
 canonical=json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()
 if len(rows)!=1 or rows[0]['review_key']!='R0349' or build['input_sha256']!=sha(canonical) or result.get('input_sha256')!=sha(canonical):raise ValueError('Exact key/input verification failed')
 if build['parent_output_sha256']!=result.get('parent_output_sha256'):raise ValueError('Parent output hash mismatch')
 recs=result.get('records',[])
 if len(recs)!=1 or recs[0].get('review_key')!='R0349':raise ValueError('Output key/coverage mismatch')
 row=rows[0];rec=recs[0];d=rec.get('decision') or {}
 for field,text in [('source_sha256',row['source_question_en']),('translation_sha256',row['translated_question']),('backtranslation_sha256',row['backtranslation_en'])]:
  if rec.get(field)!=sha(text.encode()):raise ValueError(f'{field} mismatch')
 if collections.Counter(MARKER.findall(row['source_question_en']))!=collections.Counter(MARKER.findall(row['translated_question'])) or collections.Counter(MARKER.findall(row['source_question_en']))!=collections.Counter(MARKER.findall(row['backtranslation_en'])):raise ValueError('Placeholder multisets differ')
 if result.get('model_id')!='Qwen/Qwen2.5-14B-Instruct' or result.get('model_revision')!='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8' or result.get('quantization')!='4-bit NF4' or result.get('gpu')!='Tesla T4':raise ValueError('Runtime pin mismatch')
 if result.get('answers_or_splits_sent') is not False or result.get('human_translation_reviewed') is not False or result.get('release_approved') is not False or result.get('training_eligible') is not False:raise ValueError('Diagnostic labels missing')
 if 'error' in d:label='BLOCKED'
 elif d.get('key')=='R0349' and type(d.get('semantic_pass')) is bool and type(d.get('fluency_pass')) is bool and isinstance(d.get('reason'),str) and d['reason'].strip():label='PASS' if d['semantic_pass'] and d['fluency_pass'] else 'FLAG'
 else:raise ValueError('Decision schema/key invalid')
 report={'status':'VERIFIED_ONE_ROW_KEY_CONSTRAINED_DIAGNOSTIC_ONLY','rows':1,'review_key':'R0349','decision':label,
  'input_sha256':sha(canonical),'parent_output_sha256':build['parent_output_sha256'],'output_sha256':sha(OUTPUT.read_bytes()),
  'human_translation_reviewed':False,'release_approved':False,'training_eligible':False,
  'interpretation':'A valid AI-only decision for the row; no human translation certification or release approval.'}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
