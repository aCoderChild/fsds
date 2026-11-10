#!/usr/bin/env python3
"""Verify the three-row retry output against exact inputs, source, and parent run."""
from __future__ import annotations
import collections, hashlib, json, sys
import base64, re
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
from verify_qwen_recheck import parse_decision, unique_object

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--remote-dir',type=Path,required=True)
    p.add_argument('--pulled-script',type=Path,required=True)
    a=p.parse_args()
    local=(HERE/'kaggle_kernel/retry.py').read_bytes()
    pulled=a.pulled_script.read_bytes()
    if local!=pulled: raise ValueError('pulled Kaggle source differs from reviewed local source')
    manifest=json.loads((HERE/'manifest.json').read_text())
    raw=(HERE/'retry_input.json').read_bytes().rstrip(b'\n')
    rows=json.loads(raw)
    parent=BASE/'qwen_remote_output/constrained_v2/qwen_recheck.json'
    parent_script=(BASE/'qwen_kernel/review.py').read_text(encoding='utf-8')
    parent_encoded=re.search(r'^RAW_INPUT_B64 = "([^"]+)"$',parent_script,re.M)
    if not parent_encoded: raise ValueError('parent run input is missing')
    parent_rows=json.loads(base64.b64decode(parent_encoded.group(1)))
    parent_output=json.loads(parent.read_text())
    malformed={r['review_key'] for r in parent_output['records'] if 'error' in r['decision']}
    if (len(rows)!=3 or sha(local)!=manifest['script_sha256']
        or sha(raw)!=manifest['input_sha256']
        or sha(parent.read_bytes())!=manifest['parent_output_sha256']):
        raise ValueError('input or parent-run hash mismatch')
    expected_parent={r['review_key']:r for r in parent_rows}
    if (set(malformed)!=set(manifest['review_keys'])
        or any(expected_parent.get(r['review_key'])!=r for r in rows)):
        raise ValueError('retry is not the exact malformed subset of the parent 31-row input')
    output_path=a.remote_dir/'qwen_recheck.json'
    output=json.loads(output_path.read_text(),object_pairs_hook=unique_object)
    if (output.get('status')!='DIAGNOSTIC_COMPLETE'
        or output.get('model_id')!=manifest['model_id']
        or output.get('model_revision')!=manifest['model_revision']
        or output.get('input_sha256')!=manifest['input_sha256']
        or output.get('parent_output_sha256')!=manifest['parent_output_sha256']
        or output.get('quantization')!='4-bit NF4'
        or output.get('lm_format_enforcer')!='0.11.1'
        or output.get('answers_or_splits_sent') is not False
        or output.get('human_translation_reviewed') is not False):
        raise ValueError('run metadata, scope, or model pin mismatch')
    records=output.get('records')
    if not isinstance(records,list) or len(records)!=3:
        raise ValueError('expected exactly three retry records')
    expected={r['review_key']:r for r in rows}
    observed={}
    for rec in records:
        k=rec.get('review_key')
        if k in observed: raise ValueError(f'duplicate review key: {k}')
        observed[k]=rec
    if set(observed)!=set(expected): raise ValueError('retry key coverage mismatch')
    for key, rec in observed.items():
        src=expected[key]
        if (rec.get('language')!=src['language'] or rec.get('kind')!=src['kind']
            or rec.get('source_sha256')!=sha(src['source_question_en'].encode())
            or rec.get('translation_sha256')!=sha(src['translated_question'].encode())
            or rec.get('back_translation_sha256')!=sha(src['back_translation_en'].encode())):
            raise ValueError(f'{key}: row binding mismatch')
        d=parse_decision(rec.get('raw_output',''),key)
        if rec.get('decision')!=d: raise ValueError(f'{key}: normalized decision mismatch')
    counts=collections.Counter()
    for d in [rec['decision'] for rec in observed.values()]:
        counts['PASS' if d['semantic_pass'] and d['fluency_pass'] else 'FLAG']+=1
    if output.get('counts')!=dict(counts): raise ValueError('reported counts differ from verified rows')
    old={r['review_key']:r for r in parent_output['records']}
    for key,rec in observed.items():
        if 'error' not in old[key]['decision']: raise ValueError(f'{key}: row was not malformed in parent run')
    receipt={'status':'VERIFIED_THREE_ROW_FORMAT_RETRY_DIAGNOSTIC_ONLY',
        'rows':3,'counts':dict(counts),'release_approved':False,'training_eligible':False,
        'model_id':manifest['model_id'],'model_revision':manifest['model_revision'],
        'input_sha256':manifest['input_sha256'],'parent_output_sha256':manifest['parent_output_sha256'],
        'submitted_script_sha256':sha(local),'pulled_script_sha256':sha(pulled),
        'output_sha256':sha(output_path.read_bytes()),'human_translation_reviewed':False,
        'answers_or_splits_sent':False}
    (a.remote_dir/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
