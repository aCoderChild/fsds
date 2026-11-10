#!/usr/bin/env python3
"""Build a source-bound, diagnostic-only reconciliation of 31 formerly blocked rows."""
from __future__ import annotations
import base64, collections, hashlib, json, re
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=Path(__file__).resolve().parents[6]
MERGED=ROOT/'tmp/r1_review_merge_verification_2026-11-04/merged_full799_diagnostic_reviews.jsonl'
ALIAS_DIR=BASE
QWEN_DIR=BASE/'qwen_remote_output/constrained_v2'
RETRY_DIR=BASE/'qwen_retry3_2026-11-04/remote_output'
RETRY_LOCAL=BASE/'qwen_retry3_2026-11-04'
MODEL='Qwen/Qwen2.5-14B-Instruct'
REVISION='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
MARKER=re.compile(r'\[(?:ENTITY|NUMBER)_[A-Z]+\]')

def sha(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def jsonl(path:Path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def read(path:Path): return json.loads(path.read_text(encoding='utf-8'))
def main():
    alias_output_path=ALIAS_DIR/'remote_output/alias_probe.json'
    alias_receipt_path=ALIAS_DIR/'remote_output/verification.json'
    qwen_output_path=QWEN_DIR/'qwen_recheck.json'
    qwen_receipt_path=QWEN_DIR/'verification.json'
    retry_output_path=RETRY_DIR/'qwen_recheck.json'
    retry_receipt_path=RETRY_DIR/'verification.json'
    qwen_script=(ALIAS_DIR/'qwen_kernel/review.py').read_text(encoding='utf-8')
    encoded=re.search(r'^RAW_INPUT_B64 = "([^"]+)"$',qwen_script,re.M)
    input_hash=re.search(r'^INPUT_SHA256 = "([^"]+)"$',qwen_script,re.M)
    if not encoded or not input_hash: raise ValueError('31-row submitted source lacks embedded input')
    raw_input=base64.b64decode(encoded.group(1))
    inputs=json.loads(raw_input)
    if len(inputs)!=31 or sha(raw_input)!=input_hash.group(1): raise ValueError('31-row input/hash mismatch')
    alias_receipt=read(alias_receipt_path); alias=read(alias_output_path)
    if (alias_receipt.get('status')!='VERIFIED_ALIAS_MARKER_SURVIVAL_DIAGNOSTIC_ONLY'
        or alias_receipt.get('rows')!=28 or alias_receipt.get('counts')!={'PASS_ALIAS_COUNT_ONLY':28}
        or alias_receipt.get('output_sha256')!=sha(alias_output_path.read_bytes())):
        raise ValueError('28-row alias evidence not verified')
    alias_by_key={r['review_key']:r for r in alias['rows']}
    qwen_receipt=read(qwen_receipt_path); qwen=read(qwen_output_path)
    if (qwen_receipt.get('status')!='VERIFIED_BLINDED_QWEN_RECHECK_DIAGNOSTIC_ONLY'
        or qwen_receipt.get('output_sha256')!=sha(qwen_output_path.read_bytes())
        or qwen_receipt.get('input_sha256')!=input_hash.group(1)
        or qwen.get('model_id')!=MODEL or qwen.get('model_revision')!=REVISION):
        raise ValueError('31-row Qwen run verification failed')
    retry_receipt=read(retry_receipt_path); retry=read(retry_output_path)
    retry_manifest=read(RETRY_LOCAL/'manifest.json')
    if (retry_receipt.get('status')!='VERIFIED_THREE_ROW_FORMAT_RETRY_DIAGNOSTIC_ONLY'
        or retry_receipt.get('output_sha256')!=sha(retry_output_path.read_bytes())
        or retry_manifest.get('parent_output_sha256')!=sha(qwen_output_path.read_bytes())):
        raise ValueError('three-row retry is not bound to the parent run')
    primary={r['review_key']:r for r in qwen['records']}
    retries={r['review_key']:r for r in retry['records']}
    if len(primary)!=31 or set(retries)!=set(retry_manifest['review_keys']): raise ValueError('row coverage mismatch')
    final={**primary,**retries}
    merged=jsonl(MERGED)
    if len(merged)!=799: raise ValueError('expected archived 799-row diagnostic merge')
    old_by_key={r['review_key']:r for r in merged}
    blocked={k for k,r in old_by_key.items() if r.get('review_status')=='AI_ONLY_REPAIR_REVIEW_BLOCKED'}
    if len(blocked)!=31 or set(final)!=blocked: raise ValueError('targeted rows are not the exact blocked set')
    counts=collections.Counter(); by_lang=collections.defaultdict(collections.Counter); by_kind=collections.defaultdict(collections.Counter)
    rows=[]
    for src in inputs:
        key=src['review_key']; rec=final[key]; old=old_by_key[key]
        if rec['source_sha256']!=sha(src['source_question_en'].encode()) or rec['translation_sha256']!=sha(src['translated_question'].encode()):
            raise ValueError(f'{key}: source/translation hash mismatch')
        if rec.get('language')!=src['language'] or rec.get('kind')!=src['kind']:
            raise ValueError(f'{key}: language or example kind mismatch')
        if rec['back_translation_sha256']!=sha(src['back_translation_en'].encode()):
            raise ValueError(f'{key}: back-translation hash mismatch')
        if old.get('source_sha256')!=rec['source_sha256'] or old.get('translation_sha256')!=rec['translation_sha256']:
            raise ValueError(f'{key}: not the blocked frozen-source candidate')
        decision=rec.get('decision')
        if 'error' in decision: raise ValueError(f'{key}: unresolved malformed response remains')
        if decision.get('key')!=key or type(decision.get('semantic_pass')) is not bool or type(decision.get('fluency_pass')) is not bool:
            raise ValueError(f'{key}: invalid normalized decision')
        alias_row=alias_by_key.get(key)
        if alias_row:
            if alias_row.get('status')!='PASS_ALIAS_COUNT_ONLY' or alias_row.get('restored_backtranslation_en')!=src['back_translation_en']:
                raise ValueError(f'{key}: alias back-translation differs from reviewed text')
            method='alias_restored_backtranslation'
        else:
            method='previous_hash_bound_backtranslation'
        status='AI_ONLY_REPAIR_REVIEW_PASS' if decision['semantic_pass'] and decision['fluency_pass'] else 'AI_ONLY_REPAIR_REVIEW_FLAGGED'
        counts[status]+=1; by_lang[src['language']][status]+=1; by_kind[src['kind']][status]+=1
        rows.append({'review_key':key,'language':src['language'],'kind':src['kind'],
            'prompt_id':old['prompt_id'],'source_question_en':src['source_question_en'],
            'translated_question':src['translated_question'],'backtranslation_en':src['back_translation_en'],
            'source_sha256':rec['source_sha256'],'translation_sha256':rec['translation_sha256'],
            'backtranslation_sha256':rec['back_translation_sha256'],
            'review_status':status,'decision':{'semantic_pass':decision['semantic_pass'],
                'fluency_pass':decision['fluency_pass'],'reason':decision['reason']},
            'review_method':method,'review_model_id':MODEL,'review_model_revision':REVISION,
            'human_translation_reviewed':False,'release_approved':False,'training_eligible':False})
    payload=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)
    out=HERE/'rechecked_blocked_31_rows.jsonl'; out.write_text(payload,encoding='utf-8')
    receipt={'status':'VERIFIED_31_ROW_ALIAS_RECHECK_DIAGNOSTIC_ONLY','rows':len(rows),
        'counts':dict(counts),'counts_by_language':{k:dict(v) for k,v in sorted(by_lang.items())},
        'counts_by_kind':{k:dict(v) for k,v in sorted(by_kind.items())},
        'formerly_blocked_rows_rechecked':31,'alias_marker_count_restored':len(alias_by_key),
        'semantic_reviews_complete':len(rows),'malformed_or_blocked_after_retry':0,
        'source_799_rows_sha256':sha(MERGED.read_bytes()),
        'alias_output_sha256':sha(alias_output_path.read_bytes()),
        'qwen_recheck_output_sha256':sha(qwen_output_path.read_bytes()),
        'qwen_recheck_verification_sha256':sha(qwen_receipt_path.read_bytes()),
        'three_row_retry_output_sha256':sha(retry_output_path.read_bytes()),
        'three_row_retry_verification_sha256':sha(retry_receipt_path.read_bytes()),
        'result_rows_sha256':sha(out.read_bytes()),'human_translation_reviewed':False,
        'answers_or_splits_sent':False,'release_approved':False,'training_eligible':False,
        'limitation':'AI-only diagnostic judgment; semantic/fluency accuracy is not human-certified.'}
    (HERE/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
