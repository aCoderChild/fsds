#!/usr/bin/env python3
"""Create a blinded direct source-to-target review for the exact repaired 31 rows."""
import base64, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
rows_path=BASE/'final_diagnostic_overlay_2026-11-04/rechecked_blocked_31_rows.jsonl'
receipt_path=BASE/'final_diagnostic_overlay_2026-11-04/verification.json'
receipt=json.loads(receipt_path.read_text())
rows=[json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
if receipt.get('status')!='VERIFIED_31_ROW_ALIAS_RECHECK_DIAGNOSTIC_ONLY' or len(rows)!=31 or receipt.get('result_rows_sha256')!=hashlib.sha256(rows_path.read_bytes()).hexdigest():
    raise ValueError('parent 31-row diagnostic overlay failed verification')
inputs=[{'review_key':r['review_key'],'language':r['language'],'kind':r['kind'],'source_question_en':r['source_question_en'],'translated_question':r['translated_question']} for r in rows]
raw=json.dumps(inputs,ensure_ascii=False,separators=(',',':')).encode()
input_sha=hashlib.sha256(raw).hexdigest()
parent_sha=hashlib.sha256(rows_path.read_bytes()).hexdigest()
code=r'''#!/usr/bin/env python3
"""Blinded direct source-to-target translation screen; no back-translation is provided."""
import base64, collections, hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0", "bitsandbytes==0.50.2", "lm-format-enforcer==0.11.1"], check=True)
import torch, transformers, accelerate, bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
MODEL_ID="Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_SHA256="__INPUT_SHA256__"
PARENT_OVERLAY_SHA256="__PARENT_SHA256__"
RAW_INPUT_B64="__INPUT_B64__"
LANG_NAMES={"fr":"French","de":"German","hi":"Hindi","th":"Thai"}
OUT=Path('/kaggle/working/direct_recheck.json')
SYSTEM=("You judge whether a target-language synthetic QA question faithfully asks the same thing as its English source. "
"Preserve relation, direction, polarity, scope, tense/time, and answer type. Accept contextual synonyms and natural paraphrases. "
"Judge the target-language question directly; no back-translation is available. Do not answer the question or infer facts. "
"Names and bracketed entity/number markers are opaque placeholders. If meaning or fluency is uncertain, use false. "
"The reason must be one short sentence of at most 12 words. Return exactly one JSON object with a decisions array containing "
"one item with the supplied key and fields semantic_pass, fluency_pass, reason; no other text.")
SCHEMA={"type":"object","properties":{"decisions":{"type":"array","items":{"type":"object","properties":{"key":{"type":"string"},"semantic_pass":{"type":"boolean"},"fluency_pass":{"type":"boolean"},"reason":{"type":"string"},},"required":["key","semantic_pass","fluency_pass","reason"],"additionalProperties":False},"minItems":1,"maxItems":1}},"required":["decisions"],"additionalProperties":False}
def sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def unique_object(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError(f'duplicate JSON key: {k}')
        out[k]=v
    return out
def parse(raw,key):
    text=raw.strip()
    if not text.startswith('{'): raise ValueError('response does not start with JSON object')
    obj,end=json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(text)
    if text[end:].strip(): raise ValueError('trailing text after JSON')
    if not isinstance(obj,dict) or set(obj)!={'decisions'}: raise ValueError('invalid JSON wrapper')
    decisions=obj['decisions']
    if not isinstance(decisions,list) or len(decisions)!=1: raise ValueError('expected one decision')
    d=decisions[0]
    if not isinstance(d,dict) or set(d)!={'key','semantic_pass','fluency_pass','reason'} or d.get('key')!=key or type(d.get('semantic_pass')) is not bool or type(d.get('fluency_pass')) is not bool or not isinstance(d.get('reason'),str) or not d['reason'].strip():
        raise ValueError('decision schema/key invalid')
    return d
def main():
    raw=base64.b64decode(RAW_INPUT_B64)
    if hashlib.sha256(raw).hexdigest()!=INPUT_SHA256: raise ValueError('input hash mismatch')
    rows=json.loads(raw)
    keys=[r.get('review_key') for r in rows]
    if len(rows)!=31 or len(set(keys))!=31: raise ValueError('expected 31 unique rows')
    allowed={'review_key','language','kind','source_question_en','translated_question'}
    if any(set(r)!=allowed for r in rows): raise ValueError('unexpected field; no BT/answers/splits allowed')
    if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable; refusing alternate device')
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,torch_dtype=torch.float16,quantization_config=quant,device_map='auto')
    model.eval(); prefix=build_transformers_prefix_allowed_tokens_fn(tok,JsonSchemaParser(SCHEMA))
    result={'status':'IN_PROGRESS','model_id':MODEL_ID,'model_revision':MODEL_REVISION,'input_sha256':INPUT_SHA256,'parent_overlay_sha256':PARENT_OVERLAY_SHA256,'torch':torch.__version__,'transformers':transformers.__version__,'accelerate':accelerate.__version__,'bitsandbytes':bitsandbytes.__version__,'lm_format_enforcer':version('lm-format-enforcer'),'output_schema':'single-item decisions array','quantization':'4-bit NF4','gpu':torch.cuda.get_device_name(0),'human_translation_reviewed':False,'answers_or_splits_sent':False,'backtranslation_provided':False,'records':[],'started_at_utc':datetime.now(timezone.utc).isoformat()}
    for i,row in enumerate(rows,1):
        payload={'target_language':LANG_NAMES[row['language']],'items':[{'key':row['review_key'],'source_question_en':row['source_question_en'],'translated_question':row['translated_question']}]}
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        encoded=tok(prompt,return_tensors='pt').to(model.device)
        with torch.inference_mode(): generated=model.generate(**encoded,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id,prefix_allowed_tokens_fn=prefix)
        raw_out=tok.decode(generated[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
        try: decision=parse(raw_out,row['review_key'])
        except Exception as exc: decision={'error':type(exc).__name__,'detail':str(exc)}
        rec={'review_key':row['review_key'],'language':row['language'],'kind':row['kind'],'source_sha256':sha(row['source_question_en']),'translation_sha256':sha(row['translated_question']),'raw_output':raw_out,'decision':decision}
        result['records'].append(rec)
        tmp=OUT.with_suffix('.tmp'); tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(OUT)
        print(f'Reviewed {i}/{len(rows)}',flush=True)
        del encoded,generated; torch.cuda.empty_cache()
    counts=collections.Counter()
    for rec in result['records']:
        d=rec['decision']
        if 'error' in d: counts['MALFORMED_BLOCKED']+=1
        elif d['semantic_pass'] and d['fluency_pass']: counts['PASS']+=1
        else: counts['FLAG']+=1
    result.update({'status':'DIAGNOSTIC_COMPLETE','counts':dict(counts),'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'counts':result['counts']},indent=2),flush=True)
if __name__=='__main__': main()
'''
code=code.replace('__INPUT_SHA256__',input_sha).replace('__PARENT_SHA256__',parent_sha).replace('__INPUT_B64__',base64.b64encode(raw).decode())
kernel=HERE/'kaggle_kernel'; kernel.mkdir(exist_ok=True)
(kernel/'direct_review.py').write_text(code)
metadata={'id':'bunnoob2005/synthworlds-r1-direct-same31-review-2026-11-04','title':'SynthWorlds R1 direct same 31 translation review 2026-11-04','code_file':'direct_review.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(kernel/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
(HERE/'review_input.json').write_bytes(raw+b'\n')
manifest={'status':'SOURCE_BOUND_BLINDED_DIRECT_REVIEW_READY_NOT_APPROVAL','rows':31,'input_sha256':input_sha,'parent_overlay_sha256':parent_sha,'script_sha256':hashlib.sha256(code.encode()).hexdigest(),'model_id':'Qwen/Qwen2.5-14B-Instruct','model_revision':'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8','output_schema':'single-item decisions array; lm-format-enforcer 0.11.1','review_mode':'direct English source vs target language only; no back-translation, prior decision, answer, fact, or split label','release_approved':False,'training_eligible':False,'human_translation_reviewed':False,'answers_or_splits_sent':False}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
compile(code,str(kernel/'direct_review.py'),'exec')
print(json.dumps(manifest,indent=2))
