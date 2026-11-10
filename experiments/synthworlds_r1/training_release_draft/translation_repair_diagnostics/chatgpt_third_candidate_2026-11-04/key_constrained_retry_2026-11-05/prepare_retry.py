#!/usr/bin/env python3
"""Prepare a one-row Qwen retry with the output key constrained by schema."""
from __future__ import annotations
import ast, base64, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
THIRD=HERE.parent
QWEN_DIR=THIRD/'qwen_only_retry_2026-11-05'
SOURCE=QWEN_DIR/'review_input.json'
PARENT=QWEN_DIR/'remote_output/qwen_only_retry.json'
KERNEL=HERE/'kaggle_kernel'

TEMPLATE=r'''#!/usr/bin/env python3
"""Strict one-row Qwen retry; exact expected key is enforced by the JSON schema."""
import base64,collections,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
subprocess.run([sys.executable,"-m","pip","install","-q","transformers==4.51.3","accelerate==1.15.0","lm-format-enforcer==0.11.1","bitsandbytes==0.50.2"],check=True)
import torch,transformers,accelerate,bitsandbytes
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
MODEL="Qwen/Qwen2.5-14B-Instruct"; REV="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
RAW_B64="__RAW_B64__"; INPUT_SHA="__INPUT_SHA__"; PARENT_SHA="__PARENT_SHA__"
OUT=Path('/kaggle/working/key_constrained_retry.json')
LANG={'fr':'French','de':'German','hi':'Hindi','th':'Thai'}
SYSTEM=("You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
"Decide whether the target question asks the same thing as the English source, preserving relation, direction, polarity, "
"scope, tense/time, and answer type. Accept faithful paraphrases and contextual synonyms; do not require word-for-word equality. "
"Use the English back-translation as diagnostic evidence, not as a literal-match requirement. Judge target-language grammar and "
"naturalness separately. Names/numbers are opaque placeholders. Do not answer questions or infer facts. If a material issue is "
"uncertain, return false. Give one short reason of at most 12 words. Return only one JSON decision.")
def sha(b):return hashlib.sha256(b).hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate JSON key')
  d[k]=v
 return d
def main():
 raw=base64.b64decode(RAW_B64)
 if sha(raw)!=INPUT_SHA:raise ValueError('embedded input hash mismatch')
 rows=json.loads(raw)
 if len(rows)!=1:raise ValueError('expected one row')
 row=rows[0]; key=row['review_key']
 if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
 schema={'type':'object','properties':{'decisions':{'type':'array','items':{
  'type':'object','properties':{'key':{'type':'string','enum':[key]},'semantic_pass':{'type':'boolean'},
  'fluency_pass':{'type':'boolean'},'reason':{'type':'string'}},'required':['key','semantic_pass','fluency_pass','reason'],
  'additionalProperties':False},'minItems':1,'maxItems':1}},'required':['decisions'],'additionalProperties':False}
 q={'status':'IN_PROGRESS','mode':'key_constrained_single_row_retry','input_sha256':INPUT_SHA,'parent_output_sha256':PARENT_SHA,
  'model_id':MODEL,'model_revision':REV,'torch':torch.__version__,'transformers':transformers.__version__,
  'accelerate':accelerate.__version__,'bitsandbytes':bitsandbytes.__version__,'quantization':'4-bit NF4',
  'gpu':torch.cuda.get_device_name(0),'answers_or_splits_sent':False,'human_translation_reviewed':False,
  'release_approved':False,'training_eligible':False,'started_at_utc':datetime.now(timezone.utc).isoformat(),'records':[]}
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
 tok=AutoTokenizer.from_pretrained(MODEL,revision=REV); model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float16,quantization_config=quant,device_map='auto');model.eval()
 prefix=build_transformers_prefix_allowed_tokens_fn(tok,JsonSchemaParser(schema))
 payload={'target_language':LANG[row['language']],'items':[{'key':key,'source_question_en':row['source_question_en'],
  'translated_question':row['translated_question'],'back_translation_en':row['backtranslation_en']}]}
 prompt=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],tokenize=False,add_generation_prompt=True)
 encoded=tok(prompt,return_tensors='pt').to(model.device); raw_out=''; decision=None; error=None
 for _ in range(2):
  try:
   with torch.inference_mode():out=model.generate(**encoded,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id,prefix_allowed_tokens_fn=prefix)
   raw_out=tok.decode(out[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
   parsed,end=json.JSONDecoder(object_pairs_hook=unique).raw_decode(raw_out.strip())
   if end!=len(raw_out.strip()):raise ValueError('trailing output')
   d=parsed['decisions'][0]
   if d['key']!=key or type(d['semantic_pass']) is not bool or type(d['fluency_pass']) is not bool or not d['reason'].strip():raise ValueError('wrong key or malformed decision')
   decision=d;error=None;break
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
 rec={'review_key':key,'language':row['language'],'kind':row['kind'],
  'source_sha256':sha(row['source_question_en'].encode()),'translation_sha256':sha(row['translated_question'].encode()),
  'backtranslation_sha256':sha(row['backtranslation_en'].encode()),'raw_output':raw_out,
  'decision':decision if decision else {'error':error}}
 q['records']=[rec];q['status']='COMPLETE_DIAGNOSTIC_ONLY';q['completed_at_utc']=datetime.now(timezone.utc).isoformat()
 q['counts']={'BLOCKED':1} if not decision else ({'PASS':1} if decision['semantic_pass'] and decision['fluency_pass'] else {'FLAG':1})
 OUT.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':q['status'],'key':key,'counts':q['counts']},indent=2))
if __name__=='__main__':main()
'''

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 rows=json.loads(SOURCE.read_text()); parent=json.loads(PARENT.read_text())
 bad=[r for r in parent['records'] if 'error' in (r.get('decision') or {})]
 if len(bad)!=1 or bad[0]['review_key']!='R0349':raise ValueError('Expected exactly the verified wrong-key row R0349')
 source={r['review_key']:r for r in rows}['R0349']
 row={k:source[k] for k in ('review_key','language','kind','source_question_en','translated_question','backtranslation_en')}
 raw=json.dumps([row],ensure_ascii=False,separators=(',',':')).encode()
 if KERNEL.exists():raise FileExistsError(KERNEL)
 KERNEL.mkdir(parents=True)
 script=TEMPLATE.replace('__RAW_B64__',base64.b64encode(raw).decode()).replace('__INPUT_SHA__',sha(raw)).replace('__PARENT_SHA__',sha(PARENT.read_bytes()))
 ast.parse(script);(KERNEL/'retry.py').write_text(script)
 meta={'id':'bunnoob2005/sw-third-key-retry-2026-11-05','title':'sw-third-key-retry-2026-11-05','code_file':'retry.py',
  'language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,
  'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
 (KERNEL/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 receipt={'status':'PREPARED_ONE_ROW_SCHEMA_CONSTRAINED_RETRY_NOT_APPROVAL','review_key':'R0349','input_sha256':sha(raw),
  'parent_output_sha256':sha(PARENT.read_bytes()),'script_sha256':sha((KERNEL/'retry.py').read_bytes()),
  'answers_or_splits_sent':False,'human_translation_reviewed':False,'release_approved':False,'training_eligible':False}
 (HERE/'build_manifest.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');(HERE/'review_input.json').write_bytes(raw+b'\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
