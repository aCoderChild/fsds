#!/usr/bin/env python3
"""Qwen-only review of 15 rows with alias-safe, hash-verified backtranslations."""
import base64, collections, hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
subprocess.run([sys.executable,"-m","pip","install","-q","transformers==4.51.3",
    "accelerate==1.15.0","lm-format-enforcer==0.11.1","bitsandbytes==0.50.2"],check=True)
import torch,transformers,accelerate,bitsandbytes
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

MODEL_ID="Qwen/Qwen2.5-14B-Instruct"
MODEL_REV="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
RAW_B64="W3sicmV2aWV3X2tleSI6IlIwMDM2IiwibGFuZ3VhZ2UiOiJmciIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgd2FzIFtFTlRJVFlfQV0gdGhlIGFyY2hpdGVjdCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJFbiBxdWVsbGUgYW5uw6llIFtFTlRJVFlfQV0gYS10LWlsIMOpdMOpIGzigJlhcmNoaXRlY3RlIGRlIFtFTlRJVFlfQl0gPyIsImJhY2t0cmFuc2xhdGlvbl9lbiI6IldoYXQgeWVhciB3YXMgW0VOVElUWV9BXSBbRU5USVRZX0JdJ3MgYXJjaGl0ZWN0PyJ9LHsicmV2aWV3X2tleSI6IlIwMDY2IiwibGFuZ3VhZ2UiOiJmciIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGljaCBlbnRpdHkgaXMgW0VOVElUWV9BXSBkaWZmZXJlbnQgZnJvbT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiUXVlbGxlIGVzdCBs4oCZZW50aXTDqSBkb250IFtFTlRJVFlfQV0gZGlmZsOocmUgPyIsImJhY2t0cmFuc2xhdGlvbl9lbiI6IldoYXQgaXMgW0VOVElUWV9BXSdzIG90aGVyIGVudGl0eT8ifSx7InJldmlld19rZXkiOiJSMDE3OSIsImxhbmd1YWdlIjoiZGUiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGJlY29tZSB0aGUgY291bnRyeSB0aGF0IFtFTlRJVFlfQl0gaXMgaW4/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IkFuIHdlbGNoZW0gRGF0dW0gd3VyZGUgW0VOVElUWV9BXSB6dSBkZW0gTGFuZCwgaW4gZGVtIFtFTlRJVFlfQl0gbGllZ3Q/IiwiYmFja3RyYW5zbGF0aW9uX2VuIjoiV2hlbiBkaWQgW0VOVElUWV9BXSBiZWNvbWUgdGhlIGNvdW50cnkgd2hlcmUgW0VOVElUWV9CXSBsaWVzPyJ9LHsicmV2aWV3X2tleSI6IlIwMjA2IiwibGFuZ3VhZ2UiOiJkZSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGF0IGlzIHRoZSBbRU5USVRZX0FdIGRpZmZlcmVudCBmcm9tPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJXb3ZvbiB1bnRlcnNjaGVpZGV0IHNpY2ggW0VOVElUWV9BXT8iLCJiYWNrdHJhbnNsYXRpb25fZW4iOiJXaGF0J3MgW0VOVElUWV9BXSBkaWZmZXJlbnQgYWJvdXQ/In0seyJyZXZpZXdfa2V5IjoiUjAyMjIiLCJsYW5ndWFnZSI6ImRlIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoaWNoIGVudGl0eSBpcyBbRU5USVRZX0FdIGRpZmZlcmVudCBmcm9tPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJWb24gd2VsY2hlciBFbnRpdMOkdCB1bnRlcnNjaGVpZGV0IHNpY2ggW0VOVElUWV9BXT8iLCJiYWNrdHJhbnNsYXRpb25fZW4iOiJXaGF0IGVudGl0eSBpcyBbRU5USVRZX0FdIGRpZmZlcmVudCBmcm9tPyJ9LHsicmV2aWV3X2tleSI6IlIwMzQ5IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gYmVjb21lIGEgcmVsYXRpdmUgb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSV4KS/4KS4IOCkpOCkvuCksOClgOCkliDgpJXgpYsgW0VOVElUWV9BXSDgpJTgpLAgW0VOVElUWV9CXSDgpJXgpYcg4KSs4KWA4KSaIOCkquCkvuCksOCkv+CkteCkvuCksOCkv+CklSDgpLjgpILgpKzgpILgpKcg4KS44KWN4KSl4KS+4KSq4KS/4KSkIOCkueClgeCkhj8iLCJiYWNrdHJhbnNsYXRpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIHRoZSBmYW1pbHkgcmVsYXRpb25zaGlwIGJldHdlZW4gW0VOVElUWV9BXSBhbmQgW0VOVElUWV9CXSBiZWdpbj8ifSx7InJldmlld19rZXkiOiJSMDM2MCIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGJlY29tZSB0aGUgY291bnRyeSB0aGF0IFtFTlRJVFlfQl0gaXMgaW4/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpKTgpL7gpLDgpYDgpJYg4KSV4KWLIFtFTlRJVFlfQV0g4KS14KS5IOCkpuClh+CktiDgpKzgpKjgpL4g4KSc4KS/4KS44KSu4KWH4KSCIFtFTlRJVFlfQl0g4KS44KWN4KSl4KS/4KSkIOCkueCliD8iLCJiYWNrdHJhbnNsYXRpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gbWFrZSB0aGUgY291bnRyeSBpbiB3aGljaCBbRU5USVRZX0JdIGlzIGxvY2F0ZWQ/In0seyJyZXZpZXdfa2V5IjoiUjAzNzkiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBjZWFzZSB0byBiZSB0aGUgY291bnRyeSBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSWIOCkleCliyBbRU5USVRZX0FdLCBbRU5USVRZX0JdIOCkleCkviDgpKbgpYfgpLYg4KSo4KS54KWA4KSCIOCksOCkueCkvj8iLCJiYWNrdHJhbnNsYXRpb25fZW4iOiJXaGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0sIFtFTlRJVFlfQl0ncyBjb3VudHJ5IG5vdCBnbz8ifSx7InJldmlld19rZXkiOiJSMDQwNyIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdIGJlY29tZSB0aGUgc3BvdXNlIG9mIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpLXgpLDgpY3gpLcgW0VOVElUWV9BXSwgW0VOVElUWV9CXSDgpJXgpL4g4KSc4KWA4KS14KSo4KS44KS+4KSl4KWAIOCkrOCkqOCkvj8iLCJiYWNrdHJhbnNsYXRpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gYmVjb21lIFtFTlRJVFlfQl0ncyB3aWZlPyJ9LHsicmV2aWV3X2tleSI6IlIwNDE3IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gc3RvcCBiZWluZyB0aGUgY2FwaXRhbCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSWIOCkleCliyBbRU5USVRZX0FdIOCkqOClhyBbRU5USVRZX0JdIOCkleClgCDgpLDgpL7gpJzgpKfgpL7gpKjgpYAg4KS54KWL4KSo4KS+IOCkrOCkguCkpiDgpJXgpL/gpK/gpL4/IiwiYmFja3RyYW5zbGF0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGNlYXNlIHRvIGJlIFtFTlRJVFlfQl0ncyBjYXBpdGFsPyJ9LHsicmV2aWV3X2tleSI6IlIwNDM2IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGVuIGRpZCBbRU5USVRZX0FdIGJlY29tZSBwYXJ0IG9mIHRoZSBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSWIOCkleCliyBbRU5USVRZX0FdLCBbRU5USVRZX0JdIOCkleCkviDgpLngpL/gpLjgpY3gpLjgpL4g4KSs4KSo4KS+PyIsImJhY2t0cmFuc2xhdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSwgW0VOVElUWV9CXSwgYmVjb21lIHBhcnQgb2Y/In0seyJyZXZpZXdfa2V5IjoiUjA0NDkiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSBzdG9wIHNlcnZpbmcgYXMgYSBib2FyZCBtZW1iZXIgb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSV4KS/4KS4IOCkteCksOCljeCktyBbRU5USVRZX0FdIOCkqOClhyBbRU5USVRZX0JdIOCkleClhyDgpKzgpYvgpLDgpY3gpKEg4KS44KSm4KS44KWN4KSvIOCkleClhyDgpLDgpYLgpKog4KSu4KWH4KSCIOCkleCkvuCkriDgpJXgpLDgpKjgpL4g4KSs4KSC4KSmIOCkleCkv+Ckr+Ckvj8iLCJiYWNrdHJhbnNsYXRpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gY2Vhc2UgdG8gc2VydmUgYXMgYSBib2FyZCBtZW1iZXIgb2YgW0VOVElUWV9CXT8ifSx7InJldmlld19rZXkiOiJSMDQ2NCIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiY29tcG9zaXRpb24iLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGF0IGlzIGEgcGFydCBvZiB0aGUgbmV4dCBoaWdoZXIgcmFuayBhYm92ZSB0aGUgY2xhc3Mgb2Ygd2hpY2ggdGhlIFtFTlRJVFlfQV0gaXMgYSBzdWJjbGFzcz8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSc4KS/4KS4IOCkteCksOCljeCklyDgpJXgpYAg4KSJ4KSq4KS24KWN4KSw4KWH4KSj4KWAIFtFTlRJVFlfQV0g4KS54KWILCDgpIngpLjgpLjgpYcg4KSP4KSVIOCkuOCljeCkpOCksCDgpIrgpKrgpLAg4KSV4KWAIOCktuCljeCksOClh+Cko+ClgCDgpJXgpL4g4KSV4KWM4KSoLeCkuOCkviDgpK3gpL7gpJcg4KS54KWIPyIsImJhY2t0cmFuc2xhdGlvbl9lbiI6IldoaWNoIG9mIHRoZSBzdWJjYXRlZ29yaWVzIG9mIHRoZSBjbGFzcyBbRU5USVRZX0FdIGlzIGEgbGV2ZWwgYWJvdmUgdGhlIGNsYXNzPyJ9LHsicmV2aWV3X2tleSI6IlIwNTQ5IiwibGFuZ3VhZ2UiOiJ0aCIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0ncyB0ZW51cmUgYXMgZGlyZWN0b3Igb2YgW0VOVElUWV9CXSBlbmQ/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuC4p+C4suC4o+C4sOC4geC4suC4o+C4lOC4s+C4o+C4h+C4leC4s+C5geC4q+C4meC5iOC4h+C4nOC4ueC5ieC4reC4s+C4meC4p+C4ouC4geC4suC4o+C4guC4reC4hyBbRU5USVRZX0FdIOC4l+C4teC5iCBbRU5USVRZX0JdIOC4quC4tOC5ieC4meC4quC4uOC4lOC4peC4h+C5gOC4oeC4t+C5iOC4reC4p+C4seC4meC4l+C4teC5iOC5gOC4l+C5iOC4suC5g+C4lD8iLCJiYWNrdHJhbnNsYXRpb25fZW4iOiJXaGVuIGRpZCBbRU5USVRZX0FdJ3MgdGVybSBhcyBkaXJlY3RvciBhdCBbRU5USVRZX0JdJ3MgZW5kPyJ9LHsicmV2aWV3X2tleSI6IlIwNjc3IiwibGFuZ3VhZ2UiOiJ0aCIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgZGlkIFtFTlRJVFlfQV0gcmVjZWl2ZSB0aGUgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDguYTguJTguYnguKPguLHguJogW0VOVElUWV9CXSDguYPguJnguJvguLXguYPguJQ/IiwiYmFja3RyYW5zbGF0aW9uX2VuIjoiW0VOVElUWV9BXSBnb3QgW0VOVElUWV9CXSBpbiB3aGF0IHllYXI/In1d"
INPUT_SHA256="ae9e971f55064f1d059cbd6cd4d61d2763c5eb78a123b1280eeaf08b2ffdeeda"
ALIAS_OUTPUT_SHA256="49ef405a582c6af101a3293bdede84f88a1509f73256c9754e4b34412b468590"
OUT=Path("/kaggle/working/qwen_only_retry.json")
MARKER=re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
LANG_NAMES={"fr":"French","de":"German","hi":"Hindi","th":"Thai"}
SYSTEM=("You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
"Decide whether the target question asks the same thing as the English source, preserving relation, direction, "
"polarity, scope, tense/time, and answer type. Accept faithful paraphrases and contextual synonyms; do not "
"require word-for-word equality. Use the English back-translation as diagnostic evidence, not a literal-match "
"requirement. Judge target-language grammar and naturalness separately. Names/numbers are opaque placeholders. "
"Do not answer questions or infer facts. If a material issue is uncertain, return false. Give one short reason "
"of at most 12 words. Return exactly one JSON decision and no text outside it.")
SCHEMA={"type":"object","properties":{"decisions":{"type":"array","items":{
"type":"object","properties":{"key":{"type":"string"},"semantic_pass":{"type":"boolean"},
"fluency_pass":{"type":"boolean"},"reason":{"type":"string"}},"required":["key","semantic_pass","fluency_pass","reason"],
"additionalProperties":False},"minItems":1,"maxItems":1}},"required":["decisions"],"additionalProperties":False}

def sha(b): return hashlib.sha256(b).hexdigest()
def unique_object(pairs):
    obj={}
    for k,v in pairs:
        if k in obj: raise ValueError(f"duplicate JSON key {k}")
        obj[k]=v
    return obj
def parse(raw,key):
    text=raw.strip(); start=text.find("{")
    if start<0: raise ValueError("no JSON object")
    value,end=json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(text[start:])
    if text[start+end:].strip(): raise ValueError("trailing text")
    if not isinstance(value,dict) or set(value)!={"decisions"} or not isinstance(value["decisions"],list) or len(value["decisions"])!=1:
        raise ValueError("invalid top-level decision coverage")
    d=value["decisions"][0]
    if (not isinstance(d,dict) or set(d)!={"key","semantic_pass","fluency_pass","reason"} or d.get("key")!=key
       or type(d.get("semantic_pass")) is not bool or type(d.get("fluency_pass")) is not bool
       or not isinstance(d.get("reason"),str) or not d["reason"].strip()): raise ValueError("invalid decision schema/key")
    return d
def save(result):
    t=OUT.with_suffix(".tmp"); t.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); t.replace(OUT)
def main():
    raw=base64.b64decode(RAW_B64)
    if sha(raw)!=INPUT_SHA256: raise ValueError("embedded Qwen input hash mismatch")
    rows=json.loads(raw)
    if len(rows)!=15 or len({r["review_key"] for r in rows})!=15: raise ValueError("expected exact 15-row input")
    if not torch.cuda.is_available(): raise RuntimeError("GPU unavailable; refusing CPU fallback")
    result={"status":"IN_PROGRESS","mode":"qwen_only_alias_safe_backtranslation_retry",
      "input_sha256":INPUT_SHA256,"alias_retry_output_sha256":ALIAS_OUTPUT_SHA256,
      "model_id":MODEL_ID,"model_revision":MODEL_REV,"torch":torch.__version__,
      "transformers":transformers.__version__,"accelerate":accelerate.__version__,"bitsandbytes":bitsandbytes.__version__,
      "quantization":"4-bit NF4","gpu":torch.cuda.get_device_name(0),"answers_or_splits_sent":False,
      "human_translation_reviewed":False,"release_approved":False,"training_eligible":False,
      "started_at_utc":datetime.now(timezone.utc).isoformat(),"records":[]}
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,torch_dtype=torch.float16,quantization_config=quant,device_map="auto")
    model.eval(); prefix=build_transformers_prefix_allowed_tokens_fn(tok,JsonSchemaParser(SCHEMA))
    for i,row in enumerate(rows,1):
        payload={"target_language":LANG_NAMES[row["language"]],"items":[{"key":row["review_key"],
          "source_question_en":row["source_question_en"],"translated_question":row["translated_question"],
          "back_translation_en":row["backtranslation_en"]}]}
        prompt=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(payload,ensure_ascii=False)}],tokenize=False,add_generation_prompt=True)
        encoded=tok(prompt,return_tensors="pt").to(model.device); raw_output=""; decision=None; error=None
        for attempt in range(2):
            try:
                with torch.inference_mode(): out=model.generate(**encoded,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id,prefix_allowed_tokens_fn=prefix)
                raw_output=tok.decode(out[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
                decision=parse(raw_output,row["review_key"]); break
            except Exception as exc: error=f"{type(exc).__name__}: {exc}"
        record={"review_key":row["review_key"],"language":row["language"],"kind":row["kind"],
          "source_sha256":sha(row["source_question_en"].encode()),"translation_sha256":sha(row["translated_question"].encode()),
          "backtranslation_en":row["backtranslation_en"],"backtranslation_sha256":sha(row["backtranslation_en"].encode()),
          "raw_output":raw_output,"decision":decision if decision else {"error":error}}
        result["records"].append(record); save(result); print(f"Qwen {i}/15: {row['review_key']}",flush=True)
        del encoded
    counts=collections.Counter()
    for rec in result["records"]:
        d=rec["decision"]
        if "error" in d: counts["BLOCKED"]+=1
        elif d["semantic_pass"] and d["fluency_pass"]: counts["PASS"]+=1
        else: counts["FLAG"]+=1
    result.update({"status":"COMPLETE_DIAGNOSTIC_ONLY","counts":dict(counts),"completed_at_utc":datetime.now(timezone.utc).isoformat()})
    save(result); print(json.dumps({"status":result["status"],"counts":result["counts"],"rows":len(rows)},indent=2),flush=True)
if __name__=="__main__": main()
