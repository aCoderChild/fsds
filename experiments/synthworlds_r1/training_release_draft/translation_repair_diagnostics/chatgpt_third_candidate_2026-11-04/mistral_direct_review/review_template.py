#!/usr/bin/env python3
"""Blind direct bilingual cross-review of masked translations; diagnostic only."""
import gc, hashlib, json, re, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.51.3", "accelerate==1.15.0", "bitsandbytes==0.50.2",
    "lm-format-enforcer==0.11.1"], check=True)
import torch, transformers, accelerate, bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

MODE = "third_candidate_mistral_direct_187"
INPUT_SHA256 = "REPLACE_INPUT_SHA"
RAW_INPUT = "REPLACE_RAW_INPUT"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
OUT = Path("/kaggle/working/mistral_direct_review.json")
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
LANGUAGES = {"fr":"French", "de":"German", "hi":"Hindi", "th":"Thai"}


def sha(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()

def unique_object(pairs):
    obj={}
    for key,value in pairs:
        if key in obj: raise ValueError(f"Duplicate JSON key: {key}")
        obj[key]=value
    return obj

def schema():
    item={"type":"object","properties":{"key":{"type":"string"},"semantic_pass":{"type":"boolean"},"fluency_pass":{"type":"boolean"},"reason":{"type":"string"}},"required":["key","semantic_pass","fluency_pass","reason"],"additionalProperties":False}
    return {"type":"object","properties":{"decisions":{"type":"array","items":item,"minItems":1,"maxItems":1}},"required":["decisions"],"additionalProperties":False}

def save(status, rows, records, runtime):
    tmp=OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"status":status,"mode":MODE,"input_sha256":INPUT_SHA256,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,"records":records,**runtime},ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(OUT)

def review_one(row, model, tokenizer, prefix_function):
    system=("You are an independent multilingual translation reviewer. Compare the English source question directly with the target-language question. Judge whether the translation preserves the relation, direction, polarity, scope, tense, and answer type, and whether it is grammatical and natural in the named language. Accept faithful idiomatic paraphrases. Do not require literal wording. Bracketed entity and number placeholders are opaque. Do not answer the question or infer facts. Do not use outside knowledge. If unsure about a material semantic change or serious fluency problem, set the relevant field false. Return one concise reason and exactly one decision for the supplied key.")
    payload={"target_language":LANGUAGES[row["language"]],"items":[{"key":row["review_key"],"source_question_en":row["source_question_en"],"translated_question":row["translated_question"]}]}
    messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(payload,ensure_ascii=False)}]
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated=model.generate(**inputs,max_new_tokens=384,do_sample=False,pad_token_id=tokenizer.eos_token_id,prefix_allowed_tokens_fn=prefix_function)
    raw=tokenizer.decode(generated[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
    match=re.search(r"\{.*\}",raw,re.S)
    parsed=json.loads(match.group(0),object_pairs_hook=unique_object) if match else None
    decisions=parsed.get("decisions") if isinstance(parsed,dict) else None
    if not isinstance(decisions,list) or len(decisions)!=1: raise ValueError("invalid decision coverage")
    d=decisions[0]
    if (not isinstance(d,dict) or d.get("key")!=row["review_key"] or type(d.get("semantic_pass")) is not bool or type(d.get("fluency_pass")) is not bool or not isinstance(d.get("reason"),str) or not d["reason"].strip()):
        raise ValueError("invalid decision schema/key")
    return raw,d

def main():
    rows=json.loads(RAW_INPUT)
    if sha(RAW_INPUT)!=INPUT_SHA256: raise ValueError("embedded input hash mismatch")
    if len(rows)!=187 or len({r.get("review_key") for r in rows})!=187: raise ValueError("expected 187 unique review rows")
    for r in rows:
        if set(r)!={"review_key","language","kind","source_question_en","translated_question"} or r["language"] not in LANGUAGES or r["kind"] not in {"atomic","composition"}: raise ValueError("invalid blinded input schema")
        if not r["source_question_en"].strip() or not r["translated_question"].strip(): raise ValueError("empty translation/source")
        if Counter(PLACEHOLDER.findall(r["source_question_en"]))!=Counter(PLACEHOLDER.findall(r["translated_question"])): raise ValueError("placeholder mismatch in input")
    if not torch.cuda.is_available(): raise RuntimeError("GPU unavailable")
    runtime={"torch":torch.__version__,"transformers":transformers.__version__,"accelerate":accelerate.__version__,"bitsandbytes":bitsandbytes.__version__,"quantization":"4-bit NF4","gpu":torch.cuda.get_device_name(0),"answers_or_splits_sent":False,"qwen_decisions_or_backtranslations_sent":False,"human_translation_reviewed":False,"started_at_utc":datetime.now(timezone.utc).isoformat()}
    records=[]
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,torch_dtype=torch.float16,quantization_config=quant,device_map="auto")
    model.eval()
    prefix=build_transformers_prefix_allowed_tokens_fn(tokenizer,JsonSchemaParser(schema()))
    for idx,row in enumerate(rows,1):
        record={"review_key":row["review_key"],"language":row["language"],"kind":row["kind"],"source_sha256":sha(row["source_question_en"]),"translation_sha256":sha(row["translated_question"]),"decision":None,"raw_output":None}
        error=None
        for _ in range(2):
            try:
                raw,d=review_one(row,model,tokenizer,prefix);record["raw_output"]=raw;record["decision"]=d;break
            except Exception as exc:error=exc
        if record["decision"] is None:record["decision"]={"error":f"{type(error).__name__}: {error}"}
        records.append(record)
        save("IN_PROGRESS",rows,records,runtime)
        print(f"Reviewed {idx}/{len(rows)}",flush=True)
    blocked=sum("error" in (r.get("decision") or {}) for r in records)
    runtime.update({"completed_at_utc":datetime.now(timezone.utc).isoformat(),"reviewed_rows":len(rows)-blocked,"blocked_rows":blocked})
    status="MISTRAL_DIRECT_REVIEW_COMPLETE_DIAGNOSTIC_ONLY" if blocked==0 else "MISTRAL_DIRECT_REVIEW_HAS_BLOCKED_ROWS"
    save(status,rows,records,runtime)
    print(json.dumps({"status":status,"rows":len(rows),"reviewed_rows":len(rows)-blocked,"blocked_rows":blocked,"gpu":runtime["gpu"],"quantization":runtime["quantization"]},indent=2),flush=True)
    del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    return 0

if __name__=="__main__":raise SystemExit(main())
