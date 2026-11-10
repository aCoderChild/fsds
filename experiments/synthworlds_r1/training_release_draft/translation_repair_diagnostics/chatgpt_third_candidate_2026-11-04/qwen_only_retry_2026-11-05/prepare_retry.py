#!/usr/bin/env python3
"""Prepare the Qwen-only half of an 18-row repair, reusing verified saved backtranslations."""
from __future__ import annotations

import ast
import base64
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
THIRD = HERE.parent
ALIAS_DIR = THIRD / "marker_safe_retry_2026-11-05"
SOURCE = ALIAS_DIR / "review_input.json"
ALIAS_OUTPUT = ALIAS_DIR / "remote_output/marker_safe_retry.json"
ORIGINAL_REVIEW = THIRD / "kaggle_review/remote_output/repair_review.json"
KERNEL = HERE / "kaggle_kernel"
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")

TEMPLATE = r'''#!/usr/bin/env python3
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
RAW_B64="__RAW_B64__"
INPUT_SHA256="__INPUT_SHA256__"
ALIAS_OUTPUT_SHA256="__ALIAS_OUTPUT_SHA256__"
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
'''


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    retry = json.loads(ALIAS_OUTPUT.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_REVIEW.read_text(encoding="utf-8"))
    if len(rows) != 18 or len(retry.get("records", [])) != 18 or len(original.get("records", [])) != 187:
        raise ValueError("Frozen 18-row retry/source/187-row parent files are incomplete")
    retry_by_key = {r["review_key"]: r for r in retry["records"]}
    original_by_key = {r["review_key"]: r for r in original["records"]}
    qwen_rows = []
    for source in rows:
        key = source["review_key"]
        first_retry = retry_by_key[key]
        prior = original_by_key[key]
        if prior.get("backtranslation_error"):
            if first_retry.get("backtranslation_error") != "NameError: name 'bt_by_key' is not defined":
                raise ValueError(f"Unexpected alias-retry failure for {key}")
            if (first_retry.get("source_sha256") != sha(source["source_question_en"].encode())
                or first_retry.get("translation_sha256") != sha(source["translated_question"].encode())):
                raise ValueError(f"Source/candidate hash mismatch {key}")
            bt = first_retry.get("backtranslation_en", "")
            if not bt or collections.Counter(MARKER.findall(bt)) != collections.Counter(MARKER.findall(source["source_question_en"])):
                raise ValueError(f"The restored alias backtranslation failed marker integrity {key}")
            if first_retry.get("backtranslation_sha256") != sha(bt.encode()):
                raise ValueError(f"Restored backtranslation hash mismatch {key}")
            qwen_rows.append({"review_key": key, "language": source["language"], "kind": source["kind"],
                "source_question_en": source["source_question_en"], "translated_question": source["translated_question"],
                "backtranslation_en": bt})
        else:
            decision = first_retry.get("decision") or {}
            if "error" in decision or not first_retry.get("backtranslation_en"):
                raise ValueError(f"Three-row format retry did not produce a usable decision: {key}")
    if len(qwen_rows) != 15:
        raise ValueError(f"Expected exactly 15 Qwen-only rows, got {len(qwen_rows)}")
    raw = json.dumps(qwen_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if KERNEL.exists():
        raise FileExistsError(f"Refusing to overwrite {KERNEL}")
    KERNEL.mkdir(parents=True)
    script = (TEMPLATE.replace("__RAW_B64__", base64.b64encode(raw).decode())
        .replace("__INPUT_SHA256__", sha(raw)).replace("__ALIAS_OUTPUT_SHA256__", sha(ALIAS_OUTPUT.read_bytes())))
    ast.parse(script)
    (KERNEL / "review.py").write_text(script, encoding="utf-8")
    metadata = {"id":"bunnoob2005/sw-third-qwen-retry-2026-11-05","title":"sw-third-qwen-retry-2026-11-05",
        "code_file":"review.py","language":"python","kernel_type":"script","is_private":True,
        "enable_gpu":True,"enable_internet":True,"dataset_sources":[],"competition_sources":[],
        "kernel_sources":[],"model_sources":[]}
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    receipt = {"status":"PREPARED_15_ROW_QWEN_RETRY_NOT_APPROVAL","rows":15,
        "input_sha256":sha(raw),"alias_retry_output_sha256":sha(ALIAS_OUTPUT.read_bytes()),
        "prior_187_output_sha256":sha(ORIGINAL_REVIEW.read_bytes()),
        "script_sha256":sha((KERNEL / "review.py").read_bytes()),
        "answers_or_splits_sent":False,"human_translation_reviewed":False,
        "release_approved":False,"training_eligible":False,
        "interpretation":"Reuses the exact restored backtranslations from the alias retry after verifying source/candidate hashes and placeholder multisets; no NLLB rerun."}
    (HERE / "build_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "review_input.json").write_bytes(raw + b"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
