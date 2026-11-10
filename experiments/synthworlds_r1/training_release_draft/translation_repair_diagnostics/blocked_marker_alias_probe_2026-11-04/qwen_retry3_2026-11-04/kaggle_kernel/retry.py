#!/usr/bin/env python3
"""Blinded, one-row Qwen semantic/fluency recheck with raw-output provenance."""
import base64, collections, hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0", "bitsandbytes==0.50.2", "lm-format-enforcer==0.11.1"], check=True)
import torch, transformers, accelerate, bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import lmformatenforcer
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
PARENT_OUTPUT_SHA256 = "0586215735ec2f4b6b046ccac08b124a82907b0cff4c4581e7a5716a622cb2db"
INPUT_SHA256 = "d7fa988d54547b19cea3b531e18fb1ee47bc93e98f3146018e431920b6346f1e"
RAW_INPUT_B64 = "W3sicmV2aWV3X2tleSI6IlIwMjIyIiwibGFuZ3VhZ2UiOiJkZSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGljaCBlbnRpdHkgaXMgW0VOVElUWV9BXSBkaWZmZXJlbnQgZnJvbT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiVm9uIHdlbGNoZXIgRW50aXTDpHQgdW50ZXJzY2hlaWRldCBzaWNoIFtFTlRJVFlfQV0/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6IldoYXQgZW50aXR5IGlzIFtFTlRJVFlfQV0gZGlmZmVyZW50IGZyb20/In0seyJyZXZpZXdfa2V5IjoiUjA0MjEiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoYXQgaW1tZWRpYXRlbHkgZm9sbG93cyBbRU5USVRZX0FdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleClhyDgpKTgpYHgpLDgpILgpKQg4KSs4KS+4KSmIOCkleCljeCkr+CkviDgpIbgpKTgpL4g4KS54KWIPyIsImJhY2tfdHJhbnNsYXRpb25fZW4iOiJXaGF0IGNvbWVzIGFmdGVyIFtFTlRJVFlfQV0/In0seyJyZXZpZXdfa2V5IjoiUjA1MjEiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImNvbXBvc2l0aW9uIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hlcmUgaXMgdGhlIGVudGl0eSB0aGF0IGJvcmRlcnMgdGhlIHBsYWNlIFtFTlRJVFlfQV0gaXMgaW5kaWdlbm91cyB0byBoZWFkcXVhcnRlcmVkPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJzgpL/gpLgg4KS44KWN4KSl4KS+4KSoIOCkleCkviBbRU5USVRZX0FdIOCkruClguCksiDgpKjgpL/gpLXgpL7gpLjgpYAg4KS54KWILCDgpIngpLjgpJXgpYAg4KS44KWA4KSu4KS+IOCkuOClhyDgpLLgpJfgpYAg4KSH4KSV4KS+4KSIIOCkleCkviDgpK7gpYHgpJbgpY3gpK/gpL7gpLLgpK8g4KSV4KS54KS+4KSBIOCkuOCljeCkpeCkv+CkpCDgpLngpYg/IiwiYmFja190cmFuc2xhdGlvbl9lbiI6IldoZXJlIGlzIHRoZSBoZWFkcXVhcnRlcnMgb2YgdGhlIHVuaXQgbG9jYXRlZCBvbiB0aGUgYm9yZGVyIG9mIHRoZSBwbGFjZSB3aGVyZSBbRU5USVRZX0FdIGlzIG5hdGl2ZT8ifV0="
LANG_NAMES = {"fr":"French", "de":"German", "hi":"Hindi", "th":"Thai"}
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
OUT = Path("/kaggle/working/qwen_recheck.json")

SYSTEM = (
    "You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
    "For semantic_pass, decide whether the target-language question asks the same thing as the English "
    "source: preserve relation, direction, polarity, scope, tense/time, and answer type. Accept faithful "
    "paraphrases, inflection changes, and contextual synonyms. Do not reject merely because a word has "
    "another dictionary sense; judge the translated question in context. Use the English back-translation "
    "as independent diagnostic evidence, not as a requirement for word-for-word equality. For fluency_pass, "
    "judge only whether the translated question is grammatical and natural in the named target language. "
    "Names/numbers are opaque placeholders. Do not answer the questions or infer their facts. If a material "
    "meaning change or substantial fluency problem is uncertain, return false. Give one short sentence of at most 12 words as the reason; do not elaborate. "
    "Return exactly one JSON object with a `decisions` array containing one item with the supplied key "
    "and the fields `semantic_pass`, `fluency_pass`, and `reason`; return no text outside that JSON."
)

SCHEMA = {"type":"object", "properties":{"decisions":{"type":"array", "items":{
    "type":"object", "properties":{"key":{"type":"string"},
        "semantic_pass":{"type":"boolean"}, "fluency_pass":{"type":"boolean"},
        "reason":{"type":"string"}},
    "required":["key","semantic_pass","fluency_pass","reason"], "additionalProperties":False},
    "minItems":1, "maxItems":1}}, "required":["decisions"], "additionalProperties":False}

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out

def parse_decision(raw, expected_key):
    text = raw.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("model response contains no JSON object")
    parsed, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(text[start:])
    if text[start + end:].strip():
        raise ValueError("non-whitespace trailing text after JSON")
    if not isinstance(parsed, dict):
        raise ValueError("top-level response is not an object")
    if set(parsed) == {"decisions"}:
        decisions = parsed["decisions"]
        if not isinstance(decisions, list) or len(decisions) != 1:
            raise ValueError("expected exactly one decision in decisions array")
        decision = decisions[0]
    elif set(parsed) == {expected_key} and isinstance(parsed[expected_key], dict):
        decision = {"key": expected_key, **parsed[expected_key]}
    else:
        decision = parsed
    if (not isinstance(decision, dict)
            or set(decision) != {"key", "semantic_pass", "fluency_pass", "reason"}
            or decision.get("key") != expected_key
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str)
            or not decision["reason"].strip()):
        raise ValueError("single decision failed exact key/schema checks")
    return decision

def main():
    raw_input = base64.b64decode(RAW_INPUT_B64)
    if hashlib.sha256(raw_input).hexdigest() != INPUT_SHA256:
        raise ValueError("embedded input hash mismatch")
    rows = json.loads(raw_input)
    if len(rows) != 3 or len({r.get("review_key") for r in rows}) != 3:
        raise ValueError("expected exactly three unique truncated-response rows")
    for row in rows:
        if set(row) != {"review_key", "language", "kind", "source_question_en", "translated_question", "back_translation_en"}:
            raise ValueError("unexpected field; only question text is allowed")
        if not (collections.Counter(MARKER.findall(row["source_question_en"]))
                == collections.Counter(MARKER.findall(row["translated_question"]))
                == collections.Counter(MARKER.findall(row["back_translation_en"]))):
            raise ValueError(f"{row['review_key']}: placeholder sequences differ")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing a different-device model review")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float16, quantization_config=quant, device_map="auto")
    model.eval()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tok, JsonSchemaParser(SCHEMA))
    result = {"status":"IN_PROGRESS", "model_id":MODEL_ID, "model_revision":MODEL_REVISION,
        "input_sha256":INPUT_SHA256, "parent_output_sha256":PARENT_OUTPUT_SHA256, "torch":torch.__version__,
        "transformers":transformers.__version__, "accelerate":accelerate.__version__,
        "bitsandbytes":bitsandbytes.__version__, "lm_format_enforcer":version("lm-format-enforcer"),
        "output_schema":"single-item decisions array", "quantization":"4-bit NF4",
        "gpu":torch.cuda.get_device_name(0), "answers_or_splits_sent":False,
        "human_translation_reviewed":False, "records":[],
        "started_at_utc":datetime.now(timezone.utc).isoformat()}
    for i, row in enumerate(rows, 1):
        payload = {"target_language":LANG_NAMES[row["language"]], "items":[{
            "key":row["review_key"], "source_question_en":row["source_question_en"],
            "translated_question":row["translated_question"],
            "back_translation_en":row["back_translation_en"]}]}
        messages = [{"role":"system", "content":SYSTEM},
                    {"role":"user", "content":json.dumps(payload, ensure_ascii=False)}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=128, do_sample=False,
                pad_token_id=tok.eos_token_id, prefix_allowed_tokens_fn=prefix_function)
        raw = tok.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            decision = parse_decision(raw, row["review_key"])
        except Exception as exc:
            decision = {"error":type(exc).__name__, "detail":str(exc)}
        record = {"review_key":row["review_key"], "language":row["language"], "kind":row["kind"],
            "source_sha256":sha(row["source_question_en"]),
            "translation_sha256":sha(row["translated_question"]),
            "back_translation_sha256":sha(row["back_translation_en"]),
            "raw_output":raw, "decision":decision}
        result["records"].append(record)
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(OUT)
        print(f"Reviewed {i}/{len(rows)}", flush=True)
        del encoded, generated
        torch.cuda.empty_cache()
    counts = collections.Counter()
    for record in result["records"]:
        d = record["decision"]
        if "error" in d: counts["MALFORMED_BLOCKED"] += 1
        elif d["semantic_pass"] and d["fluency_pass"]: counts["PASS"] += 1
        else: counts["FLAG"] += 1
    result.update({"status":"DIAGNOSTIC_COMPLETE", "counts":dict(counts),
        "completed_at_utc":datetime.now(timezone.utc).isoformat()})
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status":result["status"], "counts":result["counts"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
