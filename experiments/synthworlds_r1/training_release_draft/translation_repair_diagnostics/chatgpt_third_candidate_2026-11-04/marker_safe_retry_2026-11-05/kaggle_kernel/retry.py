#!/usr/bin/env python3
"""Retry only blocked third-candidate rows using alias-safe NLLB and strict Qwen JSON."""
import base64, collections, gc, hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.51.3", "accelerate==1.15.0", "lm-format-enforcer==0.11.1",
    "bitsandbytes==0.50.2"], check=True)
import torch, transformers, accelerate, bitsandbytes
from transformers import (AutoModelForSeq2SeqLM, NllbTokenizerFast,
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REV = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REV = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
NLLB_CODES = {"fr":"fra_Latn", "de":"deu_Latn", "hi":"hin_Deva", "th":"tha_Thai"}
LANG_NAMES = {"fr":"French", "de":"German", "hi":"Hindi", "th":"Thai"}
ALIASES = {"[ENTITY_A]":"Alice", "[ENTITY_B]":"Bob"}
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
RAW_B64 = "W3sicmV2aWV3X2tleSI6IlIwMDM2IiwibGFuZ3VhZ2UiOiJmciIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJJbiB3aGF0IHllYXIgd2FzIFtFTlRJVFlfQV0gdGhlIGFyY2hpdGVjdCBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJFbiBxdWVsbGUgYW5uw6llIFtFTlRJVFlfQV0gYS10LWlsIMOpdMOpIGzigJlhcmNoaXRlY3RlIGRlIFtFTlRJVFlfQl0gPyJ9LHsicmV2aWV3X2tleSI6IlIwMDY2IiwibGFuZ3VhZ2UiOiJmciIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGljaCBlbnRpdHkgaXMgW0VOVElUWV9BXSBkaWZmZXJlbnQgZnJvbT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiUXVlbGxlIGVzdCBs4oCZZW50aXTDqSBkb250IFtFTlRJVFlfQV0gZGlmZsOocmUgPyJ9LHsicmV2aWV3X2tleSI6IlIwMTc5IiwibGFuZ3VhZ2UiOiJkZSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJPbiB3aGF0IGRhdGUgZGlkIFtFTlRJVFlfQV0gYmVjb21lIHRoZSBjb3VudHJ5IHRoYXQgW0VOVElUWV9CXSBpcyBpbj8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiQW4gd2VsY2hlbSBEYXR1bSB3dXJkZSBbRU5USVRZX0FdIHp1IGRlbSBMYW5kLCBpbiBkZW0gW0VOVElUWV9CXSBsaWVndD8ifSx7InJldmlld19rZXkiOiJSMDIwNiIsImxhbmd1YWdlIjoiZGUiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiV2hhdCBpcyB0aGUgW0VOVElUWV9BXSBkaWZmZXJlbnQgZnJvbT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiV292b24gdW50ZXJzY2hlaWRldCBzaWNoIFtFTlRJVFlfQV0/In0seyJyZXZpZXdfa2V5IjoiUjAyMjIiLCJsYW5ndWFnZSI6ImRlIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoaWNoIGVudGl0eSBpcyBbRU5USVRZX0FdIGRpZmZlcmVudCBmcm9tPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJWb24gd2VsY2hlciBFbnRpdMOkdCB1bnRlcnNjaGVpZGV0IHNpY2ggW0VOVElUWV9BXT8ifSx7InJldmlld19rZXkiOiJSMDM0OSIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIGJlY29tZSBhIHJlbGF0aXZlIG9mIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpKTgpL7gpLDgpYDgpJYg4KSV4KWLIFtFTlRJVFlfQV0g4KSU4KSwIFtFTlRJVFlfQl0g4KSV4KWHIOCkrOClgOCkmiDgpKrgpL7gpLDgpL/gpLXgpL7gpLDgpL/gpJUg4KS44KSC4KSs4KSC4KSnIOCkuOCljeCkpeCkvuCkquCkv+CkpCDgpLngpYHgpIY/In0seyJyZXZpZXdfa2V5IjoiUjAzNjAiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBiZWNvbWUgdGhlIGNvdW50cnkgdGhhdCBbRU5USVRZX0JdIGlzIGluPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSWIOCkleCliyBbRU5USVRZX0FdIOCkteCkuSDgpKbgpYfgpLYg4KSs4KSo4KS+IOCknOCkv+CkuOCkruClh+CkgiBbRU5USVRZX0JdIOCkuOCljeCkpeCkv+CkpCDgpLngpYg/In0seyJyZXZpZXdfa2V5IjoiUjAzNzkiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6Ik9uIHdoYXQgZGF0ZSBkaWQgW0VOVElUWV9BXSBjZWFzZSB0byBiZSB0aGUgY291bnRyeSBvZiBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLgpJXgpL/gpLgg4KSk4KS+4KSw4KWA4KSWIOCkleCliyBbRU5USVRZX0FdLCBbRU5USVRZX0JdIOCkleCkviDgpKbgpYfgpLYg4KSo4KS54KWA4KSCIOCksOCkueCkvj8ifSx7InJldmlld19rZXkiOiJSMDQwNyIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiSW4gd2hhdCB5ZWFyIGRpZCBbRU5USVRZX0FdIGJlY29tZSB0aGUgc3BvdXNlIG9mIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpLXgpLDgpY3gpLcgW0VOVElUWV9BXSwgW0VOVElUWV9CXSDgpJXgpL4g4KSc4KWA4KS14KSo4KS44KS+4KSl4KWAIOCkrOCkqOCkvj8ifSx7InJldmlld19rZXkiOiJSMDQxNyIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdIHN0b3AgYmVpbmcgdGhlIGNhcGl0YWwgb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSV4KS/4KS4IOCkpOCkvuCksOClgOCkliDgpJXgpYsgW0VOVElUWV9BXSDgpKjgpYcgW0VOVElUWV9CXSDgpJXgpYAg4KSw4KS+4KSc4KSn4KS+4KSo4KWAIOCkueCli+CkqOCkviDgpKzgpILgpKYg4KSV4KS/4KSv4KS+PyJ9LHsicmV2aWV3X2tleSI6IlIwNDIxIiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJhdG9taWMiLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGF0IGltbWVkaWF0ZWx5IGZvbGxvd3MgW0VOVElUWV9BXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoiW0VOVElUWV9BXSDgpJXgpYcg4KSk4KWB4KSw4KSC4KSkIOCkrOCkvuCkpiDgpJXgpY3gpK/gpL4g4KSG4KSk4KS+IOCkueCliD8iLCJwcmlvcl9iYWNrdHJhbnNsYXRpb25fZW4iOiJXaGF0IGNvbWVzIGFmdGVyIFtFTlRJVFlfQV0/In0seyJyZXZpZXdfa2V5IjoiUjA0MzYiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoZW4gZGlkIFtFTlRJVFlfQV0gYmVjb21lIHBhcnQgb2YgdGhlIFtFTlRJVFlfQl0/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IuCkleCkv+CkuCDgpKTgpL7gpLDgpYDgpJYg4KSV4KWLIFtFTlRJVFlfQV0sIFtFTlRJVFlfQl0g4KSV4KS+IOCkueCkv+CkuOCljeCkuOCkviDgpKzgpKjgpL4/In0seyJyZXZpZXdfa2V5IjoiUjA0NDMiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoaWNoIHRvd24gaXMgW0VOVElUWV9BXSB0d2lubmVkIHdpdGg/IiwidHJhbnNsYXRlZF9xdWVzdGlvbiI6IltFTlRJVFlfQV0g4KSV4KS+IOCknOClgeCkoeCkvOCkteCkvuCkgSDgpLbgpLngpLAg4KSV4KWM4KSoLeCkuOCkviDgpLngpYg/IiwicHJpb3JfYmFja3RyYW5zbGF0aW9uX2VuIjoiV2hhdCBpcyB0aGUgdHdpbiBjaXR5IG9mIFtFTlRJVFlfQV0/In0seyJyZXZpZXdfa2V5IjoiUjA0NDkiLCJsYW5ndWFnZSI6ImhpIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSBzdG9wIHNlcnZpbmcgYXMgYSBib2FyZCBtZW1iZXIgb2YgW0VOVElUWV9CXT8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSV4KS/4KS4IOCkteCksOCljeCktyBbRU5USVRZX0FdIOCkqOClhyBbRU5USVRZX0JdIOCkleClhyDgpKzgpYvgpLDgpY3gpKEg4KS44KSm4KS44KWN4KSvIOCkleClhyDgpLDgpYLgpKog4KSu4KWH4KSCIOCkleCkvuCkriDgpJXgpLDgpKjgpL4g4KSs4KSC4KSmIOCkleCkv+Ckr+Ckvj8ifSx7InJldmlld19rZXkiOiJSMDQ2NCIsImxhbmd1YWdlIjoiaGkiLCJraW5kIjoiY29tcG9zaXRpb24iLCJzb3VyY2VfcXVlc3Rpb25fZW4iOiJXaGF0IGlzIGEgcGFydCBvZiB0aGUgbmV4dCBoaWdoZXIgcmFuayBhYm92ZSB0aGUgY2xhc3Mgb2Ygd2hpY2ggdGhlIFtFTlRJVFlfQV0gaXMgYSBzdWJjbGFzcz8iLCJ0cmFuc2xhdGVkX3F1ZXN0aW9uIjoi4KSc4KS/4KS4IOCkteCksOCljeCklyDgpJXgpYAg4KSJ4KSq4KS24KWN4KSw4KWH4KSj4KWAIFtFTlRJVFlfQV0g4KS54KWILCDgpIngpLjgpLjgpYcg4KSP4KSVIOCkuOCljeCkpOCksCDgpIrgpKrgpLAg4KSV4KWAIOCktuCljeCksOClh+Cko+ClgCDgpJXgpL4g4KSV4KWM4KSoLeCkuOCkviDgpK3gpL7gpJcg4KS54KWIPyJ9LHsicmV2aWV3X2tleSI6IlIwNTM3IiwibGFuZ3VhZ2UiOiJoaSIsImtpbmQiOiJjb21wb3NpdGlvbiIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IldoaWNoIG5vYmxlIHRpdGxlIGRvZXMgW0VOVElUWV9BXSBob2xkPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOCkleClhyDgpKrgpL7gpLgg4KSV4KWM4KSoLeCkuOClgCDgpJXgpYHgpLLgpYDgpKgg4KSJ4KSq4KS+4KSn4KS/IOCkueCliD8iLCJwcmlvcl9iYWNrdHJhbnNsYXRpb25fZW4iOiJXaGF0IGlzIHRoZSB0aXRsZSBvZiB0aGUgW0VOVElUWV9BXT8ifSx7InJldmlld19rZXkiOiJSMDU0OSIsImxhbmd1YWdlIjoidGgiLCJraW5kIjoiYXRvbWljIiwic291cmNlX3F1ZXN0aW9uX2VuIjoiT24gd2hhdCBkYXRlIGRpZCBbRU5USVRZX0FdJ3MgdGVudXJlIGFzIGRpcmVjdG9yIG9mIFtFTlRJVFlfQl0gZW5kPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiLguKfguLLguKPguLDguIHguLLguKPguJTguLPguKPguIfguJXguLPguYHguKvguJnguYjguIfguJzguLnguYnguK3guLPguJnguKfguKLguIHguLLguKPguILguK3guIcgW0VOVElUWV9BXSDguJfguLXguYggW0VOVElUWV9CXSDguKrguLTguYnguJnguKrguLjguJTguKXguIfguYDguKHguLfguYjguK3guKfguLHguJnguJfguLXguYjguYDguJfguYjguLLguYPguJQ/In0seyJyZXZpZXdfa2V5IjoiUjA2NzciLCJsYW5ndWFnZSI6InRoIiwia2luZCI6ImF0b21pYyIsInNvdXJjZV9xdWVzdGlvbl9lbiI6IkluIHdoYXQgeWVhciBkaWQgW0VOVElUWV9BXSByZWNlaXZlIHRoZSBbRU5USVRZX0JdPyIsInRyYW5zbGF0ZWRfcXVlc3Rpb24iOiJbRU5USVRZX0FdIOC5hOC4lOC5ieC4o+C4seC4miBbRU5USVRZX0JdIOC5g+C4meC4m+C4teC5g+C4lD8ifV0="
INPUT_SHA256 = "5c4044643e876db21ce5979c7377f7e5745d7a18fec5d21fda15fa3177049849"
PRIOR_OUTPUT_SHA256 = "774b1fa1e3a555df6495525910f452f301d1743a4c07d1b4f5c62e6b95fbab70"
OUT = Path("/kaggle/working/marker_safe_retry.json")

SYSTEM = (
    "You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
    "Decide whether the target question asks the same thing as the English source, preserving relation, "
    "direction, polarity, scope, tense/time, and answer type. Accept faithful paraphrases and contextual "
    "synonyms; do not require word-for-word equality. The English back-translation is diagnostic evidence, "
    "not a literal-match requirement. Judge target-language grammar and naturalness separately. Names and "
    "numbers are opaque placeholders. Do not answer questions or infer facts. If a material issue is uncertain, "
    "return false. Give one short reason of at most 12 words. Return exactly one JSON decision and no other text."
)
SCHEMA = {"type":"object", "properties":{"decisions":{"type":"array", "items":{
    "type":"object", "properties":{"key":{"type":"string"}, "semantic_pass":{"type":"boolean"},
    "fluency_pass":{"type":"boolean"}, "reason":{"type":"string"}},
    "required":["key","semantic_pass","fluency_pass","reason"], "additionalProperties":False},
    "minItems":1,"maxItems":1}}, "required":["decisions"], "additionalProperties":False}

def sha(data): return hashlib.sha256(data).hexdigest()

def checkpoint(payload):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT)

def alias_mask(text):
    counts = collections.Counter()
    marker_for_alias = {}
    def sub(match):
        marker = match.group(0)
        alias = ALIASES.get(marker)
        if alias is None:
            raise ValueError(f"No collision-safe alias configured for {marker}")
        if alias.casefold() in text.casefold().replace(marker.casefold(), ""):
            raise ValueError(f"Alias collision for {marker}")
        counts[alias] += 1
        marker_for_alias[alias] = marker
        return alias
    masked = MARKER.sub(sub, text)
    if not counts:
        raise ValueError("No supported protected entity placeholder in translation")
    return masked, marker_for_alias, counts

def restore_aliases(text, marker_for_alias, expected):
    found = collections.Counter({a: text.count(a) for a in marker_for_alias})
    if found != expected:
        raise ValueError(f"alias multiset changed: expected={dict(expected)}, found={dict(found)}")
    for alias, marker in marker_for_alias.items():
        text = text.replace(alias, marker)
    if any(alias in text for alias in marker_for_alias):
        raise ValueError("alias restoration incomplete")
    return text.strip()

def parse_decision(raw, expected_key):
    text = raw.strip()
    start = text.find("{")
    if start < 0: raise ValueError("no JSON object in response")
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj: raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj
    parsed, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(text[start:])
    if text[start + end:].strip(): raise ValueError("trailing text after JSON")
    if not isinstance(parsed, dict) or set(parsed) != {"decisions"}:
        raise ValueError("unexpected top-level JSON schema")
    decisions = parsed["decisions"]
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("expected exactly one decision")
    d = decisions[0]
    if (not isinstance(d, dict) or set(d) != {"key","semantic_pass","fluency_pass","reason"}
        or d.get("key") != expected_key or type(d.get("semantic_pass")) is not bool
        or type(d.get("fluency_pass")) is not bool or not isinstance(d.get("reason"), str)
        or not d["reason"].strip()):
        raise ValueError("decision key/schema mismatch")
    return d

def main():
    raw = base64.b64decode(RAW_B64)
    if sha(raw) != INPUT_SHA256: raise ValueError("embedded retry input hash mismatch")
    rows = json.loads(raw)
    if len(rows) != 18 or len({r["review_key"] for r in rows}) != 18:
        raise ValueError("expected exactly 18 unique blocked rows")
    if not torch.cuda.is_available(): raise RuntimeError("GPU unavailable; refusing a different-device run")
    runtime = {"status":"IN_PROGRESS", "mode":"third_candidate_marker_safe_18_row_retry",
        "input_sha256":INPUT_SHA256, "prior_output_sha256":PRIOR_OUTPUT_SHA256,
        "nllb_model_id":NLLB_ID, "nllb_model_revision":NLLB_REV,
        "qwen_model_id":QWEN_ID, "qwen_model_revision":QWEN_REV,
        "torch":torch.__version__, "transformers":transformers.__version__,
        "accelerate":accelerate.__version__, "bitsandbytes":bitsandbytes.__version__,
        "quantization":"4-bit NF4", "gpu":torch.cuda.get_device_name(0),
        "answers_or_splits_sent":False, "human_translation_reviewed":False,
        "release_approved":False, "training_eligible":False,
        "started_at_utc":datetime.now(timezone.utc).isoformat(), "records":[]}
    nllb_rows = [r for r in rows if not r.get("prior_backtranslation_en")]
    rec_by_key = {}
    for row in rows:
        rec = {"review_key":row["review_key"],"language":row["language"],"kind":row["kind"],
            "source_sha256":sha(row["source_question_en"].encode()),
            "translation_sha256":sha(row["translated_question"].encode()),
            "backtranslation_en":row.get("prior_backtranslation_en", ""),
            "backtranslation_error":None,"decision":None}
        if rec["backtranslation_en"]:
            rec["backtranslation_sha256"] = sha(rec["backtranslation_en"].encode())
        runtime["records"].append(rec)
        rec_by_key[row["review_key"]] = rec
    tok = NllbTokenizerFast.from_pretrained(NLLB_ID, revision=NLLB_REV)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID, revision=NLLB_REV,
        torch_dtype=torch.float16, device_map="auto")
    model.eval()
    for i,row in enumerate(nllb_rows,1):
        rec = rec_by_key[row["review_key"]]
        encoded = generated = None
        try:
            source_markers = collections.Counter(MARKER.findall(row["source_question_en"]))
            target_markers = collections.Counter(MARKER.findall(row["translated_question"]))
            if source_markers != target_markers: raise ValueError("source/target marker multiset mismatch")
            protected, marker_for_alias, expected = alias_mask(row["translated_question"])
            tok.src_lang = NLLB_CODES[row["language"]]
            encoded = tok(protected, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded,
                    forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                    max_new_tokens=128, do_sample=False)
            bt_raw = tok.batch_decode(generated, skip_special_tokens=True)[0]
            restored = restore_aliases(bt_raw, marker_for_alias, expected)
            if collections.Counter(MARKER.findall(restored)) != source_markers:
                raise ValueError("restored backtranslation marker multiset mismatch")
            if not restored: raise ValueError("empty backtranslation")
            rec["backtranslation_en"] = restored
            rec["backtranslation_sha256"] = sha(restored.encode())
            bt_by_key[row["review_key"]] = restored
        except Exception as exc:
            rec["backtranslation_error"] = f"{type(exc).__name__}: {exc}"
        checkpoint(runtime)
        print(f"Alias backtranslation {i}/{len(nllb_rows)}: {row['review_key']}", flush=True)
        del encoded, generated
    del model,tok
    gc.collect(); torch.cuda.empty_cache()

    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    qtok = AutoTokenizer.from_pretrained(QWEN_ID,revision=QWEN_REV)
    qmodel = AutoModelForCausalLM.from_pretrained(QWEN_ID,revision=QWEN_REV,
        torch_dtype=torch.float16,quantization_config=quant,device_map="auto")
    qmodel.eval()
    prefix = build_transformers_prefix_allowed_tokens_fn(qtok,JsonSchemaParser(SCHEMA))
    for i,row in enumerate(rows,1):
        rec=rec_by_key.get(row["review_key"])
        if rec.get("backtranslation_error"):
            rec["decision"]={"error":"backtranslation_failed"}
            continue
        payload={"target_language":LANG_NAMES[row["language"]],"items":[{
            "key":row["review_key"],"source_question_en":row["source_question_en"],
            "translated_question":row["translated_question"],"back_translation_en":rec["backtranslation_en"]}]}
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(payload,ensure_ascii=False)}]
        prompt=qtok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        encoded=qtok(prompt,return_tensors="pt").to(qmodel.device)
        last_error=None
        for attempt in range(2):
            try:
                with torch.inference_mode():
                    out=qmodel.generate(**encoded,max_new_tokens=160,do_sample=False,
                        pad_token_id=qtok.eos_token_id,prefix_allowed_tokens_fn=prefix)
                raw=qtok.decode(out[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
                decision=parse_decision(raw,row["review_key"])
                rec["raw_output"]=raw; rec["decision"]=decision; last_error=None; break
            except Exception as exc:
                last_error=f"{type(exc).__name__}: {exc}"
        if last_error:
            rec["decision"]={"error":last_error}
        checkpoint(runtime)
        print(f"Qwen retry {i}/18: {row['review_key']}",flush=True)
        del encoded
    counts=collections.Counter()
    for rec in runtime["records"]:
        d=rec.get("decision") or {}
        if rec.get("backtranslation_error") or "error" in d: counts["BLOCKED"]+=1
        elif d.get("semantic_pass") is True and d.get("fluency_pass") is True: counts["PASS"]+=1
        else: counts["FLAG"]+=1
    runtime.update({"status":"COMPLETE_DIAGNOSTIC_ONLY","counts":dict(counts),
        "completed_at_utc":datetime.now(timezone.utc).isoformat()})
    checkpoint(runtime)
    print(json.dumps({"status":runtime["status"],"counts":runtime["counts"],
        "rows":len(runtime["records"])},indent=2),flush=True)

if __name__=="__main__": main()
