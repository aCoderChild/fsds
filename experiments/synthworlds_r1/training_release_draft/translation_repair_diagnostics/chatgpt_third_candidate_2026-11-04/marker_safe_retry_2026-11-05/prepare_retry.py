#!/usr/bin/env python3
"""Prepare an answer-blind, source-bound retry for the 18 blocked rows only."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
THIRD = HERE.parent
KERNEL = HERE / "kaggle_kernel"
SOURCE = THIRD / "kaggle_review/review_input.json"
PRIOR = THIRD / "kaggle_review/remote_output/repair_review.json"
EXPECTED_BT_FAILURES = 15
EXPECTED_JUDGE_FAILURES = 3

TEMPLATE = r'''#!/usr/bin/env python3
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
RAW_B64 = "__RAW_B64__"
INPUT_SHA256 = "__INPUT_SHA256__"
PRIOR_OUTPUT_SHA256 = "__PRIOR_OUTPUT_SHA256__"
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
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source_rows = read_json(SOURCE)
    prior = read_json(PRIOR)
    if len(source_rows) != 187 or len(prior.get("records", [])) != 187:
        raise ValueError("The frozen 187-row source/review files are incomplete")
    source_by_key = {r["review_key"]: r for r in source_rows}
    prior_by_key = {r["review_key"]: r for r in prior["records"]}
    if list(source_by_key) != [r["review_key"] for r in prior["records"]]:
        raise ValueError("Prior output coverage/order differs from source input")
    rows = []
    bt_fail = judge_fail = 0
    for source in source_rows:
        prior_row = prior_by_key[source["review_key"]]
        bt_error = prior_row.get("backtranslation_error")
        judge_error = (prior_row.get("decision") or {}).get("error")
        if not bt_error and not judge_error:
            continue
        row = {k: source[k] for k in (
            "review_key", "language", "kind", "source_question_en", "translated_question")}
        if bt_error:
            if "placeholder" not in bt_error.lower():
                raise ValueError(f"Unexpected non-placeholder backtranslation failure: {row['review_key']}")
            bt_fail += 1
        elif judge_error:
            if not prior_row.get("backtranslation_en"):
                raise ValueError(f"Judge-failure row has no valid prior backtranslation: {row['review_key']}")
            row["prior_backtranslation_en"] = prior_row["backtranslation_en"]
            judge_fail += 1
        rows.append(row)
    if (bt_fail, judge_fail) != (EXPECTED_BT_FAILURES, EXPECTED_JUDGE_FAILURES) or len(rows) != 18:
        raise ValueError(f"Expected 15 placeholder and 3 judge failures; found {bt_fail}/{judge_fail}")
    for row in rows:
        import re
        marker = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
        if collections_counter := __import__("collections").Counter(marker.findall(row["source_question_en"])):
            if collections_counter != __import__("collections").Counter(marker.findall(row["translated_question"])):
                raise ValueError(f"Source/candidate marker mismatch: {row['review_key']}")
        elif marker.search(row["translated_question"]):
            raise ValueError(f"Unexpected marker in candidate: {row['review_key']}")
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if KERNEL.exists():
        raise FileExistsError(f"Refusing to overwrite {KERNEL}")
    KERNEL.mkdir(parents=True)
    script = (TEMPLATE.replace("__RAW_B64__", base64.b64encode(raw).decode("ascii"))
        .replace("__INPUT_SHA256__", sha(raw))
        .replace("__PRIOR_OUTPUT_SHA256__", sha(PRIOR.read_bytes())))
    ast.parse(script)
    if "__RAW_B64__" in script or "__INPUT_SHA256__" in script or "__PRIOR_OUTPUT_SHA256__" in script:
        raise ValueError("Template substitutions incomplete")
    (KERNEL / "retry.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/sw-third-marker-retry-2026-11-05",
        "title": "sw-third-marker-retry-2026-11-05",
        "code_file": "retry.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    readback = ast.parse(script)
    payload = next(ast.literal_eval(n.value) for n in readback.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name) and t.id == "RAW_B64")
    receipt = {
        "status": "PREPARED_18_ROW_ALIAS_SAFE_RETRY_NOT_APPROVAL", "rows": len(rows),
        "alias_backtranslation_rows": bt_fail, "qwen_format_retry_rows": judge_fail,
        "input_sha256": sha(raw), "prior_output_sha256": sha(PRIOR.read_bytes()),
        "script_sha256": sha((KERNEL / "retry.py").read_bytes()),
        "decoded_row_count": len(json.loads(base64.b64decode(payload))),
        "models": {"backtranslation": "facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d",
                   "judge": "Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"},
        "answers_or_splits_sent": False, "human_translation_reviewed": False,
        "release_approved": False, "training_eligible": False,
        "limitations": ["Alias preservation is not semantic validation.",
                        "Qwen decisions remain AI-only diagnostics, not human-gold judgments.",
                        "No source corpus or 169 already reviewed rows are changed."]}
    (HERE / "build_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "review_input.json").write_bytes(raw + b"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
