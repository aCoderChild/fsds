#!/usr/bin/env python3
"""Retry only the 17 blocked diagnostic rows; never approves release."""
import base64
import gc
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.51.3", "accelerate==1.15.0",
    "lm-format-enforcer==0.11.1", "bitsandbytes==0.50.2",
], check=True)

import torch
import transformers
import accelerate
import bitsandbytes
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

INPUT_B64 = "__RETRY_INPUT_B64__"
INPUT_SHA256 = "__RETRY_INPUT_SHA256__"
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
OUT = Path("/kaggle/working/retry_output.json")
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NLLB_SOURCE_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
MARKER = re.compile(r"\[(ENTITY|NUMBER)_([A-Z]+)\]")
ALIAS = {"[ENTITY_A]": "Alice", "[ENTITY_B]": "Bob"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_input():
    raw = base64.b64decode(INPUT_B64, validate=True)
    if sha(raw) != INPUT_SHA256:
        raise ValueError("retry input SHA-256 mismatch")
    payload = json.loads(raw, object_pairs_hook=unique_object)
    if (payload.get("schema") != "synthworlds-targeted-gap-retry-v1"
            or payload.get("mode") != "AI_ONLY_DIAGNOSTIC_RETRY_NO_RELEASE_APPROVAL"
            or payload.get("human_translation_reviewed") is not False
            or payload.get("release_approved") is not False
            or payload.get("training_eligible") is not False
            or payload.get("answers_or_split_labels_included") is not False):
        raise ValueError("retry input scope/approval flags are invalid")
    rows = payload.get("rows")
    if (not isinstance(rows, list) or len(rows) != 17
            or len({row.get("review_key") for row in rows}) != 17):
        raise ValueError("expected exactly 17 unique blinded retry rows")
    for row in rows:
        if row["language"] not in LANGUAGE_NAMES or row["kind"] not in {"atomic", "composition"}:
            raise ValueError(f"invalid language/kind for {row.get('review_key')}")
        if sha(row["source_question_en"].encode()) != row["source_sha256"]:
            raise ValueError(f"source hash mismatch: {row['review_key']}")
        if sha(row["translated_question"].encode()) != row["translation_sha256"]:
            raise ValueError(f"candidate hash mismatch: {row['review_key']}")
        if row["retry_backtranslation"] not in {"ALIAS_NLLB", "REUSE_VERIFIED_PRIOR"}:
            raise ValueError(f"invalid retry mode: {row['review_key']}")
    return payload, rows


def mask_aliases(text):
    originals, expected = {}, Counter()
    def replace(match):
        marker = match.group(0)
        alias = ALIAS.get(marker)
        if alias is None:
            raise ValueError(f"no approved alias for placeholder {marker}")
        if alias.casefold() in text.casefold().replace(marker.casefold(), ""):
            raise ValueError(f"alias collision for placeholder {marker}")
        originals[alias] = marker
        expected[alias] += 1
        return alias
    masked = MARKER.sub(replace, text)
    if not expected:
        raise ValueError("candidate contains no protected placeholder")
    return masked, originals, expected


def restore_aliases(text, originals, expected):
    observed = Counter({alias: text.count(alias) for alias in originals})
    if observed != expected:
        raise ValueError(f"alias multiplicities changed: expected={dict(expected)}, observed={dict(observed)}")
    for alias, marker in originals.items():
        text = text.replace(alias, marker)
    if any(alias in text for alias in originals):
        raise ValueError("one or more aliases were not restored")
    return text.strip()


def schema():
    decision = {"type": "object", "properties": {
        "key": {"type": "string"}, "semantic_pass": {"type": "boolean"},
        "fluency_pass": {"type": "boolean"}, "reason": {"type": "string", "maxLength": 120}},
        "required": ["key", "semantic_pass", "fluency_pass", "reason"],
        "additionalProperties": False}
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": decision, "minItems": 1, "maxItems": 1}},
        "required": ["decisions"], "additionalProperties": False}


def judge(row, backtranslation, model, tokenizer, prefix_function, max_new_tokens):
    system = (
        "Adjudicate this synthetic QA question translation. Do not answer it or infer facts. "
        "Compare the English source, target text, and English back-translation. Accept faithful paraphrases. "
        "semantic_pass is false only for a material change to relation, direction, polarity, scope, time, "
        "or answer type. fluency_pass judges grammar and naturalness in the named target language. "
        "If uncertain about a material defect, return false. Give a brief reason of at most 12 words. "
        "Return one decision for the exact supplied key."
    )
    payload = {"target_language": LANGUAGE_NAMES[row["language"]], "items": [{
        "key": row["review_key"], "source_question_en": row["source_question_en"],
        "translated_question": row["translated_question"], "back_translation_en": backtranslation}]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id,
                                   prefix_allowed_tokens_fn=prefix_function)
    raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    start = raw.find("{")
    if start < 0:
        raise ValueError("judge output contains no JSON object")
    parsed, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(raw, start)
    if raw[end:].strip():
        raise ValueError("judge output contains trailing non-JSON text")
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("judge output has invalid decision coverage")
    decision = decisions[0]
    if (not isinstance(decision, dict) or decision.get("key") != row["review_key"]
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
            or len(decision["reason"]) > 120):
        raise ValueError("judge output violates the bounded decision schema or key")
    return raw, decision


def atomic_save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    payload, rows = load_input()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing a CPU fallback")
    runtime = {"status": "RUNNING_AI_ONLY_DIAGNOSTIC_RETRY", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "torch": torch.__version__, "transformers": transformers.__version__,
               "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__,
               "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "quantization": "NF4, double quantization, float16 compute",
               "human_translation_reviewed": False, "release_approved": False,
               "training_eligible": False, "answers_or_split_labels_included": False}
    records = []
    needs_nllb = any(r["retry_backtranslation"] == "ALIAS_NLLB" for r in rows)
    bt_tokenizer = bt_model = None
    if needs_nllb:
        bt_tokenizer = NllbTokenizerFast.from_pretrained(NLLB_ID, revision=NLLB_REVISION)
        bt_model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_ID, revision=NLLB_REVISION, torch_dtype=torch.float16, device_map="auto")
        bt_model.eval()
    record_by_key = {}
    for index, row in enumerate(rows, 1):
        record = {"review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
                  "source_sha256": row["source_sha256"], "translation_sha256": row["translation_sha256"],
                  "retry_backtranslation": row["retry_backtranslation"], "backtranslation_en": "",
                  "backtranslation_raw": "", "alias_counts_expected": {}, "alias_counts_observed": {},
                  "backtranslation_sha256": None, "backtranslation_error": None,
                  "decision": None, "judge_attempts": [], "raw_output": None}
        try:
            if row["retry_backtranslation"] == "REUSE_VERIFIED_PRIOR":
                text = row["prior_backtranslation_en"]
                if sha(text.encode()) != row["prior_backtranslation_sha256"]:
                    raise ValueError("prior backtranslation hash mismatch")
                record["backtranslation_en"] = text
                record["backtranslation_sha256"] = sha(text.encode())
            else:
                source_markers = Counter(MARKER.findall(row["source_question_en"]))
                target_markers = Counter(MARKER.findall(row["translated_question"]))
                if source_markers != target_markers:
                    raise ValueError("source/candidate placeholder multiplicities differ")
                protected, originals, expected = mask_aliases(row["translated_question"])
                bt_tokenizer.src_lang = NLLB_SOURCE_CODES[row["language"]]
                encoded = bt_tokenizer(protected, return_tensors="pt", truncation=True, max_length=256).to(bt_model.device)
                with torch.inference_mode():
                    generated = bt_model.generate(**encoded,
                        forced_bos_token_id=bt_tokenizer.convert_tokens_to_ids("eng_Latn"),
                        max_new_tokens=128, do_sample=False)
                translated = bt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                record["backtranslation_raw"] = translated
                record["alias_counts_expected"] = dict(expected)
                record["alias_counts_observed"] = {alias: translated.count(alias) for alias in originals}
                restored = restore_aliases(translated, originals, expected)
                if Counter(MARKER.findall(restored)) != source_markers:
                    raise ValueError("restored marker multiset differs from source")
                record["backtranslation_en"] = restored
                record["backtranslation_sha256"] = sha(restored.encode())
        except Exception as exc:
            record["backtranslation_error"] = f"{type(exc).__name__}: {exc}"
        record_by_key[row["review_key"]] = record
        records.append(record)
        atomic_save(OUT, {"status": "IN_PROGRESS_ALIAS_BACKTRANSLATION", "input_sha256": INPUT_SHA256,
            "input_payload": payload, "records": records, **runtime})
        print(f"Backtranslation {index}/{len(rows)}", flush=True)
    if bt_model is not None:
        del bt_model, bt_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
        torch_dtype=torch.float16, quantization_config=quantization, device_map="auto")
    model.eval()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema()))
    for index, row in enumerate(rows, 1):
        record = record_by_key[row["review_key"]]
        if record["backtranslation_error"]:
            record["decision"] = {"error": "backtranslation_failed; judgment withheld"}
        else:
            for max_new_tokens in (128, 192, 256):
                try:
                    raw, decision = judge(row, record["backtranslation_en"], model, tokenizer,
                                          prefix_function, max_new_tokens)
                    record["raw_output"] = raw
                    record["decision"] = decision
                    record["judge_attempts"].append({"max_new_tokens": max_new_tokens, "status": "VALID"})
                    break
                except Exception as exc:
                    record["judge_attempts"].append({"max_new_tokens": max_new_tokens,
                        "status": "INVALID", "error": f"{type(exc).__name__}: {exc}"})
            if record["decision"] is None:
                record["decision"] = {"error": "judge_invalid_after_three_bounded_retries"}
        atomic_save(OUT, {"status": "IN_PROGRESS_QWEN_RETRY", "input_sha256": INPUT_SHA256,
            "input_payload": payload, "records": records, **runtime})
        print(f"Judge {index}/{len(rows)}", flush=True)

    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["reviewed_rows"] = sum("error" not in (r.get("decision") or {}) for r in records)
    runtime["blocked_rows"] = len(records) - runtime["reviewed_rows"]
    runtime["status"] = "AI_ONLY_DIAGNOSTIC_RETRY_COMPLETE" if runtime["blocked_rows"] == 0 else "AI_ONLY_DIAGNOSTIC_RETRY_HAS_BLOCKED_ROWS"
    result = {"status": runtime["status"], "input_sha256": INPUT_SHA256,
        "retry_input_payload": payload, "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
        "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
        "records": records, **runtime}
    atomic_save(OUT, result)
    print(json.dumps({"status": runtime["status"], "rows": len(rows),
        "reviewed_rows": runtime["reviewed_rows"], "blocked_rows": runtime["blocked_rows"],
        "gpu": runtime["gpu"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
