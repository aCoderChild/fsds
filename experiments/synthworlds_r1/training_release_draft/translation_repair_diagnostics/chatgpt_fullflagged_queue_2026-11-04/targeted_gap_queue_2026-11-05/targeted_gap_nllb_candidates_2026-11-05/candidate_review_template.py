#!/usr/bin/env python3
"""Translate unresolved source questions with NLLB, then screen with pinned Qwen."""
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

INPUT_B64 = "__INPUT_B64__"
INPUT_SHA256 = "__INPUT_SHA256__"
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
OUTPUT = Path("/kaggle/working/candidate_review.json")
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NLLB_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
MARKER = re.compile(r"\[(ENTITY|NUMBER)_([A-Z]+)\]")
ALIASES = {"[ENTITY_A]": "Alice", "[ENTITY_B]": "Bob"}


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
        raise ValueError("candidate input SHA-256 mismatch")
    payload = json.loads(raw, object_pairs_hook=unique_object)
    if (payload.get("schema") != "synthworlds-nllb-alternative-translation-v1"
            or payload.get("mode") != "AI_ONLY_DIAGNOSTIC_CANDIDATE_NO_RELEASE_APPROVAL"
            or payload.get("answers_or_split_labels_included") is not False
            or payload.get("human_translation_reviewed") is not False
            or payload.get("release_approved") is not False
            or payload.get("training_eligible") is not False):
        raise ValueError("candidate input scope or approval flags invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 148 or len({r.get("review_key") for r in rows}) != 148:
        raise ValueError("expected 148 unique unresolved answer-blind question rows")
    for row in rows:
        if set(row) != {"review_key", "language", "kind", "source_question_en", "source_sha256"}:
            raise ValueError(f"unexpected input fields for {row.get('review_key')}")
        if row["language"] not in LANGUAGE_NAMES or row["kind"] not in {"atomic", "composition"}:
            raise ValueError(f"invalid language/kind for {row['review_key']}")
        if sha(row["source_question_en"].encode()) != row["source_sha256"]:
            raise ValueError(f"source hash mismatch for {row['review_key']}")
    return payload, rows


def mask_aliases(text):
    expected, originals = Counter(), {}
    def replace(match):
        marker = match.group(0)
        alias = ALIASES.get(marker)
        if alias is None:
            raise ValueError(f"no alias defined for {marker}")
        if alias.casefold() in text.casefold().replace(marker.casefold(), ""):
            raise ValueError(f"natural-name alias collision for {marker}")
        expected[alias] += 1
        originals[alias] = marker
        return alias
    masked = MARKER.sub(replace, text)
    if not expected:
        raise ValueError("question has no protected entity placeholder")
    return masked, originals, expected


def restore_aliases(text, originals, expected):
    observed = Counter({alias: text.count(alias) for alias in originals})
    if observed != expected:
        raise ValueError(f"placeholder alias counts changed: expected={dict(expected)}, observed={dict(observed)}")
    for alias, marker in originals.items():
        text = text.replace(alias, marker)
    if any(alias in text for alias in originals):
        raise ValueError("natural alias was not restored")
    return text.strip()


def candidate_schema():
    decision = {"type": "object", "properties": {
        "key": {"type": "string"}, "semantic_pass": {"type": "boolean"},
        "fluency_pass": {"type": "boolean"}, "reason": {"type": "string", "maxLength": 120}},
        "required": ["key", "semantic_pass", "fluency_pass", "reason"],
        "additionalProperties": False}
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": decision, "minItems": 1, "maxItems": 1}},
        "required": ["decisions"], "additionalProperties": False}


def judge(row, candidate, backtranslation, model, tokenizer, prefix_function, max_new_tokens):
    system = (
        "Adjudicate this synthetic QA translation. Do not answer the question or infer facts. "
        "Compare the English source, target text, and English back-translation. Accept faithful paraphrases. "
        "semantic_pass is false only for a material change to relation, direction, polarity, scope, time, "
        "or answer type. fluency_pass judges grammar and naturalness in the named target language. "
        "If uncertain about a material defect, return false. Give a brief reason of at most 12 words. "
        "Return exactly one decision for the supplied key."
    )
    data = {"target_language": LANGUAGE_NAMES[row["language"]], "items": [{
        "key": row["review_key"], "source_question_en": row["source_question_en"],
        "translated_question": candidate, "back_translation_en": backtranslation}]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, top_p=1.0, top_k=50, pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_function)
    raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    start = raw.find("{")
    if start < 0:
        raise ValueError("Qwen output has no JSON object")
    parsed, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(raw, start)
    if raw[end:].strip():
        raise ValueError("Qwen output has trailing non-JSON text")
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("Qwen output has invalid decision coverage")
    decision = decisions[0]
    if (not isinstance(decision, dict) or decision.get("key") != row["review_key"]
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str)
            or not decision["reason"].strip() or len(decision["reason"]) > 120):
        raise ValueError("Qwen decision violates schema or key")
    return raw, decision


def atomic_save(obj):
    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUTPUT)


def main():
    payload, rows = load_input()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA unavailable; refusing CPU fallback")
    gpus = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(gpus) < 2 or any("T4" not in gpu for gpu in gpus):
        raise RuntimeError(f"expected Kaggle T4 x2, found {gpus}")
    runtime = {"status": "RUNNING_NLLB_CANDIDATE_TRANSLATION", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__,
        "gpu": gpus, "quantization": "Qwen NF4 double quantization, float16 compute",
        "human_translation_reviewed": False, "answers_or_split_labels_included": False,
        "release_approved": False, "training_eligible": False}
    records = []
    tokenizer = NllbTokenizerFast.from_pretrained(NLLB_ID, revision=NLLB_REVISION)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID,
        revision=NLLB_REVISION, torch_dtype=torch.float16, device_map="auto")
    translation_model.eval()
    for index, row in enumerate(rows, 1):
        record = {"review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
            "source_sha256": row["source_sha256"], "candidate_translation": "", "candidate_sha256": None,
            "candidate_error": None, "backtranslation_en": "", "backtranslation_sha256": None,
            "backtranslation_error": None, "decision": None, "judge_attempts": [], "raw_output": None}
        try:
            protected_source, source_aliases, source_alias_counts = mask_aliases(row["source_question_en"])
            tokenizer.src_lang = "eng_Latn"
            encoded = tokenizer(protected_source, return_tensors="pt", truncation=True, max_length=256).to(translation_model.device)
            with torch.inference_mode():
                generated = translation_model.generate(**encoded,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(NLLB_CODES[row["language"]]),
                    max_new_tokens=160, do_sample=False)
            raw_candidate = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            candidate = restore_aliases(raw_candidate, source_aliases, source_alias_counts)
            if Counter(MARKER.findall(candidate)) != Counter(MARKER.findall(row["source_question_en"])):
                raise ValueError("candidate marker multiset differs from source")
            record["candidate_translation"] = candidate
            record["candidate_sha256"] = sha(candidate.encode())

            protected_target, target_aliases, target_alias_counts = mask_aliases(candidate)
            tokenizer.src_lang = NLLB_CODES[row["language"]]
            encoded = tokenizer(protected_target, return_tensors="pt", truncation=True, max_length=256).to(translation_model.device)
            with torch.inference_mode():
                generated = translation_model.generate(**encoded,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                    max_new_tokens=160, do_sample=False)
            raw_backtranslation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            backtranslation = restore_aliases(raw_backtranslation, target_aliases, target_alias_counts)
            if Counter(MARKER.findall(backtranslation)) != Counter(MARKER.findall(row["source_question_en"])):
                raise ValueError("back-translation marker multiset differs from source")
            record["backtranslation_en"] = backtranslation
            record["backtranslation_sha256"] = sha(backtranslation.encode())
        except Exception as exc:
            if not record["candidate_translation"]:
                record["candidate_error"] = f"{type(exc).__name__}: {exc}"
            else:
                record["backtranslation_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        atomic_save({"status": "IN_PROGRESS_NLLB_CANDIDATE_TRANSLATION", "input_sha256": INPUT_SHA256,
            "input_payload": payload, "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
            "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION, "records": records, **runtime})
        print(f"Translated {index}/{len(rows)}", flush=True)

    del translation_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
        torch_dtype=torch.float16, quantization_config=quantization, device_map="auto")
    qwen_model.eval()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(qwen_tokenizer, JsonSchemaParser(candidate_schema()))
    record_by_key = {record["review_key"]: record for record in records}
    for index, row in enumerate(rows, 1):
        record = record_by_key[row["review_key"]]
        if record["candidate_error"]:
            record["decision"] = {"error": "candidate_translation_failed; judgment withheld"}
        elif record["backtranslation_error"]:
            record["decision"] = {"error": "backtranslation_failed; judgment withheld"}
        else:
            for max_new_tokens in (128, 192, 256):
                try:
                    raw, decision = judge(row, record["candidate_translation"], record["backtranslation_en"],
                        qwen_model, qwen_tokenizer, prefix_function, max_new_tokens)
                    record["raw_output"], record["decision"] = raw, decision
                    record["judge_attempts"].append({"max_new_tokens": max_new_tokens, "status": "VALID"})
                    break
                except Exception as exc:
                    record["judge_attempts"].append({"max_new_tokens": max_new_tokens,
                        "status": "INVALID", "error": f"{type(exc).__name__}: {exc}"})
            if record["decision"] is None:
                record["decision"] = {"error": "judge_invalid_after_three_bounded_retries"}
        atomic_save({"status": "IN_PROGRESS_QWEN_CANDIDATE_SCREEN", "input_sha256": INPUT_SHA256,
            "input_payload": payload, "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
            "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION, "records": records, **runtime})
        print(f"Reviewed {index}/{len(rows)}", flush=True)

    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["reviewed_rows"] = sum("error" not in (record.get("decision") or {}) for record in records)
    runtime["blocked_rows"] = len(records) - runtime["reviewed_rows"]
    runtime["candidate_rows"] = sum(bool(record["candidate_translation"]) for record in records)
    runtime["status"] = "AI_ONLY_NLLB_CANDIDATE_REVIEW_COMPLETE" if runtime["blocked_rows"] == 0 else "AI_ONLY_NLLB_CANDIDATE_REVIEW_HAS_BLOCKS"
    atomic_save({"status": runtime["status"], "input_sha256": INPUT_SHA256,
        "input_payload": payload, "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
        "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
        "records": records, **runtime})
    print(json.dumps({"status": runtime["status"], "rows": len(rows),
        "candidate_rows": runtime["candidate_rows"], "reviewed_rows": runtime["reviewed_rows"],
        "blocked_rows": runtime["blocked_rows"], "gpu": runtime["gpu"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
