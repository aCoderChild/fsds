#!/usr/bin/env python3
"""Translate text spans around protected markers, then screen with pinned Qwen."""
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
    if (payload.get("schema") != "synthworlds-nllb-segmented-retry-v1"
            or payload.get("mode") != "AI_ONLY_DIAGNOSTIC_SEGMENTED_RETRY_NO_RELEASE_APPROVAL"
            or payload.get("answers_or_split_labels_included") is not False
            or payload.get("human_translation_reviewed") is not False
            or payload.get("release_approved") is not False
            or payload.get("training_eligible") is not False):
        raise ValueError("candidate input scope or approval flags invalid")
    rows = payload.get("rows")
    expected_rows = payload.get("expected_rows")
    if (type(expected_rows) is not int or expected_rows < 1 or not isinstance(rows, list)
            or len(rows) != expected_rows or len({r.get("review_key") for r in rows}) != expected_rows):
        raise ValueError("segmented retry row count/uniqueness does not match its manifest")
    for row in rows:
        if set(row) != {"review_key", "language", "kind", "source_question_en", "source_sha256"}:
            raise ValueError(f"unexpected input fields for {row.get('review_key')}")
        if row["language"] not in LANGUAGE_NAMES or row["kind"] not in {"atomic", "composition"}:
            raise ValueError(f"invalid language/kind for {row['review_key']}")
        if sha(row["source_question_en"].encode()) != row["source_sha256"]:
            raise ValueError(f"source hash mismatch for {row['review_key']}")
    return payload, rows


def translate_marker_spans(text, translate_plain):
    """Translate non-marker spans while copying every source marker verbatim."""
    matches = list(MARKER.finditer(text))
    if not matches:
        raise ValueError("question has no protected entity placeholder")

    def translate_fragment(fragment):
        stripped = fragment.strip()
        if not stripped:
            return fragment
        leading = fragment[:len(fragment) - len(fragment.lstrip())]
        trailing = fragment[len(fragment.rstrip()):]
        translated = translate_plain(stripped).strip()
        if not translated:
            raise ValueError("NLLB returned an empty translated text span")
        return leading + translated + trailing

    pieces = []
    cursor = 0
    for match in matches:
        pieces.append(translate_fragment(text[cursor:match.start()]))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(translate_fragment(text[cursor:]))
    return "".join(pieces).strip()


def translate_plain_text(text, source_code, target_code, tokenizer, model):
    tokenizer.src_lang = source_code
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
            max_new_tokens=160,
            do_sample=False,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def translate_with_markers(text, source_code, target_code, tokenizer, model):
    return translate_marker_spans(
        text,
        lambda fragment: translate_plain_text(
            fragment, source_code, target_code, tokenizer, model
        ),
    )


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
            pad_token_id=tokenizer.eos_token_id,
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
            candidate = translate_with_markers(
                row["source_question_en"], "eng_Latn", NLLB_CODES[row["language"]],
                tokenizer, translation_model,
            )
            if Counter(MARKER.findall(candidate)) != Counter(MARKER.findall(row["source_question_en"])):
                raise ValueError("candidate marker multiset differs from source")
            record["candidate_translation"] = candidate
            record["candidate_sha256"] = sha(candidate.encode())

            backtranslation = translate_with_markers(
                candidate, NLLB_CODES[row["language"]], "eng_Latn",
                tokenizer, translation_model,
            )
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
        atomic_save({"status": "IN_PROGRESS_NLLB_SEGMENTED_RETRY_TRANSLATION", "input_sha256": INPUT_SHA256,
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
        atomic_save({"status": "IN_PROGRESS_QWEN_SEGMENTED_RETRY_SCREEN", "input_sha256": INPUT_SHA256,
            "input_payload": payload, "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
            "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION, "records": records, **runtime})
        print(f"Reviewed {index}/{len(rows)}", flush=True)

    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["reviewed_rows"] = sum("error" not in (record.get("decision") or {}) for record in records)
    runtime["blocked_rows"] = len(records) - runtime["reviewed_rows"]
    runtime["candidate_rows"] = sum(bool(record["candidate_translation"]) for record in records)
    runtime["status"] = "AI_ONLY_NLLB_SEGMENTED_RETRY_COMPLETE" if runtime["blocked_rows"] == 0 else "AI_ONLY_NLLB_SEGMENTED_RETRY_HAS_BLOCKS"
    atomic_save({"status": runtime["status"], "input_sha256": INPUT_SHA256,
        "input_payload": payload, "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
        "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
        "records": records, **runtime})
    print(json.dumps({"status": runtime["status"], "rows": len(rows),
        "candidate_rows": runtime["candidate_rows"], "reviewed_rows": runtime["reviewed_rows"],
        "blocked_rows": runtime["blocked_rows"], "gpu": runtime["gpu"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
