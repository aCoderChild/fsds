#!/usr/bin/env python3
"""AI-only round-trip and semantic/fluency screen for masked repair questions."""
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

MODE = "qwen3_translation_pilot_qwen14_roundtrip_screen_16"
INPUT_SHA256 = "e2b120430cda89f777daacdd991270b5c97607bbd4781f21d929f56880b4a1fa"
RAW_INPUT = '[{"review_key":"R0008","language":"fr","kind":"atomic","source_question_en":"Who is a doctoral student of [ENTITY_A]?","translated_question":"Qui est un étudiant en doctorat de [ENTITY_A] ?"},{"review_key":"R0011","language":"fr","kind":"atomic","source_question_en":"On what date did [ENTITY_A] become a relative of [ENTITY_B]?","translated_question":"Quelle est la date à laquelle [ENTITY_A] est devenu un parent de [ENTITY_B] ?"},{"review_key":"R0123","language":"fr","kind":"composition","source_question_en":"Where was the [ENTITY_A] formed?","translated_question":"Où a été formé le [ENTITY_A] ?"},{"review_key":"R0125","language":"fr","kind":"composition","source_question_en":"Which administrative territorial entity does [ENTITY_A] contain?","translated_question":"Quelle entité territoriale administrative [ENTITY_A] contient-elle ?"},{"review_key":"R0177","language":"de","kind":"atomic","source_question_en":"What does [ENTITY_A] replace?","translated_question":"Was ersetzt [ENTITY_A]?"},{"review_key":"R0179","language":"de","kind":"atomic","source_question_en":"On what date did [ENTITY_A] become the country that [ENTITY_B] is in?","translated_question":"An welchem Datum wurde [ENTITY_A] zum Land, in dem [ENTITY_B] ist?"},{"review_key":"R0277","language":"de","kind":"composition","source_question_en":"Who is the father of the unmarried partner of the spouse of [ENTITY_A]?","translated_question":"Wer ist der Vater des unverheirateten Partners des Ehegatten von [ENTITY_A]?"},{"review_key":"R0282","language":"de","kind":"composition","source_question_en":"Which administrative territorial entity does [ENTITY_A] contain?","translated_question":"Welche verwaltende territoriale Einheit enthält [ENTITY_A]?"},{"review_key":"R0345","language":"hi","kind":"atomic","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"[ENTITY_A] के साथ किसकी सीमा साझा है?"},{"review_key":"R0347","language":"hi","kind":"atomic","source_question_en":"What is [ENTITY_A]\'s field of work?","translated_question":"[ENTITY_A] के काम का क्षेत्र क्या है?"},{"review_key":"R0464","language":"hi","kind":"composition","source_question_en":"What is a part of the next higher rank above the class of which the [ENTITY_A] is a subclass?","translated_question":"[ENTITY_A] के उप-वर्ग के बराबर वर्ग के अगले उच्चतर रैंक का कौन-सा भाग है?"},{"review_key":"R0465","language":"hi","kind":"composition","source_question_en":"What is a part of the [ENTITY_A]?","translated_question":"[ENTITY_A] का एक भाग क्या है?"},{"review_key":"R0547","language":"th","kind":"atomic","source_question_en":"On what date was [ENTITY_A] incorporated?","translated_question":"วันที่ [ENTITY_A] ได้รับการจดทะเบียนคือเมื่อใด"},{"review_key":"R0549","language":"th","kind":"atomic","source_question_en":"On what date did [ENTITY_A]\'s tenure as director of [ENTITY_B] end?","translated_question":"[ENTITY_A] ดำรงตำแหน่งผู้อำนวยการของ [ENTITY_B] จนถึงวันที่ใด?"},{"review_key":"R0701","language":"th","kind":"composition","source_question_en":"What is a part of the next higher rank above the class of which the [ENTITY_A] is a subclass?","translated_question":"สิ่งใดเป็นส่วนหนึ่งของลำดับชั้นที่สูงกว่าของชั้นที่ [ENTITY_A] เป็นsubclass ของชั้นนั้น?"},{"review_key":"R0702","language":"th","kind":"composition","source_question_en":"Of what class is the [ENTITY_A] a subclass?","translated_question":"[ENTITY_A] เป็น subclass ของ class ใด"}]'
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NLLB_SOURCE_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
PLACEHOLDER = re.compile(r"\[(ENTITY|NUMBER)_([A-Z]+)\]")
OUT = Path("/kaggle/working/repair_review.json")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def save(result):
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUT)


def mask_placeholders(text):
    counts = Counter()
    originals = {}
    def replace(match):
        token = f"ZXQ{match.group(1)}{match.group(2)}QXZ"
        counts[token] += 1
        originals[token] = match.group(0)
        return token
    return PLACEHOLDER.sub(replace, text), originals, counts


def restore_placeholders(text, originals, expected):
    found = Counter({token: text.count(token) for token in originals})
    if found != expected:
        raise ValueError(f"backtranslation changed opaque placeholders: expected={dict(expected)}, found={dict(found)}")
    for token, original in originals.items():
        text = text.replace(token, original)
    if any(token in text for token in originals):
        raise ValueError("unrestored opaque marker")
    return text.strip()


def checkpoint(status, rows, records, runtime):
    save({"status": status, "mode": MODE, "input_sha256": INPUT_SHA256,
          "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
          "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
          "records": records, **runtime})


def make_qwen_schema():
    item = {"type": "object", "properties": {
        "key": {"type": "string"}, "semantic_pass": {"type": "boolean"},
        "fluency_pass": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["key", "semantic_pass", "fluency_pass", "reason"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": item, "minItems": 1, "maxItems": 1}},
        "required": ["decisions"], "additionalProperties": False}


def judge_one(row, backtranslation, model, tokenizer, prefix_function):
    system = (
        "You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
        "For semantic_pass, decide whether the target-language question asks the same thing as the English "
        "source: preserve relation, direction, polarity, scope, tense/time, and answer type. Accept faithful "
        "paraphrases, inflection changes, and contextual synonyms. Do not reject merely because a word has "
        "another dictionary sense; judge the translated question in context. Use the English back-translation "
        "as independent diagnostic evidence, not as a requirement for word-for-word equality. For fluency_pass, "
        "judge only whether the translated question is grammatical and natural in the named target language. "
        "Names/numbers are opaque placeholders. Do not answer the questions or infer their facts. If a material "
        "meaning change or substantial fluency problem is uncertain, return false. Give one concise reason. "
        "Return exactly one decision for the supplied key."
    )
    payload = {"target_language": LANGUAGE_NAMES[row["language"]], "items": [{
        "key": row["review_key"], "source_question_en": row["source_question_en"],
        "translated_question": row["translated_question"], "back_translation_en": backtranslation}]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=512, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id,
                                   prefix_allowed_tokens_fn=prefix_function)
    raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\{.*\}", raw, re.S)
    parsed = json.loads(match.group(0), object_pairs_hook=unique_object) if match else None
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("Qwen returned invalid decisions coverage")
    decision = decisions[0]
    if (not isinstance(decision, dict) or decision.get("key") != row["review_key"]
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
        raise ValueError("Qwen returned an invalid decision schema or key")
    return raw, decision


def main():
    rows = json.loads(RAW_INPUT)
    if sha(RAW_INPUT) != INPUT_SHA256:
        raise ValueError("embedded input hash mismatch")
    if not rows or len({r.get("review_key") for r in rows}) != len(rows):
        raise ValueError("input has no rows or duplicate review keys")
    for row in rows:
        if (set(row) != {"review_key", "language", "kind", "source_question_en", "translated_question"}
                or row["language"] not in LANGUAGE_NAMES or row["kind"] not in {"atomic", "composition"}
                or not row["source_question_en"].strip() or not row["translated_question"].strip()):
            raise ValueError(f"invalid blinded review input: {row.get('review_key')}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing a slow or different-device run")

    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__,
               "quantization": "4-bit NF4", "gpu": torch.cuda.get_device_name(0),
               "human_translation_reviewed": False, "answers_or_splits_sent": False,
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
    records = []
    bt_tokenizer = NllbTokenizerFast.from_pretrained(NLLB_ID, revision=NLLB_REVISION)
    bt_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_ID, revision=NLLB_REVISION, torch_dtype=torch.float16, device_map="auto")
    bt_model.eval()
    encoded = generated = translated = None
    for index, row in enumerate(rows, 1):
        record = {"review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
                  "source_sha256": sha(row["source_question_en"]),
                  "translation_sha256": sha(row["translated_question"]),
                  "backtranslation_en": "", "backtranslation_error": None, "decision": None, "raw_output": None}
        try:
            protected, originals, expected = mask_placeholders(row["translated_question"])
            bt_tokenizer.src_lang = NLLB_SOURCE_CODES[row["language"]]
            encoded = bt_tokenizer(protected, return_tensors="pt", truncation=True, max_length=256).to(bt_model.device)
            with torch.inference_mode():
                generated = bt_model.generate(**encoded,
                    forced_bos_token_id=bt_tokenizer.convert_tokens_to_ids("eng_Latn"),
                    max_new_tokens=128, do_sample=False)
            translated = bt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            record["backtranslation_en"] = restore_placeholders(translated, originals, expected)
            if not record["backtranslation_en"]:
                raise ValueError("empty backtranslation")
            record["backtranslation_sha256"] = sha(record["backtranslation_en"])
        except Exception as exc:
            record["backtranslation_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        checkpoint("IN_PROGRESS_BACKTRANSLATION", rows, records, runtime)
        print(f"Backtranslated {index}/{len(rows)}", flush=True)

    encoded = generated = translated = None
    del bt_model, bt_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, revision=QWEN_REVISION, torch_dtype=torch.float16,
        quantization_config=quantization, device_map="auto")
    qwen_model.eval()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(qwen_tokenizer, JsonSchemaParser(make_qwen_schema()))
    record_by_key = {r["review_key"]: r for r in records}
    for index, row in enumerate(rows, 1):
        record = record_by_key[row["review_key"]]
        if record["backtranslation_error"]:
            record["decision"] = {"error": "backtranslation_failed; semantic decision withheld"}
        else:
            last_error = None
            for attempt in range(2):
                try:
                    raw, decision = judge_one(row, record["backtranslation_en"], qwen_model, qwen_tokenizer, prefix_function)
                    record["raw_output"], record["decision"] = raw, decision
                    break
                except Exception as exc:
                    last_error = exc
            if record["decision"] is None:
                record["decision"] = {"error": f"{type(last_error).__name__}: {last_error}"}
        checkpoint("IN_PROGRESS_QWEN14B_REVIEW", rows, records, runtime)
        print(f"Reviewed {index}/{len(rows)}", flush=True)

    blocked = sum(
        "error" in (r.get("decision") or {}) or r.get("backtranslation_error") is not None
        for r in records
    )
    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["reviewed_rows"] = sum(isinstance(r.get("decision"), dict) and "error" not in r["decision"] for r in records)
    runtime["blocked_rows"] = blocked
    runtime["status"] = "AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY" if blocked == 0 else "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS"
    save({"status": runtime["status"], "mode": MODE, "input_sha256": INPUT_SHA256,
          "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
          "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
          "records": records, **runtime})
    print(json.dumps({"status": runtime["status"], "input_rows": len(rows),
                      "reviewed_rows": runtime["reviewed_rows"], "blocked_rows": blocked,
                      "gpu": runtime["gpu"], "quantization": runtime["quantization"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
