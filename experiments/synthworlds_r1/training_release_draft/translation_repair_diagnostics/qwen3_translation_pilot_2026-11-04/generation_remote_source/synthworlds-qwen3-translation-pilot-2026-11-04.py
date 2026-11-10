#!/usr/bin/env python3
"""Generate a bounded, masked multilingual translation candidate pilot."""
import gc
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.51.3", "accelerate==1.15.0",
                "bitsandbytes==0.50.2"], check=True)
import torch
import transformers
import accelerate
import bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODE = "qwen3_14b_translation_pilot_16"
MODEL_ID = "Qwen/Qwen3-14B"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
INPUT_SHA256 = "c9e41c15d3f60cb77d62dcfdef75fc3a1ae5d72ccd10c7bbc8e06d907f3cf1fc"
RAW_INPUT = '[{"review_key":"R0008","language":"fr","kind":"atomic","source_question_en":"Who is a doctoral student of [ENTITY_A]?"},{"review_key":"R0011","language":"fr","kind":"atomic","source_question_en":"On what date did [ENTITY_A] become a relative of [ENTITY_B]?"},{"review_key":"R0123","language":"fr","kind":"composition","source_question_en":"Where was the [ENTITY_A] formed?"},{"review_key":"R0125","language":"fr","kind":"composition","source_question_en":"Which administrative territorial entity does [ENTITY_A] contain?"},{"review_key":"R0177","language":"de","kind":"atomic","source_question_en":"What does [ENTITY_A] replace?"},{"review_key":"R0179","language":"de","kind":"atomic","source_question_en":"On what date did [ENTITY_A] become the country that [ENTITY_B] is in?"},{"review_key":"R0277","language":"de","kind":"composition","source_question_en":"Who is the father of the unmarried partner of the spouse of [ENTITY_A]?"},{"review_key":"R0282","language":"de","kind":"composition","source_question_en":"Which administrative territorial entity does [ENTITY_A] contain?"},{"review_key":"R0345","language":"hi","kind":"atomic","source_question_en":"What shares a border with [ENTITY_A]?"},{"review_key":"R0347","language":"hi","kind":"atomic","source_question_en":"What is [ENTITY_A]\'s field of work?"},{"review_key":"R0464","language":"hi","kind":"composition","source_question_en":"What is a part of the next higher rank above the class of which the [ENTITY_A] is a subclass?"},{"review_key":"R0465","language":"hi","kind":"composition","source_question_en":"What is a part of the [ENTITY_A]?"},{"review_key":"R0547","language":"th","kind":"atomic","source_question_en":"On what date was [ENTITY_A] incorporated?"},{"review_key":"R0549","language":"th","kind":"atomic","source_question_en":"On what date did [ENTITY_A]\'s tenure as director of [ENTITY_B] end?"},{"review_key":"R0701","language":"th","kind":"composition","source_question_en":"What is a part of the next higher rank above the class of which the [ENTITY_A] is a subclass?"},{"review_key":"R0702","language":"th","kind":"composition","source_question_en":"Of what class is the [ENTITY_A] a subclass?"}]'
OUT = Path("/kaggle/working/qwen3_translation_pilot.json")
LANGUAGES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save(result: dict) -> None:
    temporary = OUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)


class InvalidCandidate(ValueError):
    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


def translate_one(row: dict, model, tokenizer) -> tuple[str, str]:
    system = (
        "Translate one English synthetic question into the named target language. Preserve its exact meaning, "
        "including relation, direction, polarity, scope, tense/time, and answer type. Use natural, grammatical "
        "target-language wording. Treat bracketed ENTITY and NUMBER markers as opaque strings: copy every marker "
        "exactly once and do not translate or alter them. Do not answer the question or add facts. Return only "
        "the translated question, with no JSON, quotation marks, labels, or explanation."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Target language: {LANGUAGES[row['language']]}\n"
            f"Question: {row['source_question_en']}"
        )},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    translation = raw.strip().strip("\"'“”‘’ ")
    if not translation:
        raise InvalidCandidate("empty translation", raw)
    source_markers = Counter(PLACEHOLDER.findall(row["source_question_en"]))
    target_markers = Counter(PLACEHOLDER.findall(translation))
    if source_markers != target_markers:
        raise InvalidCandidate("translation changed entity/number placeholders", raw)
    if row["language"] == "hi" and not any("\u0900" <= char <= "\u097f" for char in translation):
        raise InvalidCandidate("Hindi output lacks Devanagari script", raw)
    if row["language"] == "th" and not any("\u0e00" <= char <= "\u0e7f" for char in translation):
        raise InvalidCandidate("Thai output lacks Thai script", raw)
    return raw, translation


def main() -> None:
    rows = json.loads(RAW_INPUT)
    if sha(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))) != INPUT_SHA256:
        raise ValueError("embedded input hash mismatch")
    if len(rows) != 16 or len({row.get("review_key") for row in rows}) != 16:
        raise ValueError("expected 16 unique pilot rows")
    for row in rows:
        if (set(row) != {"review_key", "language", "kind", "source_question_en"}
                or row["language"] not in LANGUAGES
                or row["kind"] not in {"atomic", "composition"}
                or not row["source_question_en"].strip()):
            raise ValueError(f"invalid blinded input row {row.get('review_key')}")
        if not PLACEHOLDER.findall(row["source_question_en"]):
            raise ValueError(f"expected opaque placeholders in {row['review_key']}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing a slow or different-device run")

    runtime = {
        "status": "STARTED",
        "mode": MODE,
        "input_sha256": INPUT_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "quantization": "4-bit NF4",
        "gpu": torch.cuda.get_device_name(0),
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    records = []
    checkpoint = {**runtime, "records": records}
    save(checkpoint)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float16,
        quantization_config=quant,
        device_map="auto",
    )
    model.eval()
    for index, row in enumerate(rows, 1):
        record = {
            "review_key": row["review_key"],
            "language": row["language"],
            "kind": row["kind"],
            "source_sha256": sha(row["source_question_en"]),
            "translation": None,
            "translation_sha256": None,
            "raw_output": None,
            "error": None,
        }
        last_error = None
        for _ in range(2):
            try:
                raw, translation = translate_one(row, model, tokenizer)
                record.update({"translation": translation, "translation_sha256": sha(translation), "raw_output": raw})
                break
            except Exception as exc:
                last_error = exc
                if isinstance(exc, InvalidCandidate):
                    record["raw_output"] = exc.raw_output
        if record["translation"] is None:
            record["error"] = f"{type(last_error).__name__}: {last_error}"
        records.append(record)
        checkpoint = {**runtime, "records": records, "completed_rows": len(records)}
        save(checkpoint)
        print(f"Translated {index}/{len(rows)}", flush=True)

    blocked = sum(record["error"] is not None for record in records)
    runtime.update({
        "status": "AI_ONLY_TRANSLATION_PILOT_COMPLETE" if blocked == 0 else "AI_ONLY_TRANSLATION_PILOT_BLOCKED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "translated_rows": len(records) - blocked,
        "blocked_rows": blocked,
    })
    save({**runtime, "records": records})
    print(json.dumps({key: runtime[key] for key in (
        "status", "translated_rows", "blocked_rows", "model_id", "model_revision", "gpu", "quantization"
    )}, indent=2), flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
