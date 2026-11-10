#!/usr/bin/env python3
"""Blinded direct source-to-target translation review; diagnostic only."""
from __future__ import annotations

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
], check=True)

import accelerate
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE = "blinded_direct_translation_screen_with_sentinels"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_SHA256 = ""
PROMPT_SHA256 = ""
RAW_INPUT = "[]"
PROMPT_INSTRUCTION = ""
OUT = Path("/kaggle/working/paired_direct_review.json")
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_decision(raw_output: str) -> dict:
    match = re.search(r"\{.*\}", raw_output, re.S)
    parsed = json.loads(match.group(0), object_pairs_hook=unique_object) if match else None
    if (not isinstance(parsed, dict)
            or set(parsed) != {"semantic_pass", "fluency_pass", "reason"}
            or type(parsed["semantic_pass"]) is not bool
            or type(parsed["fluency_pass"]) is not bool
            or not isinstance(parsed["reason"], str)
            or not parsed["reason"].strip()):
        raise ValueError("invalid direct-review JSON decision")
    return parsed


def write_result(result: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUT)


def main() -> None:
    rows = json.loads(RAW_INPUT)
    if sha_text(RAW_INPUT) != INPUT_SHA256:
        raise ValueError("embedded input SHA-256 mismatch")
    if len(rows) != 48 or len({row.get("review_key") for row in rows}) != 48:
        raise ValueError("expected 48 unique blinded review rows")
    expected_fields = {"review_key", "language", "kind", "source_question_en", "translated_question"}
    for row in rows:
        if (set(row) != expected_fields
                or row["language"] not in {"fr", "de", "hi", "th"}
                or row["kind"] not in {"atomic", "composition"}
                or not row["source_question_en"].strip()
                or not row["translated_question"].strip()
                or Counter(PLACEHOLDER.findall(row["source_question_en"]))
                != Counter(PLACEHOLDER.findall(row["translated_question"]))):
            raise ValueError(f"invalid or placeholder-mismatched input row: {row.get('review_key')}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This pinned FP16 reviewer requires the validated two-GPU Kaggle T4 runtime")

    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "gpu_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "cuda_device_count": torch.cuda.device_count(),
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "review_method": "direct English-source to target-question; no machine back-translation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    runtime["device_map"] = {key: str(value) for key, value in model.hf_device_map.items()}
    records = []
    for index, row in enumerate(rows, 1):
        payload = (
            PROMPT_INSTRUCTION
            + "\nTarget language: " + row["language"]
            + "\nEnglish source: " + row["source_question_en"]
            + "\nCandidate translation: " + row["translated_question"]
        )
        messages = [
            {"role": "system", "content": "You are a careful multilingual translation adjudicator."},
            {"role": "user", "content": payload},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        raw_output = ""
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=192,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw_output = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            decision = parse_decision(raw_output)
        except Exception as exc:
            decision = {"error": type(exc).__name__, "detail": str(exc)}
        records.append({
            "review_key": row["review_key"],
            "language": row["language"],
            "kind": row["kind"],
            "source_sha256": sha_text(row["source_question_en"]),
            "translation_sha256": sha_text(row["translated_question"]),
            "raw_output": raw_output,
            "decision": decision,
        })
        result = {
            "status": "IN_PROGRESS_DIRECT_SCREEN_DIAGNOSTIC_ONLY",
            "mode": MODE,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "input_sha256": INPUT_SHA256,
            "prompt_sha256": PROMPT_SHA256,
            **runtime,
            "records": records,
        }
        write_result(result)
        print(f"Reviewed {index}/{len(rows)}", flush=True)

    blocked = sum("error" in row["decision"] for row in records)
    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result.update({
        "status": "AI_ONLY_DIRECT_SCREEN_COMPLETE_DIAGNOSTIC_ONLY" if blocked == 0
        else "AI_ONLY_DIRECT_SCREEN_WITH_BLOCKED_ROWS",
        "reviewed_rows": len(records) - blocked,
        "blocked_rows": blocked,
        "completed_at_utc": runtime["completed_at_utc"],
        "records": records,
    })
    write_result(result)
    print(json.dumps({
        key: result[key] for key in (
            "status", "reviewed_rows", "blocked_rows", "model_id", "model_revision", "gpu_devices")
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
