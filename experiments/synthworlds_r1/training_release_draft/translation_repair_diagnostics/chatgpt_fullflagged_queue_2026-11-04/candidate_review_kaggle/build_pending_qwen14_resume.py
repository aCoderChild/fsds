#!/usr/bin/env python3
"""Build a hash-bound Kaggle continuation for Qwen rows left pending by a killed run."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE.parent / "independent_review_inputs_2026-11-04"
DEFAULT_OUTPUT = HERE / "qwen14_pending_resume_2026-11-04"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")

REVIEW_SYSTEM = (
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

REMOTE_TEMPLATE = r'''#!/usr/bin/env python3
"""Resume only the still-pending Qwen14 decisions from the hash-bound v5 review."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

MODE = "qwen14_pending_resume"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
SOURCE_MODE = "full799"
SOURCE_ROWS = 799
SOURCE_PROGRESS = 733
SOURCE_PARTIAL_OUTPUT_SHA256 = "__PARTIAL_OUTPUT_SHA256__"
SOURCE_PARTIAL_LOG_SHA256 = "__PARTIAL_LOG_SHA256__"
SOURCE_ORIGINAL_SCRIPT_SHA256 = "__ORIGINAL_SCRIPT_SHA256__"
PENDING_ROWS_JSON = __PENDING_ROWS_LITERAL__
PENDING_ROWS = json.loads(PENDING_ROWS_JSON)
PENDING_INPUT_SHA256 = "__PENDING_INPUT_SHA256__"
REVIEW_SYSTEM = __REVIEW_SYSTEM_LITERAL__
OUT = Path("/kaggle/working/pending_qwen14_review.json")
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


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


def make_qwen_schema():
    item = {"type": "object", "properties": {
        "key": {"type": "string"}, "semantic_pass": {"type": "boolean"},
        "fluency_pass": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["key", "semantic_pass", "fluency_pass", "reason"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": item, "minItems": 1, "maxItems": 1}},
        "required": ["decisions"], "additionalProperties": False}


def validate_input():
    if len(PENDING_ROWS) != 66 or len({row.get("review_key") for row in PENDING_ROWS}) != len(PENDING_ROWS):
        raise ValueError("Pending batch must contain 66 unique review rows")
    compact = json.dumps(PENDING_ROWS, ensure_ascii=False, separators=(",", ":"))
    if sha_text(compact) != PENDING_INPUT_SHA256:
        raise ValueError("Pending batch hash mismatch")
    keys = []
    for row in PENDING_ROWS:
        expected = {"review_key", "language", "kind", "source_question_en", "translated_question",
                    "backtranslation_en", "source_sha256", "translation_sha256", "backtranslation_sha256"}
        if set(row) != expected or row["language"] not in LANGUAGE_NAMES:
            raise ValueError(f"Invalid masked pending row schema: {row.get('review_key')}")
        if row["kind"] not in {"atomic", "composition"}:
            raise ValueError(f"Invalid question kind: {row.get('review_key')}")
        for text_key, hash_key in (("source_question_en", "source_sha256"),
                                   ("translated_question", "translation_sha256"),
                                   ("backtranslation_en", "backtranslation_sha256")):
            text = row[text_key]
            if not isinstance(text, str) or not text.strip() or sha_text(text) != row[hash_key]:
                raise ValueError(f"Text/hash mismatch for {row.get('review_key')}:{text_key}")
        marker = Counter(PLACEHOLDER.findall(row["source_question_en"]))
        if marker != Counter(PLACEHOLDER.findall(row["translated_question"])):
            raise ValueError(f"Source/translation placeholder mismatch: {row['review_key']}")
        if marker != Counter(PLACEHOLDER.findall(row["backtranslation_en"])):
            raise ValueError(f"Source/backtranslation placeholder mismatch: {row['review_key']}")
        keys.append(row["review_key"])
    if keys != [f"R{i:04d}" for i in range(SOURCE_PROGRESS + 1, SOURCE_ROWS + 1)]:
        raise ValueError("Pending keys are not exactly the suffix left by the verified 733-row run")
    return compact


def judge_one(row, model, tokenizer, prefix_function):
    payload = {"target_language": LANGUAGE_NAMES[row["language"]], "items": [{
        "key": row["review_key"], "source_question_en": row["source_question_en"],
        "translated_question": row["translated_question"],
        "back_translation_en": row["backtranslation_en"]}]}
    messages = [{"role": "system", "content": REVIEW_SYSTEM},
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
    compact = validate_input()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "Tesla T4":
        raise RuntimeError("Expected a Kaggle Tesla T4 GPU; refusing a different runtime")
    if transformers.__version__ != "4.51.3" or accelerate.__version__ != "1.15.0" or bitsandbytes.__version__ != "0.50.2":
        raise RuntimeError("Pinned review stack mismatch")

    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__,
               "quantization": "4-bit NF4", "gpu": torch.cuda.get_device_name(0),
               "answers_or_splits_sent": False, "human_translation_reviewed": False,
               "source_mode": SOURCE_MODE, "source_rows": SOURCE_ROWS,
               "source_progress_before_resume": SOURCE_PROGRESS,
               "source_partial_output_sha256": SOURCE_PARTIAL_OUTPUT_SHA256,
               "source_partial_log_sha256": SOURCE_PARTIAL_LOG_SHA256,
               "source_original_script_sha256": SOURCE_ORIGINAL_SCRIPT_SHA256,
               "pending_input_sha256": sha_text(compact),
               "resume_script_sha256": sha_bytes(Path(__file__).read_bytes()),
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
    result = {"status": "IN_PROGRESS_QWEN14_PENDING_RESUME", "mode": MODE,
              "model_id": QWEN_ID, "model_revision": QWEN_REVISION,
              "pending_rows": len(PENDING_ROWS), "records": [], **runtime}
    save(result)

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    model = AutoModelForCausalLM.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
        torch_dtype=torch.float16, quantization_config=quantization, device_map="auto")
    model.eval()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(make_qwen_schema()))
    records = []
    for index, row in enumerate(PENDING_ROWS, 1):
        last_error = None
        record = {"review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
                  "source_sha256": row["source_sha256"], "translation_sha256": row["translation_sha256"],
                  "backtranslation_sha256": row["backtranslation_sha256"], "decision": None, "raw_output": None}
        for _ in range(2):
            try:
                raw, decision = judge_one(row, model, tokenizer, prefix_function)
                record["raw_output"], record["decision"] = raw, decision
                break
            except Exception as exc:
                last_error = exc
        if record["decision"] is None:
            record["decision"] = {"error": f"{type(last_error).__name__}: {last_error}"}
        records.append(record)
        result.update({"records": records, "completed_pending_rows": index,
                       "latest_review_key": row["review_key"],
                       "status": "IN_PROGRESS_QWEN14_PENDING_RESUME"})
        save(result)
        print(f"Reviewed {SOURCE_PROGRESS + index}/{SOURCE_ROWS} ({index}/{len(PENDING_ROWS)})", flush=True)
        if index % 8 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    blocked = sum("error" in (row.get("decision") or {}) for row in records)
    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["reviewed_rows"] = len(records) - blocked
    runtime["blocked_rows"] = blocked
    runtime["completed_pending_rows"] = len(records)
    runtime["status"] = "AI_ONLY_QWEN14_PENDING_RESUME_COMPLETE_DIAGNOSTIC_ONLY"
    save({"status": runtime["status"], "mode": MODE, "model_id": QWEN_ID,
          "model_revision": QWEN_REVISION, "pending_rows": len(PENDING_ROWS),
          "records": records, **runtime})
    print(json.dumps({"status": runtime["status"], "pending_rows": len(records),
                      "reviewed_rows": len(records) - blocked, "blocked_rows": blocked,
                      "gpu": runtime["gpu"], "quantization": runtime["quantization"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(args: argparse.Namespace) -> dict:
    rows_path = INPUT_DIR / "full799.jsonl"
    launch_path = INPUT_DIR / "full799.kaggle_launch_manifest.json"
    original_source = HERE.parent / "remote_output/full799/version_5/source/synthworlds-ai-only-translation-repair-review.py"
    rows = read_jsonl(rows_path)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    partial = json.loads(args.partial_review.read_text(encoding="utf-8"))
    log_bytes = args.partial_log.read_bytes()
    output_bytes = args.partial_review.read_bytes()
    if (launch.get("rows") != 799 or launch.get("mode") != "full799"
            or sha_bytes(original_source.read_bytes()) != launch.get("review_script_sha256")
            or partial.get("input_sha256") != launch.get("input_compact_array_sha256")
            or partial.get("mode") != "full799"
            or partial.get("status") != "IN_PROGRESS_QWEN14B_REVIEW"
            or partial.get("qwen_model_id") != QWEN_ID
            or partial.get("qwen_model_revision") != QWEN_REVISION
            or partial.get("nllb_model_id") != NLLB_ID
            or partial.get("nllb_model_revision") != NLLB_REVISION):
        raise ValueError("Original review launch, model provenance, or partial checkpoint mismatch")
    if (sha_bytes(rows_path.read_bytes()) != launch.get("input_jsonl_sha256")
            or not args.partial_log.is_file() or b"Killed" not in log_bytes):
        raise ValueError("Input hash mismatch or partial run lacks a logged worker kill")
    progress = [int(m.group(1)) for m in re.finditer(rb"Reviewed (\d+)/799", log_bytes)]
    if not progress or max(progress) != 733:
        raise ValueError(f"Expected a last verified progress marker of 733/799, found {max(progress, default=0)}")
    partial_by_key = {row.get("review_key"): row for row in partial.get("records", [])}
    input_by_key = {row.get("review_key"): row for row in rows}
    if len(partial_by_key) != 799 or set(partial_by_key) != set(input_by_key):
        raise ValueError("Partial result does not cover exactly the 799 source-bound review keys")
    pending_keys = [row["review_key"] for row in rows if not isinstance(partial_by_key[row["review_key"]].get("decision"), dict)]
    expected_pending = [f"R{i:04d}" for i in range(734, 800)]
    if pending_keys != expected_pending:
        raise ValueError("Pending records do not match the exact unprocessed 734-799 suffix")
    pending = []
    for row in rows:
        key = row["review_key"]
        source_record = partial_by_key[key]
        if (source_record.get("language") != row["language"] or source_record.get("kind") != row["kind"]
                or source_record.get("source_sha256") != sha_text(row["source_question_en"])
                or source_record.get("translation_sha256") != sha_text(row["translated_question"])):
            raise ValueError(f"Source/translation identity mismatch for {key}")
        if key not in pending_keys:
            continue
        backtranslation = source_record.get("backtranslation_en")
        bt_hash = source_record.get("backtranslation_sha256")
        if source_record.get("backtranslation_error") is not None or not isinstance(backtranslation, str) or not backtranslation:
            raise ValueError(f"Pending row {key} lacks a successful backtranslation")
        if bt_hash != sha_text(backtranslation):
            raise ValueError(f"Backtranslation hash mismatch for {key}")
        if (collections.Counter(PLACEHOLDER.findall(row["source_question_en"]))
                != collections.Counter(PLACEHOLDER.findall(row["translated_question"]))
                or collections.Counter(PLACEHOLDER.findall(row["source_question_en"]))
                != collections.Counter(PLACEHOLDER.findall(backtranslation))):
            raise ValueError(f"Protected placeholder mismatch for {key}")
        pending.append({
            "review_key": key, "language": row["language"], "kind": row["kind"],
            "source_question_en": row["source_question_en"],
            "translated_question": row["translated_question"],
            "backtranslation_en": backtranslation,
            "source_sha256": source_record["source_sha256"],
            "translation_sha256": source_record["translation_sha256"],
            "backtranslation_sha256": bt_hash,
        })
    compact_pending = json.dumps(pending, ensure_ascii=False, separators=(",", ":"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_input_path = output_dir / "pending_qwen14_input.jsonl"
    batch_input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pending), encoding="utf-8")
    script_text = REMOTE_TEMPLATE
    replacements = {
        "__PARTIAL_OUTPUT_SHA256__": sha_bytes(output_bytes),
        "__PARTIAL_LOG_SHA256__": sha_bytes(log_bytes),
        "__ORIGINAL_SCRIPT_SHA256__": launch["review_script_sha256"],
        "__PENDING_ROWS_LITERAL__": repr(compact_pending),
        "__PENDING_INPUT_SHA256__": sha_text(compact_pending),
        "__REVIEW_SYSTEM_LITERAL__": repr(REVIEW_SYSTEM),
    }
    for marker, value in replacements.items():
        if marker not in script_text:
            raise ValueError(f"Missing template placeholder {marker}")
        script_text = script_text.replace(marker, value)
    script_path = output_dir / "review_pending_qwen14.py"
    script_path.write_text(script_text, encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/synthworlds-qwen14-pending-review-20261104",
        "title": "SynthWorlds Qwen14 Pending Translation Review",
        "code_file": "review_pending_qwen14.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "keywords": ["translation", "review", "ai-only"], "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    metadata_path = output_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "PREPARED_KAGGLE_PENDING_QWEN14_REVIEW_DIAGNOSTIC_ONLY",
        "mode": "qwen14_pending_resume", "source_mode": "full799",
        "source_rows": len(rows), "source_progress_before_resume": max(progress),
        "pending_rows": len(pending), "pending_keys": pending_keys,
        "input_jsonl_sha256": sha_bytes(rows_path.read_bytes()),
        "original_launch_manifest_sha256": sha_bytes(launch_path.read_bytes()),
        "original_review_script_sha256": launch["review_script_sha256"],
        "original_partial_output_sha256": sha_bytes(output_bytes),
        "original_partial_log_sha256": sha_bytes(log_bytes),
        "pending_input_sha256": sha_text(compact_pending),
        "pending_jsonl_sha256": sha_bytes(batch_input_path.read_bytes()),
        "resume_script_sha256": sha_bytes(script_path.read_bytes()),
        "kernel_metadata_sha256": sha_bytes(metadata_path.read_bytes()),
        "review_system_prompt_sha256": sha_text(REVIEW_SYSTEM),
        "model_id": QWEN_ID, "model_revision": QWEN_REVISION,
        "backtranslation_model_id": NLLB_ID, "backtranslation_model_revision": NLLB_REVISION,
        "answers_or_splits_sent": False, "human_translation_reviewed": False,
        "release_approved": False, "training_eligible": False,
        "limitations": ["AI-only review is not human linguistic certification.",
                        "This batch only completes pending Qwen judgments; it does not approve translations.",
                        "The reviewer remains Qwen2.5-14B and can share correlated errors with Qwen2.5-7B."],
    }
    (output_dir / "resume_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = (
        "# Pending Qwen14 translation review continuation\n\n"
        "This private Kaggle script resumes the exact 66 rows left without a Qwen decision by the hash-verified 733/799 v5 run. "
        "It reuses each row's already verified NLLB back-translation, preserves the production Qwen system prompt, model revision, "
        "NF4 quantization, constrained JSON schema, and one-row-per-call behavior. It sends masked source/translation questions and "
        "back-translations only; it includes no answers or split labels. Output is diagnostic-only, never translation approval or an SFT release.\n\n"
        "Before upload, inspect `resume_manifest.json` and `review_pending_qwen14.py`. The kernel is private, uses a single Kaggle Tesla T4, "
        "and downloads only the public pinned Qwen model.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-review", type=Path, required=True)
    parser.add_argument("--partial-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({key: manifest[key] for key in (
        "status", "source_rows", "source_progress_before_resume", "pending_rows", "pending_keys",
        "pending_input_sha256", "resume_script_sha256", "answers_or_splits_sent",
        "release_approved", "training_eligible")}, indent=2))


if __name__ == "__main__":
    main()
