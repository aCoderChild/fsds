#!/usr/bin/env python3
"""Build a blinded 31-row Qwen recheck using verified back-translations."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKBENCH = HERE.parent / "chatgpt_fullflagged_queue_2026-11-04"
FULL_INPUT = WORKBENCH / "independent_review_inputs_2026-11-04/full799.jsonl"
PARTIAL_OUTPUT = WORKBENCH / "remote_output/full799/version_5/repair_review.json"
ALIAS_OUTPUT = HERE / "remote_output/alias_probe.json"
ALIAS_VERIFICATION = HERE / "remote_output/verification.json"
MERGED_DIAGNOSTIC = Path("tmp/r1_review_merge_verification_2026-11-04/merged_full799_diagnostic_reviews.jsonl")
KERNEL = HERE / "qwen_kernel"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

TEMPLATE = r'''#!/usr/bin/env python3
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
INPUT_SHA256 = "__INPUT_SHA256__"
RAW_INPUT_B64 = "__INPUT_B64__"
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
    "meaning change or substantial fluency problem is uncertain, return false. Give one concise reason. "
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
    if len(rows) != 31 or len({r.get("review_key") for r in rows}) != 31:
        raise ValueError("expected 31 unique unresolved rows")
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
        "input_sha256":INPUT_SHA256, "torch":torch.__version__,
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
            generated = model.generate(**encoded, max_new_tokens=256, do_sample=False,
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
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    merged = [json.loads(line) for line in MERGED_DIAGNOSTIC.read_text(encoding="utf-8").splitlines() if line]
    unresolved = {row["review_key"] for row in merged if row.get("review_status") == "AI_ONLY_REPAIR_REVIEW_BLOCKED"}
    full_rows = [json.loads(line) for line in FULL_INPUT.read_text(encoding="utf-8").splitlines() if line]
    full_by_key = {row["review_key"]: row for row in full_rows}
    original = json.loads(PARTIAL_OUTPUT.read_text(encoding="utf-8"))
    original_by_key = {row["review_key"]: row for row in original["records"]}
    alias_verification = json.loads(ALIAS_VERIFICATION.read_text(encoding="utf-8"))
    if (alias_verification.get("status") != "VERIFIED_ALIAS_MARKER_SURVIVAL_DIAGNOSTIC_ONLY"
            or alias_verification.get("counts") != {"PASS_ALIAS_COUNT_ONLY": 28}
            or alias_verification.get("output_sha256") != sha(ALIAS_OUTPUT.read_bytes())):
        raise ValueError("Alias-survival probe is not a complete, hash-bound 28-row result")
    alias = json.loads(ALIAS_OUTPUT.read_text(encoding="utf-8"))
    alias_by_key = {row["review_key"]: row for row in alias["rows"]}
    if len(unresolved) != 31 or len(alias_by_key) != 28:
        raise ValueError("Input state changed; expected 31 blocked rows and 28 alias probes")
    inputs = []
    for key in sorted(unresolved):
        row = full_by_key[key]
        old = original_by_key[key]
        bt = alias_by_key[key]["restored_backtranslation_en"] if key in alias_by_key else old.get("backtranslation_en", "")
        if not bt or MARKER_COUNTS(row["source_question_en"]) != MARKER_COUNTS(row["translated_question"]) or MARKER_COUNTS(bt) != MARKER_COUNTS(row["source_question_en"]):
            raise ValueError(f"{key}: source, candidate, and back-translation marker multisets do not match")
        inputs.append({"review_key":key, "language":row["language"], "kind":row["kind"],
            "source_question_en":row["source_question_en"], "translated_question":row["translated_question"],
            "back_translation_en":bt})
    raw = json.dumps(inputs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    KERNEL.mkdir(parents=True, exist_ok=True)
    code = TEMPLATE.replace("__INPUT_SHA256__", sha(raw)).replace(
        "__INPUT_B64__", base64.b64encode(raw).decode("ascii"))
    (KERNEL / "review.py").write_text(code, encoding="utf-8")
    metadata = {"id":"bunnoob2005/synthworlds-r1-blocked-qwen-recheck-2026-11-04",
        "title":"SynthWorlds R1 blocked Qwen recheck 2026-11-04", "code_file":"review.py",
        "language":"python", "kernel_type":"script", "is_private":True,
        "enable_gpu":True, "enable_internet":True, "dataset_sources":[],
        "competition_sources":[], "kernel_sources":[], "model_sources":[]}
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {"status":"SOURCE_BOUND_31_ROW_BLINDED_RECHECK_READY_NOT_APPROVAL",
        "rows":len(inputs), "input_sha256":sha(raw), "review_keys_sha256":sha("\n".join(sorted(unresolved)).encode()),
        "alias_output_sha256":sha(ALIAS_OUTPUT.read_bytes()),
        "alias_verification_sha256":sha(ALIAS_VERIFICATION.read_bytes()),
        "prior_partial_output_sha256":sha(PARTIAL_OUTPUT.read_bytes()),
        "qwen_model_id":MODEL, "qwen_revision":REVISION,
        "nllb_model_id":"facebook/nllb-200-distilled-600M", "nllb_revision":"f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "backtranslation_input":"verified alias-count-preserving NLLB outputs for 28 rows; prior hash-bound outputs for 3 rows",
        "output_schema":"single-item decisions array constrained with lm-format-enforcer==0.11.1",
        "answers_or_splits_sent":False, "human_translation_reviewed":False,
        "training_eligible":False, "release_approved":False}
    (HERE / "qwen_recheck_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def MARKER_COUNTS(text: str) -> dict[str, int]:
    import re
    from collections import Counter
    return dict(Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", text)))


if __name__ == "__main__":
    main()
