#!/usr/bin/env python3
"""Build a private, answer-blind Kaggle review for the nine Qwen-flagged rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "review.json"
OUT = HERE.parent / "mistral_cross_review9_kaggle"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    review = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for row in review["records"]:
        decision = row["decision"]
        if not (decision["semantic_pass"] and decision["fluency_pass"]):
            rows.append({key: row[key] for key in ("prompt_id", "language", "kind", "source", "translation")})
    if len(rows) != 9:
        raise ValueError(f"Expected exactly nine Qwen-flagged rows, found {len(rows)}")
    raw_input = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    input_hash = sha(raw_input.encode("utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input.jsonl").write_text(raw_input, encoding="utf-8")

    embedded = json.dumps(raw_input, ensure_ascii=False)
    script = f'''#!/usr/bin/env python3
"""Cross-model review of nine disputed ChatGPT translation candidates; diagnostic only."""
import hashlib, json, re, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0"], check=True)
import torch, transformers, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = {MODEL_ID!r}
MODEL_REVISION = {MODEL_REVISION!r}
RAW_INPUT = {embedded}
INPUT_SHA256 = {input_hash!r}
rows = [json.loads(line) for line in RAW_INPUT.splitlines() if line.strip()]
assert len(rows) == 9
assert hashlib.sha256(RAW_INPUT.encode("utf-8")).hexdigest() == INPUT_SHA256
assert torch.cuda.is_available(), "GPU required"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map="auto")
model.eval()
out = Path("/kaggle/working/review.json")
records = []
instruction = (
    "Independently review this translation of a question. Judge exact meaning, relation direction, "
    "scope, polarity, tense, and question/answer type, plus naturalness in the target language. "
    "Do not answer the question. The ZXQ markers are protected entities. Return only JSON with "
    "boolean semantic_pass, boolean fluency_pass, and a brief reason. If uncertain, set the relevant "
    "value to false; do not invent a correction or explanation."
)
for index, row in enumerate(rows, 1):
    prompt = (instruction + "\\nLanguage: " + row["language"] + "\\nEnglish source: " + row["source"]
              + "\\nCandidate translation: " + row["translation"])
    messages = [{{"role":"system", "content":"You are an independent multilingual translation reviewer."}},
                {{"role":"user", "content":prompt}}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=192, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    raw_output = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        match = re.search(r"\\{{.*\\}}", raw_output, re.S)
        parsed = json.loads(match.group(0)) if match else None
        if (not isinstance(parsed, dict) or type(parsed.get("semantic_pass")) is not bool
                or type(parsed.get("fluency_pass")) is not bool or not isinstance(parsed.get("reason"), str)):
            raise ValueError("Invalid review schema")
        decision = parsed
    except Exception as exc:
        decision = {{"error": type(exc).__name__}}
    records.append({{**row, "raw_output":raw_output, "decision":decision}})
    result = {{"status":"IN_PROGRESS_DIAGNOSTIC_ONLY", "model_id":MODEL_ID,
        "model_revision":MODEL_REVISION, "input_sha256":INPUT_SHA256,
        "transformers":transformers.__version__, "accelerate":accelerate.__version__,
        "torch":torch.__version__, "gpu":torch.cuda.get_device_name(0), "records":records}}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"Reviewed {{index}} / {{len(rows)}}", flush=True)
errors = sum("error" in r["decision"] for r in records)
result["status"] = "DIAGNOSTIC_CROSS_REVIEW_COMPLETE" if errors == 0 else "DIAGNOSTIC_CROSS_REVIEW_SCHEMA_ERRORS"
result["schema_errors"] = errors
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("Final status:", result["status"], "schema_errors:", errors, flush=True)
'''
    (OUT / "review.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/synthworlds-chatgpt-cal32-mistral-cross-review-9",
        "title": "SynthWorlds ChatGPT Cal32 Mistral Cross Review 9",
        "code_file": "review.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "DIAGNOSTIC_ONLY_NOT_RELEASE",
        "rows": len(rows),
        "input_sha256": input_hash,
        "source_review_sha256": sha(SOURCE.read_bytes()),
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "review_script_sha256": sha((OUT / "review.py").read_bytes()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "answers_included": False,
        "target_marker_policy": "ZXQ markers preserved exactly; no mapping or answers included",
        "human_translation_reviewed": False,
    }
    (OUT / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "rows": len(rows), "input_sha256": input_hash,
                      "review_script_sha256": manifest["review_script_sha256"], "package": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
