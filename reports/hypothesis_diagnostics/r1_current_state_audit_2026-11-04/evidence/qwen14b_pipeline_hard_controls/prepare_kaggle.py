#!/usr/bin/env python3
"""Create a pinned Qwen14B run that mirrors the production review prompt and inputs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = (HERE / "input_array.json").read_text(encoding="utf-8").strip()
rows = json.loads(raw)
manifest = json.loads((HERE / "control_manifest.json").read_text(encoding="utf-8"))
input_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
assert input_hash == manifest["input_sha256_compact_array"] and len(rows) == 16
assert len({r["control_id"] for r in rows}) == 16

review = '''#!/usr/bin/env python3
"""Run Qwen2.5-14B on blinded relation controls with production back-translation evidence."""
import hashlib, json, re, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0"], check=True)
import torch, transformers, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_SHA256 = "''' + input_hash + '''"
RAW_INPUT = ''' + repr(raw) + '''
rows = json.loads(RAW_INPUT)
assert len(rows) == 16 and hashlib.sha256(RAW_INPUT.encode("utf-8")).hexdigest() == INPUT_SHA256
assert len({r["control_id"] for r in rows}) == 16
assert all(r.get("back_translation_en") for r in rows)
assert torch.cuda.is_available(), "GPU required"

LANG_NAMES = {"fr":"French", "de":"German", "hi":"Hindi", "th":"Thai"}
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map="auto")
model.eval()
out = Path("/kaggle/working/review.json")
records = []
system = (
    "You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
    "For semantic_pass, decide whether the target-language question asks the same thing as the English "
    "source: preserve relation, direction, polarity, scope, tense/time, and answer type. Accept faithful "
    "paraphrases, inflection changes, and contextual synonyms. Do not reject merely because a word has "
    "another dictionary sense; judge the translated question in context. Use the English back-translation "
    "as independent diagnostic evidence, not as a requirement for word-for-word equality. For fluency_pass, "
    "judge only whether the translated question is grammatical and natural in the named target language. "
    "Names/numbers are opaque placeholders. Do not answer the questions or infer their facts. If a material "
    "meaning change or a substantial fluency problem is uncertain, return false. Give one concise reason. "
    "Return exactly one decision for each key, preserving the key strings and input order."
)
for i, row in enumerate(rows, 1):
    lang = row["language"]
    items = [{"key":row["control_id"], "source_question_en":row["source"],
              "translated_question":row["translation"], "back_translation_en":row["back_translation_en"]}]
    payload = {"target_language":LANG_NAMES[lang], "items":items}
    messages = [{"role":"system", "content":system},
                {"role":"user", "content":json.dumps(payload, ensure_ascii=False)}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    raw_output = tok.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        match = re.search(r"\\{.*\\}", raw_output, re.S)
        parsed = json.loads(match.group(0)) if match else None
        decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
        if not isinstance(decisions, list) or len(decisions) != 1:
            raise ValueError("Invalid decisions list or row count")
        decision = decisions[0]
        if (not isinstance(decision, dict) or decision.get("key") != row["control_id"]
                or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
            raise ValueError("Invalid decision schema or key")
    except Exception as exc:
        decision = {"error":type(exc).__name__, "detail":str(exc)}
    records.append({"control_id":row["control_id"], "language":lang, "kind":row["kind"],
        "source":row["source"], "translation_sha256":hashlib.sha256(row["translation"].encode()).hexdigest(),
        "back_translation_sha256":hashlib.sha256(row["back_translation_en"].encode()).hexdigest(),
        "raw_output":raw_output, "decision":decision})
    result = {"status":"IN_PROGRESS_DIAGNOSTIC_ONLY", "model_id":MODEL_ID, "model_revision":MODEL_REVISION,
        "input_sha256":INPUT_SHA256, "torch":torch.__version__, "transformers":transformers.__version__,
        "accelerate":accelerate.__version__, "gpu":torch.cuda.get_device_name(0), "records":records}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    del encoded, generated
    torch.cuda.empty_cache()
    print(f"Reviewed {i}/{len(rows)} controls ({lang})", flush=True)
errors = sum("error" in r["decision"] for r in records)
result.update({"status":"DIAGNOSTIC_COMPLETE" if errors == 0 else "DIAGNOSTIC_SCHEMA_ERRORS",
    "schema_errors":errors,
    "semantic_rejects_on_known_mismatches":sum(r["decision"].get("semantic_pass") is False for r in records),
    "semantic_passes_on_known_mismatches":sum(r["decision"].get("semantic_pass") is True for r in records),
    "fluency_passes":sum(r["decision"].get("fluency_pass") is True for r in records),
    "both_pass_on_known_mismatches":sum(r["decision"].get("semantic_pass") is True and r["decision"].get("fluency_pass") is True for r in records)})
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:result[k] for k in ("status","schema_errors","semantic_rejects_on_known_mismatches","semantic_passes_on_known_mismatches","fluency_passes","both_pass_on_known_mismatches")}, indent=2), flush=True)
'''
(HERE / "review.py").write_text(review, encoding="utf-8")
metadata = {
    "id":"bunnoob2005/synthworlds-qwen14b-pipeline-controls-2026-11-04",
    "title":"SynthWorlds Qwen14B Pipeline Controls 2026-11-04",
    "code_file":"review.py", "language":"python", "kernel_type":"script",
    "is_private":True, "enable_gpu":True, "enable_internet":True,
    "dataset_sources":[], "competition_sources":[], "kernel_sources":[], "model_sources":[]
}
(HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"rows":len(rows), "input_sha256":input_hash, "kernel":metadata["id"],
                  "review_sha256":hashlib.sha256(review.encode()).hexdigest()}, indent=2))
