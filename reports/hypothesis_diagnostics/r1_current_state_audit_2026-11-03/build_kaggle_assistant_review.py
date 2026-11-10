#!/usr/bin/env python3
"""Build a private, answer-blind Kaggle review job for the 15-row probe."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_source_bound_calibration import verify, TRAINING
import verify_assistant_candidate_probe as probe
sys.path.insert(0, str(TRAINING / "scripts"))
from review_masking_v2 import mask_triplet

REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def main():
    probe.main()
    selected, _, _ = verify()
    source = {(x["language"], x["row"]["kind"], x["row"]["prompt_id"]): x["row"] for x in selected}
    candidates = json.loads((HERE / "assistant_candidate_probe.json").read_text())["candidates"]
    inputs = []
    for c in candidates:
        key = (c["language"], c["kind"], c["prompt_id"])
        row = source[key]
        en, target, _ = mask_triplet({**row, "text_translation": c["translation"], "backtranslation_en": row["text_en"]})
        inputs.append({"language": key[0], "kind": key[1], "prompt_id": key[2], "source": en, "translation": target})
    out = HERE / "kaggle_assistant_review"
    out.mkdir(exist_ok=True)
    input_json = json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
    input_hash = hashlib.sha256(input_json.encode()).hexdigest()
    template = '''import json, re, subprocess, sys, hashlib, platform
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.45,<4.52", "accelerate>=0.34,<2"], check=True)
import torch, transformers, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
assert torch.cuda.is_available(), "GPU required"
model_id = "Qwen/Qwen2.5-7B-Instruct"
revision = "__REVISION__"
raw = __INPUT_REPR__
assert hashlib.sha256(raw.encode()).hexdigest() == "__INPUT_HASH__"
rows = json.loads(raw)
tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16, device_map="auto")
model.eval()
out = Path("/kaggle/working/review.json")
records = []
instruction = "Independently review the translation of this question. Entity and number placeholders are protected. Judge whether the translation preserves the exact meaning, relationship direction and question type, and whether it is natural in the target language. Do not answer the question. Return only JSON with boolean semantic_pass, boolean fluency_pass, and a brief reason. If uncertain, set the relevant value to false."
for row in rows:
    prompt = instruction + "\\nLanguage: " + row["language"] + "\\nEnglish: " + row["source"] + "\\nTranslation: " + row["translation"]
    messages = [{"role":"system", "content":"You are a careful multilingual translation reviewer."},{"role":"user","content":prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    raw_output = tok.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        match = re.search(r"\\{.*\\}", raw_output, re.S)
        parsed = json.loads(match.group(0)) if match else None
        if not isinstance(parsed, dict) or type(parsed.get("semantic_pass")) is not bool or type(parsed.get("fluency_pass")) is not bool or not isinstance(parsed.get("reason"), str):
            raise ValueError("Invalid review schema")
        decision = parsed
    except Exception as exc:
        decision = {"error": type(exc).__name__}
    records.append({"language":row["language"],"kind":row["kind"],"prompt_id":row["prompt_id"],"raw_output":raw_output,"decision":decision})
    result = {"status":"DIAGNOSTIC_AI_REVIEW_ONLY", "model_id":model_id,"model_revision":revision,"input_sha256":"__INPUT_HASH__", "generator_revision":"not recorded", "torch":torch.__version__,"transformers":transformers.__version__,"accelerate":accelerate.__version__,"gpu":torch.cuda.get_device_name(0),"records":records}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("Reviewed", len(records), "/", len(rows), flush=True)
'''
    script = template.replace("__REVISION__", REVISION).replace("__INPUT_REPR__", repr(input_json)).replace("__INPUT_HASH__", input_hash)
    (out / "review.py").write_text(script, encoding="utf-8")
    metadata = {"id":"bunnoob2005/synthworlds-assistant-review-probe", "title":"SynthWorlds assistant review probe", "code_file":"review.py", "language":"python", "kernel_type":"script", "is_private":"true", "enable_gpu":"true", "enable_internet":"true", "dataset_sources":[], "competition_sources":[], "kernel_sources":[], "model_sources":[]}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"rows":len(inputs), "masked_input_sha256":input_hash, "script_sha256":hashlib.sha256(script.encode()).hexdigest(), "output":str(out)}, indent=2))


if __name__ == "__main__":
    main()
