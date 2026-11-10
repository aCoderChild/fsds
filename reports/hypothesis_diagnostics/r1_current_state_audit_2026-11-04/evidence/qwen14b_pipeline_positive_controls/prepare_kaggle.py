#!/usr/bin/env python3
"""Build a JSON-schema-constrained HF/Kaggle reviewer for AI-only positive controls."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEG = HERE.parent / "qwen14b_pipeline_hard_controls"
TEMPLATE_PATH = NEG / "remote_output/version_2/source/synthworlds-qwen14b-pipeline-controls-2026-11-04.py"
template = TEMPLATE_PATH.read_text(encoding="utf-8")
raw = (HERE / "input_array.json").read_text(encoding="utf-8").strip()
rows = json.loads(raw)
manifest = json.loads((HERE / "control_manifest.json").read_text(encoding="utf-8"))
input_hash = hashlib.sha256(raw.encode()).hexdigest()
assert input_hash == manifest["input_sha256_compact_array"] and len(rows) == 16
assert len({r["control_id"] for r in rows}) == 16

review = re.sub(r'(?m)^INPUT_SHA256 = "[0-9a-f]+"$', f'INPUT_SHA256 = "{input_hash}"', template, count=1)
review, raw_count = re.subn(r"(?m)^RAW_INPUT = .*?$", "RAW_INPUT = " + repr(raw), review, count=1)
assert raw_count == 1
review = review.replace(
    '"transformers==4.51.3", "accelerate==1.15.0"',
    '"transformers==4.51.3", "accelerate==1.15.0", "lm-format-enforcer==0.11.1", "bitsandbytes==0.50.2"',
)
review = review.replace(
    "from transformers import AutoModelForCausalLM, AutoTokenizer",
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
    "from transformers import BitsAndBytesConfig\n"
    "from lmformatenforcer import JsonSchemaParser\n"
    "from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn",
)
load_old = 'model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map="auto")'
load_new = '''quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
    quantization_config=quantization, device_map="auto",
)'''
assert review.count(load_old) == 1
review = review.replace(load_old, load_new)
review = review.replace("import torch, transformers, accelerate", "import torch, transformers, accelerate, bitsandbytes")
tok_line = 'tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)'
schema_setup = '''tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
DECISION_SCHEMA = {
    "type":"object",
    "properties":{
        "key":{"type":"string"},
        "semantic_pass":{"type":"boolean"},
        "fluency_pass":{"type":"boolean"},
        "reason":{"type":"string"},
    },
    "required":["key","semantic_pass","fluency_pass","reason"],
    "additionalProperties":False,
}
OUTPUT_SCHEMA = {
    "type":"object",
    "properties":{"decisions":{"type":"array","items":DECISION_SCHEMA,"minItems":1,"maxItems":1}},
    "required":["decisions"],
    "additionalProperties":False,
}
prefix_function = build_transformers_prefix_allowed_tokens_fn(tok, JsonSchemaParser(OUTPUT_SCHEMA))'''
assert review.count(tok_line) == 1
review = review.replace(tok_line, schema_setup)
gen_old = 'generated = model.generate(**encoded, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)'
gen_new = 'generated = model.generate(**encoded, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id, prefix_allowed_tokens_fn=prefix_function)'
assert review.count(gen_old) == 1
review = review.replace(gen_old, gen_new)
review = review.replace(
    'parsed = json.loads(match.group(0)) if match else None',
    'parsed = json.loads(match.group(0), object_pairs_hook=unique_object) if match else None',
)
review = review.replace(
    '"accelerate":accelerate.__version__, "gpu":torch.cuda.get_device_name(0)',
    '"accelerate":accelerate.__version__, "bitsandbytes":bitsandbytes.__version__, "quantization":"4-bit NF4", "gpu":torch.cuda.get_device_name(0)',
)
insert_at = 'LANG_NAMES = {"fr":"French", "de":"German", "hi":"Hindi", "th":"Thai"}'
duplicate_guard = '''def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result

LANG_NAMES = {"fr":"French", "de":"German", "hi":"Hindi", "th":"Thai"}'''
assert review.count(insert_at) == 1
review = review.replace(insert_at, duplicate_guard)
review = review.replace(
    '"semantic_rejects_on_known_mismatches"', '"semantic_rejects_on_ai_screened_positive_controls"'
).replace(
    '"semantic_passes_on_known_mismatches"', '"semantic_passes_on_ai_screened_positive_controls"'
).replace(
    '"both_pass_on_known_mismatches"', '"both_pass_on_ai_screened_positive_controls"'
)
assert "prefix_allowed_tokens_fn=prefix_function" in review
assert "lm-format-enforcer==0.11.1" in review
assert "bitsandbytes==0.50.2" in review and "load_in_4bit=True" in review
compile(review, "review.py", "exec")
(HERE / "review.py").write_text(review, encoding="utf-8")
metadata = {
    "id": "bunnoob2005/synthworlds-qwen14b-positive-controls-2026-11-04",
    "title": "SynthWorlds Qwen14B Positive Controls 2026-11-04",
    "code_file": "review.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
}
(HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"rows": len(rows), "input_sha256": input_hash, "kernel": metadata["id"],
                  "review_sha256": hashlib.sha256(review.encode()).hexdigest(),
                  "format_constraint": "production-equivalent one-decision JSON schema"}, indent=2))
