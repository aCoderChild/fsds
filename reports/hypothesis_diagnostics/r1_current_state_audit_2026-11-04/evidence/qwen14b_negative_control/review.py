#!/usr/bin/env python3
"""Run Qwen2.5-14B on the fixed 15-row, intentionally mismatched translation control.
Diagnostic only: this control is deliberately easy and does not estimate subtle-error detection.
"""
import hashlib, json, re, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0"], check=True)
import torch, transformers, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_SHA256 = "8e4a157d862439d71150cf341728826e880ca66dd590237ebb0b27f2df0ba7e0"
RAW_INPUT = '[{"language":"fr","kind":"atomic","prompt_id":"0050-qa-sm:r1","source":"What color is [ENTITY_A]?","translation":"De qui [ENTITY_A] est-il le parrain ou la marraine ?"},{"language":"fr","kind":"composition","prompt_id":"disjoint-b:0348-qa-sm:q2","source":"What color is [ENTITY_A]?","translation":"Qui est le beau-parent d’[ENTITY_A] ?"},{"language":"de","kind":"atomic","prompt_id":"0169-qa-sm:r1","source":"What color is [ENTITY_A]?","translation":"Wer ist der Herausgeber oder die Herausgeberin von [ENTITY_A]?"},{"language":"de","kind":"atomic","prompt_id":"0196-qa-sm:r1","source":"What color is [ENTITY_A]?","translation":"Wer war Doktorand oder Doktorandin von [ENTITY_A]?"},{"language":"de","kind":"atomic","prompt_id":"0152-qa-sm:r2","source":"What color is [ENTITY_A]?","translation":"In welchem Werk wird [ENTITY_A] beschrieben?"},{"language":"de","kind":"composition","prompt_id":"disjoint-b:0238-qa-sm:multi","source":"What color is [ENTITY_A]?","translation":"Welches politische Amt des Staates, der das Land mit der Hauptstadt [ENTITY_A] ablöste, wird von dessen Staatsoberhaupt bekleidet?"},{"language":"de","kind":"composition","prompt_id":"disjoint-b:0203-qa-sm:q3","source":"What color is [ENTITY_A]?","translation":"Wer ist der Stiefelternteil von [ENTITY_A]?"},{"language":"hi","kind":"atomic","prompt_id":"0072-qa-sm:r1","source":"What color is [ENTITY_A]?","translation":"[ENTITY_A] की सीमा पर क्या है?"},{"language":"hi","kind":"atomic","prompt_id":"0078-qa-sm:r2","source":"What color is [ENTITY_A]?","translation":"किस कृति में [ENTITY_A] का वर्णन किया गया है?"},{"language":"hi","kind":"composition","prompt_id":"disjoint-b:0300-qa-sm:q2","source":"What color is [ENTITY_A]?","translation":"[ENTITY_A] के धर्म-माता या धर्म-पिता कौन हैं?"},{"language":"th","kind":"atomic","prompt_id":"0044-qa-sm:r2","source":"What color is [ENTITY_A]?","translation":"[ENTITY_A] ตั้งอยู่บนลักษณะภูมิประเทศใด?"},{"language":"th","kind":"atomic","prompt_id":"0152-qa-sm:r1","source":"What color is [ENTITY_A]?","translation":"คำยกย่องที่ใช้เป็นคำนำหน้าชื่อของ [ENTITY_A] คืออะไร?"},{"language":"th","kind":"atomic","prompt_id":"0055-qa-sm:r1","source":"What color is [ENTITY_A]?","translation":"ใครเป็นผู้แสดง [ENTITY_A]?"},{"language":"th","kind":"composition","prompt_id":"disjoint-b:0382-qa-sm:multi","source":"What color is [ENTITY_A]?","translation":"บิดาของผู้สร้างบรรดาศักดิ์ที่ [ENTITY_A] ถือครองคือใคร?"},{"language":"th","kind":"composition","prompt_id":"disjoint-b:0300-qa-sm:q2","source":"What color is [ENTITY_A]?","translation":"พ่อทูนหัวหรือแม่ทูนหัวของ [ENTITY_A] คือใคร?"}]'
rows = json.loads(RAW_INPUT)
assert len(rows) == 15 and hashlib.sha256(RAW_INPUT.encode("utf-8")).hexdigest() == INPUT_SHA256
assert all(row["source"] == "What color is [ENTITY_A]?" for row in rows)
assert torch.cuda.is_available(), "GPU required"
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map="auto")
model.eval()
out = Path("/kaggle/working/review.json")
records = []
instruction = ("Review whether this target-language question is a faithful translation of the English source. "
    "Preserve relation, direction, scope, polarity, tense, and question/answer type; accept faithful paraphrases. "
    "Judge grammaticality and naturalness in the target language separately. Do not answer the question. "
    "[ENTITY_A] is an opaque protected placeholder. Return only JSON with boolean semantic_pass, boolean "
    "fluency_pass, and a concise reason. If meaning is not equivalent, semantic_pass must be false.")
for i, row in enumerate(rows, 1):
    payload = instruction + "\nTarget language: " + row["language"] + "\nEnglish source: " + row["source"] + "\nCandidate translation: " + row["translation"]
    messages = [{"role":"system", "content":"You are a careful multilingual translation adjudicator."}, {"role":"user", "content":payload}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=192, do_sample=False, pad_token_id=tok.eos_token_id)
    raw_output = tok.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        match = re.search(r"\{.*\}", raw_output, re.S)
        parsed = json.loads(match.group(0)) if match else None
        if (not isinstance(parsed, dict) or type(parsed.get("semantic_pass")) is not bool
                or type(parsed.get("fluency_pass")) is not bool or not isinstance(parsed.get("reason"), str)):
            raise ValueError("Invalid review schema")
        decision = parsed
    except Exception as exc:
        decision = {"error": type(exc).__name__, "detail": str(exc)}
    records.append({"language": row["language"], "kind": row["kind"], "prompt_id": row["prompt_id"], "raw_output": raw_output, "decision": decision})
    result = {"status":"IN_PROGRESS_DIAGNOSTIC_ONLY", "model_id":MODEL_ID, "model_revision":MODEL_REVISION,
        "input_sha256":INPUT_SHA256, "torch":torch.__version__, "transformers":transformers.__version__,
        "accelerate":accelerate.__version__, "gpu":torch.cuda.get_device_name(0), "records":records}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"Reviewed {i}/15", flush=True)
errors = sum("error" in r["decision"] for r in records)
semantic_passes = sum(r["decision"].get("semantic_pass") is True for r in records)
fluency_passes = sum(r["decision"].get("fluency_pass") is True for r in records)
result.update({"status":"DIAGNOSTIC_COMPLETE" if errors == 0 else "DIAGNOSTIC_SCHEMA_ERRORS", "schema_errors":errors,
    "semantic_passes_on_known_mismatches":semantic_passes, "fluency_passes":fluency_passes,
    "both_pass_on_known_mismatches":sum(r["decision"].get("semantic_pass") is True and r["decision"].get("fluency_pass") is True for r in records)})
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:result[k] for k in ("status","schema_errors","semantic_passes_on_known_mismatches","fluency_passes","both_pass_on_known_mismatches")}, indent=2), flush=True)
