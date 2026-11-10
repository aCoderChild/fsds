#!/usr/bin/env python3
"""Cross-model review of nine disputed ChatGPT translation candidates; diagnostic only."""
import hashlib, json, re, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0"], check=True)
import torch, transformers, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
MODEL_REVISION = 'c170c708c41dac9275d15a8fff4eca08d52bab71'
RAW_INPUT = "{\"kind\": \"atomic\", \"language\": \"fr\", \"prompt_id\": \"0134-qa-sm:r1\", \"source\": \"What did ZXQENTAQXZ replace?\", \"translation\": \"Qu'est-ce que ZXQENTAQXZ a remplacé ?\"}\n{\"kind\": \"composition\", \"language\": \"fr\", \"prompt_id\": \"disjoint-b:0348-qa-sm:q2\", \"source\": \"Who is ZXQENTAQXZ's stepparent?\", \"translation\": \"Qui est le beau-père ou la belle-mère de ZXQENTAQXZ ?\"}\n{\"kind\": \"composition\", \"language\": \"de\", \"prompt_id\": \"disjoint-b:0203-qa-sm:q3\", \"source\": \"Who is ZXQENTAQXZ's stepparent?\", \"translation\": \"Wer ist der Stiefvater oder die Stiefmutter von ZXQENTAQXZ?\"}\n{\"kind\": \"composition\", \"language\": \"hi\", \"prompt_id\": \"disjoint-b:0267-qa-sm:multi\", \"source\": \"Who founded the organization that the rank above the ZXQENTAQXZ is part of?\", \"translation\": \"उस संगठन की स्थापना किसने की, जिसका हिस्सा ZXQENTAQXZ से ऊपर का पद है?\"}\n{\"kind\": \"composition\", \"language\": \"hi\", \"prompt_id\": \"disjoint-b:0300-qa-sm:q2\", \"source\": \"Who is ZXQENTAQXZ's godparent?\", \"translation\": \"ZXQENTAQXZ के धर्मपिता या धर्ममाता कौन हैं?\"}\n{\"kind\": \"atomic\", \"language\": \"th\", \"prompt_id\": \"0044-qa-sm:r2\", \"source\": \"On which physical feature is ZXQENTAQXZ located?\", \"translation\": \"ZXQENTAQXZ ตั้งอยู่บนลักษณะภูมิประเทศใด?\"}\n{\"kind\": \"atomic\", \"language\": \"th\", \"prompt_id\": \"0115-qa-sm:r2\", \"source\": \"Where is the ZXQENTAQXZ language indigenous to?\", \"translation\": \"ภาษา ZXQENTAQXZ เป็นภาษาพื้นเมืองของที่ใด?\"}\n{\"kind\": \"composition\", \"language\": \"th\", \"prompt_id\": \"disjoint-b:0382-qa-sm:multi\", \"source\": \"Who is the father of the creator of the noble title held by ZXQENTAQXZ?\", \"translation\": \"ใครคือบิดาของผู้สถาปนาบรรดาศักดิ์ที่ ZXQENTAQXZ ถือครอง?\"}\n{\"kind\": \"composition\", \"language\": \"th\", \"prompt_id\": \"disjoint-b:0291-qa-sm:q2\", \"source\": \"Who is the unmarried partner of ZXQENTAQXZ?\", \"translation\": \"ใครคือคู่ครองที่ไม่ได้สมรสกับ ZXQENTAQXZ?\"}\n"
INPUT_SHA256 = '56794242bf15f01619252bf5e118ed17ee927fb5865acd95d63b10ac27755c3a'
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
    prompt = (instruction + "\nLanguage: " + row["language"] + "\nEnglish source: " + row["source"]
              + "\nCandidate translation: " + row["translation"])
    messages = [{"role":"system", "content":"You are an independent multilingual translation reviewer."},
                {"role":"user", "content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=192, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    raw_output = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        match = re.search(r"\{.*\}", raw_output, re.S)
        parsed = json.loads(match.group(0)) if match else None
        if (not isinstance(parsed, dict) or type(parsed.get("semantic_pass")) is not bool
                or type(parsed.get("fluency_pass")) is not bool or not isinstance(parsed.get("reason"), str)):
            raise ValueError("Invalid review schema")
        decision = parsed
    except Exception as exc:
        decision = {"error": type(exc).__name__}
    records.append({**row, "raw_output":raw_output, "decision":decision})
    result = {"status":"IN_PROGRESS_DIAGNOSTIC_ONLY", "model_id":MODEL_ID,
        "model_revision":MODEL_REVISION, "input_sha256":INPUT_SHA256,
        "transformers":transformers.__version__, "accelerate":accelerate.__version__,
        "torch":torch.__version__, "gpu":torch.cuda.get_device_name(0), "records":records}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"Reviewed {index} / {len(rows)}", flush=True)
errors = sum("error" in r["decision"] for r in records)
result["status"] = "DIAGNOSTIC_CROSS_REVIEW_COMPLETE" if errors == 0 else "DIAGNOSTIC_CROSS_REVIEW_SCHEMA_ERRORS"
result["schema_errors"] = errors
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("Final status:", result["status"], "schema_errors:", errors, flush=True)
