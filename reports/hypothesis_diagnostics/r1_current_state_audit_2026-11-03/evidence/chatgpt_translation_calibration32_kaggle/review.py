#!/usr/bin/env python3
"""Diagnostic-only cross-model review of 32 ChatGPT UI translations."""
import json, re, subprocess, sys, hashlib, os
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0"], check=True)
import torch, transformers, accelerate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
RAW_INPUT = "{\"prompt_id\":\"0134-qa-sm:r1\",\"language\":\"fr\",\"kind\":\"atomic\",\"source\":\"What did ZXQENTAQXZ replace?\",\"translation\":\"Qu'est-ce que ZXQENTAQXZ a remplacé ?\"}\n{\"prompt_id\":\"0050-qa-sm:r1\",\"language\":\"fr\",\"kind\":\"atomic\",\"source\":\"Who is ZXQENTAQXZ the godparent of?\",\"translation\":\"De qui ZXQENTAQXZ est-il le parrain ou la marraine ?\"}\n{\"prompt_id\":\"0172-qa-sm:r2\",\"language\":\"fr\",\"kind\":\"atomic\",\"source\":\"What assessment was used to evaluate ZXQENTAQXZ?\",\"translation\":\"Quelle évaluation a été utilisée pour évaluer ZXQENTAQXZ ?\"}\n{\"prompt_id\":\"0014-qa-sm:r2\",\"language\":\"fr\",\"kind\":\"atomic\",\"source\":\"When did ZXQENTBQXZ become part of the ZXQENTAQXZ?\",\"translation\":\"Quand ZXQENTBQXZ a-t-il commencé à faire partie de ZXQENTAQXZ ?\"}\n{\"prompt_id\":\"disjoint-b:0202-qa-sm:q1\",\"language\":\"fr\",\"kind\":\"composition\",\"source\":\"Of what class is the ZXQENTAQXZ a subclass?\",\"translation\":\"De quelle classe ZXQENTAQXZ est-il une sous-classe ?\"}\n{\"prompt_id\":\"disjoint-b:0348-qa-sm:q2\",\"language\":\"fr\",\"kind\":\"composition\",\"source\":\"Who is ZXQENTAQXZ's stepparent?\",\"translation\":\"Qui est le beau-père ou la belle-mère de ZXQENTAQXZ ?\"}\n{\"prompt_id\":\"disjoint-b:0305-qa-sm:multi\",\"language\":\"fr\",\"kind\":\"composition\",\"source\":\"What is the family name of the entity that immediately follows the family to which ZXQENTAQXZ belongs?\",\"translation\":\"Quel est le nom de famille de l'entité qui suit immédiatement la famille à laquelle ZXQENTAQXZ appartient ?\"}\n{\"prompt_id\":\"disjoint-b:0305-qa-sm:q3\",\"language\":\"fr\",\"kind\":\"composition\",\"source\":\"What is ZXQENTAQXZ's family name?\",\"translation\":\"Quel est le nom de famille de ZXQENTAQXZ ?\"}\n{\"prompt_id\":\"0169-qa-sm:r1\",\"language\":\"de\",\"kind\":\"atomic\",\"source\":\"Who is the editor of ZXQENTAQXZ?\",\"translation\":\"Wer ist der Herausgeber von ZXQENTAQXZ?\"}\n{\"prompt_id\":\"0035-qa-sm:r2\",\"language\":\"de\",\"kind\":\"atomic\",\"source\":\"What is the native language of ZXQENTAQXZ?\",\"translation\":\"Was ist die Muttersprache von ZXQENTAQXZ?\"}\n{\"prompt_id\":\"0196-qa-sm:r1\",\"language\":\"de\",\"kind\":\"atomic\",\"source\":\"Who was a doctoral student of ZXQENTAQXZ?\",\"translation\":\"Wer war Doktorand von ZXQENTAQXZ?\"}\n{\"prompt_id\":\"0152-qa-sm:r2\",\"language\":\"de\",\"kind\":\"atomic\",\"source\":\"In what work is ZXQENTAQXZ described?\",\"translation\":\"In welchem Werk wird ZXQENTAQXZ beschrieben?\"}\n{\"prompt_id\":\"disjoint-b:0238-qa-sm:multi\",\"language\":\"de\",\"kind\":\"composition\",\"source\":\"Which political office of the state that replaced the country whose capital is ZXQENTAQXZ is held by its head of state?\",\"translation\":\"Welches politische Amt des Staates, der das Land ablöste, dessen Hauptstadt ZXQENTAQXZ ist, wird vom Staatsoberhaupt dieses Staates bekleidet?\"}\n{\"prompt_id\":\"disjoint-b:0295-qa-sm:q1\",\"language\":\"de\",\"kind\":\"composition\",\"source\":\"Which work is a part of ZXQENTAQXZ?\",\"translation\":\"Welches Werk ist ein Teil von ZXQENTAQXZ?\"}\n{\"prompt_id\":\"disjoint-b:0203-qa-sm:q3\",\"language\":\"de\",\"kind\":\"composition\",\"source\":\"Who is ZXQENTAQXZ's stepparent?\",\"translation\":\"Wer ist der Stiefvater oder die Stiefmutter von ZXQENTAQXZ?\"}\n{\"prompt_id\":\"disjoint-b:0269-qa-sm:q2\",\"language\":\"de\",\"kind\":\"composition\",\"source\":\"Which work does ZXQENTAQXZ reference?\",\"translation\":\"Auf welches Werk verweist ZXQENTAQXZ?\"}\n{\"prompt_id\":\"0046-qa-sm:r2\",\"language\":\"hi\",\"kind\":\"atomic\",\"source\":\"In which work does ZXQENTAQXZ appear?\",\"translation\":\"ZXQENTAQXZ किस कृति में प्रकट होता है?\"}\n{\"prompt_id\":\"0092-qa-sm:r2\",\"language\":\"hi\",\"kind\":\"atomic\",\"source\":\"In which area does the ZXQENTAQXZ operate?\",\"translation\":\"ZXQENTAQXZ किस क्षेत्र में कार्य करता है?\"}\n{\"prompt_id\":\"0072-qa-sm:r1\",\"language\":\"hi\",\"kind\":\"atomic\",\"source\":\"What is on the boundary of ZXQENTAQXZ?\",\"translation\":\"ZXQENTAQXZ की सीमा पर क्या है?\"}\n{\"prompt_id\":\"0078-qa-sm:r2\",\"language\":\"hi\",\"kind\":\"atomic\",\"source\":\"In which work is ZXQENTAQXZ described?\",\"translation\":\"ZXQENTAQXZ का वर्णन किस कृति में किया गया है?\"}\n{\"prompt_id\":\"disjoint-b:0267-qa-sm:multi\",\"language\":\"hi\",\"kind\":\"composition\",\"source\":\"Who founded the organization that the rank above the ZXQENTAQXZ is part of?\",\"translation\":\"उस संगठन की स्थापना किसने की, जिसका हिस्सा ZXQENTAQXZ से ऊपर का पद है?\"}\n{\"prompt_id\":\"disjoint-b:0295-qa-sm:q1\",\"language\":\"hi\",\"kind\":\"composition\",\"source\":\"Which work is a part of ZXQENTAQXZ?\",\"translation\":\"कौन-सी कृति ZXQENTAQXZ का हिस्सा है?\"}\n{\"prompt_id\":\"disjoint-b:0218-qa-sm:q2\",\"language\":\"hi\",\"kind\":\"composition\",\"source\":\"Who is the editor of ZXQENTAQXZ?\",\"translation\":\"ZXQENTAQXZ का संपादक कौन है?\"}\n{\"prompt_id\":\"disjoint-b:0300-qa-sm:q2\",\"language\":\"hi\",\"kind\":\"composition\",\"source\":\"Who is ZXQENTAQXZ's godparent?\",\"translation\":\"ZXQENTAQXZ के धर्मपिता या धर्ममाता कौन हैं?\"}\n{\"prompt_id\":\"0044-qa-sm:r2\",\"language\":\"th\",\"kind\":\"atomic\",\"source\":\"On which physical feature is ZXQENTAQXZ located?\",\"translation\":\"ZXQENTAQXZ ตั้งอยู่บนลักษณะภูมิประเทศใด?\"}\n{\"prompt_id\":\"0152-qa-sm:r1\",\"language\":\"th\",\"kind\":\"atomic\",\"source\":\"What honorific prefix is used for ZXQENTAQXZ?\",\"translation\":\"ใช้คำยกย่องนำหน้าชื่อใดสำหรับ ZXQENTAQXZ?\"}\n{\"prompt_id\":\"0115-qa-sm:r2\",\"language\":\"th\",\"kind\":\"atomic\",\"source\":\"Where is the ZXQENTAQXZ language indigenous to?\",\"translation\":\"ภาษา ZXQENTAQXZ เป็นภาษาพื้นเมืองของที่ใด?\"}\n{\"prompt_id\":\"0055-qa-sm:r1\",\"language\":\"th\",\"kind\":\"atomic\",\"source\":\"Who performed ZXQENTAQXZ?\",\"translation\":\"ใครเป็นผู้แสดง ZXQENTAQXZ?\"}\n{\"prompt_id\":\"disjoint-b:0382-qa-sm:multi\",\"language\":\"th\",\"kind\":\"composition\",\"source\":\"Who is the father of the creator of the noble title held by ZXQENTAQXZ?\",\"translation\":\"ใครคือบิดาของผู้สถาปนาบรรดาศักดิ์ที่ ZXQENTAQXZ ถือครอง?\"}\n{\"prompt_id\":\"disjoint-b:0300-qa-sm:q2\",\"language\":\"th\",\"kind\":\"composition\",\"source\":\"Who is ZXQENTAQXZ's godparent?\",\"translation\":\"ใครคือพ่อทูนหัวหรือแม่ทูนหัวของ ZXQENTAQXZ?\"}\n{\"prompt_id\":\"disjoint-b:0291-qa-sm:q2\",\"language\":\"th\",\"kind\":\"composition\",\"source\":\"Who is the unmarried partner of ZXQENTAQXZ?\",\"translation\":\"ใครคือคู่ครองที่ไม่ได้สมรสกับ ZXQENTAQXZ?\"}\n{\"prompt_id\":\"disjoint-b:0277-qa-sm:multi\",\"language\":\"th\",\"kind\":\"composition\",\"source\":\"Where is the mother of the person ZXQENTAQXZ works for buried?\",\"translation\":\"มารดาของบุคคลที่ ZXQENTAQXZ ทำงานให้ถูกฝังอยู่ที่ใด?\"}\n"
INPUT_SHA256 = "7fe75e5cb4ab9b3ca9bb96c4cbd713d0a88a39feffc4b7af330d3e18259ccc14"
rows = [json.loads(line) for line in RAW_INPUT.splitlines() if line.strip()]
assert len(rows) == 32
assert hashlib.sha256(RAW_INPUT.encode("utf-8")).hexdigest() == INPUT_SHA256
assert torch.cuda.is_available(), "GPU required"
config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
assert getattr(config, "use_sliding_window", False) is False, "Unexpected model attention config"
# This checkpoint disables sliding-window attention. Clear its unused window size to avoid
# Transformers 4.51.3's misleading warning; this does not alter the checkpoint's layer pattern.
config.sliding_window = None
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, config=config, torch_dtype=torch.float16,
    device_map="auto", attn_implementation="sdpa"
)
model.eval()
out = Path("/kaggle/working/review.json")
records = []
instruction = (
    "Independently review this translation of a question. Judge whether it preserves the exact "
    "meaning, relation direction, scope, polarity, tense, and question/answer type, and whether "
    "it is natural in the target language. ZXQ markers are protected entities/numbers; ignore their "
    "surface form and verify they remain in corresponding semantic positions. Do not answer the "
    "question. Return only JSON with boolean semantic_pass, boolean fluency_pass, and a brief reason. "
    "If uncertain, set the relevant value to false."
)
for index, row in enumerate(rows, 1):
    prompt = (instruction + "\nLanguage: " + row["language"] + "\nEnglish source: " + row["source"]
              + "\nCandidate translation: " + row["translation"])
    messages = [{"role":"system", "content":"You are an independent multilingual translation reviewer."},
                {"role":"user", "content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=160, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id, use_cache=True)
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
    result = {
        "status":"IN_PROGRESS_DIAGNOSTIC_ONLY", "model_id":MODEL_ID,
        "model_revision":MODEL_REVISION, "input_sha256":INPUT_SHA256,
        "transformers":transformers.__version__, "accelerate":accelerate.__version__,
        "torch":torch.__version__, "gpu":torch.cuda.get_device_name(0),
        "attention_backend":model.config._attn_implementation,
        "use_sliding_window":model.config.use_sliding_window,
        "records":records
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"Reviewed {index} / {len(rows)}", flush=True)
errors = sum("error" in r["decision"] for r in records)
result["status"] = "DIAGNOSTIC_AI_REVIEW_COMPLETE" if errors == 0 else "DIAGNOSTIC_AI_REVIEW_COMPLETE_WITH_SCHEMA_ERRORS"
result["schema_errors"] = errors
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("Final status:", result["status"], "schema_errors:", errors, flush=True)
