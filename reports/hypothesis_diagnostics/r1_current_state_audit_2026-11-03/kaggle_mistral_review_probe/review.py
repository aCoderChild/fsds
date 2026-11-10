import json, re, subprocess, sys, hashlib, platform
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.45,<4.52", "accelerate>=0.34,<2"], check=True)
import torch, transformers, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
assert torch.cuda.is_available(), "GPU required"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
revision = "c170c708c41dac9275d15a8fff4eca08d52bab71"
raw = '[{"language":"fr","kind":"atomic","prompt_id":"0050-qa-sm:r1","source":"Who is [ENTITY_A] the godparent of?","translation":"De qui [ENTITY_A] est-il le parrain ou la marraine ?"},{"language":"fr","kind":"composition","prompt_id":"disjoint-b:0348-qa-sm:q2","source":"Who is [ENTITY_A]\'s stepparent?","translation":"Qui est le beau-parent d’[ENTITY_A] ?"},{"language":"de","kind":"atomic","prompt_id":"0169-qa-sm:r1","source":"Who is the editor of [ENTITY_A]?","translation":"Wer ist der Herausgeber oder die Herausgeberin von [ENTITY_A]?"},{"language":"de","kind":"atomic","prompt_id":"0196-qa-sm:r1","source":"Who was a doctoral student of [ENTITY_A]?","translation":"Wer war Doktorand oder Doktorandin von [ENTITY_A]?"},{"language":"de","kind":"atomic","prompt_id":"0152-qa-sm:r2","source":"In what work is [ENTITY_A] described?","translation":"In welchem Werk wird [ENTITY_A] beschrieben?"},{"language":"de","kind":"composition","prompt_id":"disjoint-b:0238-qa-sm:multi","source":"Which political office of the state that replaced the country whose capital is [ENTITY_A] is held by its head of state?","translation":"Welches politische Amt des Staates, der das Land mit der Hauptstadt [ENTITY_A] ablöste, wird von dessen Staatsoberhaupt bekleidet?"},{"language":"de","kind":"composition","prompt_id":"disjoint-b:0203-qa-sm:q3","source":"Who is [ENTITY_A]\'s stepparent?","translation":"Wer ist der Stiefelternteil von [ENTITY_A]?"},{"language":"hi","kind":"atomic","prompt_id":"0072-qa-sm:r1","source":"What is on the boundary of [ENTITY_A]?","translation":"[ENTITY_A] की सीमा पर क्या है?"},{"language":"hi","kind":"atomic","prompt_id":"0078-qa-sm:r2","source":"In which work is [ENTITY_A] described?","translation":"किस कृति में [ENTITY_A] का वर्णन किया गया है?"},{"language":"hi","kind":"composition","prompt_id":"disjoint-b:0300-qa-sm:q2","source":"Who is [ENTITY_A]\'s godparent?","translation":"[ENTITY_A] के धर्म-माता या धर्म-पिता कौन हैं?"},{"language":"th","kind":"atomic","prompt_id":"0044-qa-sm:r2","source":"On which physical feature is [ENTITY_A] located?","translation":"[ENTITY_A] ตั้งอยู่บนลักษณะภูมิประเทศใด?"},{"language":"th","kind":"atomic","prompt_id":"0152-qa-sm:r1","source":"What honorific prefix is used for [ENTITY_A]?","translation":"คำยกย่องที่ใช้เป็นคำนำหน้าชื่อของ [ENTITY_A] คืออะไร?"},{"language":"th","kind":"atomic","prompt_id":"0055-qa-sm:r1","source":"Who performed [ENTITY_A]?","translation":"ใครเป็นผู้แสดง [ENTITY_A]?"},{"language":"th","kind":"composition","prompt_id":"disjoint-b:0382-qa-sm:multi","source":"Who is the father of the creator of the noble title held by [ENTITY_A]?","translation":"บิดาของผู้สร้างบรรดาศักดิ์ที่ [ENTITY_A] ถือครองคือใคร?"},{"language":"th","kind":"composition","prompt_id":"disjoint-b:0300-qa-sm:q2","source":"Who is [ENTITY_A]\'s godparent?","translation":"พ่อทูนหัวหรือแม่ทูนหัวของ [ENTITY_A] คือใคร?"}]'
assert hashlib.sha256(raw.encode()).hexdigest() == "38ff32f0036ce23e2a2daf0fa3c3d12fcf692eb3454d2ee45a6336e2f6d000ee"
rows = json.loads(raw)
tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16, device_map="auto")
model.eval()
out = Path("/kaggle/working/review.json")
records = []
instruction = "Independently review the translation of this question. Entity and number placeholders are protected. Judge whether the translation preserves the exact meaning, relationship direction and question type, and whether it is natural in the target language. Do not answer the question. Return only JSON with boolean semantic_pass, boolean fluency_pass, and a brief reason. If uncertain, set the relevant value to false."
for row in rows:
    prompt = instruction + "\nLanguage: " + row["language"] + "\nEnglish: " + row["source"] + "\nTranslation: " + row["translation"]
    messages = [{"role":"system", "content":"You are a careful multilingual translation reviewer."},{"role":"user","content":prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    raw_output = tok.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        match = re.search(r"\{.*\}", raw_output, re.S)
        parsed = json.loads(match.group(0)) if match else None
        if not isinstance(parsed, dict) or type(parsed.get("semantic_pass")) is not bool or type(parsed.get("fluency_pass")) is not bool or not isinstance(parsed.get("reason"), str):
            raise ValueError("Invalid review schema")
        decision = parsed
    except Exception as exc:
        decision = {"error": type(exc).__name__}
    records.append({"language":row["language"],"kind":row["kind"],"prompt_id":row["prompt_id"],"raw_output":raw_output,"decision":decision})
    result = {"status":"DIAGNOSTIC_AI_REVIEW_ONLY", "model_id":model_id,"model_revision":revision,"input_sha256":"38ff32f0036ce23e2a2daf0fa3c3d12fcf692eb3454d2ee45a6336e2f6d000ee", "generator_revision":"not recorded", "torch":torch.__version__,"transformers":transformers.__version__,"accelerate":accelerate.__version__,"gpu":torch.cuda.get_device_name(0),"records":records}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("Reviewed", len(records), "/", len(rows), flush=True)
