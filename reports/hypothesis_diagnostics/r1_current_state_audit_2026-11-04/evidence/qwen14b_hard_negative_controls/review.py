#!/usr/bin/env python3
"""Qwen14B blinded relation-level mismatch diagnostic; not corpus approval."""
import hashlib, json, re, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0"], check=True)
import torch, transformers, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_SHA256 = "a0cfaf53556f292294c09655106f28c0cb23b8baa6265a6654df4ce71218fcad"
RAW_INPUT = '[{"control_id":"HN001","language":"fr","kind":"atomic","source":"Who is the father of [ENTITY_A]?","translation":"Qui est la mère de [ENTITY_A] ?"},{"control_id":"HN002","language":"fr","kind":"atomic","source":"On what date did [ENTITY_A] officially close?","translation":"À quelle date [ENTITY_A] a-t-il officiellement ouvert ses portes ?"},{"control_id":"HN003","language":"fr","kind":"composition","source":"Who is the mother of [ENTITY_A]?","translation":"Qui est le père de [ENTITY_A] ?"},{"control_id":"HN004","language":"fr","kind":"composition","source":"Who is the sibling of the person for whom the place where [ENTITY_A] worked was named?","translation":"Qui est le conjoint de la personne pour qui le lieu où [ENTITY_A] travaillait a été nommé?"},{"control_id":"HN005","language":"de","kind":"atomic","source":"Who is [ENTITY_A]\'s mother?","translation":"Wer ist [ENTITY_A]s Vater?"},{"control_id":"HN006","language":"de","kind":"atomic","source":"In what year did [ENTITY_A] officially close?","translation":"In welchem Jahr wurde [ENTITY_A] offiziell eröffnet?"},{"control_id":"HN007","language":"de","kind":"composition","source":"Who is the significant person linked to the father of the person to whom the [ENTITY_A] is dedicated?","translation":"Wer ist die bedeutende Person, die mit der Mutter der Person verbunden ist, der das [ENTITY_A] gewidmet ist?"},{"control_id":"HN008","language":"de","kind":"composition","source":"Who is the editor of [ENTITY_A]?","translation":"Wer ist der Direktor von [ENTITY_A]?"},{"control_id":"HN009","language":"hi","kind":"atomic","source":"Who is [ENTITY_A]\'s father?","translation":"[ENTITY_A] की माँ कौन है?"},{"control_id":"HN010","language":"hi","kind":"atomic","source":"On what date did [ENTITY_A] officially close?","translation":"[ENTITY_A] को आधिकारिक तौर पर किस तारीख को खोला गया था?"},{"control_id":"HN011","language":"hi","kind":"composition","source":"Which family does the father of the main subject of the [ENTITY_A] belong to?","translation":"[ENTITY_A] के मुख्य विषय की माँ किस परिवार से संबंधित है?"},{"control_id":"HN012","language":"hi","kind":"composition","source":"Who is the editor of the crew that the [ENTITY_A] crew member belongs to?","translation":"[ENTITY_A] चालक दल के सदस्य का चालक दल का निदेशक कौन है?"},{"control_id":"HN013","language":"th","kind":"atomic","source":"Who is the father of [ENTITY_A]?","translation":"ใครคือแม่ของ [ENTITY_A]?"},{"control_id":"HN014","language":"th","kind":"atomic","source":"In what year did [ENTITY_A] officially open?","translation":"ในปีใดที่ [ENTITY_A] ได้ปิดงานอย่างเป็นทางการ?"},{"control_id":"HN015","language":"th","kind":"composition","source":"Who is the father of [ENTITY_A]?","translation":"ใครคือแม่ของ [ENTITY_A]?"},{"control_id":"HN016","language":"th","kind":"composition","source":"Who is [ENTITY_A]\'s sibling?","translation":"ใครคือคู่ครองของ [ENTITY_A]?"}]'
rows = json.loads(RAW_INPUT)
assert len(rows) == 16 and hashlib.sha256(RAW_INPUT.encode("utf-8")).hexdigest() == INPUT_SHA256
assert len({r["control_id"] for r in rows}) == 16
assert torch.cuda.is_available(), "GPU required"

tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map="auto")
model.eval()
out = Path("/kaggle/working/review.json")
records = []
instruction = ("Review whether this target-language question faithfully expresses the English source. "
    "Carefully compare the central relation, including kinship or role (for example father vs mother, "
    "spouse vs sibling, director vs editor), direction, scope, polarity, tense, and question/answer type. "
    "Accept faithful paraphrases. Judge target-language grammaticality and naturalness separately. "
    "Do not answer the question. [ENTITY_A] is an opaque protected placeholder. Return only JSON with "
    "boolean semantic_pass, boolean fluency_pass, and a concise reason. If meaning differs, semantic_pass must be false.")
for i, row in enumerate(rows, 1):
    payload = instruction + "\nTarget language: " + row["language"] + "\nEnglish source: " + row["source"] + "\nCandidate translation: " + row["translation"]
    messages = [{"role":"system", "content":"You are a careful multilingual translation adjudicator."}, {"role":"user", "content":payload}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(prompt, return_tensors="pt").to(model.device)
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
    records.append({"control_id":row["control_id"], "language":row["language"], "kind":row["kind"],
        "source":row["source"], "translation_sha256":hashlib.sha256(row["translation"].encode("utf-8")).hexdigest(),
        "raw_output":raw_output, "decision":decision})
    result = {"status":"IN_PROGRESS_DIAGNOSTIC_ONLY", "model_id":MODEL_ID, "model_revision":MODEL_REVISION,
        "input_sha256":INPUT_SHA256, "torch":torch.__version__, "transformers":transformers.__version__,
        "accelerate":accelerate.__version__, "gpu":torch.cuda.get_device_name(0), "records":records}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"Reviewed {i}/16", flush=True)
errors = sum("error" in r["decision"] for r in records)
result.update({"status":"DIAGNOSTIC_COMPLETE" if errors == 0 else "DIAGNOSTIC_SCHEMA_ERRORS",
    "schema_errors":errors,
    "semantic_rejects_on_known_mismatches":sum(r["decision"].get("semantic_pass") is False for r in records),
    "semantic_passes_on_known_mismatches":sum(r["decision"].get("semantic_pass") is True for r in records),
    "fluency_passes":sum(r["decision"].get("fluency_pass") is True for r in records),
    "both_pass_on_known_mismatches":sum(r["decision"].get("semantic_pass") is True and r["decision"].get("fluency_pass") is True for r in records)})
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:result[k] for k in ("status","schema_errors","semantic_rejects_on_known_mismatches","semantic_passes_on_known_mismatches","fluency_passes","both_pass_on_known_mismatches")}, indent=2), flush=True)
