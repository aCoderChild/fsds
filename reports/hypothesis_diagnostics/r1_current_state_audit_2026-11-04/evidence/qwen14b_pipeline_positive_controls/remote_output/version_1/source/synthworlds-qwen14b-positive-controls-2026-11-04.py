#!/usr/bin/env python3
"""Run Qwen2.5-14B on blinded relation controls with production back-translation evidence."""
import hashlib, json, re, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.15.0"], check=True)
import torch, transformers, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_SHA256 = "0179d6f1cdcf9d98e238181726b21ec92681abebee216adc0261eed0fbb6ca85"
RAW_INPUT = '[{"control_id":"PP001","language":"fr","kind":"atomic","source":"Who is the mother of [ENTITY_A]?","translation":"Qui est la mère de [ENTITY_A] ?","back_translation_en":"Who is the mother of [ENTITY_A]?"},{"control_id":"PP002","language":"fr","kind":"atomic","source":"On what date did [ENTITY_A] officially open?","translation":"À quelle date [ENTITY_A] a-t-il officiellement ouvert ses portes ?","back_translation_en":"On what date did [ENTITY_A] officially open?"},{"control_id":"PP003","language":"fr","kind":"composition","source":"Who is the father of [ENTITY_A]?","translation":"Qui est le père de [ENTITY_A] ?","back_translation_en":"Who is the father of [ENTITY_A]?"},{"control_id":"PP004","language":"fr","kind":"composition","source":"Who is the spouse of the person for whom the place where [ENTITY_A] worked was named?","translation":"Qui est le conjoint de la personne pour qui le lieu où [ENTITY_A] travaillait a été nommé?","back_translation_en":"Who is the spouse of the person for whom the place where [ENTITY_A] worked was named?"},{"control_id":"PP005","language":"de","kind":"atomic","source":"Who is [ENTITY_A]\'s father?","translation":"Wer ist [ENTITY_A]s Vater?","back_translation_en":"Who is the father of [ENTITY_A]?"},{"control_id":"PP006","language":"de","kind":"atomic","source":"In what year did [ENTITY_A] officially open?","translation":"In welchem Jahr wurde [ENTITY_A] offiziell eröffnet?","back_translation_en":"In what year was [ENTITY_A] officially opened?"},{"control_id":"PP007","language":"de","kind":"composition","source":"Who is the significant person linked to the mother of the person to whom the [ENTITY_A] is dedicated?","translation":"Wer ist die bedeutende Person, die mit der Mutter der Person verbunden ist, der das [ENTITY_A] gewidmet ist?","back_translation_en":"Who is the significant person associated with the mother of the person to whom [ENTITY_A] is dedicated?"},{"control_id":"PP008","language":"de","kind":"composition","source":"Who is the director of [ENTITY_A]?","translation":"Wer ist der Direktor von [ENTITY_A]?","back_translation_en":"Who is the director of [ENTITY_A]?"},{"control_id":"PP009","language":"hi","kind":"atomic","source":"Who is [ENTITY_A]\'s mother?","translation":"[ENTITY_A] की माँ कौन है?","back_translation_en":"Who is the mother of [ENTITY_A]?"},{"control_id":"PP010","language":"hi","kind":"atomic","source":"On what date did [ENTITY_A] officially open?","translation":"[ENTITY_A] को आधिकारिक तौर पर किस तारीख को खोला गया था?","back_translation_en":"On what date was [ENTITY_A] officially opened?"},{"control_id":"PP011","language":"hi","kind":"composition","source":"Which family does the mother of the main subject of the [ENTITY_A] belong to?","translation":"[ENTITY_A] के मुख्य विषय की माँ किस परिवार से संबंधित है?","back_translation_en":"What family does the mother of the main subject of [ENTITY_A] belong to?"},{"control_id":"PP012","language":"hi","kind":"composition","source":"Who is the director of the crew that the [ENTITY_A] crew member belongs to?","translation":"[ENTITY_A] चालक दल के सदस्य का चालक दल का निदेशक कौन है?","back_translation_en":"Who is the director of the crew that [ENTITY_A]\'s crew member belongs to?"},{"control_id":"PP013","language":"th","kind":"atomic","source":"Who is the mother of [ENTITY_A]?","translation":"ใครคือแม่ของ [ENTITY_A]?","back_translation_en":"Who is the mother of [ENTITY_A]?"},{"control_id":"PP014","language":"th","kind":"atomic","source":"In what year did [ENTITY_A] officially close?","translation":"ในปีใดที่ [ENTITY_A] ได้ปิดงานอย่างเป็นทางการ?","back_translation_en":"In what year did [ENTITY_A] officially close its operations?"},{"control_id":"PP015","language":"th","kind":"composition","source":"Who is the mother of [ENTITY_A]?","translation":"ใครคือแม่ของ [ENTITY_A]?","back_translation_en":"Who is the mother of [ENTITY_A]?"},{"control_id":"PP016","language":"th","kind":"composition","source":"Who is [ENTITY_A]\'s spouse?","translation":"ใครคือคู่ครองของ [ENTITY_A]?","back_translation_en":"Who is the partner of [ENTITY_A]?"}]'
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
        match = re.search(r"\{.*\}", raw_output, re.S)
        parsed = json.loads(match.group(0)) if match else None
        if isinstance(parsed, dict) and isinstance(parsed.get("decisions"), list):
            decisions = parsed["decisions"]
            if len(decisions) != 1 or not isinstance(decisions[0], dict):
                raise ValueError("Invalid decisions array or row count")
            wrapped = decisions[0]
            if wrapped.get("key") != row["control_id"]:
                raise ValueError("Decision key mismatch")
            decision = {k:v for k,v in wrapped.items() if k != "key"}
            response_format = "decisions_array"
        elif isinstance(parsed, dict) and set(parsed) == {row["control_id"]}:
            decision = parsed[row["control_id"]]
            response_format = "keyed_object"
        else:
            raise ValueError("Expected one keyed decision or decisions array")
        if (not isinstance(decision, dict) or set(decision) != {"semantic_pass", "fluency_pass", "reason"}
                or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
            raise ValueError("Invalid decision schema")
    except Exception as exc:
        decision = {"error":type(exc).__name__, "detail":str(exc)}
        response_format = "invalid"
    records.append({"control_id":row["control_id"], "language":lang, "kind":row["kind"],
        "source":row["source"], "translation_sha256":hashlib.sha256(row["translation"].encode()).hexdigest(),
        "back_translation_sha256":hashlib.sha256(row["back_translation_en"].encode()).hexdigest(),
        "raw_output":raw_output, "response_format":response_format, "decision":decision})
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
    "semantic_rejects_on_ai_screened_positive_controls":sum(r["decision"].get("semantic_pass") is False for r in records),
    "semantic_passes_on_ai_screened_positive_controls":sum(r["decision"].get("semantic_pass") is True for r in records),
    "fluency_passes":sum(r["decision"].get("fluency_pass") is True for r in records),
    "both_pass_on_ai_screened_positive_controls":sum(r["decision"].get("semantic_pass") is True and r["decision"].get("fluency_pass") is True for r in records)})
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:result[k] for k in ("status","schema_errors","semantic_rejects_on_ai_screened_positive_controls","semantic_passes_on_ai_screened_positive_controls","fluency_passes","both_pass_on_ai_screened_positive_controls")}, indent=2), flush=True)
