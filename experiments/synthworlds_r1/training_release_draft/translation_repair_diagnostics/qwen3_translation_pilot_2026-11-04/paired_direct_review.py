#!/usr/bin/env python3
"""Blinded direct source-to-target translation review; diagnostic only."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.51.3", "accelerate==1.15.0",
], check=True)

import accelerate
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE = "blinded_direct_translation_screen_with_sentinels"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_SHA256 = "63928c1ff976329a231bd719f6e41ab931a44b6b45ffd87fbb87c532f45b78b4"
PROMPT_SHA256 = "c614f74527022c6f3e4b03b18ca2cd81a894619ee40e02f185d50f417e2206a4"
RAW_INPUT = '[{"review_key":"R1A82B7AD2B4","language":"hi","kind":"atomic","source_question_en":"Who is [ENTITY_A]\'s father?","translated_question":"[ENTITY_A] की माँ कौन है?"},{"review_key":"P85BA522530","language":"hi","kind":"atomic","source_question_en":"What is [ENTITY_A]\'s field of work?","translated_question":"[ENTITY_A] का कार्यक्षेत्र क्या है?"},{"review_key":"PEF304C762D","language":"de","kind":"atomic","source_question_en":"On what date did [ENTITY_A] become the country that [ENTITY_B] is in?","translated_question":"An welchem Datum wurde [ENTITY_A] zum Land, in dem [ENTITY_B] ist?"},{"review_key":"P643392C3A2","language":"th","kind":"composition","source_question_en":"What is a part of the next higher rank above the class of which the [ENTITY_A] is a subclass?","translated_question":"อะไรเป็นส่วนหนึ่งของลำดับชั้นที่สูงขึ้นถัดไปเหนือหมวดหมู่ที่ [ENTITY_A] เป็นหมวดหมู่ย่อยของ?"},{"review_key":"P75075A2957","language":"de","kind":"composition","source_question_en":"Which administrative territorial entity does [ENTITY_A] contain?","translated_question":"Welche administrative Gebietseinheit liegt innerhalb von [ENTITY_A]?"},{"review_key":"PD1E890DC78","language":"de","kind":"composition","source_question_en":"Who is the father of the unmarried partner of the spouse of [ENTITY_A]?","translated_question":"Wer ist der Vater des unverheirateten Partners des Ehegatten von [ENTITY_A]?"},{"review_key":"P9C46DD0049","language":"de","kind":"atomic","source_question_en":"On what date did [ENTITY_A] become the country that [ENTITY_B] is in?","translated_question":"An welchem Datum wurde [ENTITY_A] zu dem Land, in dem sich [ENTITY_B] befindet?"},{"review_key":"PBD7986949D","language":"th","kind":"atomic","source_question_en":"On what date was [ENTITY_A] incorporated?","translated_question":"วันที่ [ENTITY_A] ได้รับการจดทะเบียนคือเมื่อใด"},{"review_key":"R24393D95773","language":"fr","kind":"composition","source_question_en":"Who is the mother of [ENTITY_A]?","translated_question":"Qui est le père de [ENTITY_A] ?"},{"review_key":"R1B779901FF4","language":"fr","kind":"composition","source_question_en":"Who is the sibling of the person for whom the place where [ENTITY_A] worked was named?","translated_question":"Qui est le conjoint de la personne pour qui le lieu où [ENTITY_A] travaillait a été nommé?"},{"review_key":"PB2B23CADD3","language":"th","kind":"composition","source_question_en":"What is a part of the next higher rank above the class of which the [ENTITY_A] is a subclass?","translated_question":"สิ่งใดเป็นส่วนหนึ่งของลำดับชั้นที่สูงกว่าของชั้นที่ [ENTITY_A] เป็นsubclass ของชั้นนั้น?"},{"review_key":"R137AA8FF779","language":"de","kind":"composition","source_question_en":"Who is the significant person linked to the father of the person to whom the [ENTITY_A] is dedicated?","translated_question":"Wer ist die bedeutende Person, die mit der Mutter der Person verbunden ist, der das [ENTITY_A] gewidmet ist?"},{"review_key":"PBC897486F4","language":"hi","kind":"composition","source_question_en":"What is a part of the next higher rank above the class of which the [ENTITY_A] is a subclass?","translated_question":"[ENTITY_A] के उप-वर्ग के बराबर वर्ग के अगले उच्चतर रैंक का कौन-सा भाग है?"},{"review_key":"R84826DA8AEF","language":"hi","kind":"composition","source_question_en":"Which family does the father of the main subject of the [ENTITY_A] belong to?","translated_question":"[ENTITY_A] के मुख्य विषय की माँ किस परिवार से संबंधित है?"},{"review_key":"P5D159C4886","language":"fr","kind":"composition","source_question_en":"Which administrative territorial entity does [ENTITY_A] contain?","translated_question":"Quelle entité territoriale administrative [ENTITY_A] contient-elle ?"},{"review_key":"PBB78CE9183","language":"fr","kind":"atomic","source_question_en":"On what date did [ENTITY_A] become a relative of [ENTITY_B]?","translated_question":"À quelle date le lien de parenté entre [ENTITY_A] et [ENTITY_B] a-t-il commencé ?"},{"review_key":"R3AD9B994AFA","language":"th","kind":"atomic","source_question_en":"In what year did [ENTITY_A] officially open?","translated_question":"ในปีใดที่ [ENTITY_A] ได้ปิดงานอย่างเป็นทางการ?"},{"review_key":"R9CF669BD864","language":"de","kind":"composition","source_question_en":"Who is the editor of [ENTITY_A]?","translated_question":"Wer ist der Direktor von [ENTITY_A]?"},{"review_key":"R56B2C0EAFAB","language":"fr","kind":"atomic","source_question_en":"On what date did [ENTITY_A] officially close?","translated_question":"À quelle date [ENTITY_A] a-t-il officiellement ouvert ses portes ?"},{"review_key":"P5E1BB40B91","language":"hi","kind":"atomic","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"कौन-सा स्थान [ENTITY_A] के साथ सीमा साझा करता है?"},{"review_key":"PAEEDF23209","language":"de","kind":"atomic","source_question_en":"What does [ENTITY_A] replace?","translated_question":"Was ersetzt [ENTITY_A]?"},{"review_key":"PD9FCEEB948","language":"fr","kind":"atomic","source_question_en":"Who is a doctoral student of [ENTITY_A]?","translated_question":"Qui prépare un doctorat sous la direction de [ENTITY_A] ?"},{"review_key":"P012B410986","language":"th","kind":"atomic","source_question_en":"On what date was [ENTITY_A] incorporated?","translated_question":"นิติบุคคล [ENTITY_A] จดทะเบียนจัดตั้งเมื่อวันที่ใด"},{"review_key":"R8BC75358C92","language":"th","kind":"composition","source_question_en":"Who is the father of [ENTITY_A]?","translated_question":"ใครคือแม่ของ [ENTITY_A]?"},{"review_key":"R7748A4B0398","language":"fr","kind":"atomic","source_question_en":"Who is the father of [ENTITY_A]?","translated_question":"Qui est la mère de [ENTITY_A] ?"},{"review_key":"R09FDCF28F01","language":"hi","kind":"composition","source_question_en":"Who is the editor of the crew that the [ENTITY_A] crew member belongs to?","translated_question":"[ENTITY_A] चालक दल के सदस्य का चालक दल का निदेशक कौन है?"},{"review_key":"P17A7C5817E","language":"hi","kind":"composition","source_question_en":"What is a part of the [ENTITY_A]?","translated_question":"[ENTITY_A] का कौन-सा भाग है?"},{"review_key":"P8DADAA1597","language":"fr","kind":"composition","source_question_en":"Where was the [ENTITY_A] formed?","translated_question":"Où la formation de [ENTITY_A] a-t-elle eu lieu ?"},{"review_key":"P4B7D1E131B","language":"fr","kind":"atomic","source_question_en":"Who is a doctoral student of [ENTITY_A]?","translated_question":"Qui est un étudiant en doctorat de [ENTITY_A] ?"},{"review_key":"PC02A31B0A9","language":"th","kind":"composition","source_question_en":"Of what class is the [ENTITY_A] a subclass?","translated_question":"[ENTITY_A] เป็น subclass ของ class ใด"},{"review_key":"P276AC476CE","language":"fr","kind":"composition","source_question_en":"Which administrative territorial entity does [ENTITY_A] contain?","translated_question":"Quelle entité administrative territoriale se trouve à l’intérieur de [ENTITY_A] ?"},{"review_key":"RB342B28B602","language":"th","kind":"atomic","source_question_en":"Who is the father of [ENTITY_A]?","translated_question":"ใครคือแม่ของ [ENTITY_A]?"},{"review_key":"PF4EFD8AEB5","language":"hi","kind":"atomic","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"[ENTITY_A] के साथ किसकी सीमा साझा है?"},{"review_key":"P3B150C5B8E","language":"de","kind":"atomic","source_question_en":"What does [ENTITY_A] replace?","translated_question":"Was wird von [ENTITY_A] ersetzt?"},{"review_key":"PF4CC6335A2","language":"th","kind":"atomic","source_question_en":"On what date did [ENTITY_A]\'s tenure as director of [ENTITY_B] end?","translated_question":"[ENTITY_A] ดำรงตำแหน่งผู้อำนวยการของ [ENTITY_B] จนถึงวันที่ใด?"},{"review_key":"PE8EA07BEAE","language":"th","kind":"composition","source_question_en":"Of what class is the [ENTITY_A] a subclass?","translated_question":"[ENTITY_A] เป็นหมวดหมู่ย่อยของหมวดหมู่ใด?"},{"review_key":"P5C6A40F184","language":"hi","kind":"atomic","source_question_en":"What is [ENTITY_A]\'s field of work?","translated_question":"[ENTITY_A] के काम का क्षेत्र क्या है?"},{"review_key":"PCE400F9414","language":"de","kind":"composition","source_question_en":"Which administrative territorial entity does [ENTITY_A] contain?","translated_question":"Welche verwaltende territoriale Einheit enthält [ENTITY_A]?"},{"review_key":"R63CCA6E9691","language":"de","kind":"atomic","source_question_en":"Who is [ENTITY_A]\'s mother?","translated_question":"Wer ist [ENTITY_A]s Vater?"},{"review_key":"P42C79D3D36","language":"th","kind":"atomic","source_question_en":"On what date did [ENTITY_A]\'s tenure as director of [ENTITY_B] end?","translated_question":"วาระการดำรงตำแหน่งกรรมการของ [ENTITY_A] ใน [ENTITY_B] สิ้นสุดเมื่อวันที่ใด"},{"review_key":"PE5EC24109F","language":"de","kind":"composition","source_question_en":"Who is the father of the unmarried partner of the spouse of [ENTITY_A]?","translated_question":"Wer ist der Vater des unverheirateten Partners des Ehepartners von [ENTITY_A]?"},{"review_key":"R47B46AB1E6A","language":"th","kind":"composition","source_question_en":"Who is [ENTITY_A]\'s sibling?","translated_question":"ใครคือคู่ครองของ [ENTITY_A]?"},{"review_key":"RC3368582C20","language":"de","kind":"atomic","source_question_en":"In what year did [ENTITY_A] officially close?","translated_question":"In welchem Jahr wurde [ENTITY_A] offiziell eröffnet?"},{"review_key":"P51D03B0723","language":"hi","kind":"composition","source_question_en":"What is a part of the [ENTITY_A]?","translated_question":"[ENTITY_A] का एक भाग क्या है?"},{"review_key":"P0340B729BD","language":"fr","kind":"atomic","source_question_en":"On what date did [ENTITY_A] become a relative of [ENTITY_B]?","translated_question":"Quelle est la date à laquelle [ENTITY_A] est devenu un parent de [ENTITY_B] ?"},{"review_key":"R775C7BB11C2","language":"hi","kind":"atomic","source_question_en":"On what date did [ENTITY_A] officially close?","translated_question":"[ENTITY_A] को आधिकारिक तौर पर किस तारीख को खोला गया था?"},{"review_key":"P8EF83E88F3","language":"hi","kind":"composition","source_question_en":"What is a part of the next higher rank above the class of which the [ENTITY_A] is a subclass?","translated_question":"उस वर्ग के ठीक ऊपर वाले स्तर का कौन-सा भाग है, जिसका [ENTITY_A] एक उपवर्ग है?"},{"review_key":"P22170E6E51","language":"fr","kind":"composition","source_question_en":"Where was the [ENTITY_A] formed?","translated_question":"Où a été formé le [ENTITY_A] ?"}]'
PROMPT_INSTRUCTION = 'Review whether this target-language question faithfully expresses the English source. Carefully compare the central relation, including kinship or role (for example father vs mother, spouse vs sibling, director vs editor), direction, scope, polarity, tense, and question/answer type. Accept faithful paraphrases. Judge target-language grammaticality and naturalness separately. Do not answer the question. [ENTITY_A] is an opaque protected placeholder. Return only JSON with boolean semantic_pass, boolean fluency_pass, and a concise reason. If meaning differs, semantic_pass must be false.'
OUT = Path("/kaggle/working/paired_direct_review.json")
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_decision(raw_output: str) -> dict:
    match = re.search(r"\{.*\}", raw_output, re.S)
    parsed = json.loads(match.group(0), object_pairs_hook=unique_object) if match else None
    if (not isinstance(parsed, dict)
            or set(parsed) != {"semantic_pass", "fluency_pass", "reason"}
            or type(parsed["semantic_pass"]) is not bool
            or type(parsed["fluency_pass"]) is not bool
            or not isinstance(parsed["reason"], str)
            or not parsed["reason"].strip()):
        raise ValueError("invalid direct-review JSON decision")
    return parsed


def write_result(result: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUT)


def main() -> None:
    rows = json.loads(RAW_INPUT)
    if sha_text(RAW_INPUT) != INPUT_SHA256:
        raise ValueError("embedded input SHA-256 mismatch")
    if len(rows) != 48 or len({row.get("review_key") for row in rows}) != 48:
        raise ValueError("expected 48 unique blinded review rows")
    expected_fields = {"review_key", "language", "kind", "source_question_en", "translated_question"}
    for row in rows:
        if (set(row) != expected_fields
                or row["language"] not in {"fr", "de", "hi", "th"}
                or row["kind"] not in {"atomic", "composition"}
                or not row["source_question_en"].strip()
                or not row["translated_question"].strip()
                or Counter(PLACEHOLDER.findall(row["source_question_en"]))
                != Counter(PLACEHOLDER.findall(row["translated_question"]))):
            raise ValueError(f"invalid or placeholder-mismatched input row: {row.get('review_key')}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This pinned FP16 reviewer requires the validated two-GPU Kaggle T4 runtime")

    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "gpu_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "cuda_device_count": torch.cuda.device_count(),
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "review_method": "direct English-source to target-question; no machine back-translation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    runtime["device_map"] = {key: str(value) for key, value in model.hf_device_map.items()}
    records = []
    for index, row in enumerate(rows, 1):
        payload = (
            PROMPT_INSTRUCTION
            + "\nTarget language: " + row["language"]
            + "\nEnglish source: " + row["source_question_en"]
            + "\nCandidate translation: " + row["translated_question"]
        )
        messages = [
            {"role": "system", "content": "You are a careful multilingual translation adjudicator."},
            {"role": "user", "content": payload},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        raw_output = ""
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=192,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw_output = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            decision = parse_decision(raw_output)
        except Exception as exc:
            decision = {"error": type(exc).__name__, "detail": str(exc)}
        records.append({
            "review_key": row["review_key"],
            "language": row["language"],
            "kind": row["kind"],
            "source_sha256": sha_text(row["source_question_en"]),
            "translation_sha256": sha_text(row["translated_question"]),
            "raw_output": raw_output,
            "decision": decision,
        })
        result = {
            "status": "IN_PROGRESS_DIRECT_SCREEN_DIAGNOSTIC_ONLY",
            "mode": MODE,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "input_sha256": INPUT_SHA256,
            "prompt_sha256": PROMPT_SHA256,
            **runtime,
            "records": records,
        }
        write_result(result)
        print(f"Reviewed {index}/{len(rows)}", flush=True)

    blocked = sum("error" in row["decision"] for row in records)
    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result.update({
        "status": "AI_ONLY_DIRECT_SCREEN_COMPLETE_DIAGNOSTIC_ONLY" if blocked == 0
        else "AI_ONLY_DIRECT_SCREEN_WITH_BLOCKED_ROWS",
        "reviewed_rows": len(records) - blocked,
        "blocked_rows": blocked,
        "records": records,
    })
    write_result(result)
    print(json.dumps({
        key: result[key] for key in (
            "status", "reviewed_rows", "blocked_rows", "model_id", "model_revision", "gpu_devices")
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
