#!/usr/bin/env python3
"""AI-only round-trip and semantic/fluency screen for masked repair questions."""
import gc
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
    "lm-format-enforcer==0.11.1", "bitsandbytes==0.50.2",
], check=True)

import torch
import transformers
import accelerate
import bitsandbytes
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

MODE = 'tg_qwen7_complement_v1'
INPUT_SHA256 = '53c3c0abc151fe1a0b5f7b2b089bf3c5fbbed1e8db068da6149f18e518fe437d'
RAW_INPUT = '[{"review_key":"T0001","language":"fr","kind":"atomic","source_question_en":"What is the standard plug type for mains electricity in [ENTITY_A]?","translated_question":"Quel est le type de prise standard utilisé pour l\'électricité secteur à [ENTITY_A] ?"},{"review_key":"T0002","language":"fr","kind":"atomic","source_question_en":"Which organization confers the [ENTITY_A]?","translated_question":"Quelle organisation attribue le prix [ENTITY_A] ?"},{"review_key":"T0003","language":"fr","kind":"atomic","source_question_en":"Who is [ENTITY_A]\'s stepparent?","translated_question":"Qui est le beau-père de [ENTITY_A] ?"},{"review_key":"T0004","language":"fr","kind":"atomic","source_question_en":"Who is the exhibited creator of [ENTITY_A]?","translated_question":"Quel est l\'artiste présenté dans l\'exposition \\"[ENTITY_A]\\"?"},{"review_key":"T0005","language":"fr","kind":"atomic","source_question_en":"What is the equivalent given name of the opposite gender for [ENTITY_A]?","translated_question":"Quel est le prénom masculin correspondant à [ENTITY_A] ?"},{"review_key":"T0006","language":"fr","kind":"atomic","source_question_en":"Which continent is [ENTITY_A] part of?","translated_question":"Quel est le continent auquel appartient [ENTITY_A] ?"},{"review_key":"T0007","language":"fr","kind":"atomic","source_question_en":"What was [ENTITY_A] nominated for?","translated_question":"[ENTITY_A] a-t-elle été nominée pour quoi ?"},{"review_key":"T0008","language":"fr","kind":"atomic","source_question_en":"Which continent is [ENTITY_A] part of?","translated_question":"Quelle est le continent auquel appartient [ENTITY_A] ?"},{"review_key":"T0009","language":"fr","kind":"atomic","source_question_en":"Who was a doctoral student of [ENTITY_A]?","translated_question":"Quel était le doctorant de [ENTITY_A] ?"},{"review_key":"T0010","language":"fr","kind":"composition","source_question_en":"Who is [ENTITY_A]\'s stepparent?","translated_question":"Qui est le/la beau-parent(e) de [ENTITY_A] ?"},{"review_key":"T0011","language":"fr","kind":"composition","source_question_en":"Who represents the creator of the participant in [ENTITY_A]?","translated_question":"Qui représente le créateur de l\'utilisateur dans [ENTITY_A] ?"},{"review_key":"T0012","language":"fr","kind":"composition","source_question_en":"Who is [ENTITY_A]\'s spouse?","translated_question":"Qui est le/la conjoint(e) de [ENTITY_A] ?"},{"review_key":"T0013","language":"fr","kind":"composition","source_question_en":"Who is [ENTITY_A]\'s stepparent?","translated_question":"Quel est le parent adoptif d\'[ENTITY_A] ?"},{"review_key":"T0014","language":"de","kind":"atomic","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"Welche Region grenzt an [ENTITY_A]?"},{"review_key":"T0015","language":"de","kind":"atomic","source_question_en":"Which military branch does [ENTITY_A] belong to?","translated_question":"Welche militärische Organisation gehört [ENTITY_A] an?"},{"review_key":"T0016","language":"de","kind":"atomic","source_question_en":"Who is [ENTITY_A] named after?","translated_question":"Nach wem ist der \\"[ENTITY_A]\\" benannt?"},{"review_key":"T0017","language":"de","kind":"atomic","source_question_en":"Where is [ENTITY_A] indigenous to?","translated_question":"Wo ist die ursprüngliche Heimat von [ENTITY_A]?"},{"review_key":"T0018","language":"de","kind":"atomic","source_question_en":"In which work does [ENTITY_A] appear?","translated_question":"In welchem Werk erscheint [ENTITY_A]?"},{"review_key":"T0019","language":"de","kind":"atomic","source_question_en":"Who performed [ENTITY_A]?","translated_question":"Wer hat die Rolle von [ENTITY_A] gespielt?"},{"review_key":"T0020","language":"de","kind":"atomic","source_question_en":"Which work is a part of [ENTITY_A]?","translated_question":"Welches Werk ist Teil von [ENTITY_A]?"},{"review_key":"T0021","language":"de","kind":"atomic","source_question_en":"On what date was [ENTITY_A] first performed?","translated_question":"Wann wurde das Stück \\"[ENTITY_A]\\" zum ersten Mal aufgeführt?"},{"review_key":"T0022","language":"de","kind":"atomic","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"Welche Länder grenzen an [ENTITY_A]?"},{"review_key":"T0023","language":"de","kind":"atomic","source_question_en":"Who was [ENTITY_A]\'s doctoral advisor?","translated_question":"Wer war der Doktorvater von [ENTITY_A]?"},{"review_key":"T0024","language":"de","kind":"atomic","source_question_en":"Who was [ENTITY_A] a student of?","translated_question":"Unter wem war [ENTITY_A] ein Schüler?"},{"review_key":"T0025","language":"de","kind":"atomic","source_question_en":"What significant place is associated with [ENTITY_A]?","translated_question":"Welcher bedeutender Ort ist mit [ENTITY_A] verbunden?"},{"review_key":"T0026","language":"de","kind":"atomic","source_question_en":"What is the equivalent given name of the opposite gender for [ENTITY_A]?","translated_question":"Welches ist der passende, gegengeschlechtliche Vornames für [ENTITY_A]?"},{"review_key":"T0027","language":"de","kind":"atomic","source_question_en":"On or next to which body of water is [ENTITY_A] located?","translated_question":"[ENTITY_A] liegt auf oder in der Nähe welcher Wasserfläche?"},{"review_key":"T0028","language":"de","kind":"atomic","source_question_en":"Which work does [ENTITY_A] follow in its series?","translated_question":"Welche Serie setzt \\"[ENTITY_A]\\" fort?"},{"review_key":"T0029","language":"de","kind":"atomic","source_question_en":"On what date did [ENTITY_A] stop being the capital of [ENTITY_B]?","translated_question":"Zu welchem Datum hörte [ENTITY_A] auf, die Hauptstadt von [ENTITY_B] zu sein?"},{"review_key":"T0030","language":"de","kind":"atomic","source_question_en":"Who is [ENTITY_A] named after?","translated_question":"Wer ist der Name [ENTITY_A] nach einer Person benannt?"},{"review_key":"T0031","language":"de","kind":"atomic","source_question_en":"In which document is [ENTITY_A] attested?","translated_question":"In welchem Dokument wird [ENTITY_A] erwähnt/belegt?"},{"review_key":"T0032","language":"de","kind":"atomic","source_question_en":"On which physical feature is [ENTITY_A] located?","translated_question":"Auf welchem physikalischen Merkmal befindet sich [ENTITY_A]?"},{"review_key":"T0033","language":"de","kind":"atomic","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"Welche Region grenzt an [ENTITY_A]?"},{"review_key":"T0034","language":"de","kind":"atomic","source_question_en":"Who is the godparent of [ENTITY_A]?","translated_question":"Wer ist der Pate von [ENTITY_A]?"},{"review_key":"T0035","language":"de","kind":"atomic","source_question_en":"Who is [ENTITY_A]\'s godparent?","translated_question":"Wer ist der Pate von [ENTITY_A]?"},{"review_key":"T0036","language":"de","kind":"atomic","source_question_en":"Which military branch does [ENTITY_A] belong to?","translated_question":"Welche militärische Organisation gehört [ENTITY_A] an?"},{"review_key":"T0037","language":"de","kind":"composition","source_question_en":"Who was [ENTITY_A]\'s doctoral advisor?","translated_question":"Wer war der Doktorvater von [ENTITY_A]?"},{"review_key":"T0038","language":"de","kind":"composition","source_question_en":"Which political office of [ENTITY_A] is held by its head of state?","translated_question":"Welche politische Position in [ENTITY_A] bekleidet ihre Staatsoberhaupt?"},{"review_key":"T0039","language":"de","kind":"composition","source_question_en":"Who is [ENTITY_A] different from?","translated_question":"Wer unterscheidet sich von [ENTITY_A]?"},{"review_key":"T0040","language":"de","kind":"composition","source_question_en":"Who was [ENTITY_A]\'s doctoral advisor?","translated_question":"Wer war der Doktorvater von [ENTITY_A]?"},{"review_key":"T0041","language":"de","kind":"composition","source_question_en":"Which work is a part of [ENTITY_A]?","translated_question":"Welches Werk gehört zum Buch \\"[ENTITY_A]\\"?"},{"review_key":"T0042","language":"de","kind":"composition","source_question_en":"Who is the spouse of the godparent of the head of state of [ENTITY_A]?","translated_question":"Wer ist der Ehepartner des Taufszeugen des Staatsoberhaupts von [ENTITY_A]?"},{"review_key":"T0043","language":"de","kind":"composition","source_question_en":"Who is [ENTITY_A]\'s godparent?","translated_question":"Wer ist der Pate von [ENTITY_A]?"},{"review_key":"T0044","language":"de","kind":"composition","source_question_en":"Where is [ENTITY_A] indigenous to?","translated_question":"Wo ist die Art [ENTITY_A] heimisch?"},{"review_key":"T0045","language":"de","kind":"composition","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"Welche Länder grenzen an [ENTITY_A]?"},{"review_key":"T0046","language":"de","kind":"composition","source_question_en":"What institution holds the collection items of [ENTITY_A]?","translated_question":"Welche Institution besitzt die Sammlung von [ENTITY_A]?"},{"review_key":"T0047","language":"de","kind":"composition","source_question_en":"Where did the father of the godparent of [ENTITY_A] die?","translated_question":"Wo ist der Vater des Taufeopaters von [ENTITY_A] gestorben?"},{"review_key":"T0048","language":"th","kind":"atomic","source_question_en":"Which organization confers the [ENTITY_A]?","translated_question":"องค์กรใดเป็นผู้มอบรางวัล [ENTITY_A] ครับ"},{"review_key":"T0049","language":"th","kind":"atomic","source_question_en":"When did [ENTITY_A] become part of the [ENTITY_B]?","translated_question":"[ENTITY_A] เริ่มเป็นส่วนหนึ่งของกลุ่ม [ENTITY_B] เมื่อไหร่?"},{"review_key":"T0050","language":"th","kind":"atomic","source_question_en":"What language does [ENTITY_A] write in?","translated_question":"[ENTITY_A] เขียนด้วยภาษาอะไร?"},{"review_key":"T0051","language":"th","kind":"atomic","source_question_en":"In which work does [ENTITY_A] appear?","translated_question":"\\"[ENTITY_A]\\" ปรากฏในผลงานชิ้นไหน"},{"review_key":"T0052","language":"th","kind":"atomic","source_question_en":"In what year was [ENTITY_A] nominated for [ENTITY_B]?","translated_question":"[ENTITY_A] ได้รับการเสนอชื่อเข้าชิงรางวัล \\"[ENTITY_B]\\" ในปีใด?"},{"review_key":"T0053","language":"th","kind":"atomic","source_question_en":"Which institution holds [ENTITY_A]\'s archives?","translated_question":"สถาบันใดเป็นผู้เก็บรักษาเอกสารของ [ENTITY_A]?"},{"review_key":"T0054","language":"th","kind":"atomic","source_question_en":"Which work expands the plot of [ENTITY_A]?","translated_question":"งานชิ้นไหนที่ขยายเนื้อเรื่องของ \\"[ENTITY_A]\\"?"},{"review_key":"T0055","language":"th","kind":"atomic","source_question_en":"On what date did [ENTITY_A] cease to be the country of [ENTITY_B]?","translated_question":"[ENTITY_A] ได้เลิกเป็นประเทศ [ENTITY_B] เมื่อวันที่เท่าไหร่?"},{"review_key":"T0056","language":"th","kind":"atomic","source_question_en":"Which work is a part of [ENTITY_A]?","translated_question":"งานชิ้นไหนเป็นส่วนหนึ่งของ \\"[ENTITY_A]\\"?"},{"review_key":"T0057","language":"th","kind":"atomic","source_question_en":"What shares a border with [ENTITY_A]?","translated_question":"มีประเทศใดบ้างที่อยู่ติดกับ [ENTITY_A]?"},{"review_key":"T0058","language":"th","kind":"atomic","source_question_en":"What significant place is associated with [ENTITY_A]?","translated_question":"สถานที่สำคัญใดที่เกี่ยวข้องกับ \\"[ENTITY_A]\\"?"},{"review_key":"T0059","language":"th","kind":"atomic","source_question_en":"What work follows [ENTITY_A]?","translated_question":"ผลงานชิ้นต่อไปหลังจาก \\"[ENTITY_A]\\" คืออะไร?"},{"review_key":"T0060","language":"th","kind":"atomic","source_question_en":"Who was the director of photography for [ENTITY_A]?","translated_question":"ผู้กำกับภาพสำหรับเรื่อง \\"[ENTITY_A]\\" คือใคร"},{"review_key":"T0061","language":"th","kind":"atomic","source_question_en":"Who is [ENTITY_A]\'s child?","translated_question":"[ENTITY_A] มีลูกคนใด?"},{"review_key":"T0062","language":"th","kind":"atomic","source_question_en":"Which organization confers the [ENTITY_A]?","translated_question":"องค์กรใดเป็นผู้มอบรางวัล \\"[ENTITY_A]\\""},{"review_key":"T0063","language":"th","kind":"atomic","source_question_en":"In which work did [ENTITY_A] first appear?","translated_question":"\\"[ENTITY_A]\\" ปรากฏในผลงานชิ้นไหนเป็นครั้งแรก?"},{"review_key":"T0064","language":"th","kind":"atomic","source_question_en":"Who distributed [ENTITY_A]?","translated_question":"ใครเป็นผู้เผยแพร่ [ENTITY_A]?"},{"review_key":"T0065","language":"th","kind":"composition","source_question_en":"Who is [ENTITY_A]\'s stepparent?","translated_question":"ผู้เป็นพ่อ/แม่เลี้ยงของ [ENTITY_A] คือใคร?"},{"review_key":"T0066","language":"th","kind":"composition","source_question_en":"What immediately followed [ENTITY_A]?","translated_question":"สิ่งที่เกิดขึ้นทันทีหลังจาก \\"[ENTITY_A]\\" คืออะไร?"},{"review_key":"T0067","language":"th","kind":"composition","source_question_en":"Which work expands the plot of [ENTITY_A]?","translated_question":"งานชิ้นไหนที่ขยายเนื้อเรื่องของ \\"[ENTITY_A]\\"?"},{"review_key":"T0068","language":"th","kind":"composition","source_question_en":"Which work does [ENTITY_A] reference?","translated_question":"\\"[ENTITY_A]\\" อ้างอิงถึงงานอะไร?"},{"review_key":"T0069","language":"th","kind":"composition","source_question_en":"Which institution holds the documentation files about the person after whom the city that houses the [ENTITY_A]\'s headquarters is named?","translated_question":"สถาบันใดที่เก็บเอกสารเกี่ยวกับบุคคลที่เมืองซึ่งเป็นที่ตั้งของสำนักงานใหญ่ของ [ENTITY_A] ได้รับชื่อมาจาก?"},{"review_key":"T0070","language":"th","kind":"composition","source_question_en":"Who was [ENTITY_A]\'s doctoral advisor?","translated_question":"ผู้ที่ทำหน้าที่เป็นอาจารย์ที่ปรึกษาปริญญาเอกของ [ENTITY_A] คือใคร?"},{"review_key":"T0071","language":"th","kind":"composition","source_question_en":"What honorific prefix is used before the name of the unmarried partner of the founder of [ENTITY_A]?","translated_question":"คำสรรพนามที่ใช้ก่อนชื่อของคู่สมรสที่ไม่แต่งงานของผู้ก่อตั้ง \\"[ENTITY_A]\\" คืออะไร"},{"review_key":"T0072","language":"th","kind":"composition","source_question_en":"What honorific prefix is used before [ENTITY_A]\'s name?","translated_question":"คำสรรพนามที่ใช้ก่อนชื่อ \\"[ENTITY_A]\\" คืออะไร?"},{"review_key":"T0073","language":"th","kind":"composition","source_question_en":"Who is the child of [ENTITY_A]?","translated_question":"เด็กของ [ENTITY_A] คือใคร?"},{"review_key":"T0074","language":"th","kind":"composition","source_question_en":"Who participated in [ENTITY_A]?","translated_question":"ใครบ้างที่เข้าร่วมใน [ENTITY_A]?"},{"review_key":"T0075","language":"th","kind":"composition","source_question_en":"Who commissioned [ENTITY_A]?","translated_question":"ใครเป็นผู้ว่าจ้างบริษัท [ENTITY_A]?"}]'
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NLLB_SOURCE_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
PLACEHOLDER = re.compile(r"\[(ENTITY|NUMBER)_([A-Z]+)\]")
OUT = Path("/kaggle/working/repair_review.json")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def save(result):
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUT)


def mask_placeholders(text):
    counts = Counter()
    originals = {}
    def replace(match):
        token = f"ZXQ{match.group(1)}{match.group(2)}QXZ"
        counts[token] += 1
        originals[token] = match.group(0)
        return token
    return PLACEHOLDER.sub(replace, text), originals, counts


def restore_placeholders(text, originals, expected):
    found = Counter({token: text.count(token) for token in originals})
    if found != expected:
        raise ValueError(f"backtranslation changed opaque placeholders: expected={dict(expected)}, found={dict(found)}")
    for token, original in originals.items():
        text = text.replace(token, original)
    if any(token in text for token in originals):
        raise ValueError("unrestored opaque marker")
    return text.strip()


def checkpoint(status, rows, records, runtime):
    save({"status": status, "mode": MODE, "input_sha256": INPUT_SHA256,
          "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
          "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
          "records": records, **runtime})


def make_qwen_schema():
    item = {"type": "object", "properties": {
        "key": {"type": "string"}, "semantic_pass": {"type": "boolean"},
        "fluency_pass": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["key", "semantic_pass", "fluency_pass", "reason"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": item, "minItems": 1, "maxItems": 1}},
        "required": ["decisions"], "additionalProperties": False}


def judge_one(row, backtranslation, model, tokenizer, prefix_function):
    system = (
        "You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
        "For semantic_pass, decide whether the target-language question asks the same thing as the English "
        "source: preserve relation, direction, polarity, scope, tense/time, and answer type. Accept faithful "
        "paraphrases, inflection changes, and contextual synonyms. Do not reject merely because a word has "
        "another dictionary sense; judge the translated question in context. Use the English back-translation "
        "as independent diagnostic evidence, not as a requirement for word-for-word equality. For fluency_pass, "
        "judge only whether the translated question is grammatical and natural in the named target language. "
        "Names/numbers are opaque placeholders. Do not answer the questions or infer their facts. If a material "
        "meaning change or substantial fluency problem is uncertain, return false. Give one concise reason. "
        "Return exactly one decision for the supplied key."
    )
    payload = {"target_language": LANGUAGE_NAMES[row["language"]], "items": [{
        "key": row["review_key"], "source_question_en": row["source_question_en"],
        "translated_question": row["translated_question"], "back_translation_en": backtranslation}]}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=512, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id,
                                   prefix_allowed_tokens_fn=prefix_function)
    raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\{.*\}", raw, re.S)
    parsed = json.loads(match.group(0), object_pairs_hook=unique_object) if match else None
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("Qwen returned invalid decisions coverage")
    decision = decisions[0]
    if (not isinstance(decision, dict) or decision.get("key") != row["review_key"]
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
            or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
        raise ValueError("Qwen returned an invalid decision schema or key")
    return raw, decision


def main():
    rows = json.loads(RAW_INPUT)
    if sha(RAW_INPUT) != INPUT_SHA256:
        raise ValueError("embedded input hash mismatch")
    if not rows or len({r.get("review_key") for r in rows}) != len(rows):
        raise ValueError("input has no rows or duplicate review keys")
    for row in rows:
        if (set(row) != {"review_key", "language", "kind", "source_question_en", "translated_question"}
                or row["language"] not in LANGUAGE_NAMES or row["kind"] not in {"atomic", "composition"}
                or not row["source_question_en"].strip() or not row["translated_question"].strip()):
            raise ValueError(f"invalid blinded review input: {row.get('review_key')}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing a slow or different-device run")

    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__,
               "quantization": "4-bit NF4", "gpu": torch.cuda.get_device_name(0),
               "human_translation_reviewed": False, "answers_or_splits_sent": False,
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
    records = []
    bt_tokenizer = NllbTokenizerFast.from_pretrained(NLLB_ID, revision=NLLB_REVISION)
    bt_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_ID, revision=NLLB_REVISION, torch_dtype=torch.float16, device_map="auto")
    bt_model.eval()
    encoded = generated = translated = None
    for index, row in enumerate(rows, 1):
        record = {"review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
                  "source_sha256": sha(row["source_question_en"]),
                  "translation_sha256": sha(row["translated_question"]),
                  "backtranslation_en": "", "backtranslation_error": None, "decision": None, "raw_output": None}
        try:
            protected, originals, expected = mask_placeholders(row["translated_question"])
            bt_tokenizer.src_lang = NLLB_SOURCE_CODES[row["language"]]
            encoded = bt_tokenizer(protected, return_tensors="pt", truncation=True, max_length=256).to(bt_model.device)
            with torch.inference_mode():
                generated = bt_model.generate(**encoded,
                    forced_bos_token_id=bt_tokenizer.convert_tokens_to_ids("eng_Latn"),
                    max_new_tokens=128, do_sample=False)
            translated = bt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            record["backtranslation_en"] = restore_placeholders(translated, originals, expected)
            if not record["backtranslation_en"]:
                raise ValueError("empty backtranslation")
            record["backtranslation_sha256"] = sha(record["backtranslation_en"])
        except Exception as exc:
            record["backtranslation_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        checkpoint("IN_PROGRESS_BACKTRANSLATION", rows, records, runtime)
        print(f"Backtranslated {index}/{len(rows)}", flush=True)

    encoded = generated = translated = None
    del bt_model, bt_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, revision=QWEN_REVISION, torch_dtype=torch.float16,
        quantization_config=quantization, device_map="auto")
    qwen_model.eval()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(qwen_tokenizer, JsonSchemaParser(make_qwen_schema()))
    record_by_key = {r["review_key"]: r for r in records}
    for index, row in enumerate(rows, 1):
        record = record_by_key[row["review_key"]]
        if record["backtranslation_error"]:
            record["decision"] = {"error": "backtranslation_failed; semantic decision withheld"}
        else:
            last_error = None
            for attempt in range(2):
                try:
                    raw, decision = judge_one(row, record["backtranslation_en"], qwen_model, qwen_tokenizer, prefix_function)
                    record["raw_output"], record["decision"] = raw, decision
                    break
                except Exception as exc:
                    last_error = exc
            if record["decision"] is None:
                record["decision"] = {"error": f"{type(last_error).__name__}: {last_error}"}
        checkpoint("IN_PROGRESS_QWEN14B_REVIEW", rows, records, runtime)
        print(f"Reviewed {index}/{len(rows)}", flush=True)

    blocked = sum(
        "error" in (r.get("decision") or {}) or r.get("backtranslation_error") is not None
        for r in records
    )
    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["reviewed_rows"] = sum(isinstance(r.get("decision"), dict) and "error" not in r["decision"] for r in records)
    runtime["blocked_rows"] = blocked
    runtime["status"] = "AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY" if blocked == 0 else "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS"
    save({"status": runtime["status"], "mode": MODE, "input_sha256": INPUT_SHA256,
          "nllb_model_id": NLLB_ID, "nllb_model_revision": NLLB_REVISION,
          "qwen_model_id": QWEN_ID, "qwen_model_revision": QWEN_REVISION,
          "records": records, **runtime})
    print(json.dumps({"status": runtime["status"], "input_rows": len(rows),
                      "reviewed_rows": runtime["reviewed_rows"], "blocked_rows": blocked,
                      "gpu": runtime["gpu"], "quantization": runtime["quantization"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
