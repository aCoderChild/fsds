#!/usr/bin/env python3
"""Generate machine-translation drafts for SFT; never marks them reviewed or approved."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "facebook/nllb-200-distilled-600M"
LANG_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
NUMERIC = re.compile(r"\d+(?:[-/.]\d+)*")
TOKEN = re.compile(r"<ENTITY_[A-Z]+>")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def alpha_code(n):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    while True:
        n, r = divmod(n, len(chars))
        out = chars[r] + out
        if n == 0:
            return out
        n -= 1


def protect(text, entity_names):
    values = sorted({x for x in entity_names if isinstance(x, str) and x}, key=len, reverse=True)
    mapping, counts = {}, {}
    for i, value in enumerate(values):
        token = f"<ENTITY_{alpha_code(i)}>"
        pattern = re.compile(re.escape(value), flags=re.IGNORECASE)
        text, n = pattern.subn(token, text)
        if n:
            mapping[token] = value
            counts[token] = n
    number_index = len(mapping)

    def hide_number(match):
        nonlocal number_index
        token = f"<ENTITY_{alpha_code(number_index)}>"
        number_index += 1
        mapping[token] = match.group(0)
        counts[token] = 1
        return token

    text = NUMERIC.sub(hide_number, text)
    return text, mapping, counts


def restore(text, mapping, counts):
    remaining = counts.copy()
    repairs = []

    def restore_token(match):
        token = match.group(0)
        if token in mapping and remaining.get(token, 0) > 0:
            remaining[token] -= 1
            return mapping[token]
        repairs.append(f"unexpected_or_duplicate:{token}")
        return ""

    text = TOKEN.sub(restore_token, text).strip()
    missing = []
    for token, value in mapping.items():
        if remaining[token]:
            missing.extend([value] * remaining[token])
            repairs.extend([f"omitted:{value}"] * remaining[token])
    if missing:
        suffix = " (" + "; ".join(missing) + ")"
        text = re.sub(r"([?!.,。！？؟])$", lambda m: suffix + m.group(1), text) if text else "(" + "; ".join(missing) + ")"
    return text.strip(), repairs


def translate_batch(model, tokenizer, torch, texts, target_code, device, batch_size):
    tokenizer.src_lang = "eng_Latn"
    target_id = tokenizer.convert_tokens_to_ids(target_code)
    translated = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            out = model.generate(**batch, forced_bos_token_id=target_id, max_new_tokens=128, num_beams=4)
        translated.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return translated


def run(root, batch_size=8, model_id=MODEL_ID):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = MODEL_ID if model_id == "default" else model_id
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    report = {"status": "MACHINE_TRANSLATION_DRAFTS_PENDING_FLUENT_REVIEW", "model_id": model_id, "model_license": "CC-BY-NC-4.0; research/non-commercial use only", "device": device, "batch_size": batch_size, "languages": {}, "approval_claimed": False}
    for lang, code in LANG_CODES.items():
        lang_dir = Path(root) / "sft_translation_requests" / lang
        files = sorted(lang_dir.glob("batch_*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No translation request batches for {lang}: {lang_dir}")
        count = 0
        repaired = 0
        for source_path in files:
            rows = read_jsonl(source_path)
            out_path = Path(root) / "sft_translation_machine_drafts" / lang / source_path.name
            if out_path.is_file():
                old_rows = read_jsonl(out_path)
                if len(old_rows) == len(rows) and all(r.get("question_translation") and r.get("answer_translation") for r in old_rows):
                    for old in old_rows:
                        old["translation_model"] = model_id
                        old.setdefault("protected_text_repairs", {"question": [], "answer": []})
                        repaired += bool(old["protected_text_repairs"].get("question") or old["protected_text_repairs"].get("answer"))
                    out_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in old_rows), encoding="utf-8")
                    count += len(rows)
                    continue
            protected_rows = []
            inputs = []
            for row in rows:
                if row.get("translate_to") != lang:
                    raise ValueError(f"Language mismatch in {source_path}: {row.get('sft_id')}")
                q, qm, qc = protect(row["question_en"], row["protected_entity_names"])
                a, am, ac = protect(row["answer_en"], row["protected_entity_names"])
                protected_rows.append((qm, qc, am, ac))
                inputs.extend((q, a))
            outputs = translate_batch(model, tokenizer, torch, inputs, code, device, batch_size)
            for i, row in enumerate(rows):
                qm, qc, am, ac = protected_rows[i]
                try:
                    row["question_translation"], q_repairs = restore(outputs[2 * i], qm, qc)
                    row["answer_translation"], a_repairs = restore(outputs[2 * i + 1], am, ac)
                except ValueError as first_error:
                    retry_text = (protect(row["question_en"], row["protected_entity_names"])[0], protect(row["answer_en"], row["protected_entity_names"])[0])
                    retry_outputs = translate_batch(model, tokenizer, torch, list(retry_text), code, device, 1)
                    try:
                        row["question_translation"], q_repairs = restore(retry_outputs[0], qm, qc)
                        row["answer_translation"], a_repairs = restore(retry_outputs[1], am, ac)
                    except ValueError as retry_error:
                        raise ValueError(f"Could not preserve protected text for {lang}/{row['sft_id']}; initial={first_error}; retry={retry_error}") from retry_error
                row["translation_status"] = "PENDING_FLUENT_REVIEW"
                row["review_status"] = "PENDING"
                row["translation_method"] = "machine_translation_draft"
                row["translation_model"] = model_id
                row["protected_text_repairs"] = {"question": q_repairs, "answer": a_repairs}
                repaired += bool(q_repairs or a_repairs)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            count += len(rows)
        report["languages"][lang] = {"records": count, "batches": len(files), "records_with_protected_text_repairs": repaired, "status": "PENDING_FLUENT_REVIEW"}
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["request_manifest_sha256"] = hashlib.sha256((Path(root) / "sft_translation_requests" / "manifest.json").read_bytes()).hexdigest()
    write_json(Path(root) / "sft_translation_machine_drafts" / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        raise SystemExit("Install SentencePiece first: python -m pip install sentencepiece")
    print(json.dumps(run(a.root, a.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
