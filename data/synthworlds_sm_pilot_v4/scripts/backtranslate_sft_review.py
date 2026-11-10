#!/usr/bin/env python3
"""Back-translate SFT drafts as an AI-only semantic-drift audit; never grants human approval."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "facebook/nllb-200-distilled-600M"
LANG_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backtranslate(model, tokenizer, torch, texts, source_code, batch_size, device):
    tokenizer.src_lang = source_code
    target_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    out = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            generated = model.generate(**batch, forced_bos_token_id=target_id, max_new_tokens=128, num_beams=4)
        out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return out


def run(root, output_dir, batch_size=8):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype=dtype, local_files_only=True).to(device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "AI_BACKTRANSLATION_AUDIT_ONLY_NOT_HUMAN_APPROVAL", "model_id": MODEL_ID, "device": device, "languages": {}, "approval_claimed": False}
    for lang, source_code in LANG_CODES.items():
        rows = read_jsonl(root / "sft_translation_candidates" / f"{lang}.jsonl")
        texts = []
        records = []
        for row in rows:
            for field, source_key, target_key in (
                ("question", "question_en", "question"),
                ("answer", "answer_en", "answer_translation"),
            ):
                texts.append(row[target_key])
                records.append({"sft_id": row["sft_id"], "language": lang, "field": field, "source_en": row[source_key], "translation": row[target_key]})
        back = backtranslate(model, tokenizer, torch, texts, source_code, batch_size, device)
        for row, english in zip(records, back):
            row["backtranslation_en"] = english
        path = output_dir / f"{lang}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        summary["languages"][lang] = {"records": len(rows), "fields_backtranslated": len(records), "source_sha256": sha(root / "sft_translation_candidates" / f"{lang}.jsonl"), "audit_sha256": sha(path)}
    (output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    output = a.output_dir or a.root / "sft_translation_ai_review" / "backtranslations"
    print(json.dumps(run(a.root, output, a.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
