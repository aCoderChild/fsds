#!/usr/bin/env python3
"""Draft NLLB translations of question-only evaluation prompts.

Translations remain pending AI semantic review and human review. This script
never marks a translation approved.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from draft_sft_translations_nllb import LANG_CODES, MODEL_ID, protect, restore, translate_batch


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(root, batch_size=8, model_id=MODEL_ID):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype, local_files_only=True).to(device).eval()
    out_root = root / "evaluation_dataset" / "translations"
    report = {
        "status": "NLLB_DRAFTS_PENDING_AI_SEMANTIC_REVIEW_HUMAN_REVIEW_NOT_PERFORMED",
        "translation_model": model_id,
        "device": device,
        "languages": {},
        "approval_claimed": False,
    }
    for lang, code in LANG_CODES.items():
        requests = read_jsonl(root / "evaluation_dataset" / "translation_requests" / f"{lang}.jsonl")
        protected = []
        inputs = []
        for row in requests:
            q, mapping, counts = protect(row["source_question"], row["protected_entity_names"])
            protected.append((mapping, counts))
            inputs.append(q)
        drafts = translate_batch(model, tokenizer, torch, inputs, code, device, batch_size)
        outputs = []
        repair_count = 0
        for request, draft, (mapping, counts) in zip(requests, drafts, protected):
            restored, repairs = restore(draft, mapping, counts)
            # Detect if protected content had to be appended or repaired.
            repair_count += bool(repairs)
            outputs.append({
                "translation_id": request["translation_id"],
                "instance_id": request["instance_id"],
                "probe": request["probe"],
                "split": request["split"],
                "language": lang,
                "question": restored,
                "source_question_en": request["source_question"],
                "translation_model": model_id,
                "translation_status": "MACHINE_DRAFT_PENDING_AI_REVIEW",
                "review_status": "PENDING",
                "review_method": "NOT_STARTED",
                "human_review_status": "NOT_PERFORMED",
                "protected_text_repairs": repairs,
                "do_not_train": True,
            })
        path = out_root / f"{lang}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
        report["languages"][lang] = {
            "rows": len(outputs),
            "rows_with_protected_text_repairs": repair_count,
            "identical_to_english": sum(r["question"].strip() == r["source_question_en"].strip() for r in outputs),
            "status": "MACHINE_DRAFT_PENDING_AI_REVIEW",
        }
    (out_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        sys.exit("Install SentencePiece: python -m pip install sentencepiece")
    print(json.dumps(run(args.root, args.batch_size), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
