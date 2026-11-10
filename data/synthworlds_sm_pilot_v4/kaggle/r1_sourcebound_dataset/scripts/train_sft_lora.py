#!/usr/bin/env python3
"""Fail-closed LoRA SFT runner. Preflight is default; training requires --train."""
import argparse
import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SUPPORTED_CORPUS_STATUSES = {
    "AI_ONLY_REVIEWED_FIVE_LANGUAGE_SFT_CORPUS",
    "AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_iso_date(value):
    value = str(value or "")
    try:
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) and date.fromisoformat(value))
    except ValueError:
        return False


def encode_assistant_only(tokenizer, messages):
    """Return chat-template token IDs with prompt labels masked to -100."""
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt prefix differs from full conversation; refusing unsafe loss mask")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("No assistant answer tokens after the prompt")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def preflight(corpus_path, manifest_path, output_dir, train=False, include_splits=("forget", "retain"), seed=42):
    corpus_path, manifest_path = Path(corpus_path), Path(manifest_path)
    errors = []
    if not corpus_path.is_file() or not manifest_path.is_file():
        errors.append("Approved SFT corpus and manifest are required")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in SUPPORTED_CORPUS_STATUSES or manifest.get("do_not_train") is not False:
            errors.append("Corpus manifest is not approved for training")
        if manifest.get("translation_review_mode") != "AI_ONLY" or manifest.get("human_translation_reviewed") is not False:
            errors.append("Corpus review mode is missing or human-review status is misrepresented")
        if manifest.get("translation_review_authorization") != "AI_ONLY_REVIEW_AUTHORIZED" or manifest.get("ai_reviewed_rows") != manifest.get("unique_sft_ids_per_language", 0) * 4:
            errors.append("Corpus lacks complete user-authorized AI-only translation review metadata")
        split_approval = manifest.get("split_approval", {})
        project_root = corpus_path.parent.parent if corpus_path.parent.name == "sft_training" else corpus_path.parent
        approval_path = project_root / "split_approval.json"
        if (
            split_approval.get("status") != "APPROVED_PILOT_SPLIT"
            or not str(split_approval.get("reviewer_name", "")).strip()
            or not valid_iso_date(split_approval.get("review_date"))
            or split_approval.get("reviewer_type") != "AI"
            or not approval_path.is_file()
            or split_approval.get("record_sha256") != sha(approval_path)
        ):
            errors.append("Corpus manifest lacks an intact, explicitly AI-attributed pilot split record")
        if manifest.get("corpus_sha256") != sha(corpus_path):
            errors.append("Corpus checksum does not match manifest")
        counts = manifest.get("rows_by_language", {})
        if any(counts.get(lang) != manifest.get("unique_sft_ids_per_language") for lang in ("en", "fr", "de", "hi", "th")):
            errors.append("Five-language row counts are incomplete or unequal")
        if manifest.get("holdout_rows") != 0:
            errors.append("Holdout rows are forbidden")
        if not errors:
            with corpus_path.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            actual = {lang: sum(row.get("language") == lang for row in rows) for lang in ("en", "fr", "de", "hi", "th")}
            if actual != counts or len(rows) != manifest.get("total_rows"):
                errors.append("Corpus rows do not match manifest language counts")
            if any(row.get("split") == "holdout" for row in rows):
                errors.append("Corpus contains a holdout row")
    available = {name: __import__("importlib.util").util.find_spec(name) is not None for name in ("torch", "transformers", "peft", "accelerate", "numpy")}
    if train and not all(available.values()):
        errors.append("Missing training packages: " + ", ".join(k for k, v in available.items() if not v))
    token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token:
        try:
            from huggingface_hub import get_token
            token = bool(get_token())
        except (ImportError, OSError):
            pass
    if any(s not in {"forget", "retain", "train_only", "holdout"} for s in include_splits) or "holdout" in include_splits:
        errors.append("Training splits may include forget/retain/train_only; holdout is forbidden")
    if train and not token:
        errors.append("No Hugging Face access token found; gated Llama weights cannot be loaded")
    status = "READY_TO_TRAIN" if not errors else "BLOCKED"
    result = {"status": status, "training_started": False, "train_flag": train, "model_id": MODEL_ID, "errors": errors, "translation_review_mode": "AI_ONLY" if not errors else None, "human_translation_reviewed": False, "training_packages": available, "training_packages_ready": all(available.values()), "hf_token_present": token, "note": "Preflight never downloads weights or starts training. The approved corpus uses user-authorized AI-only translation review; no human fluency review was performed."}
    if train and not errors:
        result = run_training(corpus_path, manifest_path, output_dir, "AI_ONLY", include_splits, seed, manifest.get("training_variant", "atomic_five_language_sft"))
    return result


def run_training(corpus_path, manifest_path, output_dir, translation_review_mode, include_splits=("forget", "retain"), seed=42, corpus_variant="atomic_five_language_sft"):
    import random
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=True, torch_dtype=dtype)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = []
    with Path(corpus_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                if ex.get("split") not in include_splits:
                    continue
                ids, labels = encode_assistant_only(tok, ex["messages"])
                rows.append({"input_ids": ids, "labels": labels})

    class Collator:
        def __call__(self, batch):
            width = max(len(x["input_ids"]) for x in batch)
            ids, masks, labels = [], [], []
            for x in batch:
                n = width - len(x["input_ids"])
                ids.append(x["input_ids"] + [tok.pad_token_id] * n)
                masks.append([1] * len(x["input_ids"]) + [0] * n)
                labels.append(x["labels"] + [-100] * n)
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=1, gradient_accumulation_steps=32, learning_rate=2e-4, logging_steps=10, save_strategy="epoch", report_to="none", remove_unused_columns=False, seed=seed, data_seed=seed, fp16=(device == "cuda"))
    trainer = Trainer(model=model, args=args, train_dataset=rows, data_collator=Collator())
    trainer.train()
    output_dir = Path(output_dir)
    trainer.save_model(output_dir / "adapter")
    tok.save_pretrained(output_dir / "adapter")
    if corpus_variant == "atomic_five_language_sft":
        training_variant = "retain_only" if set(include_splits) == {"retain"} else "full_sft"
    else:
        training_variant = "composition_retain_only" if "forget" not in set(include_splits) else "composition_sft"
    run_rows = [json.loads(line) for line in Path(corpus_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    run_rows = [row for row in run_rows if row.get("split") in include_splits]
    result = {"status": "TRAINING_COMPLETE", "training_started": True, "model_id": MODEL_ID, "device": device, "seed": seed,
              "rows": len(rows), "rows_by_language": {lang: sum(row.get("language") == lang for row in run_rows) for lang in ("en", "fr", "de", "hi", "th")},
              "rows_by_split": {split: sum(row.get("split") == split for row in run_rows) for split in include_splits},
              "included_splits": list(include_splits), "training_variant": training_variant, "corpus_variant": corpus_variant,
              "corpus_sha256": sha(corpus_path), "corpus_manifest_sha256": sha(manifest_path),
              "epochs": 3, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 32,
              "learning_rate": 2e-4, "loss_mask": "assistant answer tokens only", "translation_review_mode": translation_review_mode,
              "human_translation_reviewed": False, "adapter_dir": str(output_dir / "adapter")}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parents[1] / "sft_training" / "corpus.jsonl")
    p.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "sft_training" / "manifest.json")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "sft_training" / "llama32_1b_lora")
    p.add_argument("--train", action="store_true", help="Start training only after all fail-closed checks pass")
    p.add_argument("--include-splits", nargs="+", choices=("forget", "retain", "train_only"), default=("forget", "retain"), help="Training splits; composition practice is explicitly train_only and holdout is never allowed")
    p.add_argument("--seed", type=int, default=42, help="Seed for reproducible initialization and data order")
    a = p.parse_args()
    result = preflight(a.corpus, a.manifest, a.output_dir, a.train, a.include_splits, a.seed)
    print(json.dumps(result, indent=2))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
