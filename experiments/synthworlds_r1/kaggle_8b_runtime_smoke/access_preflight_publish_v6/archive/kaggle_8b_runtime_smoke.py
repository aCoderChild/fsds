#!/usr/bin/env python3
"""Run one synthetic 8B NF4 LoRA optimizer step on a Kaggle GPU.

This checks runtime compatibility only. It does not read the research corpus,
produce a model result, or certify that an experiment dataset is ready.
"""
from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EXPECTED = {
    "transformers": "5.22.1",
    "peft": "0.21.2",
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
}


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:  # Kaggle-only secret service; never print credential.
        raise RuntimeError(f"HF_TOKEN unavailable ({type(exc).__name__})") from exc
    if not token:
        raise RuntimeError("HF_TOKEN is empty")
    return token


def main() -> int:
    output = Path(os.environ.get("R1_SMOKE_OUTPUT", "/kaggle/working/r1_8b_runtime_smoke.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict = {
        "status": "STARTED",
        "model_id": MODEL_ID,
        "scope": "SYNTHETIC_RUNTIME_SMOKE_ONLY",
        "research_data_loaded": False,
        "optimizer_steps_requested": 1,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    try:
        os.environ.setdefault("HF_HOME", "/kaggle/working/.cache/huggingface-r1-8b-smoke")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch
        from huggingface_hub import HfApi, hf_hub_download
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )

        actual = {name: version(name) for name in EXPECTED}
        receipt["package_versions"] = actual
        if actual != EXPECTED:
            raise RuntimeError(f"Pinned package mismatch: expected={EXPECTED}, actual={actual}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available")
        torch.cuda.set_device(0)
        receipt["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
        token = get_token()
        info = HfApi(token=token).model_info(MODEL_ID, token=token)
        revision = info.sha
        if not revision or len(revision) != 40:
            raise RuntimeError("Could not resolve a full immutable Hugging Face commit")
        receipt["model_revision"] = revision
        config_path = Path(
            hf_hub_download(
                repo_id=MODEL_ID,
                filename="config.json",
                revision=revision,
                token=token,
            )
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("model_type") != "llama":
            raise RuntimeError("Pinned config is not a Llama model config")
        receipt["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        receipt["config_model_type"] = config["model_type"]
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, revision=revision)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=token,
            revision=revision,
            dtype=torch.float16,
            quantization_config=quantization,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        receipt["model_load_seconds"] = round(time.monotonic() - started, 2)
        receipt["model_loaded_4bit"] = bool(getattr(model, "is_loaded_in_4bit", False))
        if not receipt["model_loaded_4bit"]:
            raise RuntimeError("Model did not report a 4-bit load")

        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        messages = [
            {"role": "user", "content": "For this software check only, reply with exactly OK."},
            {"role": "assistant", "content": "OK"},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
            raise RuntimeError("Assistant-only loss mask does not match the tokenizer chat template")
        sample = {
            "input_ids": full_ids,
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
        }

        def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
            width = max(len(row["input_ids"]) for row in batch)
            input_ids, labels, masks = [], [], []
            for row in batch:
                pad = width - len(row["input_ids"])
                input_ids.append(row["input_ids"] + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad)
                labels.append(row["labels"] + [-100] * pad)
                masks.append([1] * len(row["input_ids"]) + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

        args = TrainingArguments(
            output_dir="/kaggle/working/r1_8b_runtime_smoke_trainer",
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=0,
            disable_tqdm=True,
            seed=42,
            data_seed=42,
        )
        trainer = Trainer(model=model, args=args, train_dataset=[sample], data_collator=collate)
        trainer.train()
        steps = int(trainer.state.global_step)
        if steps != 1:
            raise RuntimeError(f"Trainer completed {steps} optimizer steps, expected 1")
        receipt.update(
            {
                "status": "PASS_RUNTIME_ONLY",
                "trainer_global_step": steps,
                "last_loss": float(trainer.state.log_history[-1]["loss"]),
                "cuda_peak_memory_gib": round(torch.cuda.max_memory_reserved(0) / 1024**3, 2),
                "research_data_loaded": False,
                "interpretation": "Exact pinned stack loaded the gated 8B checkpoint as NF4 and completed one synthetic LoRA Trainer step; this is not an experiment result.",
            }
        )
        del trainer, model, tokenizer
        torch.cuda.empty_cache()
    except Exception as exc:
        receipt.update(
            {
                "status": "FAIL_RUNTIME_ONLY",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "research_data_loaded": False,
            }
        )
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise

    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
