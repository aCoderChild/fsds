#!/usr/bin/env python3
"""Run one synthetic 8B NF4 LoRA optimizer step on a Kaggle GPU.

This checks runtime compatibility only. It does not read the research corpus,
produce a model result, or certify that an experiment dataset is ready.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PINNED_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
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
        "runner_sha256": os.environ.get("R1_SMOKE_SOURCE_SHA256"),
        "model_id": MODEL_ID,
        "scope": "SYNTHETIC_RUNTIME_SMOKE_ONLY",
        "research_data_loaded": False,
        "weights_load_attempted": False,
        "model_weights_loaded": False,
        "training_started": False,
        "optimizer_steps_requested": 1,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    try:
        os.environ.setdefault("HF_HOME", "/kaggle/working/.cache/huggingface-r1-8b-smoke")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch
        from huggingface_hub import HfApi, hf_hub_download
        import torch.nn.functional as F
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from assistant_masking import encode_assistant_only
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
        from unlearning_objectives import npo_loss

        actual = {name: version(name) for name in EXPECTED}
        receipt["package_versions"] = actual
        if actual != EXPECTED:
            raise RuntimeError(f"Pinned package mismatch: expected={EXPECTED}, actual={actual}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available")
        if torch.cuda.device_count() < 2:
            raise RuntimeError("The R1 NPO path requires two visible CUDA GPUs for policy and frozen reference models")
        torch.cuda.set_device(0)
        receipt["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
        token = get_token()
        # PEFT's local-adapter loader may perform an optional hub config lookup
        # without a token argument. Keep the already-authorized token in this
        # process environment so that lookup does not fall back to anonymous
        # access; never include it in a receipt or log.
        os.environ["HF_TOKEN"] = token
        info = HfApi(token=token).model_info(
            MODEL_ID, revision=PINNED_REVISION, token=token
        )
        revision = info.sha
        if not revision or len(revision) != 40:
            raise RuntimeError("Could not resolve a full immutable Hugging Face commit")
        if revision != PINNED_REVISION:
            raise RuntimeError("Resolved model commit differs from the frozen R1 revision")
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

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, revision=revision, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for offset-based assistant-only loss masking")

        # Validate the chat-template mask before downloading/loading the 8B
        # weights. This makes a tokenizer/template incompatibility a cheap,
        # fail-closed preflight rather than a multi-GB failed GPU run.
        messages = [
            {"role": "user", "content": "For this software check only, reply with exactly OK."},
            {"role": "assistant", "content": "OK"},
        ]
        full_ids, labels = encode_assistant_only(tokenizer, messages)
        supervised_tokens = sum(label != -100 for label in labels)
        if supervised_tokens < 1:
            raise RuntimeError("Assistant-only mask contains no supervised answer tokens")
        receipt["loss_mask_method"] = "single_render_fast_tokenizer_offsets"
        receipt["assistant_supervised_tokens"] = supervised_tokens
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        started = time.monotonic()
        receipt["weights_load_attempted"] = True
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
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
        receipt["model_weights_loaded"] = True
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
        sample = {
            "input_ids": full_ids,
            "labels": labels,
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
        receipt["training_started"] = True
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        train_result = trainer.train()
        steps = int(trainer.state.global_step)
        if steps != 1:
            raise RuntimeError(f"Trainer completed {steps} optimizer steps, expected 1")
        train_loss = train_result.metrics.get("train_loss")
        if not isinstance(train_loss, (int, float)) or not math.isfinite(float(train_loss)):
            raise RuntimeError(f"Trainer did not return a finite train_loss metric: {train_loss!r}")

        # NPO keeps a frozen copy of the SFT adapter on a second GPU. Exercise
        # that exact two-model path with the synthetic sample before accepting
        # this receipt for the main-study notebook.
        model.train()
        npo_beta = 0.1
        with tempfile.TemporaryDirectory(prefix="r1_8b_npo_reference_") as temp_dir:
            reference_adapter_dir = Path(temp_dir) / "adapter"
            model.save_pretrained(reference_adapter_dir)
            reference_base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=token,
                revision=revision,
                dtype=torch.float16,
                quantization_config=quantization,
                device_map={"": 1},
                low_cpu_mem_usage=True,
            )
            reference_base.config.use_cache = False
            reference_model = PeftModel.from_pretrained(
                reference_base, reference_adapter_dir, is_trainable=False
            )
            reference_model.eval()
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)

            policy_device = model.get_input_embeddings().weight.device
            reference_device = reference_model.get_input_embeddings().weight.device
            policy_ids = torch.tensor([full_ids], dtype=torch.long, device=policy_device)
            policy_labels = torch.tensor([sample["labels"]], dtype=torch.long, device=policy_device)
            reference_ids = torch.tensor([full_ids], dtype=torch.long, device=reference_device)
            reference_labels = torch.tensor([sample["labels"]], dtype=torch.long, device=reference_device)

            def sequence_logp(net, input_ids, labels):
                result = net(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                logits = result.logits[:, :-1].float()
                shifted_labels = labels[:, 1:]
                token_nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape_as(shifted_labels)
                answer_mask = shifted_labels.ne(-100)
                return -(token_nll * answer_mask).sum()

            with torch.no_grad():
                reference_logp = sequence_logp(reference_model, reference_ids, reference_labels)
            policy_logp = sequence_logp(model, policy_ids, policy_labels)
            npo_objective = npo_loss(policy_logp, reference_logp.to(policy_logp.device), npo_beta)
            if not torch.isfinite(npo_objective):
                raise RuntimeError("Synthetic NPO objective is not finite")
            npo_optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-4
            )
            npo_optimizer.zero_grad(set_to_none=True)
            npo_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            npo_optimizer.step()
            npo_loss_value = float(npo_objective.detach().float().cpu())
            if reference_device.index != 1:
                raise RuntimeError(f"NPO reference model loaded on unexpected device {reference_device}")
            del reference_model, reference_base, npo_optimizer

        receipt.update(
            {
                "status": "PASS_RUNTIME_ONLY",
                "trainer_global_step": steps,
                "npo_optimizer_steps_requested": 1,
                "npo_optimizer_steps_completed": 1,
                "npo_beta": npo_beta,
                "npo_loss": npo_loss_value,
                "reference_model_loaded_4bit": True,
                "npo_reference_cuda_device": 1,
                "trainer_train_loss": float(train_loss),
                "cuda_peak_memory_gib_by_gpu": [
                    round(torch.cuda.max_memory_reserved(i) / 1024**3, 2)
                    for i in range(torch.cuda.device_count())
                ],
                "research_data_loaded": False,
                "interpretation": "Exact pinned stack loaded two NF4 copies across two GPUs, completed one synthetic LoRA Trainer step and one synthetic NPO update; this is not an experiment result.",
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
