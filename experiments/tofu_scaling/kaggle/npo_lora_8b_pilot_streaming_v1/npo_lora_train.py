#!/usr/bin/env python3
"""Resource-bounded LoRA NPO for the pinned Open-Unlearning TOFU checkpoints.

The NPO and retain objectives follow Open-Unlearning at the pinned revision.
Only the LoRA adapter is trained; this is explicitly an adapter-based NPO
variant, not a reproduction of full-parameter NPO.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from tofu_official_eval import (
    CHECKPOINTS,
    DATASET_ID,
    DATASET_REVISION,
    UPSTREAM_REVISION,
    collate_scoring,
    encode_answer,
    get_bytes,
)
from npo_objective import objective_from_nlls, sequence_nll


METHOD = {
    "method": "NPO + retain NLL (LoRA adapter)",
    "beta": 0.1,
    "gamma_forget": 1.0,
    "alpha_retain": 1.0,
    "retain_loss_type": "NLL",
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "optimizer_betas": [0.9, 0.999],
    "optimizer_eps": 1e-8,
    "epochs": 10,
    "max_length": 512,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "gradient_clip_norm": 1.0,
    "batching": "2 forget + 2 retain examples per microbatch; effective paired batch 8",
    "optimizer": "torch AdamW, betas=(0.9,0.999), eps=1e-8",
    "scheduler": "linear; one epoch warmup",
    "precision": "FP16 frozen base checkpoint; FP32 LoRA parameters and loss; SDPA",
    "retain_sampling": "uniform with replacement, one retain row per forget row, as in pinned upstream anchor=forget wrapper",
}


MODEL_FAMILIES = {
    "3b": ("llama_3_2_3b_full", 2, 4),
    "8b": ("llama_3_1_8b_full", 1, 8),
}


def fetch_jsonl(filename: str) -> tuple[list[dict], str]:
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{filename}"
    raw = get_bytes(url)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str) for row in rows):
        raise ValueError(f"Invalid pinned TOFU train file: {filename}")
    return rows, hashlib.sha256(raw).hexdigest()


def collate(rows: list[tuple[list[int], list[int]]], pad_id: int, device: torch.device):
    input_ids, attention, labels = collate_scoring(rows, pad_id)
    return input_ids.to(device), attention.to(device), labels.to(device)


def encode_rows(tokenizer, rows: list[dict]) -> list[tuple[list[int], list[int]]]:
    encoded = [encode_answer(tokenizer, row["question"], row["answer"]) for row in rows]
    if any(len(ids) > METHOD["max_length"] for ids, _ in encoded):
        raise ValueError("Found a training sequence longer than the pinned 512-token limit; no silent truncation is allowed")
    return encoded


def reference_nlls(model, encoded, pad_id: int, device: torch.device, batch_size: int = 1) -> list[float]:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            ids, attention, labels = collate(encoded[start:start + batch_size], pad_id, device)
            outputs = model(input_ids=ids, attention_mask=attention, use_cache=False)
            nll, _ = sequence_nll(outputs.logits, labels)
            values.extend(float(x) for x in nll.cpu().tolist())
    if len(values) != len(encoded) or not all(math.isfinite(x) and x >= 0 for x in values):
        raise RuntimeError("Reference NLL precomputation was incomplete or non-finite")
    return values


def train(args) -> dict:
    global METHOD
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    objective_check = json.loads(args.objective_check.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        raise RuntimeError("NPO protocol is not marked preregistered; refusing to train")
    if objective_check.get("status") != "PASS":
        raise RuntimeError("Pinned-objective loss/gradient equivalence did not pass")
    expected_packages = {"transformers": "5.22.1", "accelerate": "1.15.0", "peft": "0.21.2", "scipy": "1.16.3"}
    actual_packages = {name: importlib.metadata.version(name) for name in expected_packages}
    if actual_packages != expected_packages:
        raise RuntimeError(f"Training package versions differ from the verified stack: {actual_packages}")
    torchao_spec = importlib.util.find_spec("torchao")
    if torchao_spec is not None:
        try:
            torchao_version = importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            torchao_version = "unknown-version"
        raise RuntimeError(
            f"Optional torchao {torchao_version} is installed. This unquantized LoRA run does not use torchao; "
            "remove it before loading the base model to keep PEFT on the ordinary Linear dispatcher."
        )
    recovery_protocol = protocol.get("protocol_id") in {
        "tofu_3b_npo_lora_profile_recovery_v1",
        "tofu_3b_npo_lora_profile_alpha40_followup_v1",
    }
    if recovery_protocol:
        if args.cohort not in {"dev", "final"}:
            raise RuntimeError("Recovery training requires an explicit --cohort dev|final")
        training = protocol["training"]
        allowed_alpha = protocol["selection"]["candidate_retain_nll_weights_ascending"]
        if args.alpha_retain is None or float(args.alpha_retain) not in allowed_alpha:
            raise RuntimeError(f"--alpha-retain must be one of the preregistered candidates {allowed_alpha}")
        METHOD = dict(METHOD)
        METHOD.update({
            "beta": float(protocol["objective"]["beta"]),
            "gamma_forget": float(protocol["objective"]["forget_weight_gamma"]),
            "alpha_retain": float(args.alpha_retain),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "optimizer_betas": list(training["optimizer_betas"]),
            "optimizer_eps": float(training["optimizer_eps"]),
            "scheduler": training["scheduler"],
            "precision": training["precision"],
            "epochs": int(training["epochs"]),
            "max_length": int(training["max_length"]),
            "lora_rank": int(training["lora_rank"]),
            "lora_alpha": int(training["lora_alpha"]),
            "lora_dropout": float(training["lora_dropout"]),
            "lora_target_modules": list(training["target_modules"]),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
            "batching": "2 forget + 2 retain examples per microbatch; effective paired batch 8",
            "retain_sampling": "uniform with replacement from optimizer-only retain profiles; seed 42",
        })
        expected_model_family = protocol["model"]["model_family"]
        expected_seed = int(protocol["model"]["seed"])
        locked_checkpoint = protocol["model"]
    else:
        if args.cohort is not None or args.alpha_retain is not None:
            raise RuntimeError("--cohort/--alpha-retain are only valid for the preregistered recovery protocol")
        expected_model_family = protocol.get("model_family")
        expected_seed = protocol.get("seed")
        locked_checkpoint = protocol.get("base_checkpoint", {})

    if args.model != expected_model_family or args.seed != expected_seed:
        raise RuntimeError("Requested model/seed does not match the locked NPO diagnostic protocol")
    model_key, microbatch, accumulation = MODEL_FAMILIES[args.model]
    spec = CHECKPOINTS[model_key]
    if (spec.get("repo_id"), spec.get("revision")) != (locked_checkpoint.get("repo_id"), locked_checkpoint.get("revision")):
        raise RuntimeError("Trainer checkpoint/revision differs from the preregistered protocol")
    if objective_check.get("upstream") != f"locuslab/open-unlearning@{UPSTREAM_REVISION}":
        raise RuntimeError("Objective receipt is for the wrong Open-Unlearning revision")
    if float(objective_check.get("npo_beta", -1)) != float(protocol["objective"]["beta"]):
        raise RuntimeError("Objective check beta differs from the preregistered protocol")
    if recovery_protocol:
        locked_training = protocol["training"]
        expected_modules = locked_training["target_modules"]
        expected_precision = locked_training["precision"]
        expected_scheduler = locked_training["scheduler"]
        expected_effective_batch = locked_training["effective_paired_batch_size"]
        expected_clip = locked_training["gradient_clip_norm"]
        expected_retain_loss = "NLL"
    else:
        locked_training = protocol["adaptation"]
        expected_modules = locked_training["target_modules"]
        expected_precision = locked_training["precision"]
        expected_scheduler = locked_training["scheduler"]
        expected_effective_batch = locked_training["effective_paired_batch_size"]
        expected_clip = locked_training["gradient_clip_norm"]
        expected_retain_loss = "NLL"
    if (
        METHOD["lora_rank"] != locked_training["lora_rank"]
        or METHOD["lora_alpha"] != locked_training["lora_alpha"]
        or METHOD["lora_dropout"] != locked_training["lora_dropout"]
        or METHOD["lora_target_modules"] != expected_modules
        or METHOD["learning_rate"] != locked_training["learning_rate"]
        or METHOD["weight_decay"] != (float(locked_training["weight_decay"]) if recovery_protocol else float(locked_training["optimizer"].split("weight_decay=")[-1].rstrip(")")))
        or (recovery_protocol and METHOD["optimizer_betas"] != locked_training["optimizer_betas"])
        or (recovery_protocol and METHOD["optimizer_eps"] != locked_training["optimizer_eps"])
        or METHOD["beta"] != protocol["objective"]["beta"]
        or METHOD["gamma_forget"] != protocol["objective"]["forget_weight_gamma"]
        or (not recovery_protocol and METHOD["alpha_retain"] != protocol["objective"]["retain_weight_alpha"])
        or METHOD["retain_loss_type"] != expected_retain_loss
        or METHOD["precision"] != expected_precision
        or METHOD["scheduler"] != expected_scheduler
        or METHOD["epochs"] != locked_training["epochs"]
        or METHOD["max_length"] != locked_training["max_length"]
        or METHOD["gradient_clip_norm"] != expected_clip
        or microbatch * accumulation != expected_effective_batch
    ):
        raise RuntimeError("Trainer hyperparameters differ from the preregistered NPO-LoRA protocol")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("NPO requires two GPUs; stopped before model download")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if gpu_names.count("Tesla T4") != 2:
        raise RuntimeError(f"Protocol requires Kaggle T4 x2, found {gpu_names}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    run_type = "bounded_real_data_smoke" if args.smoke else "exploratory_single_seed"
    started = time.monotonic()
    receipt = {
        "status": "RUNNING",
        "run_type": run_type,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": spec,
        "dataset": {"repo": DATASET_ID, "revision": DATASET_REVISION},
        "upstream_code_revision": UPSTREAM_REVISION,
        "method": METHOD,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "objective_equivalence": objective_check,
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "gpu_names": gpu_names,
        "package_versions": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "peft": importlib.metadata.version("peft"),
            "scipy": importlib.metadata.version("scipy"),
            "torchao": None,
        },
        "gpu_total_memory_bytes": [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())],
        "smoke_rows": 8 if args.smoke else None,
        "training_started": False,
        "adapter_saved": False,
    }
    if recovery_protocol:
        receipt["followup"] = {
            "protocol_id": protocol["protocol_id"],
            "cohort": args.cohort,
            "selected_alpha_retain": float(args.alpha_retain),
            "study_type": protocol["study_type"],
        }
    receipt_path = output_dir / "training_receipt.json"
    trace_path = output_dir / "loss_trace.jsonl"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    try:
        forget_rows, forget_sha = fetch_jsonl("forget10.json")
        retain_rows, retain_sha = fetch_jsonl("retain90.json")
        if len(forget_rows) != 400 or len(retain_rows) != 3600:
            raise ValueError(f"Unexpected pinned train counts: forget={len(forget_rows)}, retain={len(retain_rows)}")
        if recovery_protocol:
            expected_hashes = protocol["data"]["sha256"]
            if forget_sha != expected_hashes["forget10.json"] or retain_sha != expected_hashes["retain90.json"]:
                raise ValueError("Fetched training data hash differs from the preregistered recovery protocol")
            block_size = int(protocol["data"]["profile_block_size"])
            assignments = protocol["data"]["profile_assignment_ranges_inclusive"]
            forget_key = f"forget10_{args.cohort}_train_and_eval"
            if forget_key not in assignments:
                raise ValueError(f"No forget profile assignment for cohort {args.cohort}")
            forget_blocks = [b for first, last in assignments[forget_key] for b in range(first, last + 1)]
            retain_blocks = [b for first, last in assignments["retain90_optimizer_train"] for b in range(first, last + 1)]
            if len(forget_blocks) != len(set(forget_blocks)) or len(retain_blocks) != len(set(retain_blocks)):
                raise ValueError("Recovery training profile assignments contain duplicates")
            forget_source_indices = [i for i in range(len(forget_rows)) if i // block_size in forget_blocks]
            retain_source_indices = [i for i in range(len(retain_rows)) if i // block_size in retain_blocks]
            forget_rows = [forget_rows[i] for i in forget_source_indices]
            retain_rows = [retain_rows[i] for i in retain_source_indices]
            if len(forget_rows) != 200 or len(retain_rows) != 2760:
                raise ValueError(f"Wrong {args.cohort} cohort size: forget={len(forget_rows)}, retain={len(retain_rows)}")
            receipt["dataset"].update({
                "forget10_source_rows_selected": forget_source_indices,
                "forget10_profile_blocks_selected": forget_blocks,
                "retain90_source_rows_selected": retain_source_indices,
                "retain90_profile_blocks_selected": retain_blocks,
                "forget10_rows_selected": len(forget_rows),
                "retain90_rows_selected": len(retain_rows),
            })
        else:
            locked_files = protocol["dataset"]["training"]
            if forget_sha != locked_files["forget_file"]["sha256"] or retain_sha != locked_files["retain_file"]["sha256"]:
                raise ValueError("Fetched training data hash differs from the preregistered dataset revision and hash")
        if not recovery_protocol and (len(forget_rows) != 400 or len(retain_rows) != 3600):
            raise ValueError("Fetched training data hash differs from the preregistered dataset revision and hash")
        receipt["dataset"].update({
            "forget10_rows": len(forget_rows),
            "retain90_rows": len(retain_rows),
            "forget10_sha256": forget_sha,
            "retain90_sha256": retain_sha,
        })

        cache_root = Path(os.environ.get("TOFU_MODEL_CACHE_ROOT", "/kaggle/working/model_cache"))
        checkpoint_cache = cache_root / model_key
        tokenizer = AutoTokenizer.from_pretrained(
            spec["repo_id"], revision=spec["revision"], cache_dir=str(checkpoint_cache), use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not tokenizer.is_fast:
            raise RuntimeError("Pinned tokenizer must be fast")
        forget_encoded = encode_rows(tokenizer, forget_rows)
        retain_encoded = encode_rows(tokenizer, retain_rows)
        receipt["tokenization"] = {
            "max_length_observed": max(len(x[0]) for x in forget_encoded + retain_encoded),
            "truncated_examples": 0,
            "prompt_answer_prefix_check": "PASS for every sequence",
        }

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(
            spec["repo_id"],
            revision=spec["revision"],
            cache_dir=str(checkpoint_cache),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="balanced",
            max_memory={0: "13GiB", 1: "13GiB", "cpu": "40GiB"},
            attn_implementation="sdpa",
        )
        device_map = getattr(model, "hf_device_map", {})
        offload = sorted({str(v) for v in device_map.values() if str(v).lower() in {"cpu", "disk"}})
        if offload:
            raise RuntimeError(f"Refusing training with CPU/disk offload: {offload}")
        input_device = model.get_input_embeddings().weight.device
        receipt["runtime"] = {
            "device_map": {str(k): str(v) for k, v in device_map.items()},
            "cpu_or_disk_offload": False,
            "input_device": str(input_device),
            "dtype": "float16",
            "attention": "sdpa",
        }

        if args.smoke:
            forget_encoded = forget_encoded[:8]
            # Keep the full retain pool in the real-data smoke; sample only the
            # first eight target steps to bound runtime and avoid a scored run.
            steps_to_run = 1
            batch_size = 1
            accumulation_steps = 1
            epochs = 1
        else:
            steps_to_run = None
            batch_size = microbatch
            accumulation_steps = accumulation
            epochs = METHOD["epochs"]

        effective_forget = 8
        batches_per_epoch = math.ceil(len(forget_encoded) / batch_size)
        steps_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
        planned_steps = 1 if args.smoke else steps_per_epoch * epochs
        planned_trace_rows = min(batches_per_epoch * epochs, planned_steps * accumulation_steps)
        receipt["training_schedule"] = {
            "epochs": epochs,
            "forget_examples_per_microbatch": batch_size,
            "retain_examples_per_microbatch": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "forget_examples_per_optimizer_step": batch_size * accumulation_steps,
            "microbatches_per_epoch": batches_per_epoch,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "planned_optimizer_steps": planned_steps,
            "planned_loss_trace_rows": planned_trace_rows,
            "loss_trace_unit": "one mean-loss record per paired microbatch",
        }
        receipt["training_started"] = True
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        ref_nll = reference_nlls(model, forget_encoded, int(tokenizer.pad_token_id), input_device)

        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        lora = LoraConfig(
            r=METHOD["lora_rank"],
            lora_alpha=METHOD["lora_alpha"],
            lora_dropout=METHOD["lora_dropout"],
            target_modules=METHOD["lora_target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable or any(p.dtype != torch.float32 for p in trainable):
            raise RuntimeError("Expected nonempty FP32 LoRA-only trainable parameters")
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable base parameter: {name}")
        model.train()
        optimizer = torch.optim.AdamW(
            trainable,
            lr=METHOD["learning_rate"],
            betas=tuple(METHOD["optimizer_betas"]),
            eps=METHOD["optimizer_eps"],
            weight_decay=METHOD["weight_decay"],
        )

        if not args.smoke:
            total_steps = planned_steps
            warmup_steps = steps_per_epoch
        else:
            total_steps, warmup_steps = planned_steps, 0
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        rng = random.Random(args.seed)
        global_step = 0
        # PEFT initializes LoRA-A with zero gradient on the first step and
        # LoRA-B at zero; inspect every adapter tensor so the smoke checks an
        # actual update rather than whichever tensor happens to be first.
        smoke_initial = [p.detach().float().cpu().clone() for p in trainable] if args.smoke else None
        optimizer.zero_grad(set_to_none=True)
        with trace_path.open("w", encoding="utf-8") as trace:
            for epoch in range(epochs):
                order = list(range(len(forget_encoded)))
                rng.shuffle(order)
                micro_in_update = 0
                pending_losses = []
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    if not indices:
                        continue
                    f_rows = [forget_encoded[i] for i in indices]
                    retain_indices = [rng.randrange(len(retain_encoded)) for _ in indices]
                    r_rows = [retain_encoded[i] for i in retain_indices]
                    f_ids, f_mask, f_labels = collate(f_rows, int(tokenizer.pad_token_id), input_device)
                    r_ids, r_mask, r_labels = collate(r_rows, int(tokenizer.pad_token_id), input_device)
                    f_logits = model(input_ids=f_ids, attention_mask=f_mask, use_cache=False).logits
                    f_current, _ = sequence_nll(f_logits, f_labels)
                    refs = torch.tensor([ref_nll[i] for i in indices], dtype=f_current.dtype, device=f_current.device)
                    forget_loss = objective_from_nlls(f_current, refs, METHOD["beta"])
                    if not torch.isfinite(forget_loss):
                        raise FloatingPointError("Non-finite NPO forget loss")
                    forget_loss_value = float(forget_loss.detach().cpu())
                    (METHOD["gamma_forget"] * forget_loss / accumulation_steps).backward()
                    del f_logits, f_current, refs, forget_loss

                    r_logits = model(input_ids=r_ids, attention_mask=r_mask, use_cache=False).logits
                    r_nll, r_counts = sequence_nll(r_logits, r_labels)
                    retain_loss = (r_nll.sum() / r_counts.sum().to(r_nll.dtype))
                    if not torch.isfinite(retain_loss):
                        raise FloatingPointError("Non-finite retain NLL")
                    retain_loss_value = float(retain_loss.detach().cpu())
                    (METHOD["alpha_retain"] * retain_loss / accumulation_steps).backward()
                    del r_logits, r_nll, r_counts, retain_loss
                    combined_loss_value = (
                        METHOD["gamma_forget"] * forget_loss_value
                        + METHOD["alpha_retain"] * retain_loss_value
                    )
                    if not math.isfinite(combined_loss_value):
                        raise FloatingPointError("Non-finite combined NPO+retain loss")
                    micro_in_update += 1
                    pending_losses.append({
                        "epoch": epoch + 1,
                        "forget_npo_loss": forget_loss_value,
                        "retain_token_nll": retain_loss_value,
                        "combined_loss": combined_loss_value,
                        "forget_examples_in_microbatch": len(indices),
                        "retain_examples_in_microbatch": len(retain_indices),
                    })

                    final_micro = start + batch_size >= len(order)
                    if micro_in_update == accumulation_steps or final_micro:
                        # Correct for a short final accumulation window.
                        if final_micro and micro_in_update < accumulation_steps:
                            scale = accumulation_steps / micro_in_update
                            for parameter in trainable:
                                if parameter.grad is not None:
                                    parameter.grad.mul_(scale)
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, METHOD["gradient_clip_norm"])
                        if not torch.isfinite(grad_norm):
                            raise FloatingPointError("Non-finite LoRA gradient norm")
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        for record in pending_losses:
                            record.update({
                                "global_step": global_step,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "gradient_norm": float(grad_norm.detach().cpu()),
                            })
                            trace.write(json.dumps(record) + "\n")
                        trace.flush()
                        if global_step % 10 == 0 or global_step == planned_steps:
                            print(json.dumps({
                                "training_progress": {
                                    "global_step": global_step,
                                    "planned_steps": planned_steps,
                                    "epoch": epoch + 1,
                                    "runtime_seconds": round(time.monotonic() - started, 1),
                                    "mean_combined_loss_this_update": sum(row["combined_loss"] for row in pending_losses) / len(pending_losses),
                                    "learning_rate": scheduler.get_last_lr()[0],
                                }
                            }), flush=True)
                        pending_losses.clear()
                        micro_in_update = 0
                        if steps_to_run is not None and global_step >= steps_to_run:
                            break
                if steps_to_run is not None and global_step >= steps_to_run:
                    break

        if global_step == 0:
            raise RuntimeError("No optimizer update was performed")
        smoke_parameter_delta = None
        if args.smoke:
            smoke_parameter_delta = max(
                float((parameter.detach().float().cpu() - initial).abs().max())
                for parameter, initial in zip(trainable, smoke_initial)
            )
            if not math.isfinite(smoke_parameter_delta) or smoke_parameter_delta <= 0:
                raise RuntimeError("Real-data smoke loss was finite but no LoRA parameter changed")
        receipt.update({
            "status": "SMOKE_PASS" if args.smoke else "TRAINING_COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "global_steps": global_step,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "all_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_fraction": sum(p.numel() for p in trainable) / sum(p.numel() for p in model.parameters()),
            "optimizer_steps_with_finite_loss_and_gradient": global_step,
            "smoke_max_abs_lora_parameter_update": smoke_parameter_delta,
            "runtime_seconds": round(time.monotonic() - started, 2),
        })
        if not args.smoke:
            model.save_pretrained(output_dir / "adapter", safe_serialization=True)
            tokenizer.save_pretrained(output_dir / "adapter")
            receipt["adapter_saved"] = True
            receipt["adapter_files_sha256"] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((output_dir / "adapter").iterdir()) if p.is_file()
            }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2), flush=True)
        return receipt
    except Exception as error:
        receipt.update({
            "status": "ERROR",
            "error": f"{type(error).__name__}: {error}",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(time.monotonic() - started, 2),
        })
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_FAMILIES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--objective-check", type=Path, required=True)
    parser.add_argument("--cohort", choices=("dev", "final"))
    parser.add_argument("--alpha-retain", type=float)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
