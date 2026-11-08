#!/usr/bin/env python3
"""Resource-bounded LoRA adaptation of the official TOFU IDK+AP objective."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tofu_official_eval import CHECKPOINTS, DATASET_ID, DATASET_REVISION, encode_answer, get_bytes


IGNORE_INDEX = -100
MODEL_KEY = "llama_3_2_3b_full"
EXPECTED_PROTOCOL_SHA256 = "650af06eed1d254cfdd2b47714299686925cea47b5b2675edb7c07674b01c049"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fetch_jsonl(filename: str) -> tuple[list[dict], str]:
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{filename}"
    raw = get_bytes(url)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, sha256(raw)


def expand(ranges: list[list[int]]) -> list[int]:
    return [block for first, last in ranges for block in range(first, last + 1)]


def collate(rows: list[tuple[list[int], list[int]]], pad_id: int, device: torch.device):
    max_len = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = torch.full((len(rows), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, row_labels) in enumerate(rows):
        size = len(ids)
        input_ids[i, :size] = torch.tensor(ids, dtype=torch.long)
        attention[i, :size] = 1
        labels[i, :size] = torch.tensor(row_labels, dtype=torch.long)
    return input_ids.to(device), attention.to(device), labels


def per_sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = labels[..., 1:].contiguous().to(logits.device)
    losses = F.cross_entropy(
        shifted_logits.transpose(-1, -2), shifted_labels,
        ignore_index=IGNORE_INDEX, reduction="none",
    )
    counts = (shifted_labels != IGNORE_INDEX).sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("An IDK+AP example has no scored assistant tokens")
    return losses.sum(dim=-1), counts


def mean_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = labels[..., 1:].contiguous().to(logits.device)
    if not torch.any(shifted_labels != IGNORE_INDEX):
        raise ValueError("IDK forget target contains no assistant tokens")
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="mean",
    )


def check_packages() -> dict[str, str]:
    expected = {"transformers": "5.22.1", "accelerate": "1.15.0", "peft": "0.21.2", "scipy": "1.16.3", "rouge-score": "0.1.2"}
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Pinned package mismatch: {actual}")
    try:
        torchao = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao = None
    if torchao is not None:
        raise RuntimeError(f"Unexpected torchao {torchao}; the locked run does not use it")
    return actual


def prepare_rows(protocol: dict, cohort: str, idk_path: Path):
    expected_hashes = protocol["data"]["sha256"]
    forget, forget_sha = fetch_jsonl("forget10.json")
    retain, retain_sha = fetch_jsonl("retain90.json")
    if forget_sha != expected_hashes["forget10.json"] or retain_sha != expected_hashes["retain90.json"]:
        raise RuntimeError("Pinned TOFU training file hash mismatch")
    if len(forget) != 400 or len(retain) != 3600:
        raise RuntimeError(f"Unexpected TOFU source row counts: forget={len(forget)}, retain={len(retain)}")
    assignments = protocol["data"]["profile_assignment_ranges_inclusive"]
    forget_blocks = expand(assignments[f"forget10_{cohort}_train_and_eval"])
    retain_blocks = expand(assignments["retain90_optimizer_train"])
    size = int(protocol["data"]["profile_block_size"])
    forget_indices = [i for i in range(len(forget)) if i // size in forget_blocks]
    retain_indices = [i for i in range(len(retain)) if i // size in retain_blocks]
    forget_selected = [forget[i] for i in forget_indices]
    retain_selected = [retain[i] for i in retain_indices]
    if len(forget_selected) != 200 or len(retain_selected) != 2760:
        raise RuntimeError(f"Wrong {cohort} training selection: forget={len(forget_selected)}, retain={len(retain_selected)}")
    idk_bytes = idk_path.read_bytes()
    idk_responses = [line.strip() for line in idk_bytes.decode("utf-8").splitlines() if line.strip()]
    if sha256(idk_bytes) != protocol["objective"]["idk_response_sha256"] or len(idk_responses) != 100:
        raise RuntimeError("Pinned IDK responses do not match the protocol")

    # Use a deterministic, fixed pairing. This adapts the official dataset's
    # index-seeded retain/IDK sampling to the leakage-screened optimizer pool.
    paired = []
    for index, f_row in enumerate(forget_selected):
        generator = torch.Generator(device="cpu").manual_seed(index)
        retain_local = (index + int(torch.randint(len(retain_selected), (1,), generator=generator).item())) % len(retain_selected)
        idk_local = int(torch.randint(len(idk_responses), (1,), generator=generator).item())
        paired.append({
            "forget_source_index": forget_indices[index],
            "retain_source_index": retain_indices[retain_local],
            "idk_index": idk_local,
            "forget": f_row,
            "retain": retain_selected[retain_local],
            "idk": idk_responses[idk_local],
        })
    return paired, {
        "forget10_sha256": forget_sha,
        "retain90_sha256": retain_sha,
        "idk_response_sha256": sha256(idk_bytes),
        "forget_source_indices": forget_indices,
        "retain_optimizer_source_indices": retain_indices,
        "forget_profile_blocks": forget_blocks,
        "retain_optimizer_profile_blocks": retain_blocks,
        "forget_rows": len(forget_selected),
        "retain_optimizer_rows": len(retain_selected),
        "pairing_assignments": [
            {"forget_source_index": p["forget_source_index"], "retain_source_index": p["retain_source_index"],
             "idk_response_index": p["idk_index"]} for p in paired
        ],
        "pairing_sha256": sha256(json.dumps([
            [p["forget_source_index"], p["retain_source_index"], p["idk_index"]] for p in paired
        ], separators=(",", ":")).encode()),
    }


def encode_pairs(tokenizer, paired, max_length: int):
    forget_idk = []
    retain_answer = []
    retain_idk = []
    for pair in paired:
        forget_idk.append(encode_answer(tokenizer, pair["forget"]["question"], pair["idk"]))
        retain_answer.append(encode_answer(tokenizer, pair["retain"]["question"], pair["retain"]["answer"]))
        retain_idk.append(encode_answer(tokenizer, pair["retain"]["question"], pair["idk"]))
    all_encoded = forget_idk + retain_answer + retain_idk
    lengths = [len(ids) for ids, _ in all_encoded]
    if max(lengths) > max_length:
        raise ValueError(f"Untruncated sequence exceeds locked {max_length}-token limit: {max(lengths)}")
    if any(not any(label != IGNORE_INDEX for label in labels[1:]) for _, labels in all_encoded):
        raise ValueError("A source sequence has an empty assistant response mask")
    return forget_idk, retain_answer, retain_idk, max(lengths)


def run_microbatch(model, triple, pad_id, input_device, beta, do_backward: bool, gradient_scale: float = 1.0):
    f_row, answer_row, idk_row = triple
    f_ids, f_mask, f_labels = collate([f_row], pad_id, input_device)
    r_ids, r_mask, r_labels = collate([answer_row], pad_id, input_device)
    i_ids, i_mask, i_labels = collate([idk_row], pad_id, input_device)

    f_logits = model(input_ids=f_ids, attention_mask=f_mask, use_cache=False).logits
    forget_loss = mean_token_ce(f_logits, f_labels)
    if not torch.isfinite(forget_loss):
        raise FloatingPointError("Non-finite forget IDK cross-entropy")
    if do_backward:
        (forget_loss * gradient_scale).backward()
    forget_value = float(forget_loss.detach().cpu())
    del f_logits, forget_loss

    r_logits = model(input_ids=r_ids, attention_mask=r_mask, use_cache=False).logits
    answer_nll, _ = per_sequence_nll(r_logits, r_labels)
    i_logits = model(input_ids=i_ids, attention_mask=i_mask, use_cache=False).logits
    idk_nll, _ = per_sequence_nll(i_logits, i_labels)
    ap_loss = -F.logsigmoid(beta * (idk_nll - answer_nll)).mean() / beta
    if not torch.isfinite(ap_loss):
        raise FloatingPointError("Non-finite answer-preservation preference loss")
    if do_backward:
        (ap_loss * gradient_scale).backward()
    ap_value = float(ap_loss.detach().cpu())
    total = forget_value + ap_value
    del r_logits, i_logits, answer_nll, idk_nll, ap_loss
    return forget_value, ap_value, total


def train(args) -> dict:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    objective_check = json.loads(args.objective_check.read_text(encoding="utf-8"))
    data_audit = json.loads(args.data_audit.read_text(encoding="utf-8"))
    protocol_hash = sha256(args.protocol.read_bytes())
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Locked protocol SHA-256 mismatch; use a new protocol version for design changes")
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN" or objective_check.get("status") != "PASS":
        raise RuntimeError("Protocol/objective gate failed before model loading")
    if data_audit.get("status") != "PASS_IDK_AP_PROTOCOL_DATA_AUDIT":
        raise RuntimeError("Pinned data/profile audit failed before model loading")
    if data_audit.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Data audit receipt belongs to a different protocol")
    if objective_check.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Objective-check receipt belongs to a different protocol")
    packages = check_packages()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("This run requires exactly two Kaggle GPUs")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
    if gpu_names.count("Tesla T4") != 2:
        raise RuntimeError(f"Locked hardware is Tesla T4 x2, found {gpu_names}")
    model_spec = CHECKPOINTS[MODEL_KEY]
    locked = protocol["model"]
    if (locked["repo_id"], locked["revision"]) != (model_spec["repo_id"], model_spec["revision"]):
        raise RuntimeError("Pinned model checkpoint does not match tofu_official_eval")
    if args.seed != int(protocol["training"]["seed"]):
        raise RuntimeError("Requested seed differs from preregistration")
    if args.cohort not in {"dev", "final"}:
        raise RuntimeError("Explicit dev/final cohort is required")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "training_receipt.json"
    trace_path = output_dir / "loss_trace.jsonl"
    receipt = {
        "status": "PREFLIGHT",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "data_audit_sha256": sha256(args.data_audit.read_bytes()),
        "objective_equivalence": objective_check,
        "objective_equivalence_sha256": sha256(args.objective_check.read_bytes()),
        "training_script_sha256": sha256(Path(__file__).read_bytes()),
        "method": protocol["objective"],
        "training": protocol["training"],
        "cohort": args.cohort,
        "seed": args.seed,
        "checkpoint": model_spec,
        "gpus": gpu_names,
        "package_versions": {"torch": torch.__version__, **packages},
        "training_started": False,
        "adapter_saved": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        paired, selected = prepare_rows(protocol, args.cohort, args.idk)
        receipt["dataset"] = selected
        model_cache = Path(os.environ.get("TOFU_MODEL_CACHE_ROOT", "/kaggle/working/tofu_model_cache")) / MODEL_KEY
        model_cache.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_spec["repo_id"], revision=model_spec["revision"], cache_dir=str(model_cache), use_fast=True
        )
        if not tokenizer.is_fast:
            raise RuntimeError("Pinned tokenizer must be fast")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        f_encoded, r_encoded, i_encoded, max_length = encode_pairs(
            tokenizer, paired, int(protocol["training"]["max_length"])
        )
        receipt["tokenization"] = {
            "max_length_observed": max_length,
            "truncated_examples": 0,
            "all_assistant_masks_nonempty": True,
            "encoding": "pinned Llama chat template from tofu_official_eval.encode_answer",
        }

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(
            model_spec["repo_id"], revision=model_spec["revision"], cache_dir=str(model_cache), dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="balanced",
            max_memory={0: "13GiB", 1: "13GiB", "cpu": "40GiB"},
            attn_implementation="sdpa",
        )
        device_map = getattr(model, "hf_device_map", {})
        offload = sorted({str(value) for value in device_map.values() if str(value).lower() in {"cpu", "disk"}})
        if offload:
            raise RuntimeError(f"CPU/disk offload is prohibited: {offload}")
        input_device = model.get_input_embeddings().weight.device
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        t = protocol["training"]
        lora = LoraConfig(
            r=int(t["lora_rank"]), lora_alpha=int(t["lora_alpha"]),
            lora_dropout=float(t["lora_dropout"]), target_modules=list(t["target_modules"]),
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
            raise RuntimeError("Expected FP32 LoRA-only trainable parameters")
        if any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
            raise RuntimeError("A non-LoRA parameter is unexpectedly trainable")
        init_state = [parameter.detach().cpu().clone() for parameter in trainable]

        # A real-data one-step smoke validates forward/backward/optimizer behavior.
        # Restore the exact LoRA initialization so the diagnostic training starts clean.
        smoke_optimizer = torch.optim.AdamW(
            trainable, lr=float(t["learning_rate"]), betas=tuple(t["optimizer_betas"]),
            eps=float(t["optimizer_eps"]), weight_decay=float(t["weight_decay"]),
        )
        model.train()
        smoke_optimizer.zero_grad(set_to_none=True)
        beta = float(protocol["objective"]["beta"])
        smoke_triple = (f_encoded[0], r_encoded[0], i_encoded[0])
        smoke_losses = run_microbatch(model, smoke_triple, int(tokenizer.pad_token_id), input_device, beta, True)
        smoke_grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(t["gradient_clip_norm"]))
        if not torch.isfinite(smoke_grad_norm):
            raise FloatingPointError("Non-finite gradient norm in real-data smoke")
        smoke_optimizer.step()
        smoke_delta = max(float((parameter.detach().cpu() - initial).abs().max()) for parameter, initial in zip(trainable, init_state))
        if not math.isfinite(smoke_delta) or smoke_delta <= 0:
            raise RuntimeError("Smoke optimizer step did not change a LoRA parameter")
        with torch.no_grad():
            for parameter, initial in zip(trainable, init_state):
                parameter.copy_(initial.to(device=parameter.device, dtype=parameter.dtype))
        del smoke_optimizer
        for parameter in trainable:
            parameter.grad = None
        torch.cuda.empty_cache()
        receipt["runtime"] = {
            "device_map": {str(key): str(value) for key, value in device_map.items()},
            "cpu_or_disk_offload": False,
            "input_device": str(input_device),
            "base_dtype": "float16",
            "adapter_dtype": "float32",
            "attention": "sdpa",
            "real_data_smoke": {
                "status": "PASS_BOUNDED_REAL_DATA_SMOKE",
                "forget_idk_loss": smoke_losses[0],
                "retain_ap_loss": smoke_losses[1],
                "total_loss": smoke_losses[2],
                "max_abs_lora_update_before_reset": smoke_delta,
                "gradient_norm_before_clip": float(smoke_grad_norm.detach().cpu()),
                "initial_lora_restored_before_research_training": True,
            },
        }
        if any(not math.isfinite(value) for value in smoke_losses):
            raise FloatingPointError("Smoke losses were non-finite")

        epochs = int(t["epochs"])
        microbatch = int(t["microbatch_paired_examples"])
        accumulation = int(t["gradient_accumulation_steps"])
        if len(paired) % (microbatch * accumulation):
            raise RuntimeError("Locked cohort does not yield complete gradient-accumulation windows")
        steps_per_epoch = len(paired) // (microbatch * accumulation)
        total_steps = int(t["expected_optimizer_steps"])
        if steps_per_epoch * epochs != total_steps or total_steps != 125:
            raise RuntimeError("Locked optimizer schedule is internally inconsistent")
        optimizer = torch.optim.AdamW(
            trainable, lr=float(t["learning_rate"]), betas=tuple(t["optimizer_betas"]),
            eps=float(t["optimizer_eps"]), weight_decay=float(t["weight_decay"]),
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, steps_per_epoch, total_steps)
        receipt["training_schedule"] = {
            "epochs": epochs,
            "microbatch_paired_examples": microbatch,
            "gradient_accumulation_steps": accumulation,
            "effective_paired_batch_size": microbatch * accumulation,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "planned_optimizer_steps": total_steps,
            "planned_loss_trace_rows": total_steps * accumulation,
        }
        receipt["training_started"] = True
        receipt["status"] = "TRAINING"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

        rng = random.Random(args.seed)
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        with trace_path.open("w", encoding="utf-8") as trace:
            for epoch in range(epochs):
                order = list(range(len(paired)))
                rng.shuffle(order)
                pending = []
                for start in range(0, len(order), microbatch):
                    index = order[start]
                    losses = run_microbatch(
                        model,
                        (f_encoded[index], r_encoded[index], i_encoded[index]),
                        int(tokenizer.pad_token_id), input_device, beta, True, 1.0 / accumulation,
                    )
                    pending.append({
                        "epoch": epoch + 1,
                        "pair_index": index,
                        "forget_source_index": paired[index]["forget_source_index"],
                        "retain_source_index": paired[index]["retain_source_index"],
                        "idk_index": paired[index]["idk_index"],
                        "forget_idk_loss": losses[0],
                        "retain_ap_loss": losses[1],
                        "combined_loss": losses[2],
                    })
                    if len(pending) == accumulation:
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(t["gradient_clip_norm"]))
                        if not torch.isfinite(grad_norm):
                            raise FloatingPointError("Non-finite gradient norm")
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        lr_now = scheduler.get_last_lr()[0]
                        for record in pending:
                            record.update({
                                "global_step": global_step,
                                "learning_rate": lr_now,
                                "gradient_norm_preclip": float(grad_norm.detach().cpu()),
                                "clip_threshold": float(t["gradient_clip_norm"]),
                            })
                            trace.write(json.dumps(record, sort_keys=True) + "\n")
                        trace.flush()
                        pending.clear()
                        if global_step % 10 == 0:
                            print(json.dumps({"epoch": epoch + 1, "global_step": global_step, "planned_steps": total_steps,
                                              "lr": lr_now, "last_microbatch_loss": losses[2]}), flush=True)
                if pending:
                    raise RuntimeError("Unexpected incomplete gradient accumulation window")
        if global_step != total_steps:
            raise RuntimeError(f"Expected {total_steps} optimizer updates, received {global_step}")

        adapter_dir = output_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        receipt.update({
            "status": "TRAINING_COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "optimizer_steps": global_step,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "all_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "adapter_saved": True,
            "adapter_files_sha256": {path.name: sha256(path.read_bytes()) for path in sorted(adapter_dir.iterdir()) if path.is_file()},
            "loss_trace_sha256": sha256(trace_path.read_bytes()),
            "runtime_seconds": round(time.monotonic() - started, 2),
        })
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--objective-check", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--idk", type=Path, required=True)
    parser.add_argument("--cohort", choices=("dev", "final"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
