#!/usr/bin/env python3
"""Run a bounded, teacher-forced TOFU baseline diagnostic on public full checkpoints.

The scoring and Llama chat rendering follow Open-Unlearning's pinned TOFU
preprocessing/evaluation conventions. This is baseline evaluation only; it
does not train or modify model weights.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_ID = "locuslab/TOFU"
DATASET_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"
PROFILE_QA_COUNT = 20
PROFILE_SAMPLE_SEED = 42
RETAIN_PROFILE_SAMPLE_SIZE = 20
MAX_LENGTH = 512
SYSTEM_PROMPT = "You are a helpful assistant."
TEMPLATE_DATE = "10 Apr 2025"
IGNORE_INDEX = -100

MODELS = {
    "llama_3_2_3b": {
        "label": "Llama 3.2 3B",
        "repo_id": "open-unlearning/tofu_Llama-3.2-3B-Instruct_full",
        "revision": "24f31ca19f6966dcb6f6b29abc511cce71222d4a",
    },
    "llama_3_1_8b": {
        "label": "Llama 3.1 8B",
        "repo_id": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
        "revision": "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e",
    },
}
SPLIT_FILES = {
    "forget10": "forget10.json",
    "retain90": "retain90.json",
    "holdout10": "holdout10.json",
}


def get_bytes(url: str, timeout: int = 60) -> bytes:
    request = Request(url, headers={"User-Agent": "fsds-tofu-baseline/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def load_split(name: str) -> tuple[list[dict], str]:
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{SPLIT_FILES[name]}"
    raw = get_bytes(url)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str) for row in rows):
        raise ValueError(f"Unexpected row schema in {name}")
    return rows, hashlib.sha256(raw).hexdigest()


def normalize_token_ids(value) -> list[int]:
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def encode_like_open_unlearning(tokenizer, question: str, answer: str) -> tuple[list[int], list[int]]:
    """Mirror pinned Open-Unlearning preprocess_chat_instance for one QA."""
    system = {"role": "system", "content": SYSTEM_PROMPT}
    conversation = [
        system,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    date_kwargs = {"date_string": TEMPLATE_DATE}
    full_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            **date_kwargs,
        )
    )
    prompt_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            conversation[:-1],
            tokenize=True,
            add_generation_prompt=True,
            **date_kwargs,
        )
    )
    if not full_ids:
        raise ValueError("Chat template returned an empty input")
    if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
        full_ids.append(int(tokenizer.eos_token_id))
    if len(prompt_ids) > len(full_ids):
        raise ValueError("Prompt tokenization is longer than the complete conversation")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(labels) != len(full_ids):
        raise ValueError("Assistant label mask length mismatch")
    scored_tokens = sum(token_id != IGNORE_INDEX for token_id in labels[1:])
    if scored_tokens == 0:
        raise ValueError("No assistant answer tokens survived the official mask")
    # The official chat-template branch does not truncate this sequence. Keep
    # MAX_LENGTH in the run record as the benchmark config's declared limit.
    return full_ids, labels


def collate(rows: list[tuple[list[int], list[int]]], pad_token_id: int):
    max_len = max(len(input_ids) for input_ids, _ in rows)
    input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = torch.full((len(rows), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, target_labels) in enumerate(rows):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(target_labels, dtype=torch.long)
    return input_ids, attention_mask, labels


def answer_probabilities(model, encoded_rows, pad_token_id: int, batch_size: int) -> list[dict]:
    input_device = model.get_input_embeddings().weight.device
    results = []
    offset = 0
    while offset < len(encoded_rows):
        current_batch_size = min(batch_size, len(encoded_rows) - offset)
        batch = encoded_rows[offset : offset + current_batch_size]
        input_ids, attention_mask, labels = collate(batch, pad_token_id)
        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = output.logits[:, :-1, :].float()
                target = labels[:, 1:].to(logits.device)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                ).reshape(target.shape).sum(dim=1)
                denominator = (labels[:, 1:] != IGNORE_INDEX).sum(dim=1).to(loss_sum.device)
                if torch.any(denominator == 0):
                    raise ValueError("Batch contains an example with zero scored answer tokens")
                avg_loss = loss_sum / denominator
                probabilities = torch.exp(-avg_loss)
                for i in range(current_batch_size):
                    results.append(
                        {
                            "avg_answer_token_loss": float(avg_loss[i].item()),
                            "answer_probability": float(probabilities[i].item()),
                            "scored_answer_tokens": int(denominator[i].item()),
                        }
                    )
            offset += current_batch_size
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, current_batch_size // 2)
            print(f"CUDA OOM; reducing scoring batch size to {batch_size}", flush=True)
    return results


def bootstrap_profile_interval(profile_means: dict[int, float], seed: int = PROFILE_SAMPLE_SEED, draws: int = 5000):
    values = list(profile_means.values())
    if not values:
        raise ValueError("Cannot bootstrap an empty profile sample")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        estimates.append(sum(rng.choice(values) for _ in values) / len(values))
    estimates.sort()
    low = estimates[int(0.025 * (draws - 1))]
    high = estimates[int(0.975 * (draws - 1))]
    return low, high


def choose_eval_rows(split_rows: dict[str, list[dict]]) -> tuple[list[dict], list[int], list[int]]:
    retain_rows = split_rows["retain90"]
    retain_group_count = len(retain_rows) // PROFILE_QA_COUNT
    by_question: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for row_index, row in enumerate(retain_rows):
        by_question[row["question"]].append((row_index, row["answer"]))
    collision_row_indices = sorted(
        row_index
        for entries in by_question.values()
        if len(entries) > 1 and len({answer for _, answer in entries}) > 1
        for row_index, _ in entries
    )
    collision_groups = sorted({row_index // PROFILE_QA_COUNT for row_index in collision_row_indices})
    non_collision_groups = [group for group in range(retain_group_count) if group not in collision_groups]
    n_random_groups = RETAIN_PROFILE_SAMPLE_SIZE - len(collision_groups)
    if n_random_groups < 0 or n_random_groups > len(non_collision_groups):
        raise ValueError("Cannot build a retain sample containing all ambiguous-question profile blocks")
    selected_groups = sorted(
        collision_groups
        + random.Random(PROFILE_SAMPLE_SEED).sample(non_collision_groups, n_random_groups)
    )
    eval_rows = []
    for split in ("forget10", "retain90", "holdout10"):
        rows = split_rows[split]
        if len(rows) % PROFILE_QA_COUNT:
            raise ValueError(f"{split} row count is not divisible into 20-QA profile blocks")
        for row_index, row in enumerate(rows):
            group_index = row_index // PROFILE_QA_COUNT
            if split != "retain90" or group_index in selected_groups:
                eval_rows.append(
                    {
                        "split": split,
                        "row_index": row_index,
                        "profile_block": group_index,
                        "question": row["question"],
                        "answer": row["answer"],
                    }
                )
    return eval_rows, selected_groups, collision_row_indices


def summarize_scores(score_rows: list[dict]) -> dict:
    summary = {}
    for split in SPLIT_FILES:
        rows = [row for row in score_rows if row["split"] == split]
        grouped: dict[int, list[dict]] = collections.defaultdict(list)
        for row in rows:
            grouped[int(row["profile_block"])].append(row)
        profile_means = {
            group: sum(row["answer_probability"] for row in group_rows) / len(group_rows)
            for group, group_rows in grouped.items()
        }
        score_mean = sum(row["answer_probability"] for row in rows) / len(rows)
        loss_mean = sum(row["avg_answer_token_loss"] for row in rows) / len(rows)
        low, high = bootstrap_profile_interval(profile_means)
        summary[split] = {
            "n_questions": len(rows),
            "n_profile_blocks": len(profile_means),
            "mean_answer_probability": score_mean,
            "mean_answer_token_loss": loss_mean,
            "profile_cluster_bootstrap_95pct_ci_answer_probability": [low, high],
        }
    ambiguous_rows = {
        row["row_index"]
        for row in score_rows
        if row["split"] == "retain90" and row.get("ambiguous_question_collision", False)
    }
    if ambiguous_rows:
        cleaned = [
            row
            for row in score_rows
            if row["split"] != "retain90" or row["row_index"] not in ambiguous_rows
        ]
        retain_clean = [row for row in cleaned if row["split"] == "retain90"]
        grouped_clean: dict[int, list[dict]] = collections.defaultdict(list)
        for row in retain_clean:
            grouped_clean[int(row["profile_block"])].append(row)
        profile_means_clean = {
            group: sum(row["answer_probability"] for row in group_rows) / len(group_rows)
            for group, group_rows in grouped_clean.items()
        }
        low, high = bootstrap_profile_interval(profile_means_clean)
        summary["retain90_ambiguous_questions_excluded"] = {
            "n_questions": len(retain_clean),
            "n_profile_blocks": len(profile_means_clean),
            "mean_answer_probability": sum(row["answer_probability"] for row in retain_clean) / len(retain_clean),
            "profile_equal_weight_mean_answer_probability": sum(profile_means_clean.values()) / len(profile_means_clean),
            "profile_cluster_bootstrap_95pct_ci_answer_probability": [low, high],
            "excluded_row_indices": sorted(ambiguous_rows),
        }
    return summary


def evaluate_model(model_key: str, output_dir: Path, batch_size: int = 2) -> dict:
    from transformers import __version__ as transformers_version
    from accelerate import __version__ as accelerate_version

    spec = MODELS[model_key]
    model_cache = output_dir / f"hf_cache_{model_key}"
    model_cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    split_rows: dict[str, list[dict]] = {}
    dataset_hashes = {}
    for split in SPLIT_FILES:
        split_rows[split], dataset_hashes[split] = load_split(split)
    eval_rows, selected_retain_groups, collision_row_indices = choose_eval_rows(split_rows)

    tokenizer = AutoTokenizer.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        cache_dir=str(model_cache),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required for this pinned scoring implementation")
    encoded_rows = [encode_like_open_unlearning(tokenizer, row["question"], row["answer"]) for row in eval_rows]
    input_sequence_lengths = [len(input_ids) for input_ids, _ in encoded_rows]
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        cache_dir=str(model_cache),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: "12GiB", 1: "12GiB", "cpu": "40GiB"},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    actual_batch_size = batch_size
    scored = answer_probabilities(model, encoded_rows, int(tokenizer.pad_token_id), actual_batch_size)
    if len(scored) != len(eval_rows):
        raise RuntimeError(f"Expected {len(eval_rows)} score rows; got {len(scored)}")

    score_rows = []
    for row, score in zip(eval_rows, scored):
        score_rows.append(
            {
                "split": row["split"],
                "row_index": row["row_index"],
                "profile_block": row["profile_block"],
                "question_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
                "ambiguous_question_collision": row["split"] == "retain90" and row["row_index"] in collision_row_indices,
                "input_sequence_tokens": len(encoded_rows[len(score_rows)][0]),
                "avg_answer_token_loss": score["avg_answer_token_loss"],
                "answer_probability": score["answer_probability"],
                "scored_answer_tokens": score["scored_answer_tokens"],
            }
        )
    output = {
        "status": "PASS_BASELINE_DIAGNOSTIC",
        "model": spec,
        "model_config": {
            "model_type": model.config.model_type,
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "device_map": getattr(model, "hf_device_map", None),
        },
        "scoring": {
            "metric": "mean per-example exp(-mean assistant-token cross-entropy)",
            "metric_implementation": "Open-Unlearning evaluate_probability at pinned source revision",
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256_by_split": dataset_hashes,
            "system_prompt": SYSTEM_PROMPT,
            "date_string": TEMPLATE_DATE,
            "declared_max_length": MAX_LENGTH,
            "input_sequences_truncated": False,
            "profile_block_size": PROFILE_QA_COUNT,
            "retain_profile_sample_seed": PROFILE_SAMPLE_SEED,
            "retain_profile_blocks_selected": selected_retain_groups,
            "retain_ambiguous_question_collision_rows_in_sample": collision_row_indices,
            "samples_per_split": {split: sum(r["split"] == split for r in score_rows) for split in SPLIT_FILES},
            "max_input_sequence_tokens": max(input_sequence_lengths),
            "sequences_over_declared_max_length": sum(length > MAX_LENGTH for length in input_sequence_lengths),
            "all_rows_have_scored_answer_tokens": all(row["scored_answer_tokens"] > 0 for row in score_rows),
        },
        "summary": summarize_scores(score_rows),
        "scores": score_rows,
        "runtime": {
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "accelerate_version": accelerate_version,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_total_memory_bytes": [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())],
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "training_started": False,
            "research_adapter_created": False,
        },
    }
    output_path = output_dir / f"{model_key}_baseline.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_path = output_dir / f"{model_key}_scores.jsonl"
    score_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_rows), encoding="utf-8")

    del model, tokenizer, encoded_rows
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(model_cache, ignore_errors=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/tofu_baseline_results"))
    parser.add_argument("--models", default="llama_3_2_3b,llama_3_1_8b")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = set(selected_models) - set(MODELS)
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This bounded diagnostic requires Kaggle GPU T4 x2; no CUDA model run was started")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_models": selected_models,
        "model_results": {},
        "training_started": False,
    }
    status_path = args.output_dir / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    for model_key in selected_models:
        try:
            result = evaluate_model(model_key, args.output_dir, args.batch_size)
            status["model_results"][model_key] = {
                "status": result["status"],
                "summary": result["summary"],
                "result_file": f"{model_key}_baseline.json",
                "scores_file": f"{model_key}_scores.jsonl",
            }
        except Exception as error:
            status["model_results"][model_key] = {
                "status": "ERROR",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{model_key}: PASS_BASELINE_DIAGNOSTIC")
        print(json.dumps(result["summary"], indent=2))
    status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
