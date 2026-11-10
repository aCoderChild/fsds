#!/usr/bin/env python3
"""Evaluate pinned TOFU checkpoints with the core Open-Unlearning metrics.

This is inference-only. Full checkpoints receive answer likelihood, greedy
generation/ROUGE-L, and truth-ratio evaluation on the canonical TOFU sets.
The paired retain90 oracle checkpoints are scored on forget truth ratios only
to compute the benchmark's forget-quality comparison.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from scipy.stats import ks_2samp
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_ID = "locuslab/TOFU"
DATASET_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"
UPSTREAM_REVISION = "2d98027677f3468d52b6baa3fc4f2b9f2dc08820"
PROFILE_QA_COUNT = 20
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 5000
MAX_LENGTH = 512
SYSTEM_PROMPT = "You are a helpful assistant."
TEMPLATE_DATE = "10 Apr 2025"
MAX_NEW_TOKENS = 200
IGNORE_INDEX = -100
EPSILON = 1e-10

CHECKPOINTS = {
    "llama_3_2_3b_full": {
        "label": "Llama 3.2 3B full",
        "family": "llama_3_2_3b",
        "role": "full",
        "repo_id": "open-unlearning/tofu_Llama-3.2-3B-Instruct_full",
        "revision": "24f31ca19f6966dcb6f6b29abc511cce71222d4a",
    },
    "llama_3_2_3b_retain90": {
        "label": "Llama 3.2 3B retain90 oracle",
        "family": "llama_3_2_3b",
        "role": "retain90_oracle",
        "repo_id": "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90",
        "revision": "2c35819e9a731d096df9abd6efbcc361870c5333",
    },
    "llama_3_1_8b_full": {
        "label": "Llama 3.1 8B full",
        "family": "llama_3_1_8b",
        "role": "full",
        "repo_id": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
        "revision": "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e",
    },
    "llama_3_1_8b_retain90": {
        "label": "Llama 3.1 8B retain90 oracle",
        "family": "llama_3_1_8b",
        "role": "retain90_oracle",
        "repo_id": "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90",
        "revision": "f02956a9f25a18f38ad7808fbefb2f957347273e",
    },
}

DATA_FILES = (
    "forget10.json",
    "forget10_perturbed.json",
    "retain90.json",
    "retain_perturbed.json",
    "holdout10.json",
    "real_authors_perturbed.json",
    "world_facts_perturbed.json",
)


def get_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "fsds-tofu-official-eval/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def load_data() -> tuple[dict[str, list[dict]], dict[str, str]]:
    rows_by_file = {}
    hashes = {}
    for filename in DATA_FILES:
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{filename}"
        raw = get_bytes(url)
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"{filename} contains a non-object JSONL row")
        if any(not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str) for row in rows):
            raise ValueError(f"{filename} has a missing question/answer string")
        rows_by_file[filename] = rows
        hashes[filename] = hashlib.sha256(raw).hexdigest()
    return rows_by_file, hashes


def build_eval_sets(data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Build canonical evaluation rows and attach source profile blocks."""
    retain_source = data["retain90.json"]
    retain_index = collections.defaultdict(list)
    retain_question_answers = collections.defaultdict(set)
    for i, row in enumerate(retain_source):
        retain_index[(row["question"], row["answer"])].append(i)
        retain_question_answers[row["question"]].add(row["answer"])
    collision_indices = {
        i
        for i, row in enumerate(retain_source)
        if len(retain_question_answers[row["question"]]) > 1
    }

    def basic(rows: list[dict], split: str, source_indices: list[int] | None = None) -> list[dict]:
        result = []
        for i, row in enumerate(rows):
            source_i = i if source_indices is None else source_indices[i]
            result.append({
                "split": split,
                "eval_row_index": i,
                "source_row_index": source_i,
                "profile_block": source_i // PROFILE_QA_COUNT if split in {"forget10", "retain90", "holdout10"} else None,
                "question": row["question"],
                "answer": row["answer"],
                "ambiguous_question_collision": split == "retain90" and source_i in collision_indices,
                "paraphrased_answer": row.get("paraphrased_answer"),
                "perturbed_answers": row.get("perturbed_answer"),
            })
        return result

    forget = data["forget10_perturbed.json"]
    forget_source = data["forget10.json"]
    if len(forget) != len(forget_source) or any(
        (a["question"], a["answer"]) != (b["question"], b["answer"])
        for a, b in zip(forget_source, forget)
    ):
        raise ValueError("forget10_perturbed no longer matches forget10 in source order")

    retain = data["retain_perturbed.json"]
    retain_source_indices = []
    for row in retain:
        matches = retain_index.get((row["question"], row["answer"]), [])
        if len(matches) != 1:
            raise ValueError("A canonical retain_perturbed row does not map uniquely to retain90")
        retain_source_indices.append(matches[0])
    if sorted(retain_source_indices) != list(range(20 * PROFILE_QA_COUNT)):
        raise ValueError("Canonical retain_perturbed is not the pinned first-20-profile retain sample")

    return {
        "forget10": basic(forget, "forget10"),
        "retain90": basic(retain, "retain90", retain_source_indices),
        "holdout10": basic(data["holdout10.json"], "holdout10"),
        "real_authors": basic(data["real_authors_perturbed.json"], "real_authors"),
        "world_facts": basic(data["world_facts_perturbed.json"], "world_facts"),
    }


def ids(value) -> list[int]:
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def encode_answer(tokenizer, question: str, answer: str) -> tuple[list[int], list[int]]:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    kwargs = {"date_string": TEMPLATE_DATE}
    full_ids = ids(tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, **kwargs
    ))
    prompt_ids = ids(tokenizer.apply_chat_template(
        conversation[:-1], tokenize=True, add_generation_prompt=True, **kwargs
    ))
    if not full_ids or len(prompt_ids) > len(full_ids):
        raise ValueError("Invalid chat-template tokenization")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokenization is not an exact prefix of the full conversation")
    if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
        full_ids.append(int(tokenizer.eos_token_id))
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if len(labels) != len(full_ids) or not any(x != IGNORE_INDEX for x in labels[1:]):
        raise ValueError("Assistant answer mask is empty or misaligned")
    return full_ids, labels


def encode_prompt(tokenizer, question: str) -> list[int]:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    result = ids(tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        date_string=TEMPLATE_DATE,
    ))
    if not result:
        raise ValueError("Chat-template prompt was empty")
    return result


def collate_scoring(rows: list[tuple[list[int], list[int]]], pad_id: int):
    width = max(len(x[0]) for x in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for i, (tokens, targets) in enumerate(rows):
        n = len(tokens)
        input_ids[i, :n] = torch.tensor(tokens)
        attention[i, :n] = 1
        labels[i, :n] = torch.tensor(targets)
    return input_ids, attention, labels


def score_answers(model, encoded: list[tuple[list[int], list[int]]], pad_id: int, batch_size: int) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    output_scores = []
    offset = 0
    while offset < len(encoded):
        size = min(batch_size, len(encoded) - offset)
        batch = encoded[offset:offset + size]
        input_ids, attention, labels = collate_scoring(batch, pad_id)
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention.to(device),
                    use_cache=False,
                )
                logits = out.logits[:, :-1, :].float()
                targets = labels[:, 1:].to(logits.device)
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                ).reshape(targets.shape)
                tokens_per_row = (targets != IGNORE_INDEX).sum(dim=1)
                if torch.any(tokens_per_row == 0):
                    raise ValueError("Evaluation example has no scored assistant tokens")
                mean_loss = losses.sum(dim=1) / tokens_per_row
                for i in range(size):
                    value = float(mean_loss[i].item())
                    output_scores.append({
                        "avg_answer_token_loss": value,
                        "answer_probability": math.exp(-value),
                        "scored_answer_tokens": int(tokens_per_row[i].item()),
                    })
            offset += size
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, size // 2)
            print(f"CUDA OOM during scoring; reducing batch size to {batch_size}", flush=True)
    return output_scores


def collate_prompts(rows: list[list[int]], pad_id: int):
    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        input_ids[i, -len(row):] = torch.tensor(row)
        attention[i, -len(row):] = 1
    return input_ids, attention


def generate_answers(model, tokenizer, prompts: list[list[int]], batch_size: int) -> list[str]:
    device = model.get_input_embeddings().weight.device
    generations = []
    offset = 0
    model.config.use_cache = True
    while offset < len(prompts):
        size = min(batch_size, len(prompts) - offset)
        current = prompts[offset:offset + size]
        input_ids, attention = collate_prompts(current, int(tokenizer.pad_token_id))
        try:
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids.to(device),
                    attention_mask=attention.to(device),
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[:, input_ids.shape[1]:]
            texts = tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            generations.extend(text.strip() for text in texts)
            offset += size
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, size // 2)
            print(f"CUDA OOM during generation; reducing batch size to {batch_size}", flush=True)
    return generations


def build_jobs(eval_sets: dict[str, list[dict]], role: str):
    score_jobs = []
    generation_jobs = []
    for split, rows in eval_sets.items():
        for row in rows:
            key = (split, row["eval_row_index"])
            if role == "full" or split == "forget10":
                score_jobs.append({**row, "variant": "answer", "candidate_answer": row["answer"]})
                if row["paraphrased_answer"]:
                    score_jobs.append({**row, "variant": "paraphrased_answer", "candidate_answer": row["paraphrased_answer"]})
                for i, candidate in enumerate(row["perturbed_answers"] or []):
                    score_jobs.append({**row, "variant": f"perturbed_{i}", "candidate_answer": candidate})
            if role == "full":
                generation_jobs.append(row)
    return score_jobs, generation_jobs


def score_rows_by_key(jobs: list[dict], scores: list[dict]) -> dict[tuple[str, int], dict[str, dict]]:
    result = collections.defaultdict(dict)
    for job, score in zip(jobs, scores):
        key = (job["split"], job["eval_row_index"])
        if job["variant"] in result[key]:
            raise ValueError(f"Duplicate answer candidate score for {key}/{job['variant']}")
        result[key][job["variant"]] = score
    return result


def bootstrap_cluster_ci(rows: list[dict], field: str, seed: int = BOOTSTRAP_SEED) -> list[float] | None:
    groups = collections.defaultdict(list)
    for row in rows:
        if row.get("profile_block") is not None and row.get(field) is not None:
            groups[int(row["profile_block"])].append(float(row[field]))
    if not groups:
        return None
    group_values = list(groups.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [rng.choice(group_values) for _ in group_values]
        values = [value for group in sampled for value in group]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return [estimates[int(0.025 * (BOOTSTRAP_DRAWS - 1))], estimates[int(0.975 * (BOOTSTRAP_DRAWS - 1))]]


def mean_field(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def summarize_examples(examples: list[dict]) -> dict:
    summary = {"n_questions": len(examples), "n_profile_blocks": len({r["profile_block"] for r in examples if r.get("profile_block") is not None})}
    fields = (
        "answer_probability",
        "rougeL_recall",
        "truth_ratio",
        "truth_ratio_score",
        "normalized_option_probability",
    )
    for field in fields:
        value = mean_field(examples, field)
        if value is not None:
            summary[f"mean_{field}"] = value
            ci = bootstrap_cluster_ci(examples, field)
            if ci is not None:
                summary[f"profile_cluster_bootstrap_95pct_ci_{field}"] = ci
    return summary


def make_metrics(
    rows: list[dict],
    split: str,
    score_map: dict[tuple[str, int], dict[str, dict]],
    generations: dict[tuple[str, int], str],
    tokenizer,
    rouge: rouge_scorer.RougeScorer,
) -> list[dict]:
    metrics = []
    for row in rows:
        key = (split, row["eval_row_index"])
        scores = score_map.get(key, {})
        if "answer" not in scores:
            raise ValueError(f"Missing gold-answer score for {key}")
        output = {
            "split": split,
            "eval_row_index": row["eval_row_index"],
            "source_row_index": row["source_row_index"],
            "profile_block": row["profile_block"],
            "question_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
            "source_answer": row["answer"],
            "reference_answer": row["answer"],
            "ambiguous_question_collision": row["ambiguous_question_collision"],
            "answer_probability": scores["answer"]["answer_probability"],
            "answer_token_loss": scores["answer"]["avg_answer_token_loss"],
            "answer_scored_tokens": scores["answer"]["scored_answer_tokens"],
        }
        if "paraphrased_answer" in scores:
            para = scores["paraphrased_answer"]
            wrong = [scores[f"perturbed_{i}"] for i in range(len(row["perturbed_answers"] or []))]
            output["paraphrased_answer_probability"] = para["answer_probability"]
            output["paraphrased_answer_token_loss"] = para["avg_answer_token_loss"]
            output["perturbed_answer_probabilities"] = [x["answer_probability"] for x in wrong]
            output["perturbed_answer_token_losses"] = [x["avg_answer_token_loss"] for x in wrong]
            if wrong:
                # Match the pinned Open-Unlearning implementation: exponentiate
                # the mean per-candidate answer loss, i.e. the geometric mean of
                # perturbed-answer probabilities.
                geometric_wrong_probability = math.exp(-sum(x["avg_answer_token_loss"] for x in wrong) / len(wrong))
                ratio = geometric_wrong_probability / (para["answer_probability"] + EPSILON)
                output["truth_ratio"] = ratio
                output["truth_ratio_score"] = min(ratio, 1.0 / (ratio + EPSILON)) if split == "forget10" else max(0.0, 1.0 - ratio)
        elif row["perturbed_answers"]:
            wrong = [scores[f"perturbed_{i}"] for i in range(len(row["perturbed_answers"]))]
            wrong_probabilities = [x["answer_probability"] for x in wrong]
            geometric_wrong_probability = math.exp(-sum(x["avg_answer_token_loss"] for x in wrong) / len(wrong))
            ratio = geometric_wrong_probability / (scores["answer"]["answer_probability"] + EPSILON)
            output["perturbed_answer_probabilities"] = wrong_probabilities
            output["perturbed_answer_token_losses"] = [x["avg_answer_token_loss"] for x in wrong]
            output["truth_ratio"] = ratio
            output["truth_ratio_score"] = max(0.0, 1.0 - ratio)
            output["normalized_option_probability"] = scores["answer"]["answer_probability"] / (
                scores["answer"]["answer_probability"] + sum(wrong_probabilities) + EPSILON
            )
        if key in generations:
            text = generations[key]
            full_ids, _ = encode_answer(tokenizer, row["question"], row["answer"])
            prompt_ids = encode_prompt(tokenizer, row["question"])
            input_text = tokenizer.decode(
                prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            full_text = tokenizer.decode(
                full_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            reference = full_text.replace(input_text, "").strip()
            output["generation"] = text
            output["reference_answer"] = reference
            output["rougeL_recall"] = rouge.score(reference, text)["rougeL"].recall
        metrics.append(output)
    return metrics


def model_utility(metrics_by_split: dict[str, list[dict]]) -> dict:
    components = {
        "retain_probability": mean_field(metrics_by_split["retain90"], "answer_probability"),
        "retain_rougeL_recall": mean_field(metrics_by_split["retain90"], "rougeL_recall"),
        "retain_truth_ratio_score": mean_field(metrics_by_split["retain90"], "truth_ratio_score"),
        "real_authors_normalized_option_probability": mean_field(metrics_by_split["real_authors"], "normalized_option_probability"),
        "real_authors_rougeL_recall": mean_field(metrics_by_split["real_authors"], "rougeL_recall"),
        "real_authors_truth_ratio_score": mean_field(metrics_by_split["real_authors"], "truth_ratio_score"),
        "world_facts_normalized_option_probability": mean_field(metrics_by_split["world_facts"], "normalized_option_probability"),
        "world_facts_rougeL_recall": mean_field(metrics_by_split["world_facts"], "rougeL_recall"),
        "world_facts_truth_ratio_score": mean_field(metrics_by_split["world_facts"], "truth_ratio_score"),
    }
    if any(value is None for value in components.values()):
        raise ValueError(f"Cannot compute complete TOFU model utility; missing component: {components}")
    if any(value <= 0 for value in components.values()):
        aggregate = 0.0
    else:
        aggregate = len(components) / sum(1.0 / value for value in components.values())
    return {"components": components, "harmonic_mean": aggregate}


def write_reports(output_dir: Path) -> None:
    keys = ("llama_3_2_3b_full", "llama_3_1_8b_full")
    results = {key: json.loads((output_dir / f"{key}_official_eval.json").read_text(encoding="utf-8")) for key in keys}
    summaries = {key: results[key]["summary"] for key in keys}
    split_labels = {
        "forget10": "Forget10",
        "retain90": "Canonical retain eval",
        "holdout10": "Holdout10",
        "real_authors": "Real authors",
        "world_facts": "World facts",
    }
    lines = [
        "# TOFU official-metric pre-unlearning baseline",
        "",
        "**Status:** complete and inference-only. It evaluates the published full checkpoints and paired retain90 oracle checkpoints. No NPO training or research adapter was run.",
        "",
        "The metric formulas, chat preprocessing, and generated-reference extraction follow the pinned Open-Unlearning implementation. `answer_probability` is mean per-answer-token probability; `rougeL_recall` scores greedy generated answers; Truth Ratio compares the geometric mean likelihood of perturbed answers with the paraphrased correct answer (or source correct answer for real-author/world-fact utility). Kaggle T4 uses FP16 + SDPA here, while upstream configs specify BF16 + FlashAttention 2; formulas and prompts match, but numerical generations are not bitwise-identical to the upstream runtime.",
        "",
        "## Full-checkpoint metrics",
        "",
        "| Split | Model | Questions / profile blocks | Mean answer probability (95% profile CI) | Mean ROUGE-L recall (95% profile CI) | Mean truth ratio | TOFU truth-ratio score | Normalized option probability |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, label in split_labels.items():
        for key in keys:
            summary = summaries[key][split]
            def render(metric: str, ci_key: str | None = None) -> str:
                val = summary.get(f"mean_{metric}")
                if val is None:
                    return "—"
                if ci_key and ci_key in summary:
                    low, high = summary[ci_key]
                    return f"{val:.4f} [{low:.4f}, {high:.4f}]"
                return f"{val:.4f}"
            n = f"{summary['n_questions']} / {summary['n_profile_blocks']}" if summary["n_profile_blocks"] else f"{summary['n_questions']} / —"
            lines.append(
                f"| {label} | {results[key]['model']['label']} | {n} | "
                f"{render('answer_probability', 'profile_cluster_bootstrap_95pct_ci_answer_probability')} | "
                f"{render('rougeL_recall', 'profile_cluster_bootstrap_95pct_ci_rougeL_recall')} | "
                f"{render('truth_ratio')} | {render('truth_ratio_score')} | {render('normalized_option_probability')} |"
            )
    lines += ["", "## Composite utility and forget-quality comparison", ""]
    lines.append("| Model | TOFU model-utility harmonic mean | Forget-quality KS p-value vs retain90 oracle |")
    lines.append("|---|---:|---:|")
    for key in keys:
        utility = results[key]["model_utility"]["harmonic_mean"]
        fq = results[key].get("forget_quality_vs_retain90_oracle", {}).get("p_value")
        lines.append(f"| {results[key]['model']['label']} | {utility:.6f} | {fq:.6g}" if fq is not None else f"| {results[key]['model']['label']} | {utility:.6f} | pending oracle result |")

    lines += [
        "",
        "## Retain-question collision sensitivity",
        "",
        "The pinned `retain90` data repeats one generic question with different answers. The canonical 400-row retain evaluation includes one of the two conflicting rows. Primary TOFU metrics retain the official row; the sensitivity row below excludes it.",
        "",
        "| Model | Retain probability, primary → excluding collision | Retain ROUGE-L recall, primary → excluding collision | Retain truth-ratio score, primary → excluding collision |",
        "|---|---:|---:|---:|",
    ]
    for key in keys:
        rows = results[key]["per_example"]["retain90"]
        clean = [row for row in rows if not row["ambiguous_question_collision"]]
        all_summary = summarize_examples(rows)
        clean_summary = summarize_examples(clean)
        lines.append(
            f"| {results[key]['model']['label']} | {all_summary['mean_answer_probability']:.4f} → {clean_summary['mean_answer_probability']:.4f} | "
            f"{all_summary['mean_rougeL_recall']:.4f} → {clean_summary['mean_rougeL_recall']:.4f} | "
            f"{all_summary['mean_truth_ratio_score']:.4f} → {clean_summary['mean_truth_ratio_score']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Confidence intervals resample the 20 inferred profile blocks for forget10, canonical retain, and holdout10. The JSON contains question-level outputs; real-author and world-fact rows have no explicit cluster IDs, so they are reported without profile-cluster intervals.",
        "- Inference uses FP16 + SDPA because Kaggle T4 does not provide the upstream BF16 + FlashAttention 2 runtime. Small numeric and generation differences from an upstream evaluation are possible.",
        "- The 3B and 8B checkpoints come from different Llama generations and training histories. Their paired differences are descriptive checkpoint comparisons, not causal evidence about parameter count.",
        "- A small/large ROUGE or likelihood value alone does not prove forgetting. The forget-quality p-value compares distributions with a retain-only oracle and is not a standalone proof of equivalence or privacy.",
        "- TOFU profiles are fictional. These results cannot establish deletion of real personal data or real-world privacy protection.",
        "- The earlier 6 Nov likelihood-only run used a custom retain sample; this report uses the canonical `retain_perturbed` evaluation set from the pinned benchmark revision.",
        "",
        "## Reproduction metadata",
        "",
        f"Dataset revision: `{DATASET_REVISION}`. Open-Unlearning code revision: `{UPSTREAM_REVISION}`. Bootstrap: {BOOTSTRAP_DRAWS:,} profile-cluster draws, seed {BOOTSTRAP_SEED}. Greedy generation: max {MAX_NEW_TOKENS} new tokens.",
        "",
    ]
    (output_dir / "official_metric_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")

    compare = {}
    for split in split_labels:
        left = results["llama_3_2_3b_full"]["per_example"][split]
        right = results["llama_3_1_8b_full"]["per_example"][split]
        right_by_index = {row["source_row_index"]: row for row in right}
        if len(right_by_index) != len(right) or {row["source_row_index"] for row in left} != set(right_by_index):
            raise ValueError(f"Full checkpoints do not have matched source examples in {split}")
        item = {}
        for metric in ("answer_probability", "rougeL_recall", "truth_ratio", "truth_ratio_score", "normalized_option_probability"):
            pairs = [(row, right_by_index[row["source_row_index"]]) for row in left if metric in row and metric in right_by_index[row["source_row_index"]]]
            if not pairs:
                continue
            deltas = [float(b[metric]) - float(a[metric]) for a, b in pairs]
            metric_result = {"n_questions": len(pairs), "mean_8b_minus_3b": sum(deltas) / len(deltas)}
            grouped = collections.defaultdict(list)
            for (a, _), delta in zip(pairs, deltas):
                if a.get("profile_block") is not None:
                    grouped[int(a["profile_block"])].append(delta)
            if grouped:
                group_ids = list(grouped)
                rng = random.Random(BOOTSTRAP_SEED)
                estimates = []
                for _ in range(BOOTSTRAP_DRAWS):
                    sampled = [rng.choice(group_ids) for _ in group_ids]
                    vals = [value for group in sampled for value in grouped[group]]
                    estimates.append(sum(vals) / len(vals))
                estimates.sort()
                metric_result["profile_cluster_bootstrap_95pct_ci"] = [
                    estimates[int(0.025 * (BOOTSTRAP_DRAWS - 1))],
                    estimates[int(0.975 * (BOOTSTRAP_DRAWS - 1))],
                ]
            else:
                metric_result["interval"] = None
                metric_result["interval_note"] = "No explicit profile cluster IDs; no inferential interval reported."
            item[metric] = metric_result
        compare[split] = item
    compare["interpretation_limit"] = "Descriptive paired checkpoint contrasts only; model generations/training histories differ."
    (output_dir / "paired_full_checkpoint_comparison.json").write_text(json.dumps(compare, indent=2) + "\n", encoding="utf-8")


def encode_jobs(tokenizer, jobs: list[dict]) -> list[tuple[list[int], list[int]]]:
    return [encode_answer(tokenizer, job["question"], job["candidate_answer"]) for job in jobs]


def evaluate_checkpoint(
    key: str,
    output_dir: Path,
    eval_sets: dict[str, list[dict]],
    dataset_hashes: dict[str, str],
    score_batch_size: int,
    generation_batch_size: int,
    max_memory_per_gpu_gib: float,
) -> dict:
    from accelerate import __version__ as accelerate_version
    from transformers import __version__ as transformers_version

    spec = CHECKPOINTS[key]
    model_cache = Path(tempfile.gettempdir()) / "tofu_official_eval_cache" / key
    model_cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"], cache_dir=str(model_cache), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required")

    score_jobs, generation_jobs = build_jobs(eval_sets, spec["role"])
    encoded_jobs = encode_jobs(tokenizer, score_jobs)
    prompt_ids = [encode_prompt(tokenizer, row["question"]) for row in generation_jobs]
    memory_limit = f"{max_memory_per_gpu_gib:g}GiB"
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        cache_dir=str(model_cache),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        # Leave headroom for Transformers' allocator warmup, activations, and
        # CUDA context on 15 GiB T4s. The 8B checkpoint may CPU-offload a small
        # number of layers rather than OOM during weight loading.
        max_memory={0: memory_limit, 1: memory_limit, "cpu": "40GiB"},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False
    score_values = score_answers(model, encoded_jobs, int(tokenizer.pad_token_id), score_batch_size)
    if len(score_values) != len(score_jobs):
        raise RuntimeError("Scored row count does not equal job count")
    score_map = score_rows_by_key(score_jobs, score_values)
    generations = {}
    if spec["role"] == "full":
        generated_texts = generate_answers(model, tokenizer, prompt_ids, generation_batch_size)
        if len(generated_texts) != len(generation_jobs):
            raise RuntimeError("Generation row count does not equal prompt count")
        generations = {
            (row["split"], row["eval_row_index"]): text
            for row, text in zip(generation_jobs, generated_texts)
        }

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge.tokenizer = tokenizer
    splits_to_write = ("forget10",) if spec["role"] == "retain90_oracle" else tuple(eval_sets)
    metrics_by_split = {
        split: make_metrics(eval_sets[split], split, score_map, generations, tokenizer, rouge)
        for split in splits_to_write
    }
    result = {
        "status": "PASS_OFFICIAL_TOFU_BASELINE",
        "model": spec,
        "evaluation": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256_by_file": dataset_hashes,
            "open_unlearning_revision": UPSTREAM_REVISION,
            "profile_block_size": PROFILE_QA_COUNT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "use_cache": True},
            "declared_max_input_length": MAX_LENGTH,
            "input_truncation": False,
            "max_scored_sequence_tokens": max((len(tokens) for tokens, _ in encoded_jobs), default=0),
            "scored_sequences_over_declared_max_length": sum(len(tokens) > MAX_LENGTH for tokens, _ in encoded_jobs),
            "max_generation_prompt_tokens": max((len(tokens) for tokens in prompt_ids), default=0),
            "canonical_retain_eval_rows": len(eval_sets["retain90"]),
            "canonical_retain_profile_blocks": sorted({row["profile_block"] for row in eval_sets["retain90"]}),
            "score_jobs": len(score_jobs),
            "generation_jobs": len(generation_jobs),
            "metric_implementation": "Pinned Open-Unlearning probability, ROUGE-L recall, Truth Ratio, normalized option probability, model-utility harmonic mean, and KS forget-quality definitions; see README for exact conventions.",
        },
        "summary": {split: summarize_examples(rows) for split, rows in metrics_by_split.items()},
        "per_example": {split: rows for split, rows in metrics_by_split.items()},
        "model_utility": model_utility(metrics_by_split) if spec["role"] == "full" else None,
        "runtime": {
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "accelerate_version": accelerate_version,
            "rouge_score_version": __import__("importlib.metadata", fromlist=["version"]).version("rouge-score"),
            "scipy_version": __import__("scipy").__version__,
            "inference_dtype": "float16",
            "max_memory_per_gpu_gib": max_memory_per_gpu_gib,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_total_memory_bytes": [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())],
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "training_started": False,
            "research_adapter_created": False,
        },
    }
    result_path = output_dir / f"{key}_official_eval.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_path = output_dir / f"{key}_examples.jsonl"
    with score_path.open("w", encoding="utf-8") as f:
        for rows in metrics_by_split.values():
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    del model, tokenizer, encoded_jobs, prompt_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(model_cache, ignore_errors=True)
    return result


def add_forget_quality(results_dir: Path, full_key: str, oracle_key: str) -> None:
    full_path = results_dir / f"{full_key}_official_eval.json"
    oracle_path = results_dir / f"{oracle_key}_official_eval.json"
    if not full_path.exists() or not oracle_path.exists():
        return
    full = json.loads(full_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    full_rows = full["per_example"]["forget10"]
    oracle_rows = oracle["per_example"]["forget10"]
    if [(r["source_row_index"], r["question_sha256"]) for r in full_rows] != [
        (r["source_row_index"], r["question_sha256"]) for r in oracle_rows
    ]:
        raise ValueError("Full and retain90 oracle forget rows are not aligned")
    stat = ks_2samp(
        [1.0 / (row["truth_ratio"] + EPSILON) for row in full_rows],
        [1.0 / (row["truth_ratio"] + EPSILON) for row in oracle_rows],
    )
    full["forget_quality_vs_retain90_oracle"] = {
        "metric": "KS p-value on inverse forget truth-ratio distributions; monotonic inversion leaves the two-sample KS statistic and p-value unchanged",
        "n_full_checkpoint": len(full_rows),
        "n_retain90_oracle": len(oracle_rows),
        "ks_statistic": float(stat.statistic),
        "p_value": float(stat.pvalue),
        "interpretation": "A larger p-value means these two empirical distributions are less distinguishable by this KS test; it is not proof of successful unlearning.",
    }
    full_path.write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/tofu_official_eval"))
    parser.add_argument("--models", default=",".join(CHECKPOINTS))
    parser.add_argument("--score-batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--max-memory-per-gpu-gib", type=float, default=6.0)
    parser.add_argument("--resume-existing", action="store_true", help="Reuse passing, pinned result files already in output-dir; final verifier checks every row and metric.")
    args = parser.parse_args()
    selected = [x.strip() for x in args.models.split(",") if x.strip()]
    if not selected or set(selected) - set(CHECKPOINTS):
        parser.error(f"models must be selected from {sorted(CHECKPOINTS)}")
    if args.max_memory_per_gpu_gib <= 0 or args.max_memory_per_gpu_gib >= 14:
        parser.error("max-memory-per-gpu-gib must be >0 and <14 to leave T4 runtime headroom")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Requires GPU T4 x2; stopped before loading model weights")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, hashes = load_data()
    eval_sets = build_eval_sets(data)
    status_path = args.output_dir / "official_eval_status.json"
    reused_results = {}
    if args.resume_existing and status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("dataset_hashes") != hashes:
            raise RuntimeError("Cannot resume: existing receipt has different pinned dataset hashes")
        for key, entry in previous.get("model_results", {}).items():
            if entry.get("status") != "PASS_OFFICIAL_TOFU_BASELINE":
                continue
            result_path = args.output_dir / f"{key}_official_eval.json"
            examples_path = args.output_dir / f"{key}_examples.jsonl"
            if key not in CHECKPOINTS or not result_path.is_file() or not examples_path.is_file():
                raise RuntimeError(f"Cannot resume: incomplete successful receipt for {key}")
            prior_result = json.loads(result_path.read_text(encoding="utf-8"))
            spec = CHECKPOINTS[key]
            if (
                prior_result.get("status") != "PASS_OFFICIAL_TOFU_BASELINE"
                or prior_result.get("model", {}).get("repo_id") != spec["repo_id"]
                or prior_result.get("model", {}).get("revision") != spec["revision"]
                or prior_result.get("model", {}).get("role") != spec["role"]
                or prior_result.get("evaluation", {}).get("dataset_sha256_by_file") != hashes
            ):
                raise RuntimeError(f"Cannot resume: pinned model/data metadata mismatch for {key}")
            reused_results[key] = entry
        if set(selected) & set(reused_results):
            raise RuntimeError("Requested model is already marked successful; remove it from --models or start a clean output directory")

    for key in selected:
        if CHECKPOINTS[key]["role"] == "full":
            oracle_key = key.replace("_full", "_retain90")
            if oracle_key not in selected and oracle_key not in reused_results:
                parser.error(f"{key} requires its matching retain90 oracle in --models or a verified resumed output")

    status = {
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_models": selected,
        "reused_models": sorted(reused_results),
        "dataset_hashes": hashes,
        "model_results": reused_results,
        "training_started": False,
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key in selected:
        try:
            result = evaluate_checkpoint(
                key,
                args.output_dir,
                eval_sets,
                hashes,
                args.score_batch_size,
                args.generation_batch_size,
                args.max_memory_per_gpu_gib,
            )
            status["model_results"][key] = {
                "status": result["status"],
                "result_file": f"{key}_official_eval.json",
                "examples_file": f"{key}_examples.jsonl",
                "runtime_seconds": result["runtime"]["elapsed_seconds"],
            }
        except Exception as error:
            shutil.rmtree(Path(tempfile.gettempdir()) / "tofu_official_eval_cache" / key, ignore_errors=True)
            status["status"] = "ERROR"
            status["error"] = f"{type(error).__name__}: {error}"
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{key}: PASS_OFFICIAL_TOFU_BASELINE", flush=True)
    for full_key, oracle_key in (
        ("llama_3_2_3b_full", "llama_3_2_3b_retain90"),
        ("llama_3_1_8b_full", "llama_3_1_8b_retain90"),
    ):
        if full_key in status["model_results"] and oracle_key in status["model_results"]:
            add_forget_quality(args.output_dir, full_key, oracle_key)
    complete_keys = set(CHECKPOINTS)
    if complete_keys <= set(status["model_results"]):
        write_reports(args.output_dir)
    else:
        status["status"] = "PARTIAL"
        status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Partial results saved under {args.output_dir}; missing {sorted(complete_keys - set(status['model_results']))}")
        return
    status["status"] = "COMPLETE"
    status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved under {args.output_dir}")


if __name__ == "__main__":
    main()
