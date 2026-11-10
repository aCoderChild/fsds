#!/usr/bin/env python3
"""Check NPO loss/value/gradient equivalence against pinned Open-Unlearning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from npo_objective import IGNORE_INDEX, objective_from_nlls, sequence_nll


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 7)
        self.head = torch.nn.Linear(7, 13, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.head(self.embedding(input_ids))
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        return SimpleNamespace(logits=logits, loss=loss)


def max_abs_grad_difference(left, right):
    deltas = []
    for p_left, p_right in zip(left.parameters(), right.parameters()):
        if p_left.grad is None or p_right.grad is None:
            if p_left.grad is not p_right.grad:
                return float("inf")
            continue
        deltas.append(float((p_left.grad - p_right.grad).abs().max()))
    return max(deltas, default=0.0)


def verify_retain_accumulation_source(training_source: Path) -> dict:
    source = training_source.read_text(encoding="utf-8") if training_source.is_file() else ""
    source_header = "\n".join(source.splitlines()[:50])
    requirements = {
        "imports_token_mean_helper": "retain_token_mean_contribution" in source_header,
        "counts_tokens_across_update": "retain_token_count_in_update = sum(" in source,
        "normalizes_microbatch_sum_by_update_tokens": (
            "retain_token_mean_contribution(r_nll.sum(), retain_token_count_in_update)" in source
        ),
        "backpropagates_normalized_retain_contribution": (
            '(METHOD["alpha_retain"] * retain_loss).backward()' in source
        ),
    }
    return {
        "status": "PASS" if training_source.is_file() and all(requirements.values()) else "FAIL",
        "training_source": str(training_source),
        "training_source_sha256": hashlib.sha256(training_source.read_bytes()).hexdigest() if training_source.is_file() else None,
        "requirements": requirements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-src", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        raise RuntimeError("NPO pilot protocol is not registered")
    if protocol.get("objective", {}).get("source") != "locuslab/open-unlearning@2d98027677f3468d52b6baa3fc4f2b9f2dc08820":
        raise RuntimeError("Protocol does not pin the expected Open-Unlearning implementation")
    upstream_src = args.upstream_src.resolve()
    upstream_utils_candidates = (
        upstream_src / "trainer" / "utils.py",
        upstream_src / "src" / "trainer" / "utils.py",
    )
    upstream_utils = next((path for path in upstream_utils_candidates if path.is_file()), upstream_utils_candidates[0])
    if not upstream_utils.is_file():
        raise FileNotFoundError(upstream_utils)
    spec = importlib.util.spec_from_file_location("pinned_open_unlearning_utils", upstream_utils)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    compute_dpo_loss = upstream.compute_dpo_loss

    torch.manual_seed(42)
    beta = float(protocol["objective"]["beta"])
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 6, 7]]),
        "attention_mask": torch.ones((2, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4, 5], [-100, 3, 4, 6, 7]]),
    }
    base = TinyCausalLM()
    model_upstream = copy.deepcopy(base)
    model_candidate = copy.deepcopy(base)
    ref_upstream = copy.deepcopy(base)
    ref_candidate = copy.deepcopy(base)

    upstream_loss, _ = compute_dpo_loss(
        model=model_upstream,
        ref_model=ref_upstream,
        win_inputs=None,
        lose_inputs=inputs,
        beta=beta,
    )
    upstream_loss.backward()

    current_logits = model_candidate(**inputs).logits
    current_nll, _ = sequence_nll(current_logits, inputs["labels"])
    with torch.no_grad():
        reference_logits = ref_candidate(**inputs).logits
        reference_nll, _ = sequence_nll(reference_logits, inputs["labels"])
    candidate_loss = objective_from_nlls(current_nll, reference_nll, beta)
    candidate_loss.backward()

    value_diff = abs(float(upstream_loss.detach()) - float(candidate_loss.detach()))
    grad_diff = max_abs_grad_difference(model_upstream, model_candidate)
    retain_reference_loss = model_candidate(**inputs).loss.detach()
    candidate_retain_loss = current_nll.detach().sum() / (inputs["labels"][..., 1:] != IGNORE_INDEX).sum()
    retain_diff = abs(float(retain_reference_loss) - float(candidate_retain_loss))

    model_combined = copy.deepcopy(base)
    reference_combined = copy.deepcopy(base)
    combined_forget, _ = compute_dpo_loss(
        model=model_combined,
        ref_model=reference_combined,
        win_inputs=None,
        lose_inputs=inputs,
        beta=beta,
    )
    combined_retain = model_combined(**inputs).loss
    (combined_forget + combined_retain).backward()

    model_sequential = copy.deepcopy(base)
    reference_sequential = copy.deepcopy(base)
    sequential_forget, _ = compute_dpo_loss(
        model=model_sequential,
        ref_model=reference_sequential,
        win_inputs=None,
        lose_inputs=inputs,
        beta=beta,
    )
    sequential_forget.backward()
    model_sequential(**inputs).loss.backward()
    sequential_grad_diff = max_abs_grad_difference(model_combined, model_sequential)
    retain_alignment = verify_retain_accumulation_source(Path(__file__).with_name("npo_lora_train.py"))
    result = {
        "status": "PASS" if value_diff <= 1e-6 and grad_diff <= 1e-6 and retain_diff <= 1e-6 and sequential_grad_diff <= 1e-6 and retain_alignment["status"] == "PASS" else "FAIL",
        "upstream": protocol["objective"]["source"],
        "upstream_utils_sha256": hashlib.sha256(upstream_utils.read_bytes()).hexdigest(),
        "candidate_objective_sha256": hashlib.sha256(Path(__file__).with_name("npo_objective.py").read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "npo_beta": beta,
        "upstream_loss": float(upstream_loss.detach()),
        "candidate_loss": float(candidate_loss.detach()),
        "absolute_loss_difference": value_diff,
        "maximum_parameter_gradient_difference": grad_diff,
        "retain_nll_loss_difference": retain_diff,
        "combined_vs_sequential_backward_max_gradient_difference": sequential_grad_diff,
        "examples": int(inputs["input_ids"].shape[0]),
        "loss_reduction": "sum token NLL per sequence; batch mean for NPO; token mean for retain NLL",
        "retain_accumulation_source_alignment": retain_alignment,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
