#!/usr/bin/env python3
"""Verify IDK+AP loss values and gradients against the pinned official code."""

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


IGNORE_INDEX = -100
UPSTREAM_COMMIT = "b622294965f8b40feb5383c9e93dbdb26a76b6b3"
UPSTREAM_LOSS_SHA256 = "078f64bb847bde0734747c00decf8cd06d44ca587e2c4eae2db01b12e7a2534f"
EXPECTED_PROTOCOL_SHA256 = "650af06eed1d254cfdd2b47714299686925cea47b5b2675edb7c07674b01c049"


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(19, 9)
        self.head = torch.nn.Linear(9, 19, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None):
        logits = self.head(self.embedding(input_ids))
        loss = None
        if labels is not None:
            shifted_logits = logits[..., :-1, :].contiguous().float()
            shifted_labels = labels[..., 1:].contiguous().to(logits.device)
            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )
        return SimpleNamespace(logits=logits, loss=loss)


def batch_sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = labels[..., 1:].contiguous().to(logits.device)
    token_loss = F.cross_entropy(
        shifted_logits.transpose(-1, -2), shifted_labels,
        ignore_index=IGNORE_INDEX, reduction="none",
    )
    return token_loss.sum(dim=-1)


def candidate_get_loss(model, inputs, beta: float):
    forget_ids, forget_labels, forget_mask = inputs[2]
    forget = model(forget_ids, labels=forget_labels, attention_mask=forget_mask).loss
    retain_ids, retain_labels, retain_mask = inputs[1]
    idk_ids, idk_labels, idk_mask = inputs[3]
    answer_nll = batch_sequence_nll(model(retain_ids, attention_mask=retain_mask).logits, retain_labels)
    idk_nll = batch_sequence_nll(model(idk_ids, attention_mask=idk_mask).logits, idk_labels)
    ap = -F.logsigmoid(beta * (idk_nll - answer_nll)).mean() / beta
    return forget, ap


def max_grad_delta(left: torch.nn.Module, right: torch.nn.Module) -> float:
    values = []
    for a, b in zip(left.parameters(), right.parameters()):
        if a.grad is None or b.grad is None:
            if a.grad is not b.grad:
                return float("inf")
        else:
            values.append(float((a.grad - b.grad).abs().max()))
    return max(values, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-losses", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("idk_ap_lora_protocol.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_source = f"sail-sg/closer-look-LLM-unlearning@{UPSTREAM_COMMIT}"
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        raise RuntimeError("IDK+AP protocol is not preregistered")
    if hashlib.sha256(args.protocol.read_bytes()).hexdigest() != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Locked protocol SHA-256 mismatch")
    if protocol["objective"].get("source_repository") + "@" + protocol["objective"].get("source_commit") != expected_source:
        raise RuntimeError("Protocol does not pin the expected IDK+AP source")
    source_bytes = args.upstream_losses.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != UPSTREAM_LOSS_SHA256 or source_hash != protocol["objective"]["loss_source_sha256"]:
        raise RuntimeError(f"Pinned official loss source hash mismatch: {source_hash}")
    spec = importlib.util.spec_from_file_location("official_idk_ap_losses", args.upstream_losses)
    official = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(official)

    # Variable response lengths verify that AP uses each response's summed token NLL.
    inputs = [
        None,
        (torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]),
         torch.tensor([[-100, -100, -100, 4, 5, 6], [-100, -100, 4, 5, 6, 7]]),
         torch.ones((2, 6), dtype=torch.long)),
        (torch.tensor([[3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]]),
         torch.tensor([[-100, -100, 5, 6, 7, 8], [-100, -100, -100, -100, 8, 9]]),
         torch.ones((2, 6), dtype=torch.long)),
        (torch.tensor([[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]]),
         torch.tensor([[-100, -100, -100, 7, 9, 11], [-100, -100, -100, 8, 10, 12]]),
         torch.ones((2, 6), dtype=torch.long)),
    ]
    beta = float(protocol["objective"]["beta"])
    torch.manual_seed(42)
    base = TinyCausalLM()
    official_model = copy.deepcopy(base)
    candidate_model = copy.deepcopy(base)
    official_forget, official_ap = official.get_loss(official_model, None, inputs, "IDK+AP", beta=beta)
    candidate_forget, candidate_ap = candidate_get_loss(candidate_model, inputs, beta)
    official_total = official_forget + official_ap
    candidate_total = candidate_forget + candidate_ap
    official_total.backward()
    candidate_total.backward()
    forget_diff = abs(float(official_forget.detach()) - float(candidate_forget.detach()))
    ap_diff = abs(float(official_ap.detach()) - float(candidate_ap.detach()))
    total_diff = abs(float(official_total.detach()) - float(candidate_total.detach()))
    grad_diff = max_grad_delta(official_model, candidate_model)
    passed = max(forget_diff, ap_diff, total_diff, grad_diff) <= 1e-6
    result = {
        "status": "PASS" if passed else "FAIL",
        "official_source": expected_source,
        "official_losses_py_sha256": source_hash,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "beta": beta,
        "official_forget_loss": float(official_forget.detach()),
        "candidate_forget_loss": float(candidate_forget.detach()),
        "forget_loss_abs_difference": forget_diff,
        "official_ap_loss": float(official_ap.detach()),
        "candidate_ap_loss": float(candidate_ap.detach()),
        "ap_loss_abs_difference": ap_diff,
        "total_loss_abs_difference": total_diff,
        "max_parameter_gradient_abs_difference": grad_diff,
        "batch_size": 2,
        "response_token_counts_vary": True,
        "loss_reduction": "forget IDK uses token-mean CE; AP uses mean of per-sequence summed-NLL preference losses",
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
