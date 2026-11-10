"""Shared scalar objectives used by unlearning and its runtime smoke."""
from __future__ import annotations

import math

import torch.nn.functional as F


def npo_loss(policy_sequence_logp, reference_sequence_logp, beta: float):
    """Return the NPO loss for one policy/reference sequence-log-prob pair."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("NPO beta must be finite and positive")
    return -(2.0 / beta) * F.logsigmoid(
        -beta * (policy_sequence_logp - reference_sequence_logp)
    )
