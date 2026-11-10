"""Minimal NPO math shared by training and the CPU objective-equivalence check."""

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed answer NLL and scored-token count per sequence."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous().to(logits.device)
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(shift_labels.shape)
    counts = (shift_labels != IGNORE_INDEX).sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("An NPO example has no assistant answer tokens")
    return losses.sum(dim=-1), counts


def objective_from_nlls(current_nll: torch.Tensor, reference_nll: torch.Tensor, beta: float) -> torch.Tensor:
    """Pinned Open-Unlearning NPO loss for sequence-summed answer NLL."""
    # Upstream computes -2/beta * log(sigmoid(-beta * log(pi_theta/pi_ref))).
    # Since log(pi_theta/pi_ref) = reference_nll - current_nll, this is
    # beta * (current_nll - reference_nll) inside the sigmoid.
    return (-2.0 / beta * F.logsigmoid(beta * (current_nll - reference_nll))).mean()
