#!/usr/bin/env python3
"""Verify gradient accumulation preserves token-weighted retain NLL."""

from __future__ import annotations

import json

import torch

from npo_objective import retain_token_mean_contribution


def main() -> None:
    values = torch.tensor([0.2, 0.5, 0.7, 0.1, 0.4, 0.8, 0.6])

    reference_parameter = torch.nn.Parameter(torch.tensor(0.3))
    reference_token_losses = (reference_parameter - values).square()
    reference_loss = reference_token_losses.mean()
    reference_loss.backward()

    accumulated_parameter = torch.nn.Parameter(torch.tensor(0.3))
    first_microbatch = (accumulated_parameter - values[:2]).square()
    second_microbatch = (accumulated_parameter - values[2:]).square()
    accumulated_loss = (
        retain_token_mean_contribution(first_microbatch.sum(), len(values))
        + retain_token_mean_contribution(second_microbatch.sum(), len(values))
    )
    accumulated_loss.backward()

    equal_example_parameter = torch.nn.Parameter(torch.tensor(0.3))
    equal_example_loss = (
        (equal_example_parameter - values[:2]).square().mean()
        + (equal_example_parameter - values[2:]).square().mean()
    ) / 2
    equal_example_loss.backward()

    value_error = abs(float(reference_loss.detach() - accumulated_loss.detach()))
    gradient_error = abs(float(reference_parameter.grad - accumulated_parameter.grad))
    sensitivity = abs(float(reference_parameter.grad - equal_example_parameter.grad))
    passed = value_error <= 1e-7 and gradient_error <= 1e-7 and sensitivity > 1e-4
    result = {
        "status": "PASS" if passed else "FAIL",
        "fixture_token_counts_by_microbatch": [2, 5],
        "token_weighted_loss_abs_error": value_error,
        "token_weighted_gradient_abs_error": gradient_error,
        "difference_from_old_equal_example_weighting": sensitivity,
        "tolerance": 1e-7,
        "interpretation": "Unequal-length microbatches accumulate to the same token-mean value and gradient as one pooled batch.",
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
