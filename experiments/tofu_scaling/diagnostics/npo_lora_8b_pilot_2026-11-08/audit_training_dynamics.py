#!/usr/bin/env python3
"""Summarize the archived 8B NPO pilot trace without loading model weights."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACE = (
    ROOT
    / "kaggle_v2/extracted_archive/tofu_8b_npo_run_20261108T205627Z"
    / "npo_train/loss_trace.jsonl"
)
OUTPUT = ROOT / "training_dynamics_audit.json"
EXPECTED_STEPS = 500
EXPECTED_MICROBATCHES_PER_STEP = 8


def main() -> None:
    raw = TRACE.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    by_step: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_step[int(record["global_step"])].append(record)

    expected = set(range(1, EXPECTED_STEPS + 1))
    observed = set(by_step)
    errors = []
    if observed != expected:
        errors.append("global_step sequence is not exactly 1..500")
    bad_counts = {
        step: len(rows)
        for step, rows in by_step.items()
        if len(rows) != EXPECTED_MICROBATCHES_PER_STEP
    }
    if bad_counts:
        errors.append("optimizer steps do not each have exactly 8 microbatch records")

    required = ("forget_npo_loss", "retain_token_nll", "combined_loss", "gradient_norm", "learning_rate")
    for index, record in enumerate(records):
        for key in required:
            if key not in record or not math.isfinite(float(record[key])):
                errors.append(f"trace row {index} has missing/non-finite {key}")
                break

    step_means = {}
    for step in sorted(by_step):
        rows = by_step[step]
        step_means[str(step)] = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in required
        }

    selected = [1, 10, 50, 100, 250, 500]
    selected_steps = {str(step): step_means[str(step)] for step in selected if str(step) in step_means}
    first_25 = [row for step, rows in by_step.items() if step <= 25 for row in rows]
    last_25 = [row for step, rows in by_step.items() if step > EXPECTED_STEPS - 25 for row in rows]

    def means(rows: list[dict]) -> dict[str, float]:
        return {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in required
        }

    result = {
        "audit_id": "tofu_8b_npo_pilot_training_dynamics_2026-11-09",
        "trace": str(TRACE.relative_to(ROOT)),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": len(records),
        "expected_optimizer_steps": EXPECTED_STEPS,
        "observed_optimizer_steps": len(observed),
        "microbatch_records_per_optimizer_step": EXPECTED_MICROBATCHES_PER_STEP,
        "all_steps_have_expected_record_count": not bad_counts,
        "all_logged_values_finite": not any("non-finite" in error for error in errors),
        "selected_step_means": selected_steps,
        "first_25_step_microbatch_means": means(first_25),
        "last_25_step_microbatch_means": means(last_25),
        "interpretation": (
            "The trace is descriptive optimization telemetry. It records each microbatch within an optimizer update; "
            "it does not contain intermediate evaluation scores, so it cannot identify an acceptable early-stop step."
        ),
        "status": "PASS_TRACE_STRUCTURE_AND_FINITE_VALUES" if not errors else "FAIL_TRACE_AUDIT",
        "errors": errors,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
