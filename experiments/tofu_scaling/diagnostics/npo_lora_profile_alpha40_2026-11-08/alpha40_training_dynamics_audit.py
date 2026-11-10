#!/usr/bin/env python3
"""Recompute optimizer-step training dynamics from the preserved Kaggle ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_ZIP = HERE / "kaggle_output/tofu_npo_alpha40_followup_artifacts.zip"
LOCAL_TRAINER = Path(__file__).resolve().parents[2] / "npo_lora_train.py"
TRACE_MEMBER = "tofu_npo_alpha40_followup/dev_alpha_40/loss_trace.jsonl"
TRAINER_MEMBER = "tofu_npo_alpha40_followup/npo_lora_train.py"
RECEIPT_MEMBER = "tofu_npo_alpha40_followup/dev_alpha_40/training_receipt.json"
PROTOCOL_MEMBER = "tofu_npo_alpha40_followup/npo_lora_recovery_protocol.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=HERE / "alpha40_training_dynamics_audit.json")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise SystemExit(f"ZIP CRC failed at {bad_member}")
        trace_bytes = archive.read(TRACE_MEMBER)
        trainer_bytes = archive.read(TRAINER_MEMBER)
        receipt = json.loads(archive.read(RECEIPT_MEMBER))
        archived_protocol_bytes = archive.read(PROTOCOL_MEMBER)

    local_trainer_bytes = LOCAL_TRAINER.read_bytes()
    archived_trainer_hash = sha256(trainer_bytes)
    local_trainer_hash = sha256(local_trainer_bytes)
    if archived_trainer_hash != receipt.get("training_script_sha256"):
        raise SystemExit("Archived trainer hash differs from the run receipt")
    protocol = json.loads(archived_protocol_bytes)
    if sha256(archived_protocol_bytes) != receipt.get("protocol_sha256"):
        raise SystemExit("Archived protocol hash differs from the run receipt")

    # This archived run predates later edits to the shared local trainer. Audit
    # source identity against the immutable receipt, not today's live file.
    archived_source = trainer_bytes.decode("utf-8")
    per_microbatch_token_mean = (
        "retain_loss = (r_nll.sum() / r_counts.sum().to(r_nll.dtype))" in archived_source
        and "(METHOD[\"alpha_retain\"] * retain_loss / accumulation_steps).backward()" in archived_source
    )
    accumulation_steps = int(receipt["training_schedule"]["gradient_accumulation_steps"])
    protocol_objective = protocol.get("objective", {})
    protocol_scope = protocol_objective.get("retain_loss_aggregation_scope")
    objective_fixture_examples = int(receipt.get("objective_equivalence", {}).get("examples", 0))
    if not per_microbatch_token_mean:
        raise SystemExit("Archived trainer aggregation does not match the audited source pattern")

    rows = [json.loads(line) for line in trace_bytes.decode("utf-8").splitlines() if line]
    by_step: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_step[int(row["global_step"])].append(row)
        for key in ("forget_npo_loss", "retain_token_nll", "combined_loss", "gradient_norm", "learning_rate"):
            if not math.isfinite(float(row[key])):
                raise SystemExit(f"Non-finite {key} at step {row['global_step']}")

    step_rows = [by_step[key] for key in sorted(by_step)]
    expected_steps = int(receipt["training_schedule"]["planned_optimizer_steps"])
    expected_accumulation = int(receipt["training_schedule"]["gradient_accumulation_steps"])
    if len(step_rows) != expected_steps or len(rows) != int(receipt["training_schedule"]["planned_loss_trace_rows"]):
        raise SystemExit("Trace row or optimizer-step count differs from the training receipt")
    if any(len(group) != expected_accumulation for group in step_rows):
        raise SystemExit("Unexpected number of microbatch records for an optimizer step")

    steps = [group[0] for group in step_rows]
    for group in step_rows:
        for key in ("gradient_norm", "learning_rate", "epoch"):
            if len({row[key] for row in group}) != 1:
                raise SystemExit(f"Inconsistent {key} across one accumulated update")

    clip_cap = float(receipt["method"]["gradient_clip_norm"])
    norms = [float(row["gradient_norm"]) for row in steps]
    epochs: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        epochs[int(row["epoch"])].append(row)

    epoch_summary = {}
    for epoch, epoch_rows in sorted(epochs.items()):
        epoch_steps = [group[0] for group in step_rows if int(group[0]["epoch"]) == epoch]
        epoch_summary[str(epoch)] = {
            "microbatch_records": len(epoch_rows),
            "optimizer_steps": len(epoch_steps),
            "mean_forget_npo_loss": statistics.mean(float(row["forget_npo_loss"]) for row in epoch_rows),
            "mean_retain_token_nll": statistics.mean(float(row["retain_token_nll"]) for row in epoch_rows),
            "mean_combined_loss": statistics.mean(float(row["combined_loss"]) for row in epoch_rows),
            "mean_preclip_gradient_norm": statistics.mean(float(row["gradient_norm"]) for row in epoch_steps),
            "min_learning_rate": min(float(row["learning_rate"]) for row in epoch_steps),
            "max_learning_rate": max(float(row["learning_rate"]) for row in epoch_steps),
        }

    result = {
        "audit": "alpha40_training_dynamics_recomputation_v1",
        "source_archive_sha256": sha256(args.zip.read_bytes()),
        "training_source_sha256": archived_trainer_hash,
        "training_source_sha256_in_receipt": receipt["training_script_sha256"],
        "archived_training_source_matches_receipt": True,
        "current_local_training_source_sha256": local_trainer_hash,
        "current_local_training_source_matches_archive": archived_trainer_hash == local_trainer_hash,
        "source_identity_note": "The current shared trainer has changed since this archived run. The archived source hash is checked against the run receipt; current-source byte identity is reported for context only.",
        "retain_aggregation_audit": {
            "archived_behavior": "Compute a token-mean retain NLL per microbatch, then divide each microbatch contribution by gradient_accumulation_steps.",
            "gradient_accumulation_steps": accumulation_steps,
            "protocol_retain_loss_type": protocol_objective.get("retain_loss_type"),
            "protocol_aggregation_scope": protocol_scope,
            "protocol_aggregation_scope_explicit": protocol_scope is not None,
            "objective_fixture_examples": objective_fixture_examples,
            "objective_fixture_covers_full_accumulation_update": objective_fixture_examples > 0 and objective_fixture_examples >= accumulation_steps * int(receipt["training_schedule"]["retain_examples_per_microbatch"]),
            "full_update_pooled_token_mean_equivalence_verified": False,
            "interpretation": "The protocol names token_mean_nll but does not specify the aggregation scope. The recorded objective fixture covers two examples, while training accumulates four microbatches; equivalence to a token-pooled mean over the full optimizer update was not established.",
        },
        "trace_member": TRACE_MEMBER,
        "trace_sha256": sha256(trace_bytes),
        "zip_crc": "PASS",
        "trace_rows": len(rows),
        "optimizer_steps": len(steps),
        "records_per_optimizer_step": {"min": min(map(len, step_rows)), "max": max(map(len, step_rows))},
        "gradient_norm_semantics": "torch.nn.utils.clip_grad_norm_ return value, recorded once after accumulation and before clipping; repeated on each microbatch record for that optimizer step",
        "gradient_clip_cap": clip_cap,
        "optimizer_steps_above_clip_cap": sum(norm > clip_cap for norm in norms),
        "optimizer_steps_clipped_fraction": sum(norm > clip_cap for norm in norms) / len(norms),
        "preclip_gradient_norm": {
            "min": min(norms),
            "p50_nearest_rank": quantile(norms, 0.50),
            "p90_nearest_rank": quantile(norms, 0.90),
            "p95_nearest_rank": quantile(norms, 0.95),
            "max": max(norms),
            "median_clip_scale_approx": clip_cap / statistics.median(norms),
            "smallest_clip_scale_approx": clip_cap / max(norms),
        },
        "epochs": epoch_summary,
        "interpretation": "All optimizer updates exceeded the configured global-norm cap; the implementation applies clipping once after gradient accumulation. This confirms clipping was active as specified, while the large preclip norms are a sensitivity signal and do not by themselves prove instability or a code defect.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("zip_crc", "archived_training_source_matches_receipt", "current_local_training_source_matches_archive", "retain_aggregation_audit", "trace_rows", "optimizer_steps", "optimizer_steps_above_clip_cap", "preclip_gradient_norm")}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
