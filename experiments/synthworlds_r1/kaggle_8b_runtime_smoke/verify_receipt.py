#!/usr/bin/env python3
"""Verify a complete synthetic 8B runtime-smoke receipt; never an experiment result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from model_profiles import MAIN_PROFILE, get_pinned_model_revision
from validate_8b_runtime_smoke_receipt import validate_receipt


def main() -> None:
    receipt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("r1_8b_runtime_smoke.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    try:
        validate_receipt(
            receipt,
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            model_revision=get_pinned_model_revision(MAIN_PROFILE),
            config_sha256=receipt.get("config_sha256"),
        )
    except ValueError as exc:
        print(json.dumps({"status": "FAIL_RUNTIME_ONLY_RECEIPT", "error": str(exc)}, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps({
        "status": "VERIFIED_RUNTIME_ONLY_NOT_EXPERIMENT",
        "model_id": receipt["model_id"],
        "model_revision": receipt["model_revision"],
        "runner_sha256": receipt["runner_sha256"],
        "trainer_global_step": receipt["trainer_global_step"],
        "npo_optimizer_steps_completed": receipt["npo_optimizer_steps_completed"],
        "trainer_train_loss": receipt["trainer_train_loss"],
        "loss_mask_method": receipt["loss_mask_method"],
        "assistant_supervised_tokens": receipt["assistant_supervised_tokens"],
        "research_data_loaded": False,
        "warning": "Runtime compatibility check only; not SFT, unlearning, or a hypothesis result.",
    }, indent=2))


if __name__ == "__main__":
    main()
