#!/usr/bin/env python3
"""Validate the standalone synthetic 8B smoke receipt before R1 model stages."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path


EXPECTED_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EXPECTED_PACKAGES = {
    "transformers": "5.22.1",
    "peft": "0.21.2",
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
}


def validate_receipt(
    receipt: dict,
    *,
    model_id: str,
    model_revision: str,
    config_sha256: str,
) -> None:
    """Raise ValueError unless the receipt proves the exact one-step NF4 smoke."""
    runner_path = Path(__file__).with_name("kaggle_8b_runtime_smoke.py")
    expected_runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    checks = {
        "runner_source": receipt.get("runner_sha256") == expected_runner_sha256,
        "status": receipt.get("status") == "PASS_RUNTIME_ONLY",
        "model": model_id == EXPECTED_MODEL and receipt.get("model_id") == model_id,
        "scope": receipt.get("scope") == "SYNTHETIC_RUNTIME_SMOKE_ONLY",
        "revision": receipt.get("model_revision") == model_revision,
        "config": receipt.get("config_sha256") == config_sha256
        and isinstance(config_sha256, str)
        and len(config_sha256) == 64
        and all(char in "0123456789abcdef" for char in config_sha256)
        and receipt.get("config_model_type") == "llama",
        "packages": receipt.get("package_versions") == EXPECTED_PACKAGES,
        "nf4": receipt.get("model_loaded_4bit") is True,
        "one_step": receipt.get("optimizer_steps_requested") == 1
        and receipt.get("trainer_global_step") == 1,
        "assistant_only_mask": receipt.get("loss_mask_method") == "single_render_fast_tokenizer_offsets"
        and isinstance(receipt.get("assistant_supervised_tokens"), int)
        and receipt.get("assistant_supervised_tokens", 0) > 0,
        "npo_reference": receipt.get("reference_model_loaded_4bit") is True
        and receipt.get("npo_reference_cuda_device") == 1,
        "npo_step": receipt.get("npo_optimizer_steps_requested") == 1
        and receipt.get("npo_optimizer_steps_completed") == 1,
        "npo_beta": isinstance(receipt.get("npo_beta"), (int, float))
        and math.isfinite(receipt["npo_beta"])
        and receipt["npo_beta"] > 0,
        "finite_npo_loss": isinstance(receipt.get("npo_loss"), (int, float))
        and math.isfinite(receipt["npo_loss"]),
        "synthetic_only": receipt.get("research_data_loaded") is False,
        "two_gpus": isinstance(receipt.get("gpus"), list) and len(receipt["gpus"]) >= 2,
        "finite_train_loss": isinstance(receipt.get("trainer_train_loss"), (int, float))
        and math.isfinite(receipt["trainer_train_loss"]),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("8B runtime-smoke receipt failed checks: " + ", ".join(failures))
