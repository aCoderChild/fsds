#!/usr/bin/env python3
"""Check Kaggle HF_TOKEN binding and gated 8B repo access without loading weights."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def get_kaggle_token() -> str:
    """Read the notebook-attached Kaggle secret; never print or persist it."""
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is empty or unavailable to this notebook")
    return token


def main() -> int:
    output = Path(os.environ.get("R1_ACCESS_PREFLIGHT_OUTPUT", "/kaggle/working/r1_8b_hf_access_preflight.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict = {
        "status": "STARTED",
        "model_id": MODEL_ID,
        "scope": "HF_ACCESS_PREFLIGHT_ONLY",
        "gpu_required": False,
        "weights_downloaded": False,
        "research_data_loaded": False,
        "training_started": False,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    stage = "retrieve_kaggle_secret"
    try:
        token = get_kaggle_token()
        stage = "resolve_gated_repo_metadata"
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi(token=token).model_info(MODEL_ID, token=token)
        revision = info.sha
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise RuntimeError("Hugging Face did not return an immutable 40-character commit")

        stage = "download_and_validate_small_config"
        config_path = Path(
            hf_hub_download(
                repo_id=MODEL_ID,
                filename="config.json",
                revision=revision,
                token=token,
            )
        )
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if not isinstance(config, dict) or config.get("model_type") != "llama":
            raise RuntimeError("Pinned repository config is not a Llama config")
        if config.get("num_hidden_layers") != 32 or config.get("hidden_size") != 4096:
            raise RuntimeError("Pinned config does not match the expected Llama 3.1 8B architecture")

        receipt.update(
            {
                "status": "PASS_HF_ACCESS_PREFLIGHT",
                "model_revision": revision,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "config_model_type": config["model_type"],
                "config_num_hidden_layers": config["num_hidden_layers"],
                "config_hidden_size": config["hidden_size"],
                "interpretation": "This Kaggle notebook retrieved HF_TOKEN and accessed the gated model metadata/config. No model weights or research data were downloaded.",
            }
        )
    except Exception as exc:
        # Keep receipts safe: do not store exception messages that might contain credentials.
        receipt.update(
            {
                "status": "FAIL_HF_ACCESS_PREFLIGHT",
                "failed_stage": stage,
                "error_type": type(exc).__name__,
                "interpretation": "No training or model-weight download was attempted.",
            }
        )
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        raise

    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
