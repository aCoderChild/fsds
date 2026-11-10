#!/usr/bin/env python3
"""Verify Kaggle HF secret and gated Llama config access without GPU/weights."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
OUTPUT = Path("/kaggle/working/r1_8b_access_probe.json")


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    from kaggle_secrets import UserSecretsClient

    value = UserSecretsClient().get_secret("HF_TOKEN")
    if not value:
        raise RuntimeError("HF_TOKEN secret is empty")
    return value


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "STARTED",
        "model_id": MODEL_ID,
        "expected_revision": REVISION,
        "scope": "SECRET_AND_CONFIG_ACCESS_ONLY",
        "gpu_required": False,
        "weights_downloaded": False,
        "research_data_loaded": False,
        "training_started": False,
    }
    try:
        from huggingface_hub import HfApi, hf_hub_download

        token = get_token()
        info = HfApi(token=token).model_info(MODEL_ID, revision=REVISION, token=token)
        if info.sha != REVISION:
            raise RuntimeError(f"Resolved revision mismatch: {info.sha}")
        config_path = Path(
            hf_hub_download(
                repo_id=MODEL_ID,
                filename="config.json",
                revision=REVISION,
                token=token,
            )
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("model_type") != "llama":
            raise RuntimeError("Downloaded config is not a Llama model config")
        receipt.update(
            {
                "status": "PASS_ACCESS_ONLY",
                "resolved_revision": info.sha,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "config_model_type": config.get("model_type"),
                "token_value_logged": False,
            }
        )
    except Exception as exc:
        receipt.update(
            {
                "status": "FAIL_ACCESS_ONLY",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "token_value_logged": False,
            }
        )
        OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        raise
    OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
