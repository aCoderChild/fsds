#!/usr/bin/env python3
"""Validate the pinned Llama tokenizer and assistant mask without GPU/weights."""
from __future__ import annotations

import hashlib
import json
import os
import re
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from assistant_masking import encode_assistant_only


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PINNED_REVISION = os.environ.get("R1_PINNED_MODEL_REVISION", "")


def get_token() -> str:
    """Read the Kaggle-attached secret without logging or persisting it."""
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is empty or unavailable to this notebook")
    return token


def main() -> int:
    output = Path(os.environ.get("R1_MASK_PREFLIGHT_OUTPUT", "/kaggle/working/r1_8b_mask_preflight.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict = {
        "status": "STARTED",
        "model_id": MODEL_ID,
        "scope": "TOKENIZER_AND_ASSISTANT_MASK_PREFLIGHT_ONLY",
        "gpu_required": False,
        "weights_downloaded": False,
        "research_data_loaded": False,
        "training_started": False,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    stage = "retrieve_kaggle_secret"
    try:
        if re.fullmatch(r"[0-9a-f]{40}", PINNED_REVISION) is None:
            raise RuntimeError("The notebook did not provide a frozen 40-character model revision")
        token = get_token()
        stage = "verify_pinned_repo_revision"
        info = HfApi(token=token).model_info(
            MODEL_ID, revision=PINNED_REVISION, token=token
        )
        revision = info.sha
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise RuntimeError("Hugging Face did not resolve an immutable model revision")
        if revision != PINNED_REVISION:
            raise RuntimeError("Resolved model commit differs from the frozen R1 revision")

        stage = "load_pinned_fast_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, token=token, revision=revision, use_fast=True
        )
        if not tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for offset-based assistant-only masking")

        stage = "validate_synthetic_assistant_mask"
        messages = [
            {"role": "user", "content": "For this software check only, reply with exactly OK."},
            {"role": "assistant", "content": "OK"},
        ]
        input_ids, labels = encode_assistant_only(tokenizer, messages)
        supervised = [i for i, label in enumerate(labels) if label != -100]
        if len(input_ids) != len(labels) or not supervised:
            raise RuntimeError("The assistant-only mask is empty or misaligned")

        mask_sha = os.environ.get("R1_ASSISTANT_MASK_SOURCE_SHA256", "")
        if re.fullmatch(r"[0-9a-f]{64}", mask_sha) is None:
            raise RuntimeError("The notebook did not provide the verified assistant-mask source hash")
        receipt.update({
            "status": "PASS_MASK_PREFLIGHT",
            "model_revision": revision,
            "transformers_version": version("transformers"),
            "tokenizer_is_fast": bool(tokenizer.is_fast),
            "tokenizer_vocab_size": int(tokenizer.vocab_size),
            "tokenizer_json_sha256": hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode("utf-8")
            ).hexdigest(),
            "assistant_mask_source_sha256": mask_sha,
            "loss_mask_method": "single_render_fast_tokenizer_offsets",
            "input_token_count": len(input_ids),
            "assistant_supervised_token_count": len(supervised),
            "first_supervised_token_index": supervised[0],
            "interpretation": "The pinned fast tokenizer and shared assistant-only offset mask passed on a synthetic prompt. No model weights or research data were loaded; no training was started.",
        })
    except Exception as exc:
        # Avoid storing exception strings that could include request details.
        receipt.update({
            "status": "FAIL_MASK_PREFLIGHT",
            "failed_stage": stage,
            "error_type": type(exc).__name__,
        })
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        raise

    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
