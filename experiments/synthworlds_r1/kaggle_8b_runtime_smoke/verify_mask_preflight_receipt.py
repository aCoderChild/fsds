#!/usr/bin/env python3
"""Verify a successful tokenizer/mask-only preflight receipt."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from model_profiles import MAIN_PROFILE, get_pinned_model_revision


def main() -> None:
    receipt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("r1_8b_mask_preflight.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_mask_sha = hashlib.sha256((SCRIPTS / "assistant_masking.py").read_bytes()).hexdigest()
    expected = {
        "status": "PASS_MASK_PREFLIGHT",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "scope": "TOKENIZER_AND_ASSISTANT_MASK_PREFLIGHT_ONLY",
        "model_revision": get_pinned_model_revision(MAIN_PROFILE),
        "assistant_mask_source_sha256": expected_mask_sha,
        "tokenizer_is_fast": True,
        "gpu_required": False,
        "weights_downloaded": False,
        "research_data_loaded": False,
        "training_started": False,
    }
    failures = [key for key, value in expected.items() if receipt.get(key) != value]
    for key in ("input_token_count", "assistant_supervised_token_count"):
        if not isinstance(receipt.get(key), int) or receipt[key] <= 0:
            failures.append(key)
    if receipt.get("assistant_supervised_token_count", 0) > receipt.get("input_token_count", 0):
        failures.append("assistant_supervised_token_count_exceeds_input_token_count")
    if failures:
        print(json.dumps({"status": "FAIL_MASK_PREFLIGHT_RECEIPT", "failed_fields": failures}, indent=2))
        raise SystemExit(1)
    print(json.dumps({
        "status": "VERIFIED_MASK_PREFLIGHT_ONLY",
        "model_id": receipt["model_id"],
        "model_revision": receipt["model_revision"],
        "input_token_count": receipt["input_token_count"],
        "assistant_supervised_token_count": receipt["assistant_supervised_token_count"],
        "weights_downloaded": False,
        "training_started": False,
        "warning": "This verifies tokenizer/mask compatibility only; it is not SFT or an experiment result.",
    }, indent=2))


if __name__ == "__main__":
    main()
