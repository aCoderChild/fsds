#!/usr/bin/env python3
"""Build a pinned Kaggle notebook from the locked baseline runner."""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KAGGLE_DIR = ROOT / "kaggle"
PROTOCOL = ROOT / "protocol.json"
RUNNER = KAGGLE_DIR / "baseline_runner.py"
NOTEBOOK = KAGGLE_DIR / "muse_news_target_baseline.ipynb"
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def markdown(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    setup = f'''RUN_ACCESS_PREFLIGHT = False  # Set True to verify HF_TOKEN access; downloads only the small tokenizer files.
RUN_BASELINE = False  # Set True only after access preflight passes and GPU/Internet are enabled.

if RUN_ACCESS_PREFLIGHT:
    import hashlib, json
    from pathlib import Path
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import HfApi, hf_hub_download

    token = UserSecretsClient().get_secret("HF_TOKEN")
    if not token or not token.strip():
        raise RuntimeError("HF_TOKEN secret is unavailable or empty; no model weights were requested.")
    repo = "meta-llama/Llama-2-7b-hf"
    revision = "01c7f73d771dfac7d292323805ebc428287df4f9"
    expected = {{
        "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
        "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
        "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
        "tokenizer_config.json": "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a",
    }}
    info = HfApi(token=token.strip()).model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Official tokenizer revision mismatch: {{info.sha}}")
    local_dir = Path("/kaggle/working/muse_tokenizer_access_preflight")
    local_dir.mkdir(parents=True, exist_ok=True)
    observed = {{}}
    for name, expected_hash in expected.items():
        path = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision, token=token.strip(), local_dir=str(local_dir)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"Tokenizer hash mismatch for {{name}}")
        observed[name] = digest
    receipt = {{
        "status": "PASS_TOKENIZER_ACCESS_ONLY",
        "repo": repo,
        "revision": revision,
        "files_sha256": observed,
        "model_weights_downloaded": False,
        "gpu_used": False,
        "secret_value_recorded": False,
    }}
    Path("/kaggle/working/muse_tokenizer_access_preflight.json").write_text(json.dumps(receipt, indent=2) + "\\n")
    print(json.dumps(receipt, indent=2))
else:
    print("ACCESS_PREFLIGHT_SKIPPED")

if RUN_BASELINE:
    from pathlib import Path
    import hashlib, json, subprocess, sys

    PROTOCOL = {json.dumps(PROTOCOL.read_text(encoding="utf-8"))}
    PROTOCOL_SHA256 = "{PROTOCOL_SHA256}"
    protocol_bytes = PROTOCOL.encode("utf-8")
    assert hashlib.sha256(protocol_bytes).hexdigest() == PROTOCOL_SHA256
    Path("/kaggle/working/protocol_source.json").write_bytes(protocol_bytes)
    Path("/kaggle/working/protocol.json").write_bytes(protocol_bytes)

    # Keep Kaggle's CUDA-enabled torch build. Pin the inference/evaluation stack.
    packages = [
        "transformers==5.22.1",
        "accelerate==1.15.0",
        "scipy==1.16.3",
        "rouge-score==0.1.2",
        "sentencepiece==0.2.0",
        "pyarrow==19.0.1",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)
    print("SETUP_PASS; protocol_sha256=" + PROTOCOL_SHA256)
else:
    print("STAGING_ONLY: no package install, network access, model access, or GPU work. Enable the run flag only after manual Kaggle setup.")
'''
    runner = RUNNER.read_text(encoding="utf-8").replace("__PROTOCOL_SHA256__", PROTOCOL_SHA256)
    runner = runner.replace("from __future__ import annotations\n\n", "", 1)
    guarded_runner = "if RUN_BASELINE:\n" + textwrap.indent(runner, "    ") + "\nelse:\n    print(\"STAGING_ONLY: runner skipped.\")\n"
    cells = [
        markdown(
            "# MUSE-News target baseline (inference only)\n\n"
            "This notebook evaluates the official MUSE-News target checkpoint before any unlearning. It does not train, tune, or modify model weights.\n\n"
            "**Safe staging default:** both run flags are `False`. First, attach `HF_TOKEN` under Add-ons → Secrets and set `RUN_ACCESS_PREFLIGHT = True`; this checks access to the exact official tokenizer revision and its pinned file hashes without GPU use or model-weight downloads. Only after it passes, choose GPU T4 x2, enable Internet, set `RUN_ACCESS_PREFLIGHT = False` and `RUN_BASELINE = True`, and run all cells. The notebook downloads the tokenizer only from Meta's official gated repository; it does not use the public tokenizer mirror.\n\n"
            "It first validates pinned data/revisions, then checks exact parity with the pinned official MUSE evaluator on two rows, and only then evaluates all 300 rows. Outputs contain predictions and per-row scores but no model weights or raw question/answer text. Bootstrap intervals resample rows; they are not source-cluster robust. The target model's Hub card has no license tag, so this package is evaluation-only and does not redistribute it.\n\n"
            f"Locked protocol SHA-256: `{PROTOCOL_SHA256}`.\n"
        , "instructions"),
        code(setup, "run-controls-and-access-preflight"),
        code(guarded_runner, "baseline-evaluation"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": None,
            "kaggle": {"accelerator": "None", "isGpuEnabled": False, "isInternetEnabled": False},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/muse-news-target-baseline-2026-11-09",
        "title": "MUSE-News Target Baseline 2026-11-09",
        "code_file": "muse_news_target_baseline.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (KAGGLE_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {NOTEBOOK}")
    print(f"PROTOCOL_SHA256 {PROTOCOL_SHA256}")


if __name__ == "__main__":
    main()
