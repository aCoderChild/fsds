#!/usr/bin/env python3
"""Embed the hash-bound 8B runtime smoke runner in a safe-by-default Kaggle notebook."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SMOKE = REPO / "experiments/synthworlds_r1/scripts/kaggle_8b_runtime_smoke.py"
ASSISTANT_MASKING = REPO / "experiments/synthworlds_r1/scripts/assistant_masking.py"
ACCESS_PREFLIGHT = REPO / "experiments/synthworlds_r1/scripts/kaggle_8b_access_preflight.py"
NPO_OBJECTIVE = REPO / "experiments/synthworlds_r1/scripts/unlearning_objectives.py"
MASK_PREFLIGHT = REPO / "experiments/synthworlds_r1/scripts/kaggle_8b_mask_preflight.py"
OUT = Path(__file__).resolve().parent
PUBLISH_DIR = OUT / "publish_v8"
sys.path.insert(0, str(REPO / "experiments/synthworlds_r1/scripts"))
from model_profiles import MAIN_PROFILE, get_pinned_model_revision

PINNED_REVISION = get_pinned_model_revision(MAIN_PROFILE)
if not PINNED_REVISION:
    raise RuntimeError("The 8B runtime smoke requires a frozen model revision.")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "id": "scope", "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "id": "smoke-runner", "outputs": [], "source": source.splitlines(keepends=True)}


def main() -> None:
    source = SMOKE.read_text(encoding="utf-8")
    digest = sha(source.encode("utf-8"))
    masking_source = ASSISTANT_MASKING.read_text(encoding="utf-8")
    masking_digest = sha(masking_source.encode("utf-8"))
    access_source = ACCESS_PREFLIGHT.read_text(encoding="utf-8")
    access_digest = sha(access_source.encode("utf-8"))
    objective_source = NPO_OBJECTIVE.read_text(encoding="utf-8")
    objective_digest = sha(objective_source.encode("utf-8"))
    mask_preflight_source = MASK_PREFLIGHT.read_text(encoding="utf-8")
    mask_preflight_digest = sha(mask_preflight_source.encode("utf-8"))
    install_and_run = f'''# First enable Internet + HF_TOKEN and run access preflight; it does not need a GPU.
# Run the tokenizer/mask check on CPU before enabling the GPU smoke.
RUN_ACCESS_PREFLIGHT = False
RUN_MASK_PREFLIGHT = False
RUN_SMOKE = False
selected_modes = sum([RUN_ACCESS_PREFLIGHT, RUN_MASK_PREFLIGHT, RUN_SMOKE])
if selected_modes > 1:
    raise ValueError("Run access preflight, mask preflight, and GPU smoke in separate executions")
elif selected_modes == 0:
    print("Safe setup only. Run RUN_ACCESS_PREFLIGHT=True first, RUN_MASK_PREFLIGHT=True on CPU second, then enable GPU and RUN_SMOKE=True only after both pass.")
elif RUN_ACCESS_PREFLIGHT:
    import hashlib, json
    from pathlib import Path

    # huggingface_hub is already a Transformers dependency in the Kaggle image.
    # Do not downgrade it here: that can break the preinstalled Transformers/Gradio stack.
    access_source = {access_source!r}
    expected_sha256 = "{access_digest}"
    actual_sha256 = hashlib.sha256(access_source.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Access preflight runner hash mismatch: {{actual_sha256}}")
    runner = {{"__name__": "r1_8b_hf_access_preflight"}}
    exec(compile(access_source, "kaggle_8b_access_preflight.py", "exec"), runner)
    exit_code = runner["main"]()
    receipt = json.loads(Path("/kaggle/working/r1_8b_hf_access_preflight.json").read_text())
    if exit_code != 0 or receipt.get("status") != "PASS_HF_ACCESS_PREFLIGHT":
        raise RuntimeError(f"Hugging Face access preflight did not pass: {{receipt}}")
    print(json.dumps(receipt, indent=2))
elif RUN_MASK_PREFLIGHT:
    import hashlib, json, os, subprocess, sys, types
    from pathlib import Path

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1"], check=True)
    masking_source = {masking_source!r}
    masking_expected_sha256 = "{masking_digest}"
    masking_actual_sha256 = hashlib.sha256(masking_source.encode("utf-8")).hexdigest()
    if masking_actual_sha256 != masking_expected_sha256:
        raise RuntimeError(f"Assistant-mask source hash mismatch: {{masking_actual_sha256}}")
    masking_module = types.ModuleType("assistant_masking")
    exec(compile(masking_source, "assistant_masking.py", "exec"), masking_module.__dict__)
    sys.modules["assistant_masking"] = masking_module
    os.environ["R1_ASSISTANT_MASK_SOURCE_SHA256"] = masking_expected_sha256
    os.environ["R1_PINNED_MODEL_REVISION"] = "{PINNED_REVISION}"
    mask_preflight_source = {mask_preflight_source!r}
    mask_preflight_expected_sha256 = "{mask_preflight_digest}"
    mask_preflight_actual_sha256 = hashlib.sha256(mask_preflight_source.encode("utf-8")).hexdigest()
    if mask_preflight_actual_sha256 != mask_preflight_expected_sha256:
        raise RuntimeError(f"Mask-preflight runner hash mismatch: {{mask_preflight_actual_sha256}}")
    runner = {{"__name__": "r1_8b_mask_preflight"}}
    exec(compile(mask_preflight_source, "kaggle_8b_mask_preflight.py", "exec"), runner)
    exit_code = runner["main"]()
    receipt = json.loads(Path("/kaggle/working/r1_8b_mask_preflight.json").read_text())
    if exit_code != 0 or receipt.get("status") != "PASS_MASK_PREFLIGHT":
        raise RuntimeError(f"Llama tokenizer/mask preflight did not pass: {{receipt}}")
    print(json.dumps(receipt, indent=2))
else:
    import hashlib, json, os, subprocess, sys, types
    from pathlib import Path

    packages = [
        "transformers==5.22.1",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "bitsandbytes==0.50.2",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    smoke_source = {source!r}
    expected_sha256 = "{digest}"
    actual_sha256 = hashlib.sha256(smoke_source.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Smoke runner hash mismatch: {{actual_sha256}}")
    os.environ["R1_SMOKE_SOURCE_SHA256"] = expected_sha256
    objective_source = {objective_source!r}
    objective_expected_sha256 = "{objective_digest}"
    objective_actual_sha256 = hashlib.sha256(objective_source.encode("utf-8")).hexdigest()
    if objective_actual_sha256 != objective_expected_sha256:
        raise RuntimeError(f"NPO objective hash mismatch: {{objective_actual_sha256}}")
    objective_module = types.ModuleType("unlearning_objectives")
    exec(compile(objective_source, "unlearning_objectives.py", "exec"), objective_module.__dict__)
    sys.modules["unlearning_objectives"] = objective_module
    masking_source = {masking_source!r}
    masking_expected_sha256 = "{masking_digest}"
    masking_actual_sha256 = hashlib.sha256(masking_source.encode("utf-8")).hexdigest()
    if masking_actual_sha256 != masking_expected_sha256:
        raise RuntimeError(f"Assistant-mask source hash mismatch: {{masking_actual_sha256}}")
    masking_module = types.ModuleType("assistant_masking")
    exec(compile(masking_source, "assistant_masking.py", "exec"), masking_module.__dict__)
    sys.modules["assistant_masking"] = masking_module
    runner = {{"__name__": "r1_8b_runtime_smoke"}}
    exec(compile(smoke_source, "kaggle_8b_runtime_smoke.py", "exec"), runner)
    exit_code = runner["main"]()
    receipt = json.loads(Path("/kaggle/working/r1_8b_runtime_smoke.json").read_text())
    if exit_code != 0 or receipt.get("status") != "PASS_RUNTIME_ONLY":
        raise RuntimeError(f"8B smoke did not pass: {{receipt}}")
    print(json.dumps(receipt, indent=2))
'''
    notebook = {
        "cells": [
            md("""# Llama 3.1 8B runtime smoke (not an experiment)\n\nThis notebook has three separate, guarded modes: gated-repository access; a CPU-only tokenizer and assistant-mask preflight; and a GPU-only synthetic runtime smoke. The mask preflight uses the exact pinned fast tokenizer and shared offset-based mask before any checkpoint weights are downloaded. The GPU smoke runs one synthetic LoRA optimizer step and one synthetic NPO update with policy/reference models on separate GPUs. It reads no research corpus, creates no research result, and does not start SFT/unlearning.\n\nEnable Internet and the existing `HF_TOKEN` secret under Add-ons → Secrets. First set `RUN_ACCESS_PREFLIGHT = True` and run the cell. After `PASS_HF_ACCESS_PREFLIGHT`, set `RUN_MASK_PREFLIGHT = True` and run with Accelerator set to None; this fetches tokenizer files only, no model weights. Only after `PASS_MASK_PREFLIGHT`, choose GPU T4 x2, set `RUN_SMOKE = True`, and run the bounded GPU check. The token is never printed or saved.\n\nThe notebook is pinned to the immutable `meta-llama/Llama-3.1-8B-Instruct` revision in `scripts/model_profiles.py`. Each mode writes a separate receipt. None is an experiment result.\n"""),
            code(install_and_run),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.13"},
            "kaggle": {"accelerator": "None", "dataSources": [], "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ipynb = OUT / "llama31_8b_runtime_smoke.ipynb"
    ipynb.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/synthworlds-llama31-8b-runtime-smoke-2026-11-04",
        "title": "SynthWorlds Llama31 8B Runtime Smoke 2026-11-04",
        "code_file": ipynb.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    published_notebook = PUBLISH_DIR / ipynb.name
    published_metadata = PUBLISH_DIR / "kernel-metadata.json"
    published_notebook.write_bytes(ipynb.read_bytes())
    published_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    publish_manifest = {
        "status": "VERIFIED_SAFE_BY_DEFAULT_SOURCE_PACKAGE",
        "kernel_id": metadata["id"],
        "notebook_sha256": sha(published_notebook.read_bytes()),
        "metadata_sha256": sha(published_metadata.read_bytes()),
        "runner_sha256": digest,
        "mask_preflight_sha256": mask_preflight_digest,
        "run_access_preflight_default": False,
        "run_mask_preflight_default": False,
        "run_smoke_default": False,
        "gpu_enabled": metadata["enable_gpu"],
        "internet_enabled": metadata["enable_internet"],
        "attached_datasets": metadata["dataset_sources"],
        "research_data_loaded": False,
        "training_started": False,
    }
    (PUBLISH_DIR / "publish_manifest.json").write_text(json.dumps(publish_manifest, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "INTERACTIVE_RUNTIME_SMOKE_NOTEBOOK_PREPARED_SAFE_BY_DEFAULT",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision_pin": PINNED_REVISION,
        "runner_sha256": digest,
        "assistant_masking_sha256": masking_digest,
        "mask_preflight_sha256": mask_preflight_digest,
        "access_preflight_sha256": access_digest,
        "npo_objective_sha256": objective_digest,
        "notebook_sha256": sha(ipynb.read_bytes()),
        "enable_gpu_at_publish": False,
        "enable_internet_at_publish": False,
        "research_data_loaded": False,
        "run_training": False,
        "requires_user_interactive_session": True,
    }
    (OUT / "notebook_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
