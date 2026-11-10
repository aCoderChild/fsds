#!/usr/bin/env python3
"""Build the Kaggle notebook for the preregistered alpha=40 TOFU follow-up."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "kaggle" / "npo_alpha40_followup"
NOTEBOOK = OUT_DIR / "tofu_3b_npo_alpha40_followup.ipynb"
KERNEL_ID = "bunnoob2005/tofu-3b-npo-lora-alpha-40-follow-up"
REFERENCE_DATASET = "bunnoob2005/tofu-3b-profile-alpha40-reference-2026-11-08"
PROTOCOL = ROOT / "npo_lora_alpha40_followup_protocol.json"

SOURCES = (
    "tofu_official_eval.py",
    "npo_objective.py",
    "verify_npo_objective.py",
    "npo_lora_train.py",
    "validate_npo_recovery_protocol.py",
    "evaluate_npo_profile_cohort.py",
    "verify_npo_profile_recovery_run.py",
    "run_npo_profile_recovery.py",
    "requirements-npo-pilot.txt",
)


def md(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def py(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    runtime_header = repr({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "training_candidate_alpha": protocol["selection"]["candidate_retain_nll_weights_ascending"],
    })
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in SOURCES}
    cells = [
        md(
            "# TOFU Llama 3.2 3B: alpha-40 follow-up\n\n"
            "This is a **post-hoc exploratory** follow-up to the verified alpha=5/10/20 development study. Alpha=40 was chosen after observing that ROUGE-L improved as the retain-NLL weight increased; this is not confirmatory evidence and does not alter the previous result.\n\n"
            "The run executes exactly one development candidate (alpha=40) with the same seed, 250-step training procedure, profile assignments, outcomes, bootstrap, and five gates. The frozen-checkpoint reference evaluation is reused only after artifact, checkpoint, data-hash, row-coverage, and runtime checks. If any development gate fails, final-cohort training is skipped. If all pass, alpha=40 is trained/evaluated once on the disjoint final profile blocks.\n\n"
            f"Protocol SHA-256: `{protocol_sha}`. Keep Internet enabled and select Kaggle GPU T4 x2. No Hugging Face secret is needed.",
            "study-description",
        ),
        py(
            "import importlib.metadata as metadata, json, shutil, subprocess, sys, urllib.request\n"
            "from pathlib import Path\n"
            f"print({runtime_header})\n",
            "init",
        ),
        py(
            "requirements = Path('/kaggle/working/requirements-npo-pilot.txt')\n"
            f"requirements.write_text({sources['requirements-npo-pilot.txt']!r}, encoding='utf-8')\n"
            "expected = {'transformers': '5.22.1', 'accelerate': '1.15.0', 'peft': '0.21.2', 'scipy': '1.16.3', 'rouge-score': '0.1.2'}\n"
            "def installed_version(package):\n"
            "    try: return metadata.version(package)\n"
            "    except metadata.PackageNotFoundError: return None\n"
            "if any(installed_version(name) != version for name, version in expected.items()):\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '-r', str(requirements)], check=True)\n"
            "versions = {name: metadata.version(name) for name in expected}\n"
            "if versions != expected: raise RuntimeError(f'Pinned dependency mismatch: {versions}')\n"
            "try: torchao_version = metadata.version('torchao')\n"
            "except metadata.PackageNotFoundError: torchao_version = None\n"
            "if torchao_version is not None:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '--yes', 'torchao'], check=True, capture_output=True, text=True)\n"
            "    try: metadata.version('torchao'); raise RuntimeError('torchao remains installed')\n"
            "    except metadata.PackageNotFoundError: pass\n"
            "print({'packages': versions, 'torch': metadata.version('torch'), 'torchao': 'absent'})\n",
            "dependencies",
        ),
        py(
            "import torch\n"
            "if not torch.cuda.is_available() or torch.cuda.device_count() != 2: raise RuntimeError('Select Kaggle GPU T4 x2 before proceeding')\n"
            "gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]\n"
            "if gpu_names.count('Tesla T4') != 2: raise RuntimeError(f'Expected exactly two Tesla T4 GPUs; found {gpu_names}')\n"
            "with urllib.request.urlopen('https://huggingface.co', timeout=20) as response: hf_http = response.status\n"
            "if hf_http != 200: raise RuntimeError(f'Hugging Face connectivity failed: HTTP {hf_http}')\n"
            "print({'gpu_names': gpu_names, 'memory_bytes': [torch.cuda.get_device_properties(i).total_memory for i in range(2)], 'huggingface_http': hf_http})\n",
            "gpu-internet-gate",
        ),
        py(
            "upstream = Path('/kaggle/working/open-unlearning')\n"
            f"commit = {protocol['objective']['source'].split('@', 1)[1]!r}\n"
            "if not upstream.exists(): subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/locuslab/open-unlearning.git', str(upstream)], check=True)\n"
            "subprocess.run(['git', '-C', str(upstream), 'fetch', '--depth', '1', 'origin', commit], check=True)\n"
            "subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', commit], check=True)\n"
            "actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()\n"
            "if actual != commit: raise RuntimeError(f'Upstream commit mismatch: {actual}')\n"
            "print({'open_unlearning_commit': actual})\n",
            "pin-upstream",
        ),
    ]
    for index, (name, content) in enumerate(sources.items(), 1):
        cells.append(py(f"%%writefile /kaggle/working/{name}\n" + content, f"write-source-{index:02d}"))
    cells.append(py(f"%%writefile /kaggle/working/npo_lora_alpha40_followup_protocol.json\n" + PROTOCOL.read_text(encoding="utf-8"), "write-protocol"))
    cells.extend([
        py(
            "support_manifests = list(Path('/kaggle/input').rglob('reference_support_manifest.json'))\n"
            "if len(support_manifests) != 1: raise RuntimeError(f'Expected one attached verified reference-support dataset, found {support_manifests}')\n"
            "support_dir = support_manifests[0].parent\n"
            "protocol_path = Path('/kaggle/working/npo_lora_alpha40_followup_protocol.json')\n"
            "objective_path = Path('/kaggle/working/npo_alpha40_objective_equivalence.json')\n"
            "run_root = Path('/kaggle/working/tofu_npo_alpha40_followup')\n"
            "artifact_zip = Path('/kaggle/working/tofu_npo_alpha40_followup_artifacts.zip')\n"
            "command = [sys.executable, '/kaggle/working/run_npo_profile_recovery.py', '--root', str(run_root), '--protocol', str(protocol_path), '--upstream-src', str(upstream / 'src'), '--objective-check', str(objective_path), '--artifact-zip', str(artifact_zip), '--reference-support-dir', str(support_dir)]\n"
            "try:\n"
            "    subprocess.run(command, check=True)\n"
            "finally:\n"
            "    cache = Path('/kaggle/working/model_cache')\n"
            "    if cache.exists(): shutil.rmtree(cache)\n"
            "if not artifact_zip.is_file() or artifact_zip.stat().st_size == 0: raise RuntimeError('Experiment output archive is missing')\n"
            "print({'run_root': str(run_root), 'artifact_zip': str(artifact_zip), 'bytes': artifact_zip.stat().st_size})\n",
            "run-alpha40",
        ),
        py(
            "verification = Path('/kaggle/working/alpha40_independent_verification.json')\n"
            "command = [sys.executable, '/kaggle/working/verify_npo_profile_recovery_run.py', '--run-root', '/kaggle/working/tofu_npo_alpha40_followup', '--protocol', '/kaggle/working/npo_lora_alpha40_followup_protocol.json', '--artifact-zip', '/kaggle/working/tofu_npo_alpha40_followup_artifacts.zip', '--output', str(verification)]\n"
            "subprocess.run(command, check=True)\n"
            "print(json.loads(verification.read_text()))\n",
            "independent-verification",
        ),
    ])
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": "TOFU 3B NPO-LoRA Alpha 40 Follow-up",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "GPU T4 x2",
        "dataset_sources": [REFERENCE_DATASET],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}; protocol_sha256={protocol_sha}")


if __name__ == "__main__":
    main()
