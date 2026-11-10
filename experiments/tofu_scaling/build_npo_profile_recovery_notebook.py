#!/usr/bin/env python3
"""Build the private Kaggle notebook for the locked profile-cohort follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "kaggle" / "npo_profile_recovery"
OUT = OUT_DIR / "tofu_3b_npo_profile_recovery.ipynb"
KERNEL_ID = "bunnoob2005/tofu-3b-npo-lora-profile-recovery-2026-11-07"
UPSTREAM_COMMIT = "2d98027677f3468d52b6baa3fc4f2b9f2dc08820"


def markdown(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args
    sources = {
        "tofu_official_eval.py": ROOT / "tofu_official_eval.py",
        "npo_objective.py": ROOT / "npo_objective.py",
        "verify_npo_objective.py": ROOT / "verify_npo_objective.py",
        "npo_lora_train.py": ROOT / "npo_lora_train.py",
        "validate_npo_recovery_protocol.py": ROOT / "validate_npo_recovery_protocol.py",
        "evaluate_npo_profile_cohort.py": ROOT / "evaluate_npo_profile_cohort.py",
        "run_npo_profile_recovery.py": ROOT / "run_npo_profile_recovery.py",
        "npo_lora_recovery_protocol.json": ROOT / "npo_lora_recovery_protocol.json",
        "requirements-npo-pilot.txt": ROOT / "requirements-npo-pilot.txt",
    }
    contents = {name: path.read_text(encoding="utf-8") for name, path in sources.items()}
    cells = [
        markdown(
            "# TOFU 3B NPO-LoRA profile-cohort recovery diagnostic\n\n"
            "This is a post-hoc exploratory follow-up to the previous 3B one-seed run, whose four utility gates failed. It does not change that result. The locked protocol selects the smallest retain-NLL weight in `[5, 10, 20]` that passes every development gate, then evaluates one adapter on a separate final profile cohort. The run stops without final training if no development candidate passes.\n\n"
            "The dataset audit verifies pinned source hashes, contiguous profile assignments, and exclusion of the two conflicting-question profiles before model weights load. Metrics are TOFU answer probability, greedy-generation ROUGE-L recall, and forget Truth Ratio. This profile subset is not a standard full-split leaderboard evaluation and a one-seed result is not confirmatory.\n\n"
            "Select Kaggle GPU T4 x2 and enable Internet before running all cells. The notebook uses public pinned Open-Unlearning model weights; it needs no Hugging Face token. Full training is enabled because the purpose is to execute the preregistered diagnostic.",
            "readme",
        ),
        code(
            "import importlib.metadata as metadata\n"
            "import json, os, subprocess, sys, urllib.request\n"
            "from pathlib import Path\n"
            "print({'notebook': 'profile-cohort recovery', 'protocol': 'locked', 'training_will_start_after_gates': True})\n",
            "runtime-init",
        ),
        code(
            "requirements = Path('/kaggle/working/requirements-npo-pilot.txt')\n"
            f"requirements.write_text({contents['requirements-npo-pilot.txt']!r}, encoding='utf-8')\n"
            "expected = {'transformers': '5.22.1', 'accelerate': '1.15.0', 'peft': '0.21.2', 'scipy': '1.16.3', 'rouge-score': '0.1.2'}\n"
            "def installed_version(package):\n"
            "    try: return metadata.version(package)\n"
            "    except metadata.PackageNotFoundError: return None\n"
            "if any(installed_version(package) != version for package, version in expected.items()):\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '-r', str(requirements)], check=True)\n"
            "versions = {package: metadata.version(package) for package in expected}\n"
            "if versions != expected: raise RuntimeError(f'Pinned dependency verification failed: {versions}')\n"
            "try:\n"
            "    torchao_version = metadata.version('torchao')\n"
            "except metadata.PackageNotFoundError:\n"
            "    torchao_version = None\n"
            "if torchao_version is not None:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '--yes', 'torchao'], check=True, capture_output=True, text=True)\n"
            "    try: metadata.version('torchao'); raise RuntimeError('torchao remains installed')\n"
            "    except metadata.PackageNotFoundError: pass\n"
            "print({'packages': versions, 'torch': metadata.version('torch'), 'torchao': 'absent'})\n",
            "dependencies",
        ),
        code(
            "import torch\n"
            "if not torch.cuda.is_available() or torch.cuda.device_count() != 2:\n"
            "    raise RuntimeError('Select Kaggle GPU T4 x2. No model weights are loaded before this check.')\n"
            "gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]\n"
            "if gpu_names.count('Tesla T4') != 2: raise RuntimeError(f'Expected T4 x2, found {gpu_names}')\n"
            "with urllib.request.urlopen('https://huggingface.co', timeout=20) as response: internet_status = response.status\n"
            "if internet_status != 200: raise RuntimeError(f'Hugging Face connectivity failed: HTTP {internet_status}')\n"
            "print({'gpu_names': gpu_names, 'gpu_memory_bytes': [torch.cuda.get_device_properties(i).total_memory for i in range(2)], 'huggingface_http': internet_status})\n",
            "gpu-and-internet-gate",
        ),
        code(
            "upstream = Path('/kaggle/working/open-unlearning')\n"
            f"commit = {UPSTREAM_COMMIT!r}\n"
            "if not upstream.exists(): subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/locuslab/open-unlearning.git', str(upstream)], check=True)\n"
            "subprocess.run(['git', '-C', str(upstream), 'fetch', '--depth', '1', 'origin', commit], check=True)\n"
            "subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', commit], check=True)\n"
            "actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()\n"
            "if actual != commit: raise RuntimeError(f'Open-Unlearning commit mismatch: {actual}')\n"
            "print({'open_unlearning_commit': actual})\n",
            "pin-upstream",
        ),
    ]
    for index, (name, content) in enumerate(contents.items(), start=1):
        cells.append(code(f"%%writefile /kaggle/working/{name}\n" + content, f"write-source-{index:02d}"))
    cells.append(code(
        "command = [sys.executable, '/kaggle/working/run_npo_profile_recovery.py']\n"
        "subprocess.run(command, check=True)\n"
        "artifact = Path('/kaggle/working/tofu_npo_profile_recovery_artifacts.zip')\n"
        "if not artifact.is_file() or artifact.stat().st_size == 0: raise RuntimeError('Recovery result archive is missing')\n"
        "import hashlib\n"
        "print({'artifact_zip': str(artifact), 'bytes': artifact.stat().st_size, 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()})\n",
        "run-preregistered-recovery",
    ))
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
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        "title": "TOFU 3B NPO LoRA Profile Recovery 2026-11-07",
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {OUT}")


if __name__ == "__main__":
    main()
