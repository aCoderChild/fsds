#!/usr/bin/env python3
"""Build the private Kaggle notebook for the locked IDK+AP LoRA diagnostic."""

from __future__ import annotations

import hashlib
import json
import base64
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "kaggle" / "idk_ap_lora_pilot"
NOTEBOOK = OUT_DIR / "tofu_3b_idk_ap_lora_diagnostic.ipynb"
KERNEL_ID = "bunnoob2005/tofu-3b-idk-ap-lora-diagnostic-2026-11-09"
REFERENCE_DATASET = "bunnoob2005/tofu-3b-profile-alpha40-reference-2026-11-08"
PROTOCOL = HERE / "idk_ap_lora_protocol.json"
PROTOCOL_SHA256 = "650af06eed1d254cfdd2b47714299686925cea47b5b2675edb7c07674b01c049"

SOURCES = {
    "tofu_official_eval.py": ROOT / "tofu_official_eval.py",
    "evaluate_npo_profile_cohort.py": ROOT / "evaluate_npo_profile_cohort.py",
    "validate_protocol.py": HERE / "validate_protocol.py",
    "verify_objective.py": HERE / "verify_objective.py",
    "train.py": HERE / "train.py",
    "evaluate.py": HERE / "evaluate.py",
    "run.py": HERE / "run.py",
    "verify_run.py": HERE / "verify_run.py",
    "requirements.txt": HERE / "requirements.txt",
    "idontknow.jsonl": HERE / "idontknow.jsonl",
    "idk_ap_lora_protocol.json": PROTOCOL,
}


def markdown(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def main() -> None:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("Protocol changed after preregistration; make a new protocol version instead")
    sources = {name: path.read_text(encoding="utf-8") if path.suffix != ".jsonl" else path.read_text(encoding="utf-8") for name, path in SOURCES.items()}
    cells = [
        markdown(
            "# TOFU 3B IDK+AP LoRA diagnostic\n\n"
            "One-seed exploratory method diagnostic on the locked TOFU development profile blocks. It is a resource-adapted LoRA implementation of the published IDK+AP objective, not an exact reproduction of the paper's full-parameter A100 experiment.\n\n"
            "The run uses the existing fixed evaluation margins and profile-block bootstrap. It performs no parameter search. It trains the final profile blocks only if every development gate passes. The IDK marker and exact-string rates are descriptive string screens, not semantic refusal judgments.\n\n"
            f"Protocol SHA-256: `{PROTOCOL_SHA256}`. Use GPU T4 x2 with Internet enabled. No Hugging Face secret is required for the public Open-Unlearning checkpoint.",
            "study-summary",
        ),
    ]
    cells.append(code("from pathlib import Path\nPath('/kaggle/working/idk_ap_lora').mkdir(parents=True, exist_ok=True)\n", "prepare-source-directory"))
    for index, (name, content) in enumerate(sources.items(), start=1):
        target = f"/kaggle/working/{name}" if name in {"tofu_official_eval.py", "evaluate_npo_profile_cohort.py"} else f"/kaggle/working/idk_ap_lora/{name}"
        if name == "idontknow.jsonl":
            exact_bytes_b64 = base64.b64encode((HERE / name).read_bytes()).decode("ascii")
            cells.append(code(
                f"from pathlib import Path\nimport base64\nPath({target!r}).write_bytes(base64.b64decode({exact_bytes_b64!r}))\n",
                f"write-source-{index:02d}",
            ))
        else:
            cells.append(code(f"%%writefile {target}\n" + content, f"write-source-{index:02d}"))
    cells.append(code(f"%%writefile /kaggle/working/idk_ap_lora_protocol.json\n" + sources["idk_ap_lora_protocol.json"], "write-protocol"))
    cells.append(
        code(
            "import importlib.metadata as metadata, subprocess, sys, urllib.request\n"
            "EXPECTED = {'transformers': '5.22.1', 'accelerate': '1.15.0', 'peft': '0.21.2', 'scipy': '1.16.3', 'rouge-score': '0.1.2'}\n"
            "def version(name):\n"
            "    try: return metadata.version(name)\n"
            "    except metadata.PackageNotFoundError: return None\n"
            "if any(version(name) != value for name, value in EXPECTED.items()):\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '-r', '/kaggle/working/idk_ap_lora/requirements.txt'], check=True)\n"
            "actual = {name: metadata.version(name) for name in EXPECTED}\n"
            "if actual != EXPECTED: raise RuntimeError(f'Pinned dependency mismatch: {actual}')\n"
            "if version('torchao') is not None:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '--yes', 'torchao'], check=True, capture_output=True, text=True)\n"
            "    if version('torchao') is not None: raise RuntimeError('torchao remained installed')\n"
            "import torch\n"
            "if not torch.cuda.is_available() or torch.cuda.device_count() != 2: raise RuntimeError('Select Kaggle GPU T4 x2')\n"
            "gpus = [torch.cuda.get_device_name(i) for i in range(2)]\n"
            "if gpus.count('Tesla T4') != 2: raise RuntimeError(f'Expected T4 x2, found {gpus}')\n"
            "with urllib.request.urlopen('https://huggingface.co', timeout=20) as r:\n"
            "    if r.status != 200: raise RuntimeError(f'Hugging Face HTTP {r.status}')\n"
            "print({'packages': actual, 'torch': torch.__version__, 'gpus': gpus, 'internet': 'PASS'})\n",
            "runtime-preflight",
        )
    )
    cells.append(
        code(
            "UPSTREAM = Path('/kaggle/working/closer-look-LLM-unlearning')\n"
            "COMMIT = 'b622294965f8b40feb5383c9e93dbdb26a76b6b3'\n"
            "if not UPSTREAM.exists(): subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/sail-sg/closer-look-LLM-unlearning.git', str(UPSTREAM)], check=True)\n"
            "subprocess.run(['git', '-C', str(UPSTREAM), 'fetch', '--depth=1', 'origin', COMMIT], check=True)\n"
            "subprocess.run(['git', '-C', str(UPSTREAM), 'checkout', '--detach', COMMIT], check=True)\n"
            "actual_commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()\n"
            "if actual_commit != COMMIT: raise RuntimeError(f'Upstream commit mismatch: {actual_commit}')\n"
            "print({'official_idk_ap_commit': actual_commit})\n",
            "pin-official-source",
        )
    )
    cells.append(
        code(
            "import json, os, subprocess, sys\n"
            "from pathlib import Path\n"
            "INPUT = Path('/kaggle/input')\n"
            "manifests = list(INPUT.rglob('reference_support_manifest.json'))\n"
            "if len(manifests) != 1: raise RuntimeError(f'Expected exactly one verified reference-support manifest; found {manifests}')\n"
            "SUPPORT = manifests[0].parent\n"
            "RUN_ROOT = Path('/kaggle/working/tofu_3b_idk_ap_lora_diagnostic_run')\n"
            "ARTIFACT = Path('/kaggle/working/tofu_3b_idk_ap_lora_diagnostic.zip')\n"
            "os.environ['TOFU_MODEL_CACHE_ROOT'] = '/kaggle/working/tofu_model_cache'\n"
            "command = [sys.executable, '/kaggle/working/idk_ap_lora/run.py', '--protocol', '/kaggle/working/idk_ap_lora_protocol.json', '--idk', '/kaggle/working/idk_ap_lora/idontknow.jsonl', '--upstream-repo', str(UPSTREAM), '--reference-support-dir', str(SUPPORT), '--root', str(RUN_ROOT), '--artifact-zip', str(ARTIFACT)]\n"
            "subprocess.run(command, check=True)\n"
            "verification = Path('/kaggle/working/idk_ap_independent_verification.json')\n"
            "verify = [sys.executable, '/kaggle/working/idk_ap_lora/verify_run.py', '--run-root', str(RUN_ROOT), '--protocol', '/kaggle/working/idk_ap_lora_protocol.json', '--artifact-zip', str(ARTIFACT), '--output', str(verification)]\n"
            "subprocess.run(verify, check=True)\n"
            "print(json.loads((RUN_ROOT / 'run_receipt.json').read_text()))\n"
            "print(json.loads(verification.read_text()))\n"
            "print({'artifact': str(ARTIFACT), 'bytes': ARTIFACT.stat().st_size})\n",
            "run-and-verify",
        )
    )
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
    kernel_meta = {
        "id": KERNEL_ID,
        "title": "TOFU 3B IDK+AP LoRA Diagnostic 2026-11-09",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "GPU T4 x2",
        "dataset_sources": [REFERENCE_DATASET],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK}; protocol_sha256={PROTOCOL_SHA256}; cells={len(cells)}")


if __name__ == "__main__":
    main()
