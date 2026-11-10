#!/usr/bin/env python3
"""Generate the private Kaggle notebook for the pinned official-metric eval."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "kaggle" / "official_metrics"
OUT = OUT_DIR / "tofu_official_metrics_baseline.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> None:
    scripts = {
        "diagnose_tofu.py": (ROOT / "diagnose_tofu.py").read_text(encoding="utf-8"),
        "tofu_official_eval.py": (ROOT / "tofu_official_eval.py").read_text(encoding="utf-8"),
        "verify_tofu_official_eval.py": (ROOT / "verify_tofu_official_eval.py").read_text(encoding="utf-8"),
    }
    cells = [
        md(
            "# TOFU 3B/8B official-metric pre-unlearning baseline\n\n"
            "This inference-only run follows the pinned Open-Unlearning TOFU metrics: answer probability, greedy-generation ROUGE-L recall, Truth Ratio, normalized option probability, the nine-component model-utility harmonic mean, and the forget-quality KS comparison to each retain90 oracle. It uses the canonical `retain_perturbed` set (400 rows / 20 complete profile blocks), plus all 400 forget and holdout questions, all real-author/world-fact utility rows, and both retain-only oracle checkpoints.\n\n"
            "The run downloads four public checkpoints sequentially on Kaggle GPU T4 x2. Inference uses FP16 + SDPA (the upstream configs specify BF16 + FlashAttention 2), so metric formulas/prompts match but outputs need not be bitwise identical to the upstream runtime. It does not start SFT/NPO or save a research adapter. Expect a longer run than the earlier likelihood-only check."
        ),
        code(
            "import importlib.metadata as metadata\n"
            "import subprocess, sys\n"
            "required = {'transformers': '5.22.1', 'accelerate': '1.15.0', 'scipy': '1.17.1', 'rouge-score': '0.1.2'}\n"
            "missing = []\n"
            "for package, wanted in required.items():\n"
            "    try: found = metadata.version(package)\n"
            "    except metadata.PackageNotFoundError: found = None\n"
            "    if found != wanted: missing.append(f'{package}=={wanted}')\n"
            "if missing: subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing], check=True)\n"
            "print({p: metadata.version(p) for p in required})"
        ),
        code(
            "import torch\n"
            "if not torch.cuda.is_available() or torch.cuda.device_count() < 2:\n"
            "    raise RuntimeError('Select GPU T4 x2 before starting; stopped before model weights download.')\n"
            "print({'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'memory_bytes': [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]})\n"
            "print('Internet is required for the pinned public TOFU files and Open-Unlearning checkpoints. No HF_TOKEN is needed.')"
        ),
    ]
    for filename, source in scripts.items():
        cells.append(code(f"%%writefile /kaggle/working/{filename}\n" + source))
    cells += [
        code(
            "from pathlib import Path\n"
            "OUT = Path('/kaggle/working/tofu_official_eval')\n"
            "OUT.mkdir(parents=True, exist_ok=True)\n"
            "!python /kaggle/working/diagnose_tofu.py --output-dir /kaggle/working/tofu_official_eval/audit\n"
            "import json\n"
            "audit = json.loads((OUT / 'audit/tofu_readiness_audit.json').read_text())\n"
            "if not audit['data_audit']['checks']['structural_checks_pass']:\n"
            "    raise RuntimeError('Pinned TOFU data audit failed; evaluator will not run.')"
            "\nif not audit.get('checkpoint_access_checks_pass'):\n"
            "    raise RuntimeError('A pinned checkpoint config/tokenizer/weight manifest check failed; evaluator will not run.')"
        ),
        code(
            "import subprocess, sys\n"
            "run = subprocess.run([sys.executable, '/kaggle/working/tofu_official_eval.py', '--output-dir', '/kaggle/working/tofu_official_eval', '--score-batch-size', '2', '--generation-batch-size', '2'])\n"
            "subprocess.run(['bash', '-lc', 'cd /kaggle/working && zip -rq tofu_official_eval_results.zip tofu_official_eval'], check=True)\n"
            "if run.returncode:\n"
            "    raise RuntimeError(f'Official-metric evaluator failed with status {run.returncode}; download the partial receipt zip for debugging.')\n"
            "verify = subprocess.run([sys.executable, '/kaggle/working/verify_tofu_official_eval.py', '/kaggle/working/tofu_official_eval', '--audit', '/kaggle/working/tofu_official_eval/audit/tofu_readiness_audit.json'])\n"
            "subprocess.run(['bash', '-lc', 'cd /kaggle/working && zip -rq -FS tofu_official_eval_results.zip tofu_official_eval'], check=True)\n"
            "if verify.returncode:\n"
            "    raise RuntimeError(f'Independent verification failed with status {verify.returncode}; download the result and verification records.')\n"
            "print('Download /kaggle/working/tofu_official_eval_results.zip from the notebook Output panel.')"
        ),
    ]
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
    metadata_path = OUT_DIR / "kernel-metadata.json"
    metadata_path.write_text(json.dumps({
        "id": "bunnoob2005/tofu-official-metrics-baseline-3b-8b-2026-11-06",
        "title": "TOFU Official Metrics Baseline 3B 8B 2026-11-06",
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
