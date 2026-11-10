#!/usr/bin/env python3
"""Embed the pinned local diagnostic scripts in a private Kaggle notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kaggle" / "tofu_baseline_diagnostic.ipynb"


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    audit_source = (ROOT / "diagnose_tofu.py").read_text(encoding="utf-8")
    eval_source = (ROOT / "tofu_baseline_eval.py").read_text(encoding="utf-8")
    cells = [
        markdown_cell(
            "# TOFU 3B/8B pre-unlearning baseline diagnostic\n\n"
            "This notebook checks the pinned public TOFU data and evaluates the released full-model checkpoints. "
            "It does not fine-tune or unlearn. Run on Kaggle GPU T4 x2 with Internet enabled. "
            "It scores all 400 forget10 and 400 holdout10 questions, plus a fixed 400-question retain90 profile sample. "
            "The output is answer-likelihood diagnostics with profile-clustered intervals, not generation accuracy.\n\n"
            "Checkpoint downloads are large (about 6.43 GB for 3B and 16.06 GB for 8B). The script downloads and removes each model cache sequentially."
        ),
        code_cell(
            "import importlib.metadata as metadata\n"
            "import subprocess, sys\n"
            "required = {'transformers': '5.22.1', 'accelerate': '1.15.0'}\n"
            "missing = []\n"
            "for package, wanted in required.items():\n"
            "    try:\n"
            "        found = metadata.version(package)\n"
            "    except metadata.PackageNotFoundError:\n"
            "        found = None\n"
            "    if found != wanted:\n"
            "        missing.append(f'{package}=={wanted}')\n"
            "if missing:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing], check=True)\n"
            "print('Pinned model/evaluation dependencies are ready.')"
        ),
        code_cell(
            "import torch, transformers, accelerate\n"
            "if not torch.cuda.is_available() or torch.cuda.device_count() < 2:\n"
            "    raise RuntimeError('Select GPU T4 x2 before running; this guard stops before model weights download.')\n"
            "print({'torch': torch.__version__, 'transformers': transformers.__version__, 'accelerate': accelerate.__version__})\n"
            "print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
            "print('Run only the baseline diagnostic. No HF_TOKEN or gated Meta repository access is required for these public checkpoints.')"
        ),
        code_cell("%%writefile /kaggle/working/diagnose_tofu.py\n" + audit_source),
        code_cell("%%writefile /kaggle/working/tofu_baseline_eval.py\n" + eval_source),
        code_cell(
            "from pathlib import Path\n"
            "OUT = Path('/kaggle/working/tofu_baseline_results')\n"
            "OUT.mkdir(parents=True, exist_ok=True)\n"
            "!python /kaggle/working/diagnose_tofu.py --output-dir /kaggle/working/tofu_baseline_results/audit"
        ),
        code_cell(
            "import subprocess, sys\n"
            "run = subprocess.run([sys.executable, '/kaggle/working/tofu_baseline_eval.py', '--output-dir', '/kaggle/working/tofu_baseline_results', '--models', 'llama_3_2_3b,llama_3_1_8b', '--batch-size', '2'])\n"
            "subprocess.run(['bash', '-lc', 'cd /kaggle/working && zip -rq tofu_baseline_diagnostic_results.zip tofu_baseline_results'], check=True)\n"
            "if run.returncode:\n"
            "    raise RuntimeError(f'Baseline runner exited with status {run.returncode}; partial status/score files were zipped.')\n"
            "print('Download /kaggle/working/tofu_baseline_diagnostic_results.zip from the notebook Output panel.')"
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
