#!/usr/bin/env python3
"""Build deterministic Kaggle notebook and metadata for the access probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "probe.py"
NOTEBOOK = ROOT / "access_probe.ipynb"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    digest = sha(source.encode("utf-8"))
    code = f"""# Hash-bound probe; it reads no weights or research data.\nimport hashlib\nsource = {source!r}\nexpected = {digest!r}\nactual = hashlib.sha256(source.encode('utf-8')).hexdigest()\nif actual != expected:\n    raise RuntimeError(f'Probe source hash mismatch: {{actual}}')\nnamespace = {{'__name__': 'kaggle_hf_access_probe'}}\nexec(compile(source, 'probe.py', 'exec'), namespace)\nnamespace['main']()\n"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "id": "scope",
                "source": [
                    "# Llama 3.1 8B access probe\\n",
                    "No GPU, model weights, research data, or training are used. This checks only the Kaggle `HF_TOKEN` secret and access to the pinned model config.\\n",
                    "Enable Internet and select the existing `HF_TOKEN` secret under Add-ons → Secrets before running. The notebook never prints the token.\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "id": "probe",
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "None", "dataSources": [], "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_bytes = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    NOTEBOOK.write_bytes(notebook_bytes)
    metadata = {
        "id": "bunnoob2005/synthworlds-llama31-8b-access-probe-2026-11-04",
        "title": "SynthWorlds Llama31 8B Access Probe 2026-11-04",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
    }
    (ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "READY_ACCESS_ONLY_NO_GPU",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "expected_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "probe_sha256": digest,
        "notebook_sha256": sha(notebook_bytes),
        "metadata_sha256": sha((ROOT / "kernel-metadata.json").read_bytes()),
        "gpu_enabled": False,
        "internet_enabled": True,
        "weights_downloaded": False,
        "training_started": False,
        "research_data_loaded": False,
    }
    (ROOT / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
