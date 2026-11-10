#!/usr/bin/env python3
"""Build the private Kaggle continuation notebook for the locked 3B study."""

from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "kaggle" / "npo_profile_recovery_resume"
OUT = OUT_DIR / "tofu_3b_npo_profile_recovery_resume.ipynb"
KERNEL_ID = "bunnoob2005/tofu-3b-npo-profile-recovery-resume"
UPSTREAM_COMMIT = "2d98027677f3468d52b6baa3fc4f2b9f2dc08820"
ARTIFACT_DATASET = "bunnoob2005/tofu-3b-recovery-v2-2026-11-08"
ARTIFACT_SHA256 = "74a8a86c7c1f67f0c9625df2805293679a9bacf808204736a8579b839b65d370"
ARTIFACT_ZIP = ROOT / "diagnostics" / "npo_lora_profile_recovery_2026-11-07" / "kaggle_output_v2" / "tofu_npo_profile_recovery_artifacts_resumed_v2.zip"


def artifact_tree_manifest() -> dict:
    files = []
    with zipfile.ZipFile(ARTIFACT_ZIP) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Source artifact ZIP CRC failure: {bad}")
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            prefix = "tofu_npo_profile_recovery/"
            if not member.filename.startswith(prefix):
                raise RuntimeError(f"Unexpected source artifact path: {member.filename}")
            relative = member.filename[len(prefix):]
            if not relative or ".." in Path(relative).parts:
                raise RuntimeError(f"Unsafe source artifact path: {member.filename}")
            content = archive.read(member)
            files.append({
                "path": relative,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
    canonical = "".join(f"{item['path']}\t{item['sha256']}\n" for item in sorted(files, key=lambda item: item["path"]))
    return {
        "root": "tofu_npo_profile_recovery",
        "source_archive_sha256": ARTIFACT_SHA256,
        "files": files,
        "tree_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def markdown(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> None:
    manifest = artifact_tree_manifest()
    sources = {
        "tofu_official_eval.py": ROOT / "tofu_official_eval.py",
        "npo_objective.py": ROOT / "npo_objective.py",
        "verify_npo_objective.py": ROOT / "verify_npo_objective.py",
        "npo_lora_train.py": ROOT / "npo_lora_train.py",
        "validate_npo_recovery_protocol.py": ROOT / "validate_npo_recovery_protocol.py",
        "evaluate_npo_profile_cohort.py": ROOT / "evaluate_npo_profile_cohort.py",
        "verify_npo_profile_recovery_run.py": ROOT / "verify_npo_profile_recovery_run.py",
        "resume_npo_profile_recovery.py": ROOT / "resume_npo_profile_recovery.py",
        "npo_lora_recovery_protocol.json": ROOT / "npo_lora_recovery_protocol.json",
        "requirements-npo-pilot.txt": ROOT / "requirements-npo-pilot.txt",
    }
    contents = {name: path.read_text(encoding="utf-8") for name, path in sources.items()}
    cells = [
        markdown(
            "# TOFU 3B NPO-LoRA preregistered continuation\n\n"
            "This continuation reuses and verifies the completed full-checkpoint reference evaluation and alpha=5 development artifacts. "
            "It independently recomputes the alpha=5 paired gates, then runs only the remaining preregistered alpha candidates (10 and 20) in order. "
            "It reuses the exact reference outputs; it does not repeat the full baseline evaluation. A final disjoint-cohort run is allowed only if a development candidate passes every locked gate.\n\n"
            "The first Kaggle run trained alpha=5 and completed its development evaluations, but its evaluator used the wrong reference-file path. Continuation v1 stopped before training because Kaggle expanded the input ZIP. Continuation v2 completed alpha=10 training and all 800 development rows, then hit a receipt-validation NameError; alpha=10's saved metrics were independently verified and failed the locked retain-utility gates. This version verifies the mounted artifact inventory and hashes, reuses the completed alpha=10 training/evaluation without repeating GPU work, and continues at alpha=20. It keeps the locked protocol and only starts the final disjoint cohort if a development candidate passes every gate. The run removes the local model cache after saving the compact experiment archive.\n\n"
            "Protocol: one seed, exploratory profile-cohort follow-up; not a standard full-split TOFU score or confirmatory result. Keep GPU T4 x2 and Internet enabled. No Hugging Face token is required.",
            "readme",
        ),
        code(
            "import importlib.metadata as metadata, json, subprocess, sys, urllib.request\n"
            "from pathlib import Path\n"
            "print({'study': 'locked TOFU 3B profile recovery', 'mode': 'resume remaining alpha candidates', 'source_archive_sha256': " + repr(ARTIFACT_SHA256) + ", 'verified_tree_sha256': " + repr(manifest['tree_sha256']) + "})\n",
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
            "try: torchao_version = metadata.version('torchao')\n"
            "except metadata.PackageNotFoundError: torchao_version = None\n"
            "if torchao_version is not None:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '--yes', 'torchao'], check=True, capture_output=True, text=True)\n"
            "    try: metadata.version('torchao'); raise RuntimeError('torchao remains installed')\n"
            "    except metadata.PackageNotFoundError: pass\n"
            "print({'packages': versions, 'torch': metadata.version('torch'), 'torchao': 'absent'})\n",
            "dependencies",
        ),
        code(
            "import torch\n"
            "if not torch.cuda.is_available() or torch.cuda.device_count() != 2: raise RuntimeError('Select Kaggle GPU T4 x2')\n"
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
        "import hashlib, json\n"
        "manifest = " + repr(manifest) + "\n"
        "matches = [path for path in Path('/kaggle/input').rglob('tofu_npo_profile_recovery') if path.is_dir() and (path / 'npo_lora_recovery_protocol.json').is_file()]\n"
        "if len(matches) != 1: raise RuntimeError(f'Expected exactly one unpacked resume artifact tree, found {matches}')\n"
        "resume_tree = matches[0]\n"
        "manifest_path = Path('/kaggle/working/resume_input_manifest.json')\n"
        "manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')\n"
        f"expected_tree_sha256 = {manifest['tree_sha256']!r}\n"
        "command = [sys.executable, '/kaggle/working/resume_npo_profile_recovery.py', '--resume-artifacts-dir', str(resume_tree), '--resume-manifest-json', str(manifest_path), '--resume-tree-sha256', expected_tree_sha256]\n"
        "print({'resume_tree': str(resume_tree), 'expected_files': len(manifest['files']), 'tree_sha256': expected_tree_sha256})\n"
        "try:\n"
        "    subprocess.run(command, check=True)\n"
        "finally:\n"
        "    cache = Path('/kaggle/working/model_cache')\n"
        "    if cache.exists():\n"
        "        import shutil; shutil.rmtree(cache)\n"
        "artifact = Path('/kaggle/working/tofu_npo_profile_recovery_artifacts_resumed.zip')\n"
        "if not artifact.is_file() or artifact.stat().st_size == 0: raise RuntimeError('Resumed result archive is missing')\n"
        "digest = hashlib.sha256()\n"
        "with artifact.open('rb') as stream:\n"
        "    for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)\n"
        "print({'artifact_zip': str(artifact), 'bytes': artifact.stat().st_size, 'sha256': digest.hexdigest()})\n",
        "resume-preregistered-recovery",
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
    metadata = {
        "id": KERNEL_ID,
        "title": "TOFU 3B NPO Profile Recovery Resume",
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "GPU T4 x2",
        "dataset_sources": [ARTIFACT_DATASET],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {OUT}")


if __name__ == "__main__":
    main()
