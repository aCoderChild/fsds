#!/usr/bin/env python3
"""Build the private Kaggle notebook for the preregistered TOFU 3B NPO-LoRA pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "kaggle" / "npo_lora_pilot"
OUT = OUT_DIR / "tofu_3b_npo_lora_pilot.ipynb"
KERNEL_ID = "bunnoob2005/tofu-3b-npo-lora-pilot-2026-11-07"
BASELINE_DATASET = "bunnoob2005/tofu-3b-8bfull-resume-2026-11-07"
BASELINE_ARCHIVE_SHA256 = "5882fc3ebf8299413b60503edd9eeee080724909f071f5d3ba34dae683f8ea70"
BASELINE_UPLOAD_MANIFEST_SHA256 = "de117326805e5b1b77fe6f18de94f9dac5e8ee3f369ff056ed3a1f762bbfa0d2"
BASELINE_PARTIAL_MANIFEST_SHA256 = "4aa0a4d65c102feb703c057daf03023f2fec3facf7e6fe790c655b635c7648ff"
UPSTREAM_COMMIT = "2d98027677f3468d52b6baa3fc4f2b9f2dc08820"


def markdown(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-full-training", action="store_true", help="Enable the full one-seed pilot after its smoke passes.")
    args = parser.parse_args()

    sources = {
        "tofu_official_eval.py": ROOT / "tofu_official_eval.py",
        "verify_tofu_official_eval.py": ROOT / "verify_tofu_official_eval.py",
        "npo_objective.py": ROOT / "npo_objective.py",
        "verify_npo_objective.py": ROOT / "verify_npo_objective.py",
        "npo_lora_train.py": ROOT / "npo_lora_train.py",
        "evaluate_npo_adapter.py": ROOT / "evaluate_npo_adapter.py",
        "verify_npo_run.py": ROOT / "verify_npo_run.py",
        "npo_lora_pilot_protocol.json": ROOT / "npo_lora_pilot_protocol.json",
        "requirements-npo-pilot.txt": ROOT / "requirements-npo-pilot.txt",
    }
    contents = {name: path.read_text(encoding="utf-8") for name, path in sources.items()}
    cells = [
        markdown(
            "# TOFU 3B NPO-LoRA pilot\n\n"
            "This private notebook starts with a bounded, one-step smoke on eight pinned TOFU forget examples and sampled retain examples. It loads the verified Llama 3.2 3B full checkpoint on Kaggle T4 x2, performs one LoRA optimizer update, checks that the adapter actually changes, then discards that smoke adapter. The default notebook does **not** start the 10-epoch pilot.\n\n"
            "Only after the smoke receipt passes should a new notebook version be built with `--run-full-training` and pushed. The pilot is a single-seed LoRA adaptation, not full-parameter NPO or confirmatory evidence. It does not train on holdout. All model and dataset revisions, hashes, thresholds, and stopping rules are locked in the included protocol. No Hugging Face token is needed for these public Open-Unlearning checkpoints.",
            "readme",
        ),
        code(
            "import importlib.metadata as metadata\n"
            "import json, os, shutil, subprocess, sys, time, zipfile\n"
            "from datetime import datetime, timezone\n"
            "from pathlib import Path\n"
            "\n"
            "RUN_FULL_TRAINING = " + ("True" if args.run_full_training else "False") + "\n"
            "print({'full_training_enabled': RUN_FULL_TRAINING, 'notebook_mode': 'full pilot' if RUN_FULL_TRAINING else 'bounded smoke only'})\n",
            "setup-vars",
        ),
        code(
            "requirements = Path('/kaggle/working/requirements-npo-pilot.txt')\n"
            f"requirements.write_text({contents['requirements-npo-pilot.txt']!r}, encoding='utf-8')\n"
            "expected = {'transformers': '5.22.1', 'accelerate': '1.15.0', 'peft': '0.21.2', 'scipy': '1.16.3', 'rouge-score': '0.1.2'}\n"
            "mismatch = []\n"
            "for package, version in expected.items():\n"
            "    try: actual = metadata.version(package)\n"
            "    except metadata.PackageNotFoundError: actual = None\n"
            "    if actual != version: mismatch.append(package + '==' + version)\n"
            "if mismatch:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '-r', str(requirements)], check=True)\n"
            "versions = {package: metadata.version(package) for package in expected}\n"
            "if versions != expected: raise RuntimeError(f'Pinned package check failed: {versions}')\n"
            "torchao_version = None\n"
            "try: torchao_version = metadata.version('torchao')\n"
            "except metadata.PackageNotFoundError: pass\n"
            "if torchao_version is not None:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '--yes', 'torchao'], check=True, capture_output=True, text=True)\n"
            "    try: metadata.version('torchao'); raise RuntimeError('torchao remained installed after removal')\n"
            "    except metadata.PackageNotFoundError: pass\n"
            "import importlib.util\n"
            "if importlib.util.find_spec('torchao') is not None: raise RuntimeError('torchao module remains importable after removal')\n"
            "import torch\n"
            "from transformers import LlamaConfig, LlamaForCausalLM\n"
            "from peft import LoraConfig, get_peft_model\n"
            "tiny_config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)\n"
            "tiny = LlamaForCausalLM(tiny_config)\n"
            "tiny = get_peft_model(tiny, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], bias='none', task_type='CAUSAL_LM'))\n"
            "tiny_ids = torch.tensor([[1,2,3,4]])\n"
            "tiny_loss = tiny(input_ids=tiny_ids, labels=tiny_ids, use_cache=False).loss\n"
            "tiny_loss.backward()\n"
            "tiny_trainable = [parameter for parameter in tiny.parameters() if parameter.requires_grad]\n"
            "tiny_grad = sum(float(parameter.grad.float().abs().sum()) for parameter in tiny_trainable if parameter.grad is not None)\n"
            "if not torch.isfinite(tiny_loss) or not tiny_trainable or not torch.isfinite(torch.tensor(tiny_grad)) or tiny_grad <= 0: raise RuntimeError('PEFT LoRA dispatch preflight failed')\n"
            "print({'packages': versions, 'torch': metadata.version('torch'), 'torchao': 'absent', 'peft_lora_dispatch_smoke': 'PASS', 'tiny_loss': float(tiny_loss.detach()), 'tiny_adapter_gradient_l1': tiny_grad})\n",
            "install-dependencies",
        ),
        code(
            "import torch\n"
            "import urllib.request\n"
            "if not torch.cuda.is_available() or torch.cuda.device_count() != 2:\n"
            "    raise RuntimeError('Select Kaggle GPU T4 x2. Refusing model loading before two-GPU validation.')\n"
            "gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
            "if gpu_names.count('Tesla T4') != 2:\n"
            "    raise RuntimeError(f'Protocol requires exactly two Tesla T4 GPUs; found {gpu_names}')\n"
            "with urllib.request.urlopen('https://huggingface.co', timeout=20) as response: internet_status = response.status\n"
            "if internet_status != 200: raise RuntimeError(f'Internet preflight returned HTTP {internet_status}')\n"
            "print({'gpu_names': gpu_names, 'total_memory_bytes': [torch.cuda.get_device_properties(i).total_memory for i in range(2)], 'internet_http_status': internet_status})\n",
            "gpu-gate",
        ),
        code(
            "upstream = Path('/kaggle/working/open-unlearning')\n"
            "commit = " + repr(UPSTREAM_COMMIT) + "\n"
            "if not upstream.exists(): subprocess.run(['git', 'clone', '--filter=blob:none', 'https://github.com/locuslab/open-unlearning.git', str(upstream)], check=True)\n"
            "subprocess.run(['git', '-C', str(upstream), 'fetch', '--depth', '1', 'origin', commit], check=True)\n"
            "subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', commit], check=True)\n"
            "actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()\n"
            "if actual != commit: raise RuntimeError(f'Wrong upstream commit: {actual}')\n"
            "print({'open_unlearning_commit': actual})\n",
            "pin-upstream",
        ),
    ]

    for index, (name, content) in enumerate(contents.items(), start=1):
        cells.append(code(f"%%writefile /kaggle/working/{name}\n" + content, f"write-source-{index:02d}"))

    cells.extend([
        code(
            "subprocess.run([sys.executable, '/kaggle/working/verify_npo_objective.py', '--upstream-src', '/kaggle/working/open-unlearning/src', '--protocol', '/kaggle/working/npo_lora_pilot_protocol.json', '--output', '/kaggle/working/npo_objective_equivalence.json'], check=True)\n"
            "objective_check = json.loads(Path('/kaggle/working/npo_objective_equivalence.json').read_text())\n"
            "if objective_check.get('status') != 'PASS': raise RuntimeError('Pinned Open-Unlearning NPO loss/gradient equivalence did not pass')\n"
            "print({'objective_check': objective_check})\n",
            "objective-equivalence",
        ),
        code(
            "input_root = Path('/kaggle/input')\n"
            "partial_manifests = list(input_root.rglob('partial_resume_manifest.json'))\n"
            "upload_manifests = list(input_root.rglob('upload_manifest.json'))\n"
            "if len(partial_manifests) != 1 or len(upload_manifests) != 1:\n"
            "    raise RuntimeError(f'Attach exactly the verified resume dataset; partial={partial_manifests}, upload={upload_manifests}')\n"
            "partial_path, upload_path = partial_manifests[0], upload_manifests[0]\n"
            "def file_sha(path):\n"
            "    import hashlib\n"
            "    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
            f"if file_sha(upload_path) != {BASELINE_UPLOAD_MANIFEST_SHA256!r}: raise RuntimeError('Pinned upload manifest checksum mismatch')\n"
            f"if file_sha(partial_path) != {BASELINE_PARTIAL_MANIFEST_SHA256!r}: raise RuntimeError('Pinned partial resume manifest checksum mismatch')\n"
            "upload_receipt = json.loads(upload_path.read_text())\n"
            f"if upload_receipt.get('archive_sha256') != {BASELINE_ARCHIVE_SHA256!r} or upload_receipt.get('archive_crc') != 'PASS': raise RuntimeError('The uploaded baseline archive receipt is not the expected CRC-passing archive')\n"
            "manifest = json.loads(partial_path.read_text())\n"
            "run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')\n"
            "RUN_ROOT = Path('/kaggle/working') / ('tofu_npo_run_' + run_id)\n"
            "BASELINE_DIR = RUN_ROOT / 'tofu_official_eval'\n"
            "TRAIN_DIR = RUN_ROOT / 'npo_train'\n"
            "SMOKE_DIR = RUN_ROOT / 'npo_smoke'\n"
            "BASELINE_DIR.mkdir(parents=True, exist_ok=False)\n"
            "copied = []\n"
            "for relative, item in manifest['files'].items():\n"
            "    source = partial_path.parent / relative\n"
            "    if not source.is_file() or file_sha(source) != item['sha256']:\n"
            "        raise RuntimeError(f'Restored baseline input missing or hash-invalid: {relative}')\n"
            "    target = BASELINE_DIR / relative\n"
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.copy2(source, target)\n"
            "    copied.append(relative)\n"
            "shutil.copy2(partial_path, BASELINE_DIR / 'partial_resume_manifest.json')\n"
            "for name in ['tofu_official_eval.py', 'verify_tofu_official_eval.py', 'npo_objective.py', 'verify_npo_objective.py', 'npo_lora_train.py', 'evaluate_npo_adapter.py', 'verify_npo_run.py', 'npo_lora_pilot_protocol.json', 'requirements-npo-pilot.txt']:\n"
            "    shutil.copy2(Path('/kaggle/working') / name, RUN_ROOT / name)\n"
            "audit_path = BASELINE_DIR / 'audit' / 'tofu_readiness_audit.json'\n"
            "audit = json.loads(audit_path.read_text())\n"
            "if audit.get('data_audit', {}).get('checks', {}).get('structural_checks_pass') is not True:\n"
            "    raise RuntimeError('Pinned baseline data audit did not pass')\n"
            "verify = subprocess.run([sys.executable, '/kaggle/working/verify_tofu_official_eval.py', str(BASELINE_DIR), '--audit', str(audit_path), '--allow-partial'], capture_output=True, text=True)\n"
            "(RUN_ROOT / 'baseline_verification.stdout.json').write_text(verify.stdout + verify.stderr)\n"
            "if verify.returncode: raise RuntimeError('Baseline partial verifier failed: ' + verify.stdout[-3000:])\n"
            "baseline_verification = json.loads((BASELINE_DIR / 'official_eval_verification.json').read_text())\n"
            "for key in ['llama_3_2_3b_full', 'llama_3_2_3b_retain90']:\n"
            "    if baseline_verification['models'][key]['status'] != 'PASS': raise RuntimeError(f'3B prerequisite failed: {key}')\n"
            "print({'run_root': str(RUN_ROOT), 'restored_baseline_files': copied, 'baseline_verification': baseline_verification['overall_status'], 'required_3b_baselines': 'PASS'})\n",
            "restore-verified-baseline",
        ),
        code(
            "def archive_run():\n"
            "    archive = Path('/kaggle/working/tofu_npo_lora_pilot_artifacts.zip')\n"
            "    if archive.exists(): archive.unlink()\n"
            "    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:\n"
            "        for path in sorted(RUN_ROOT.rglob('*')):\n"
            "            if path.is_file(): zf.write(path, path.relative_to(RUN_ROOT.parent))\n"
            "    print({'artifact_zip': str(archive), 'bytes': archive.stat().st_size})\n"
            "\n"
            "smoke_cmd = [sys.executable, '/kaggle/working/npo_lora_train.py', '--model', '3b', '--seed', '42', '--protocol', '/kaggle/working/npo_lora_pilot_protocol.json', '--objective-check', '/kaggle/working/npo_objective_equivalence.json', '--output-dir', str(SMOKE_DIR), '--smoke']\n"
            "smoke = subprocess.run(smoke_cmd, capture_output=True, text=True)\n"
            "(RUN_ROOT / 'smoke.stdout.log').write_text(smoke.stdout + smoke.stderr)\n"
            "archive_run()\n"
            "if smoke.returncode: raise RuntimeError('Bounded real-data smoke failed; receipt/log preserved in the artifact ZIP. ' + smoke.stderr[-4000:])\n"
            "smoke_receipt = json.loads((SMOKE_DIR / 'training_receipt.json').read_text())\n"
            "if smoke_receipt.get('status') != 'SMOKE_PASS' or smoke_receipt.get('global_steps') != 1 or smoke_receipt.get('smoke_max_abs_lora_parameter_update', 0) <= 0:\n"
            "    raise RuntimeError('Smoke receipt did not pass all one-step/update checks')\n"
            "if smoke_receipt.get('runtime', {}).get('cpu_or_disk_offload') is not False:\n"
            "    raise RuntimeError('Smoke used CPU/disk offload')\n"
            "print({'bounded_smoke': 'PASS', 'global_steps': smoke_receipt['global_steps'], 'max_abs_adapter_update': smoke_receipt['smoke_max_abs_lora_parameter_update'], 'research_scores_created': False, 'full_training_enabled': RUN_FULL_TRAINING})\n",
            "bounded-real-data-smoke",
        ),
        code(
            "if RUN_FULL_TRAINING:\n"
            "    train_cmd = [sys.executable, '/kaggle/working/npo_lora_train.py', '--model', '3b', '--seed', '42', '--protocol', '/kaggle/working/npo_lora_pilot_protocol.json', '--objective-check', '/kaggle/working/npo_objective_equivalence.json', '--output-dir', str(TRAIN_DIR)]\n"
            "    train = subprocess.run(train_cmd, capture_output=True, text=True)\n"
            "    (RUN_ROOT / 'training.stdout.log').write_text(train.stdout + train.stderr)\n"
            "    archive_run()\n"
            "    if train.returncode: raise RuntimeError('NPO-LoRA pilot training failed; partial receipt and trace were archived. ' + train.stderr[-5000:])\n"
            "    train_receipt = json.loads((TRAIN_DIR / 'training_receipt.json').read_text())\n"
            "    if train_receipt.get('status') != 'TRAINING_COMPLETE' or train_receipt.get('global_steps') != 500:\n"
            "        raise RuntimeError(f'Full pilot did not meet its 500-step completion criteria: {train_receipt}')\n"
            "    eval_cmd = [sys.executable, '/kaggle/working/evaluate_npo_adapter.py', '--model', '3b', '--seed', '42', '--adapter-dir', str(TRAIN_DIR / 'adapter'), '--output-dir', str(BASELINE_DIR)]\n"
            "    evaluation = subprocess.run(eval_cmd, capture_output=True, text=True)\n"
            "    (RUN_ROOT / 'evaluation.stdout.log').write_text(evaluation.stdout + evaluation.stderr)\n"
            "    archive_run()\n"
            "    if evaluation.returncode: raise RuntimeError('Official NPO evaluation failed; training and partial score receipts were archived. ' + evaluation.stderr[-5000:])\n"
            "    verify_cmd = [sys.executable, '/kaggle/working/verify_npo_run.py', '--training-dir', str(TRAIN_DIR), '--eval-dir', str(BASELINE_DIR), '--protocol', '/kaggle/working/npo_lora_pilot_protocol.json', '--objective-check', '/kaggle/working/npo_objective_equivalence.json']\n"
            "    verification = subprocess.run(verify_cmd, capture_output=True, text=True)\n"
            "    (RUN_ROOT / 'npo_verification.stdout.log').write_text(verification.stdout + verification.stderr)\n"
            "    archive_run()\n"
            "    if verification.returncode: raise RuntimeError('Independent NPO artifact verification failed; see archived npo_verification.json. ' + verification.stderr[-5000:])\n"
            "    print('NPO-LoRA pilot finished. Inspect npo_verification.md and npo_comparison.json; artifact verification and research diagnostic gates are reported separately.')\n"
            "else:\n"
            "    print('Smoke-only version complete. Review the archived one-step receipt before building the full-training version.')\n",
            "optional-full-pilot",
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
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        "title": "TOFU 3B NPO LoRA Pilot 2026-11-07",
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [BASELINE_DATASET],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {OUT} (full training {'enabled' if args.run_full_training else 'disabled'})")


if __name__ == "__main__":
    main()
