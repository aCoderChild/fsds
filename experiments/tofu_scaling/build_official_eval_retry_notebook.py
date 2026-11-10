#!/usr/bin/env python3
"""Build a private Kaggle retry that reuses verified outputs and evaluates one missing checkpoint."""

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "kaggle" / "official_metrics_retry"
OUT = OUT_DIR / "tofu_official_metrics_retry_8b.ipynb"
PARTIAL_DATASET = "bunnoob2005/tofu-3b-8bfull-resume-2026-11-07"
PARTIAL_ARCHIVE = (
    ROOT / "diagnostics" / "kaggle_official_metrics_8b_retry_2026-11-06"
    / "version5_2026-11-07" / "retain90_retry_input" / "tofu_3b_and_8b_full_verified_resume.zip"
)


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
    source_zip = zipfile.ZipFile(PARTIAL_ARCHIVE)
    bad_member = source_zip.testzip()
    if bad_member:
        raise RuntimeError(f"Resume archive failed CRC at {bad_member}")
    required_files = (
        "tofu_official_eval/official_eval_status.json",
        "tofu_official_eval/audit/tofu_readiness_audit.json",
        "tofu_official_eval/audit/tofu_readiness_audit.md",
        "tofu_official_eval/llama_3_2_3b_full_official_eval.json",
        "tofu_official_eval/llama_3_2_3b_full_examples.jsonl",
        "tofu_official_eval/llama_3_2_3b_retain90_official_eval.json",
        "tofu_official_eval/llama_3_2_3b_retain90_examples.jsonl",
        "tofu_official_eval/llama_3_1_8b_full_official_eval.json",
        "tofu_official_eval/llama_3_1_8b_full_examples.jsonl",
    )
    missing = [name for name in required_files if name not in source_zip.namelist()]
    if missing:
        raise RuntimeError(f"First-run partial archive is missing required resume files: {missing}")
    files = {}
    for source_name in required_files:
        relative = source_name.removeprefix("tofu_official_eval/")
        content = source_zip.read(source_name)
        files[relative] = {
            "source_file": Path(relative).name,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    partial_bundle_sha256 = hashlib.sha256(PARTIAL_ARCHIVE.read_bytes()).hexdigest()
    cells = [
        md(
            "# TOFU official-metric baseline retry: isolated 8B retain90 oracle\n\n"
            "This version restores checksum-verified 3B full/oracle outputs and the completed 8B full output, then evaluates only the missing 8B retain90 oracle in a fresh Kaggle runtime. This avoids reloading and rerunning the 70-minute 8B full evaluation in the same process. It disables only the pinned Transformers 5.22.1 load-time allocator preallocation for 8B, places weights with `balanced` across both T4s under a 13 GiB-per-GPU cap, and refuses scoring if any 8B weight is on CPU/disk. A four-token generation smoke must pass first.\n\n"
            "The run is inference-only: it does not rerun full checkpoints, train NPO, or create a research adapter. Kaggle T4 inference remains FP16 + SDPA; this differs from the upstream BF16 + FlashAttention 2 runtime."
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
            "    raise RuntimeError('Select GPU T4 x2 before starting; stopped before loading checkpoint weights.')\n"
            "gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
            "print({'gpus': gpus, 'memory_bytes': [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]})\n"
            "if gpus.count('Tesla T4') < 2: raise RuntimeError(f'Expected two Tesla T4 GPUs, found {gpus}')"
        ),
    ]
    for filename, source in scripts.items():
        cells.append(code(f"%%writefile /kaggle/working/{filename}\n" + source))
    cells += [
        code(
            "from pathlib import Path\n"
            "import hashlib, json, shutil, zipfile\n"
            "OUT = Path('/kaggle/working/tofu_official_eval')\n"
            "manifest_paths = list(Path('/kaggle/input').rglob('partial_resume_manifest.json'))\n"
            "if len(manifest_paths) != 1: raise RuntimeError(f'Expected exactly one resume manifest in the private dataset; found {manifest_paths}')\n"
            "resume_root = manifest_paths[0].parent\n"
            "manifest = json.loads(manifest_paths[0].read_text())\n"
            "upload_paths = list(Path('/kaggle/input').rglob('upload_manifest.json'))\n"
            "if len(upload_paths) != 1: raise RuntimeError(f'Expected exactly one upload manifest; found {upload_paths}')\n"
            "upload_manifest = json.loads(upload_paths[0].read_text())\n"
            f"expected_zip_sha256 = {json.dumps(partial_bundle_sha256)}\n"
            "if upload_manifest.get('archive_sha256') != expected_zip_sha256 or upload_manifest.get('archive_crc') != 'PASS': raise RuntimeError('Uploaded resume-artifact receipt does not match the locally verified archive')\n"
            "OUT.mkdir(parents=True, exist_ok=True)\n"
            "for rel, item in manifest['files'].items():\n"
            "    source = resume_root / rel\n"
            "    if not source.is_file(): raise RuntimeError(f'Resume dataset is missing {rel}')\n"
            "    actual = hashlib.sha256(source.read_bytes()).hexdigest()\n"
            "    if actual != item['sha256']: raise RuntimeError(f'Resume result checksum mismatch: {rel}')\n"
            "    destination = OUT / rel\n"
            "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.copy2(source, destination)\n"
            "shutil.copy2(manifest_paths[0], OUT / 'partial_resume_manifest.json')\n"
            "restore_source = 'private Kaggle dataset; source-file SHA-256 verified (dataset platform unpacked the uploaded ZIP)'\n"
            "for rel, item in manifest['files'].items():\n"
            "    source = OUT / rel\n"
            "    if not source.is_file(): raise RuntimeError(f'Resume archive missing {rel}')\n"
            "    actual = hashlib.sha256(source.read_bytes()).hexdigest()\n"
            "    if actual != item['sha256']: raise RuntimeError(f'Restored partial result checksum mismatch: {rel}')\n"
            "for key in ('llama_3_2_3b_full', 'llama_3_2_3b_retain90', 'llama_3_1_8b_full'):\n"
            "    result = json.loads((OUT / f'{key}_official_eval.json').read_text())\n"
            "    examples_path = OUT / f'{key}_examples.jsonl'\n"
            "    if not examples_path.exists():\n"
            "        with examples_path.open('w', encoding='utf-8') as f:\n"
            "            for rows in result['per_example'].values():\n"
            "                for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\\n')\n"
            "print({'restore_source': restore_source, 'verified_files': sorted(manifest['files'])})\n"
            "!python /kaggle/working/diagnose_tofu.py --output-dir /kaggle/working/tofu_official_eval/audit\n"
            "audit = json.loads((OUT / 'audit/tofu_readiness_audit.json').read_text())\n"
            "if not audit['data_audit']['checks']['structural_checks_pass'] or not audit.get('checkpoint_access_checks_pass'):\n"
            "    raise RuntimeError('Pinned data/checkpoint access audit failed; do not evaluate.')"
        ),
        code(
            "import subprocess, sys\n"
            "run = subprocess.run([sys.executable, '/kaggle/working/tofu_official_eval.py', '--output-dir', '/kaggle/working/tofu_official_eval', '--models', 'llama_3_1_8b_retain90', '--score-batch-size', '1', '--generation-batch-size', '1', '--max-memory-per-gpu-gib', '13', '--resume-existing'])\n"
            "subprocess.run(['bash', '-lc', 'cd /kaggle/working && zip -rq -FS tofu_official_eval_results.zip tofu_official_eval'], check=True)\n"
            "if run.returncode: raise RuntimeError(f'8B retry failed with status {run.returncode}; partial receipt was archived.')\n"
            "verify = subprocess.run([sys.executable, '/kaggle/working/verify_tofu_official_eval.py', '/kaggle/working/tofu_official_eval', '--audit', '/kaggle/working/tofu_official_eval/audit/tofu_readiness_audit.json'])\n"
            "subprocess.run(['bash', '-lc', 'cd /kaggle/working && zip -rq -FS tofu_official_eval_results.zip tofu_official_eval'], check=True)\n"
            "if verify.returncode: raise RuntimeError(f'Independent verification failed with status {verify.returncode}; download the verification receipt.')\n"
            "print('All four pinned checkpoint outputs passed independent verification. Download tofu_official_eval_results.zip.')"
        ),
    ]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"tofu-retry-{index:02d}"
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
        "id": "bunnoob2005/tofu-official-metrics-8b-retry-2026-11-06",
        "title": "TOFU Official Metrics 8B Retry 2026-11-06",
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [PARTIAL_DATASET],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
