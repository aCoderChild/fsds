#!/usr/bin/env python3
"""Build a guarded Kaggle notebook for evaluating an archived 8B NPO adapter."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "kaggle" / "npo_lora_8b_eval_recovery_v1"
DIAGNOSTICS = ROOT / "diagnostics" / "npo_lora_8b_pilot_2026-11-08"
PROTOCOL = ROOT / "npo_lora_8b_pilot_protocol.json"
OBJECTIVE = DIAGNOSTICS / "objective_equivalence.json"
MANIFEST = ROOT / "kaggle" / "npo_lora_8b_pilot_full" / "source_manifest.json"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    source_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    objective = json.loads(OBJECTIVE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_json = json.dumps(protocol, ensure_ascii=False, sort_keys=True)
    objective_json = json.dumps(objective, ensure_ascii=False, sort_keys=True)
    embedded_hashes = json.dumps(source_manifest["embedded_source_sha256"], sort_keys=True)
    expected_protocol_sha = source_manifest["protocol_sha256"]
    expected_manifest_sha = source_manifest["baseline_manifest_sha256"]

    setup = f'''from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import torch

PROTOCOL = json.loads({protocol_json!r})
OBJECTIVE_CHECK = json.loads({objective_json!r})
EXPECTED_PROTOCOL_SHA256 = {expected_protocol_sha!r}
EXPECTED_BASELINE_MANIFEST_SHA256 = {expected_manifest_sha!r}
EXPECTED_SOURCE_SHA256 = json.loads({embedded_hashes!r})
LAUNCH_CUTOFF_UTC = "2026-11-08T20:58:10+00:00"
INPUT_ARCHIVE_NAME = "tofu_8b_npo_lora_pilot_artifacts.zip"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if OBJECTIVE_CHECK.get("status") != "PASS" or OBJECTIVE_CHECK.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError("Embedded upstream-objective equivalence receipt is not valid for this protocol.")

archives = list(Path("/kaggle/input").rglob(INPUT_ARCHIVE_NAME))
if len(archives) != 1:
    raise RuntimeError(f"Attach exactly one Kaggle dataset containing {{INPUT_ARCHIVE_NAME}}; found {{len(archives)}}")
input_archive = archives[0]
archive_sha256 = sha256(input_archive)
extract_dir = Path("/kaggle/working/eval_recovery_input")
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir(parents=True)

with zipfile.ZipFile(input_archive) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"Input artifact ZIP CRC failed: {{bad}}")
    roots = set()
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if member.is_absolute() or ".." in member.parts or stat.S_ISLNK(mode):
            raise RuntimeError(f"Unsafe archive member rejected: {{info.filename}}")
        if member.parts:
            roots.add(member.parts[0])
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one archived run root; found {{sorted(roots)}}")
    run_root_name = next(iter(roots))
    if not re.fullmatch(r"tofu_8b_npo_run_\\d{{8}}T\\d{{6}}Z", run_root_name):
        raise RuntimeError(f"Unexpected archived run-root name: {{run_root_name}}")
    archive.extractall(extract_dir)

RUN_ROOT = extract_dir / run_root_name
TRAIN_DIR = RUN_ROOT / "npo_train"
ADAPTER_DIR = TRAIN_DIR / "adapter"
EVAL_DIR = RUN_ROOT / "tofu_official_eval"
PROTOCOL_PATH = RUN_ROOT / "npo_lora_8b_pilot_protocol.json"
OBJECTIVE_PATH = Path("/kaggle/working/npo_objective_equivalence.json")
OBJECTIVE_PATH.write_text(json.dumps(OBJECTIVE_CHECK, indent=2) + "\\n", encoding="utf-8")

if not RUN_ROOT.is_dir() or not TRAIN_DIR.is_dir() or not EVAL_DIR.is_dir():
    raise RuntimeError("The archive is missing its run, training, or baseline evaluation directory.")
if sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError("Archived protocol differs from the preregistered local protocol.")
for filename, expected_sha in EXPECTED_SOURCE_SHA256.items():
    source = RUN_ROOT / filename
    if not source.is_file() or sha256(source) != expected_sha:
        raise RuntimeError(f"Archived source differs from its verified source hash: {{filename}}")

receipt_path = TRAIN_DIR / "training_receipt.json"
if not receipt_path.is_file():
    raise RuntimeError("Training receipt is missing; refusing evaluation.")
training_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
started_at = datetime.fromisoformat(training_receipt.get("started_at_utc", "").replace("Z", "+00:00"))
launch_cutoff = datetime.fromisoformat(LAUNCH_CUTOFF_UTC)
expected_model = PROTOCOL["base_checkpoint"]
expected_training = PROTOCOL["dataset"]["training"]
device_map = training_receipt.get("runtime", {{}}).get("device_map", {{}})
schedule = training_receipt.get("training_schedule", {{}})
if (training_receipt.get("status") != "TRAINING_COMPLETE"
        or training_receipt.get("global_steps") != 500
        or training_receipt.get("seed") != 42
        or training_receipt.get("adapter_saved") is not True
        or training_receipt.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or started_at <= launch_cutoff
        or training_receipt.get("training_started") is not True
        or training_receipt.get("model", {{}}).get("repo_id") != expected_model["repo_id"]
        or training_receipt.get("model", {{}}).get("revision") != expected_model["revision"]
        or training_receipt.get("dataset", {{}}).get("repo") != PROTOCOL["dataset"]["repo_id"]
        or training_receipt.get("dataset", {{}}).get("revision") != PROTOCOL["dataset"]["revision"]
        or training_receipt.get("dataset", {{}}).get("forget10_sha256") != expected_training["forget_file"]["sha256"]
        or training_receipt.get("dataset", {{}}).get("retain90_sha256") != expected_training["retain_file"]["sha256"]
        or training_receipt.get("dataset", {{}}).get("forget10_rows") != 400
        or training_receipt.get("dataset", {{}}).get("retain90_rows") != 3600
        or training_receipt.get("training_script_sha256") != EXPECTED_SOURCE_SHA256["npo_lora_train.py"]
        or training_receipt.get("method", {{}}).get("beta") != PROTOCOL["objective"]["beta"]
        or training_receipt.get("method", {{}}).get("epochs") != PROTOCOL["adaptation"]["epochs"]
        or training_receipt.get("method", {{}}).get("lora_rank") != PROTOCOL["adaptation"]["lora_rank"]
        or training_receipt.get("tokenization", {{}}).get("truncated_examples") != 0
        or training_receipt.get("tokenization", {{}}).get("max_length_observed", 999999) > 512
        or training_receipt.get("runtime", {{}}).get("cpu_or_disk_offload") is not False
        or any(str(device).lower() in {{"cpu", "disk"}} for device in device_map.values())
        or training_receipt.get("gpu_names", []).count("Tesla T4") != 2
        or schedule.get("planned_optimizer_steps") != 500
        or schedule.get("planned_loss_trace_rows") != 4000):
    raise RuntimeError("Archived receipt does not prove the locked 500-step seed-42 pilot completed.")
trace_path = TRAIN_DIR / "loss_trace.jsonl"
trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(trace_rows) != 4000 or any(row.get("global_step", 0) not in range(1, 501) for row in trace_rows):
    raise RuntimeError(f"Loss trace does not contain the expected 4,000 records; found {{len(trace_rows)}}.")
actual_adapter_hashes = {{p.name: sha256(p) for p in sorted(ADAPTER_DIR.iterdir()) if p.is_file()}}
if not actual_adapter_hashes or actual_adapter_hashes != training_receipt.get("adapter_files_sha256"):
    raise RuntimeError("Archived adapter files do not match the training receipt hashes.")

baseline_manifest_path = RUN_ROOT / "baseline_inputs" / "npo_8b_baseline_manifest.json"
if not baseline_manifest_path.is_file() or sha256(baseline_manifest_path) != EXPECTED_BASELINE_MANIFEST_SHA256:
    raise RuntimeError("Archived 8B baseline manifest is missing or has the wrong hash.")
baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
for relative, expected_sha in baseline_manifest["files"].items():
    source = RUN_ROOT / "baseline_inputs" / relative
    if not source.is_file() or sha256(source) != expected_sha:
        raise RuntimeError(f"Archived baseline input hash mismatch: {{relative}}")

for name in ("llama_3_1_8b_full_official_eval.json", "llama_3_1_8b_retain90_official_eval.json",
             "audit/tofu_readiness_audit.json"):
    if not (EVAL_DIR / name).is_file():
        raise RuntimeError(f"Archived evaluation directory is missing required baseline file: {{name}}")

existing_result = EVAL_DIR / "llama_3_1_8b_npo_lora_seed42_official_eval.json"
existing_comparison = EVAL_DIR / "npo_comparison.json"
needs_inference = not (existing_result.is_file() and existing_comparison.is_file())
gpu_names = []
if needs_inference:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("The adapter evaluation is missing. Select Kaggle GPU T4 x2 to run it.")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
    if gpu_names.count("Tesla T4") != 2:
        raise RuntimeError(f"This locked pilot requires two Tesla T4 GPUs; found {{gpu_names}}")

print({{
    "preflight": "PASS",
    "input_archive_sha256": archive_sha256,
    "run_root": RUN_ROOT.name,
    "protocol_sha256": sha256(PROTOCOL_PATH),
    "training_status": training_receipt["status"],
    "optimizer_steps": training_receipt["global_steps"],
    "adapter_files": len(actual_adapter_hashes),
    "baseline_manifest_sha256": sha256(baseline_manifest_path),
    "gpus": gpu_names,
    "evaluation_already_complete": not needs_inference,
    "will_retrain": False,
}}, flush=True)
'''

    run_eval = '''from pathlib import Path
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

def run_streamed(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()

result_key = "llama_3_1_8b_npo_lora_seed42_official_eval.json"
comparison_path = EVAL_DIR / "npo_comparison.json"
adapter_result_path = EVAL_DIR / result_key
if adapter_result_path.is_file() and comparison_path.is_file():
    print("Complete evaluation artifacts already exist; skipping inference and proceeding to independent verification.", flush=True)
else:
    command = [sys.executable, str(RUN_ROOT / "evaluate_npo_adapter.py"),
               "--model", "8b", "--seed", "42", "--adapter-dir", str(ADAPTER_DIR),
               "--output-dir", str(EVAL_DIR)]
    print({"evaluation": "START", "cmd": command, "time_utc": datetime.now(timezone.utc).isoformat()}, flush=True)
    code = run_streamed(command, RUN_ROOT / "evaluation_recovery.stdout.log")
    if code:
        raise RuntimeError(f"Adapter evaluation failed with exit code {{code}}; log retained in output archive.")

verify_command = [sys.executable, str(RUN_ROOT / "verify_npo_run.py"), "--model", "8b",
                  "--training-dir", str(TRAIN_DIR), "--eval-dir", str(EVAL_DIR),
                  "--protocol", str(PROTOCOL_PATH), "--objective-check", str(OBJECTIVE_PATH)]
verify_code = run_streamed(verify_command, RUN_ROOT / "npo_verification_recovery.stdout.log")
if verify_code:
    raise RuntimeError(f"Independent run verification failed with exit code {{verify_code}}.")

verification_path = EVAL_DIR / "npo_verification.json"
verification = json.loads(verification_path.read_text(encoding="utf-8"))
if verification.get("overall_status") != "PASS":
    raise RuntimeError("Independent run verifier did not record overall_status=PASS.")
provenance = {
    "recovery_kind": "evaluation_only_no_retraining",
    "input_archive": str(input_archive),
    "input_archive_sha256": archive_sha256,
    "run_root": RUN_ROOT.name,
    "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    "training_receipt_sha256": sha256(receipt_path),
    "adapter_files_sha256": actual_adapter_hashes,
    "verification_status": verification.get("overall_status"),
    "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
}
(RUN_ROOT / "evaluation_recovery_provenance.json").write_text(json.dumps(provenance, indent=2) + "\\n", encoding="utf-8")
output_archive = Path("/kaggle/working/tofu_8b_npo_eval_recovery_artifacts.zip")
if output_archive.exists():
    output_archive.unlink()
with zipfile.ZipFile(output_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RUN_ROOT.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(RUN_ROOT.parent))
print({"recovery": "PASS", "overall_verification": verification["overall_status"],
       "primary_result": verification.get("primary_result"),
       "research_diagnostic_gate_status": verification.get("research_diagnostic_gate_status"),
       "utility_gates": verification.get("utility_gates"),
       "output_archive": str(output_archive), "output_bytes": output_archive.stat().st_size}, flush=True)
'''

    notebook = {
        "cells": [
            md("""# TOFU Llama 3.1 8B NPO-LoRA evaluation recovery\n\nThis notebook only evaluates an already-trained adapter from the completed Kaggle v2 artifact. It will refuse to run unless the archive proves the exact preregistered protocol, 500 optimizer steps, seed 42, valid adapter hashes, and verified 8B baseline inputs. It does **not** train or resume training.\n\nBefore running: attach the completed kernel output ZIP as a private Kaggle dataset, enable Internet, and select GPU T4 x2. If the original run already contains complete adapter evaluation files, this notebook skips inference and independently verifies them. Otherwise it runs only the missing evaluation, streams progress to the notebook output, then runs the independent verifier."""),
            code("""import importlib.metadata as metadata\nimport importlib.util\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nrequirements = Path(\"/kaggle/working/requirements-npo-pilot.txt\")\nrequirements.write_text(\"transformers==5.22.1\\naccelerate==1.15.0\\npeft==0.21.2\\nscipy==1.16.3\\nrouge-score==0.1.2\\n\", encoding=\"utf-8\")\nexpected = {\"transformers\": \"5.22.1\", \"accelerate\": \"1.15.0\", \"peft\": \"0.21.2\", \"scipy\": \"1.16.3\", \"rouge-score\": \"0.1.2\"}\ndef versions_now():\n    found = {}\n    for package in expected:\n        try:\n            found[package] = metadata.version(package)\n        except metadata.PackageNotFoundError:\n            found[package] = None\n    return found\nif versions_now() != expected:\n    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"--quiet\", \"-r\", str(requirements)], check=True)\nif versions_now() != expected:\n    raise RuntimeError(f\"Pinned package check failed: {versions_now()}\")\ntry:\n    metadata.version(\"torchao\")\nexcept metadata.PackageNotFoundError:\n    pass\nelse:\n    subprocess.run([sys.executable, \"-m\", \"pip\", \"uninstall\", \"--yes\", \"torchao\"], check=True, capture_output=True, text=True)\n    try:\n        metadata.version(\"torchao\")\n    except metadata.PackageNotFoundError:\n        pass\n    else:\n        raise RuntimeError(\"torchao remained installed after removal\")\nif importlib.util.find_spec(\"torchao\") is not None:\n    raise RuntimeError(\"torchao module remains importable after removal\")\nimport torch\nfrom transformers import LlamaConfig, LlamaForCausalLM\nfrom peft import LoraConfig, get_peft_model\ntiny = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))\ntiny = get_peft_model(tiny, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"], bias=\"none\", task_type=\"CAUSAL_LM\"))\ntiny_ids = torch.tensor([[1, 2, 3, 4]])\ntiny_loss = tiny(input_ids=tiny_ids, labels=tiny_ids, use_cache=False).loss\ntiny_loss.backward()\ntiny_trainable = [parameter for parameter in tiny.parameters() if parameter.requires_grad]\ntiny_grad = sum(float(parameter.grad.float().abs().sum()) for parameter in tiny_trainable if parameter.grad is not None)\nif not torch.isfinite(tiny_loss) or not tiny_trainable or not torch.isfinite(torch.tensor(tiny_grad)) or tiny_grad <= 0:\n    raise RuntimeError(\"PEFT LoRA dispatch preflight failed\")\nprint({\"packages\": versions_now(), \"torch\": metadata.version(\"torch\"), \"torchao\": \"absent\", \"peft_lora_dispatch_smoke\": \"PASS\", \"tiny_loss\": float(tiny_loss.detach()), \"tiny_adapter_gradient_l1\": tiny_grad}, flush=True)"""),
            code(setup),
            code(run_eval),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "tofu_8b_npo_eval_recovery.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": "bunnoob2005/tofu-8b-npo-eval-recovery-2026-11-09",
        "title": "TOFU Llama 3.1 8B NPO LoRA Evaluation Recovery",
        "code_file": "tofu_8b_npo_eval_recovery.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (PACKAGE / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {PACKAGE / 'tofu_8b_npo_eval_recovery.ipynb'}")


if __name__ == "__main__":
    main()
