#!/usr/bin/env python3
"""Independently verify a bounded TOFU NPO-LoRA Kaggle smoke archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path


UPSTREAM = "locuslab/open-unlearning@2d98027677f3468d52b6baa3fc4f2b9f2dc08820"
EXPECTED_PACKAGES = {
    "transformers": "5.22.1",
    "accelerate": "1.15.0",
    "peft": "0.21.2",
    "scipy": "1.16.3",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--kernel-log", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    problems: list[str] = []
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    expected_protocol_hash = sha256(protocol_bytes)
    archive_bytes = args.archive.read_bytes()
    archive_sha = sha256(archive_bytes)
    kernel_log = args.kernel_log.read_text(encoding="utf-8", errors="replace")

    try:
        with zipfile.ZipFile(args.archive) as archive:
            bad_member = archive.testzip()
            check(bad_member is None, f"ZIP CRC failure at {bad_member}", problems)
            names = archive.namelist()
            receipt_paths = [name for name in names if name.endswith("/npo_smoke/training_receipt.json")]
            check(len(receipt_paths) == 1, f"expected exactly one smoke receipt, found {len(receipt_paths)}", problems)
            if len(receipt_paths) != 1:
                receipt = {}
            else:
                receipt_path = receipt_paths[0]
                run_prefix = receipt_path.removesuffix("npo_smoke/training_receipt.json")
                receipt = json.loads(archive.read(receipt_path))

                trace_paths = [name for name in names if name == run_prefix + "npo_smoke/loss_trace.jsonl"]
                check(len(trace_paths) == 1, "smoke loss trace missing or duplicated", problems)
                trace = []
                if len(trace_paths) == 1:
                    trace = [json.loads(line) for line in archive.read(trace_paths[0]).decode("utf-8").splitlines() if line.strip()]
                check(len(trace) == 1, f"expected exactly one optimizer-step trace row, found {len(trace)}", problems)

                trainer_paths = [name for name in names if name == run_prefix + "npo_lora_train.py"]
                check(len(trainer_paths) == 1, "archived trainer source missing or duplicated", problems)
                if len(trainer_paths) == 1:
                    trainer_sha = sha256(archive.read(trainer_paths[0]))
                    check(receipt.get("training_script_sha256") == trainer_sha, "training-script SHA-256 does not match archived source", problems)

                baseline_paths = [name for name in names if name.endswith("/tofu_official_eval/official_eval_verification.json")]
                check(len(baseline_paths) == 1, "8B baseline verifier receipt missing or duplicated", problems)
                if len(baseline_paths) == 1:
                    baseline = json.loads(archive.read(baseline_paths[0]))
                    models = baseline.get("models", {})
                    check(models.get("llama_3_1_8b_full", {}).get("status") == "PASS",
                          "8B full baseline verification failed", problems)
                    check(models.get("llama_3_1_8b_retain90", {}).get("status") == "PASS",
                          "8B retain90 oracle baseline verification failed", problems)
                    check(baseline.get("overall_status") == "PARTIAL"
                          and baseline.get("missing_models") == ["llama_3_2_3b_full", "llama_3_2_3b_retain90"],
                          "8B run should report only the intentionally unattached 3B baselines as missing", problems)

                source_receipt_paths = [name for name in names if name.endswith("/baseline_inputs/source_verification_receipt.json")]
                check(len(source_receipt_paths) == 1, "original baseline source-verification receipt missing or duplicated", problems)
                if len(source_receipt_paths) == 1:
                    source_receipt = json.loads(archive.read(source_receipt_paths[0]))
                    check(source_receipt.get("overall_status") == "PASS",
                          "original independently verified baseline source receipt failed", problems)
                    source_models = source_receipt.get("models", {})
                    for key in ("llama_3_1_8b_full", "llama_3_1_8b_retain90"):
                        check(source_models.get(key, {}).get("status") == "PASS",
                              f"original source verification failed for {key}", problems)
    except (OSError, zipfile.BadZipFile) as error:
        problems.append(f"cannot read artifact ZIP: {type(error).__name__}: {error}")
        names = []
        receipt = {}
        trace = []

    objective = receipt.get("objective_equivalence", {})
    runtime = receipt.get("runtime", {})
    packages = receipt.get("package_versions", {})
    dataset = receipt.get("dataset", {})
    protocol_dataset = protocol.get("dataset", {})
    training = protocol_dataset.get("training", {})
    update = float(receipt.get("smoke_max_abs_lora_parameter_update", float("nan")))

    check(protocol.get("status") == "PREREGISTERED_BEFORE_GPU_RUN", "protocol is not marked preregistered before GPU run", problems)
    check(receipt.get("protocol_sha256") == expected_protocol_hash, "smoke receipt protocol hash mismatch", problems)
    check(receipt.get("status") == "SMOKE_PASS", "bounded real-data smoke did not pass", problems)
    check(receipt.get("run_type") == "bounded_real_data_smoke", "receipt is not the bounded real-data smoke", problems)
    check(receipt.get("global_steps") == 1 and receipt.get("optimizer_steps_with_finite_loss_and_gradient") == 1,
          "smoke did not perform exactly one finite optimizer step", problems)
    check(math.isfinite(update) and update > 0, "smoke did not record a finite nonzero LoRA parameter update", problems)
    check(receipt.get("training_started") is True and receipt.get("adapter_saved") is False,
          "smoke training/save flags are inconsistent", problems)
    check(receipt.get("seed") == protocol.get("seed"), "smoke seed differs from preregistration", problems)
    check(receipt.get("model", {}).get("repo_id") == protocol.get("base_checkpoint", {}).get("repo_id")
          and receipt.get("model", {}).get("revision") == protocol.get("base_checkpoint", {}).get("revision"),
          "smoke model revision differs from preregistration", problems)
    check(dataset.get("repo") == protocol_dataset.get("repo_id") and dataset.get("revision") == protocol_dataset.get("revision"),
          "smoke dataset revision differs from preregistration", problems)
    check(dataset.get("forget10_rows") == training.get("forget_file", {}).get("rows")
          and dataset.get("forget10_sha256") == training.get("forget_file", {}).get("sha256"),
          "forget training data count/hash mismatch", problems)
    check(dataset.get("retain90_rows") == training.get("retain_file", {}).get("rows")
          and dataset.get("retain90_sha256") == training.get("retain_file", {}).get("sha256"),
          "retain training data count/hash mismatch", problems)
    check(objective.get("status") == "PASS" and objective.get("upstream") == UPSTREAM,
          "pinned NPO objective/gradient equivalence receipt failed", problems)
    check(objective.get("protocol_sha256") == expected_protocol_hash, "objective check protocol hash mismatch", problems)
    check(float(objective.get("absolute_loss_difference", float("inf"))) <= 1e-6
          and float(objective.get("maximum_parameter_gradient_difference", float("inf"))) <= 1e-6
          and float(objective.get("retain_nll_loss_difference", float("inf"))) <= 1e-6
          and float(objective.get("combined_vs_sequential_backward_max_gradient_difference", float("inf"))) <= 1e-6,
          "objective loss/gradient numerical equivalence exceeds tolerance", problems)
    check(packages.get("torchao") is None, "unused torchao package was not absent from the training runtime", problems)
    for package, version in EXPECTED_PACKAGES.items():
        check(packages.get(package) == version, f"unexpected package version for {package}", problems)
    check(runtime.get("cpu_or_disk_offload") is False, "CPU/disk model offload was used or not ruled out", problems)
    check(receipt.get("gpu_names") == ["Tesla T4", "Tesla T4"], "smoke did not run on exactly two Tesla T4 GPUs", problems)
    check(runtime.get("dtype") == "float16" and runtime.get("attention") == "sdpa", "runtime dtype/attention differs from protocol", problems)
    tokenization = receipt.get("tokenization", {})
    check(tokenization.get("truncated_examples") == 0 and tokenization.get("prompt_answer_prefix_check") == "PASS for every sequence",
          "smoke tokenization failed or truncated examples", problems)
    check(int(tokenization.get("max_length_observed", protocol["adaptation"]["max_length"] + 1)) <= protocol["adaptation"]["max_length"],
          "smoke sequence exceeded the locked maximum length", problems)
    check(len(trace) == 1 and trace[0].get("global_step") == 1, "loss trace does not identify the single optimizer step", problems)
    if len(trace) == 1:
        for field in ("forget_npo_loss", "retain_token_nll", "combined_loss", "gradient_norm"):
            check(math.isfinite(float(trace[0].get(field, float("nan")))), f"loss trace has invalid {field}", problems)
    check(not any("/npo_train/" in name or "/adapter/" in name for name in names),
          "smoke archive contains full training or a saved research adapter", problems)
    check("peft_lora_dispatch_smoke" in kernel_log and "'PASS'" in kernel_log and "'torchao': 'absent'" in kernel_log,
          "pre-weight tiny-model PEFT dispatch preflight is not recorded as passing", problems)
    check("'full_training_enabled': False" in kernel_log and "'research_scores_created': False" in kernel_log,
          "smoke-only mode or no-research-scores assertion is not recorded", problems)

    result = {
        "status": "PASS_BOUNDED_SMOKE_ONLY" if not problems else "FAIL",
        "artifact_zip_crc": "PASS" if not problems else "SEE_PROBLEMS",
        "artifact_zip_sha256": archive_sha,
        "artifact_zip_bytes": len(archive_bytes),
        "protocol_sha256": expected_protocol_hash,
        "model": receipt.get("model"),
        "gpu_names": receipt.get("gpu_names"),
        "optimizer_steps": receipt.get("global_steps"),
        "finite_update_steps": receipt.get("optimizer_steps_with_finite_loss_and_gradient"),
        "max_abs_lora_update": update if math.isfinite(update) else None,
        "smoke_runtime_seconds": receipt.get("runtime_seconds"),
        "truncated_examples": tokenization.get("truncated_examples"),
        "cpu_or_disk_offload": runtime.get("cpu_or_disk_offload"),
        "full_training_enabled": False,
        "research_result_created": False,
        "problems": problems,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Bounded TOFU NPO-LoRA smoke verification",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- Model: `{receipt.get('model', {}).get('label')}` at the preregistered revision.",
        f"- Hardware: `{', '.join(receipt.get('gpu_names', []))}`; CPU/disk offload: `{runtime.get('cpu_or_disk_offload')}`.",
        f"- Real-data optimizer steps: `{receipt.get('global_steps')}`; finite steps: `{receipt.get('optimizer_steps_with_finite_loss_and_gradient')}`.",
        f"- Largest absolute LoRA update: `{update:.9g}`; truncation count: `{tokenization.get('truncated_examples')}`.",
        f"- Artifact ZIP CRC: `{result['artifact_zip_crc']}`; SHA-256: `{archive_sha}`.",
        "- Full pilot enabled: `False`; research adapter/scores created: `False`.",
        "- Interpretation: this is an implementation/runtime smoke, not a hypothesis result.",
    ]
    if problems:
        lines.extend(["", "## Problems", ""] + [f"- {problem}" for problem in problems])
    (args.output_dir / "smoke_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
