#!/usr/bin/env python3
"""Run source-bound exploratory GA/GD/NPO pilots from a verified V4 trainer."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


UPSTREAM_SHA256 = "0eb93e4448f6a9295f2a2d6be62eb89e3bc90202eff005c62d7829d70e78cd8d"
PATCHED_SHA256 = "8932d2059fff49f37980f55bf62528c9c65b8e0d3d1ff7ef34ef29ccda8abeae"

PATCHES = (
    (
        "def train(corpus_path, corpus_manifest_path, sft_run_manifest_path, target_ids_manifest_path, output_dir, method, epochs, lr, retain_weight, beta, max_length, seed, allow_failed_gate=False):\n    check = preflight(",
        "def train(corpus_path, corpus_manifest_path, sft_run_manifest_path, target_ids_manifest_path, output_dir, method, epochs, lr, retain_weight, beta, max_length, seed, allow_failed_gate=False):\n    # Capture the exact implementation file that this process imports and uses.\n    # The path/hash are written to the run manifest so a later bundle can bind\n    # adapters and predictions to an auditable source file.\n    training_script_path = Path(__file__).resolve()\n    training_script_sha256 = sha(training_script_path)\n    check = preflight(",
    ),
    (
        '"model_id": MODEL_ID, "language": "en", "seed": seed, "epochs": epochs, "optimizer_steps": steps,\n              "english_forget_rows": len(forget),',
        '"model_id": MODEL_ID, "language": "en", "seed": seed, "epochs": epochs, "optimizer_steps": steps,\n              "training_script_path": str(training_script_path),\n              "training_script_sha256": training_script_sha256,\n              "english_forget_rows": len(forget),',
    ),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_source_bound_trainer(upstream_path, output_path):
    upstream_path = Path(upstream_path)
    input_sha256 = sha256_file(upstream_path)
    if input_sha256 == PATCHED_SHA256:
        source = upstream_path.read_text(encoding="utf-8")
    elif input_sha256 == UPSTREAM_SHA256:
        source = upstream_path.read_text(encoding="utf-8")
        for old, new in PATCHES:
            if source.count(old) != 1:
                raise RuntimeError("Trainer provenance patch did not match exactly once.")
            source = source.replace(old, new, 1)
    else:
        raise RuntimeError("Attached V4 trainer checksum is unexpected; refusing to train.")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite trainer: {output_path}")
    output_path.write_text(source, encoding="utf-8")
    if sha256_file(output_path) != PATCHED_SHA256:
        output_path.unlink()
        raise RuntimeError("Patched trainer checksum mismatch; refusing to train.")
    return output_path


def load_module(path):
    spec = importlib.util.spec_from_file_location("source_bound_unlearn_lora", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-trainer", type=Path, required=True)
    parser.add_argument("--source-bound-trainer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--sft-run-manifest", type=Path, required=True)
    parser.add_argument("--target-ids-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"Refusing to reuse an existing run directory: {args.output_root}")
    trainer_path = build_source_bound_trainer(args.upstream_trainer, args.source_bound_trainer)
    trainer_sha256 = sha256_file(trainer_path)
    module = load_module(trainer_path)
    runner_sha256 = sha256_file(__file__)

    checks = {}
    for method in ("ga", "gd", "npo"):
        check = module.preflight(
            args.corpus,
            args.corpus_manifest,
            args.sft_run_manifest,
            args.target_ids_manifest,
            method,
        )
        if check.get("status") == "READY_TO_UNLEARN":
            raise RuntimeError("The expected exploratory gate state changed; inspect the new preflight.")
        if not module.allows_failed_gate_exploration(check, True):
            raise RuntimeError(f"{method.upper()} has blockers beyond the approved feasibility gate: {check}")
        checks[method] = check

    targets = json.loads(args.target_ids_manifest.read_text(encoding="utf-8"))
    if targets.get("baseline_gate_status") != "FAIL":
        raise RuntimeError("Target manifest no longer records the failed feasibility gate.")
    expected_forget_rows = len(targets.get("selected_source_qa_ids", []))
    if not expected_forget_rows:
        raise RuntimeError("The baseline target manifest has no eligible English forget IDs.")

    summaries = []
    for method in ("ga", "gd", "npo"):
        output_dir = args.output_root / method
        result = module.train(
            args.corpus,
            args.corpus_manifest,
            args.sft_run_manifest,
            args.target_ids_manifest,
            output_dir,
            method,
            args.epochs,
            args.lr,
            args.retain_weight,
            args.beta,
            args.max_length,
            args.seed,
            allow_failed_gate=True,
        )
        manifest_path = output_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_retain = checks[method].get("english_retain_rows", 0)
        if (
            result.get("training_script_sha256") != trainer_sha256
            or manifest.get("training_script_sha256") != trainer_sha256
            or manifest.get("status") != "UNLEARNING_COMPLETE"
            or manifest.get("run_mode") != "EXPLORATORY_LOW_POWER_AFTER_FAILED_GATE"
            or manifest.get("baseline_gate_status") != "FAIL"
            or manifest.get("english_forget_rows") != expected_forget_rows
            or manifest.get("english_retain_rows") != expected_retain
            or manifest.get("holdout_rows_used") != 0
            or manifest.get("optimizer_steps") != args.epochs * expected_forget_rows
        ):
            raise RuntimeError(f"{method.upper()} output failed post-run validation: {manifest}")

        adapter = Path(manifest["adapter_dir"])
        adapter_files = sorted(path for path in adapter.rglob("*") if path.is_file())
        if not any(path.suffix == ".safetensors" for path in adapter_files):
            raise RuntimeError(f"{method.upper()} adapter weights are missing.")
        manifest["runner_script_path"] = str(Path(__file__).resolve())
        manifest["runner_script_sha256"] = runner_sha256
        input_trainer_sha256 = sha256_file(args.upstream_trainer)
        manifest["source_input_trainer_sha256"] = input_trainer_sha256
        manifest["source_input_trainer_kind"] = (
            "verified_upstream" if input_trainer_sha256 == UPSTREAM_SHA256 else "already_source_bound"
        )
        manifest["verified_upstream_trainer_sha256"] = (
            UPSTREAM_SHA256 if input_trainer_sha256 == UPSTREAM_SHA256 else None
        )
        manifest["adapter_files_sha256"] = {
            str(path.relative_to(adapter)): sha256_file(path) for path in adapter_files
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "method": method.upper(),
                "status": manifest["status"],
                "run_mode": manifest["run_mode"],
                "optimizer_steps": manifest["optimizer_steps"],
                "forget_rows": manifest["english_forget_rows"],
                "retain_rows_used": manifest.get("english_retain_regularization_rows_used", 0),
                "holdout_rows_used": manifest["holdout_rows_used"],
                "trainer_sha256": manifest["training_script_sha256"],
                "adapter_weights": sum(path.suffix == ".safetensors" for path in adapter_files),
            }
        )

    print(json.dumps({"output_root": str(args.output_root), "runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
