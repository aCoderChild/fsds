#!/usr/bin/env python3
"""Run multilingual evaluation and bind predictions to their exact inputs."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_EVALUATOR_SHA256 = "038298efc274c331bc33e141afc68ba629ff705b24db57d6138fcaf26b6abb70"
LANGUAGES = ("en", "fr", "de", "hi", "th")
MODELS = ("pretrained", "sft", "retain_only", "GA", "GD", "NPO")
EXPECTED_PREDICTIONS_PER_MODEL = 1290


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--sft-run-manifest", type=Path, required=True)
    parser.add_argument("--retain-run-manifest", type=Path, required=True)
    parser.add_argument("--unlearning-root", type=Path, required=True)
    parser.add_argument("--target-ids-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse evaluation output: {args.output_dir}")
    if sha256_file(args.evaluator) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("Evaluator source checksum is unexpected; refusing to run.")

    eval_manifest_path = args.eval_root / "manifest.json"
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if (
        eval_manifest.get("evaluation_ready") is not True
        or eval_manifest.get("ai_review_status") != "AI_ONLY_REVIEWED_COMPLETE"
        or eval_manifest.get("human_review_status") != "NOT_PERFORMED"
    ):
        raise RuntimeError("Evaluation set is incomplete or has inconsistent review disclosure.")
    prompt_paths = [args.eval_root / "prompts/en.jsonl"] + [
        args.eval_root / f"translations/ai_reviewed/{language}.jsonl"
        for language in LANGUAGES[1:]
    ]
    prompt_counts = {path.stem: len(read_jsonl(path)) for path in prompt_paths}
    if prompt_counts != {"en": 258, "fr": 258, "de": 258, "hi": 258, "th": 258}:
        raise RuntimeError(f"Unexpected prompt coverage: {prompt_counts}")

    corpus_sha256 = sha256_file(args.corpus)
    target_manifest = json.loads(args.target_ids_manifest.read_text(encoding="utf-8"))
    sft_manifest = json.loads(args.sft_run_manifest.read_text(encoding="utf-8"))
    retain_manifest = json.loads(args.retain_run_manifest.read_text(encoding="utf-8"))
    if sft_manifest.get("status") != "TRAINING_COMPLETE" or retain_manifest.get("status") != "TRAINING_COMPLETE":
        raise RuntimeError("SFT and retain-only checkpoints must both be complete.")
    if sft_manifest.get("corpus_sha256") != corpus_sha256 or retain_manifest.get("corpus_sha256") != corpus_sha256:
        raise RuntimeError("SFT/reference manifests are not bound to the evaluation training corpus.")
    if target_manifest.get("source_corpus_sha256") != corpus_sha256:
        raise RuntimeError("Unlearning target manifest is tied to another corpus.")

    unlearning_manifests = {}
    for method in ("ga", "gd", "npo"):
        manifest_path = args.unlearning_root / method / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        script = Path(manifest["training_script_path"])
        adapter = Path(manifest["adapter_dir"])
        actual_adapter_hashes = {
            str(path.relative_to(adapter)): sha256_file(path)
            for path in sorted(adapter.rglob("*"))
            if path.is_file()
        }
        if (
            manifest.get("status") != "UNLEARNING_COMPLETE"
            or manifest.get("run_mode") != "EXPLORATORY_LOW_POWER_AFTER_FAILED_GATE"
            or manifest.get("baseline_gate_status") != "FAIL"
            or manifest.get("holdout_rows_used") != 0
            or manifest.get("source_corpus_sha256") != corpus_sha256
            or manifest.get("target_ids_manifest_sha256") != sha256_file(args.target_ids_manifest)
            or manifest.get("training_script_sha256") != sha256_file(script)
            or manifest.get("adapter_files_sha256") != actual_adapter_hashes
        ):
            raise RuntimeError(f"{method.upper()} run provenance or adapter integrity check failed.")
        unlearning_manifests[method] = manifest

    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_provenance = {
        "status": "EVALUATION_RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_path": str(args.evaluator.resolve()),
        "evaluator_sha256": sha256_file(args.evaluator),
        "eval_root": str(args.eval_root.resolve()),
        "eval_manifest_sha256": sha256_file(eval_manifest_path),
        "prompt_files_sha256": {str(path.relative_to(args.eval_root)): sha256_file(path) for path in prompt_paths},
        "prompt_rows_per_language": prompt_counts,
        "translation_review_mode": "AI_ONLY",
        "human_translation_reviewed": False,
        "corpus_sha256": corpus_sha256,
        "sft_run_manifest_sha256": sha256_file(args.sft_run_manifest),
        "retain_run_manifest_sha256": sha256_file(args.retain_run_manifest),
        "unlearning_run_manifest_sha256": {
            method: sha256_file(args.unlearning_root / method / "run_manifest.json")
            for method in ("ga", "gd", "npo")
        },
        "target_ids_manifest_sha256": sha256_file(args.target_ids_manifest),
        "sft_adapter_files_sha256": {
            str(path.relative_to(Path(sft_manifest["adapter_dir"]))): sha256_file(path)
            for path in sorted(Path(sft_manifest["adapter_dir"]).rglob("*"))
            if path.is_file()
        },
        "retain_adapter_files_sha256": {
            str(path.relative_to(Path(retain_manifest["adapter_dir"]))): sha256_file(path)
            for path in sorted(Path(retain_manifest["adapter_dir"]).rglob("*"))
            if path.is_file()
        },
        "checkpoints": {
            "pretrained": "meta-llama/Llama-3.2-1B-Instruct base weights",
            "sft": sft_manifest["adapter_dir"],
            "retain_only": retain_manifest["adapter_dir"],
            **{method.upper(): unlearning_manifests[method]["adapter_dir"] for method in ("ga", "gd", "npo")},
        },
        "batch_size": args.batch_size,
        "bootstrap_resamples": args.bootstrap,
    }
    provenance_path = args.output_dir / "evaluation_provenance.json"
    provenance_path.write_text(json.dumps(input_provenance, indent=2) + "\n", encoding="utf-8")

    checkpoints = [
        ("pretrained", "base"),
        ("sft", sft_manifest["adapter_dir"]),
        ("retain_only", retain_manifest["adapter_dir"]),
        *((method.upper(), unlearning_manifests[method]["adapter_dir"]) for method in ("ga", "gd", "npo")),
    ]
    command = [
        sys.executable,
        str(args.evaluator),
        "--eval-root",
        str(args.eval_root),
        "--output-dir",
        str(args.output_dir),
        "--batch-size",
        str(args.batch_size),
        "--bootstrap",
        str(args.bootstrap),
        "--contrast-model",
        "GA",
        "--contrast-model",
        "GD",
        "--contrast-model",
        "NPO",
    ]
    for name, checkpoint in checkpoints:
        command.extend(("--checkpoint", f"{name}={checkpoint}"))
    subprocess.run(command, check=True)

    predictions_path = args.output_dir / "predictions.jsonl"
    predictions = read_jsonl(predictions_path)
    counts = Counter(row.get("model") for row in predictions)
    seen = set()
    for row in predictions:
        key = (row.get("model"), row.get("instance_id"), row.get("probe"), row.get("language"))
        if key in seen:
            raise RuntimeError(f"Duplicate prediction key: {key}")
        seen.add(key)
    expected_count = len(MODELS) * EXPECTED_PREDICTIONS_PER_MODEL
    if len(predictions) != expected_count or any(counts.get(model) != EXPECTED_PREDICTIONS_PER_MODEL for model in MODELS):
        raise RuntimeError(f"Incomplete prediction coverage: {len(predictions)} total, {dict(counts)}")

    input_provenance.update(
        {
            "status": "EVALUATION_COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "prediction_rows": len(predictions),
            "prediction_rows_by_model": dict(counts),
            "generation_failures": sum(bool(row.get("generation_failed")) for row in predictions),
            "predictions_sha256": sha256_file(predictions_path),
            "scored_predictions_sha256": sha256_file(args.output_dir / "scored_predictions.jsonl"),
            "case_metrics_sha256": sha256_file(args.output_dir / "case_metrics.jsonl"),
            "summary_json_sha256": sha256_file(args.output_dir / "summary.json"),
            "contrasts_json_sha256": sha256_file(args.output_dir / "cross_language_contrasts.json"),
            "diagnostic_report_sha256": sha256_file(args.output_dir / "diagnostic_report.md"),
            "gold_labels_sha256": sha256_file(args.eval_root / "labels/gold_labels.jsonl"),
        }
    )
    provenance_path.write_text(json.dumps(input_provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EVALUATION_COMPLETE", "rows": len(predictions), "per_model": dict(counts), "generation_failures": input_provenance["generation_failures"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
