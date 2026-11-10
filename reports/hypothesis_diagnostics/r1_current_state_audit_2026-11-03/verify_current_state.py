#!/usr/bin/env python3
"""Recheck the archived composition baseline and current v5.1 release gates.

This script uses local prediction files and synthetic fixtures only. It does not
download model weights, access the Hugging Face token, or launch training.
"""
from __future__ import annotations

import hashlib
import ast
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASELINE = ROOT / "reports/synthworlds_r1_composition_baseline_2026-11-02"
V4_PROJECT = ROOT / "data/synthworlds_sm_pilot_v4/kaggle/r1_retainfix_kaggle_upload"
V5 = ROOT / "experiments/synthworlds_r1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def main() -> int:
    evidence = OUT / "evidence"
    kaggle_evidence = evidence / "kaggle_api_setup_run"
    kaggle_evidence.mkdir(parents=True, exist_ok=True)

    # Preserve the raw read-only Kaggle output and the metadata/source actually
    # submitted. No credentials are present in these files.
    source_log = Path("/tmp/codex-kaggle-experiment-output/experiments.log")
    source_meta = Path("/tmp/codex-kaggle-experiments/kernel-metadata.json")
    source_notebook = Path("/tmp/codex-kaggle-experiments/experiments.ipynb")
    if all(path.is_file() for path in (source_log, source_meta, source_notebook)):
        for source in (source_log, source_meta, source_notebook):
            (kaggle_evidence / source.name).write_bytes(source.read_bytes())

    local_notebook = V5 / "kaggle/r1_multiseed_experiment.ipynb"
    (evidence / "v5.1_local_notebook.ipynb").write_bytes(local_notebook.read_bytes())

    # Integrity of the saved baseline package.
    index = json.loads((BASELINE / "artifact_index.json").read_text(encoding="utf-8"))
    indexed = []
    for entry in index["artifacts"]:
        path = BASELINE / entry["file"]
        observed = sha256(path) if path.is_file() else None
        observed_bytes = path.stat().st_size if path.is_file() else None
        indexed.append({
            "file": entry["file"],
            "exists": path.is_file(),
            "recorded_bytes": entry["bytes"],
            "observed_bytes": observed_bytes,
            "size_matches": observed_bytes == entry["bytes"],
            "recorded_sha256": entry["sha256"],
            "observed_sha256": observed,
            "matches": observed == entry["sha256"] and observed_bytes == entry["bytes"],
        })

    # Prove that the historical training corpus named in both run manifests is
    # available locally and has the same digest.
    corpus = ROOT / "data/synthworlds_sm_pilot_v3/sft_training/composition_corpus.jsonl"
    corpus_manifest = ROOT / "data/synthworlds_sm_pilot_v3/sft_training/composition_manifest.json"
    train_manifests = {
        "sft": json.loads((BASELINE / "composition_sft_run_manifest.json").read_text()),
        "retain_only": json.loads((BASELINE / "composition_retain_only_run_manifest.json").read_text()),
    }
    corpus_matches = all(row.get("corpus_sha256") == sha256(corpus) for row in train_manifests.values())
    corpus_manifest_matches = all(row.get("corpus_manifest_sha256") == sha256(corpus_manifest) for row in train_manifests.values())

    # Independently reproduce the historical score-only outputs from raw model
    # predictions. No checkpoints are loaded in score-only mode.
    archived_evaluator = V4_PROJECT / "scripts/evaluate_crosslingual_unlearning.py"
    eval_root = V4_PROJECT / "evaluation_dataset"
    derived_files = [
        "summary.json",
        "cross_language_contrasts.json",
        "diagnostic_report.md",
        "scored_predictions.jsonl",
        "case_metrics.jsonl",
    ]
    with tempfile.TemporaryDirectory(prefix="r1-baseline-audit-") as temporary:
        temp = Path(temporary)
        score = run([
            sys.executable, str(archived_evaluator),
            "--eval-root", str(eval_root),
            "--output-dir", str(temp),
            "--checkpoint", "pretrained=unused",
            "--checkpoint", "sft=unused",
            "--checkpoint", "retain_only=unused",
            "--predictions", str(BASELINE / "predictions.jsonl"),
            "--score-only", "--baseline-only",
        ])
        rescored = []
        for name in derived_files:
            original = BASELINE / name
            reproduced = temp / name
            old_hash = sha256(original) if original.is_file() else None
            new_hash = sha256(reproduced) if reproduced.is_file() else None
            rescored.append({"file": name, "matches": old_hash == new_hash and old_hash is not None,
                             "saved_sha256": old_hash, "reproduced_sha256": new_hash})

    # Repeat the score-only reconstruction for the corrected v4 exploratory
    # archive. Use its original three unlearning contrast checkpoints; no
    # checkpoint or model weight is loaded.
    corrected_v4 = ROOT / "reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/extracted/corrected_evaluation"
    v4_derived_files = ["scored_predictions.jsonl", "case_metrics.jsonl", "summary.json", "summary.csv",
                        "cross_language_contrasts.json", "cross_language_contrasts.csv", "diagnostic_report.md"]
    with tempfile.TemporaryDirectory(prefix="r1-corrected-v4-audit-") as temporary:
        v4_temp = Path(temporary)
        v4_command = [sys.executable, str(archived_evaluator), "--eval-root", str(eval_root),
                      "--output-dir", str(v4_temp), "--predictions", str(corrected_v4 / "predictions.jsonl"),
                      "--score-only"]
        for model in ("pretrained", "sft", "retain_only", "GA", "GD", "NPO"):
            v4_command.extend(["--checkpoint", f"{model}=unused"])
        for model in ("GA", "GD", "NPO"):
            v4_command.extend(["--contrast-model", model])
        v4_score = run(v4_command)
        v4_rescored = []
        for name in v4_derived_files:
            original, reproduced = corrected_v4 / name, v4_temp / name
            old_hash = sha256(original) if original.is_file() else None
            new_hash = sha256(reproduced) if reproduced.is_file() else None
            v4_rescored.append({"file": name, "matches": old_hash == new_hash and old_hash is not None,
                                "saved_sha256": old_hash, "reproduced_sha256": new_hash})

    # Verify the corrected result ZIP from its own embedded index, independent
    # of the unpacked copy and the saved verification summary.
    v4_bundle = ROOT / "reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/r1_composition_retainfix_verified_results_20261103_v3.zip"
    v4_saved_verification = json.loads((v4_bundle.parent / "LOCAL_VERIFICATION.json").read_text())
    with zipfile.ZipFile(v4_bundle) as archive:
        crc_bad_member = archive.testzip()
        archive_names = archive.namelist()
        archive_index = json.loads(archive.read("artifact_index.json"))
        v4_indexed = []
        for entry in archive_index["files"]:
            raw = archive.read(entry["file"])
            observed = hashlib.sha256(raw).hexdigest()
            v4_indexed.append({"file": entry["file"], "matches": len(raw) == entry["bytes"] and observed == entry["sha256"],
                               "observed_bytes": len(raw), "recorded_bytes": entry["bytes"],
                               "observed_sha256": observed, "recorded_sha256": entry["sha256"]})
    v4_bundle_sha = sha256(v4_bundle)
    v4_archive_checks = {
        "zip_crc": "PASS" if crc_bad_member is None else "FAIL",
        "crc_bad_member": crc_bad_member,
        "unique_paths": len(archive_names) == len(set(archive_names)),
        "zip_entries": len(archive_names),
        "embedded_index_entries": len(v4_indexed),
        "embedded_index_all_match": all(row["matches"] for row in v4_indexed),
        "bundle_sha256_matches_saved_verification": v4_bundle_sha == v4_saved_verification.get("bundle_sha256"),
        "bundle_sha256": v4_bundle_sha,
        "indexed_files": v4_indexed,
    }

    # Recompute the baseline gate directly from row-level case metrics.
    cases = read_jsonl(BASELINE / "case_metrics.jsonl")
    sft_rows = {(r["instance_id"], r["language"]): r for r in cases if r["model"] == "sft"}
    english = {iid for (iid, lang), row in sft_rows.items()
               if lang == "en" and row["split"] == "forget" and row["direct_correct"] and row["multi_correct"]}
    gate_by_language = {}
    for language in ("fr", "de", "hi", "th"):
        target = {iid for (iid, lang), row in sft_rows.items()
                  if lang == language and row["split"] == "forget" and row["direct_correct"] and row["multi_correct"]}
        paired = english & target
        groups = {sft_rows[(iid, language)]["bootstrap_cluster_id"] for iid in paired}
        gate_by_language[language] = {"bilingual_precompetent_cases": len(paired), "entity_groups": len(groups)}

    predictions = read_jsonl(BASELINE / "predictions.jsonl")
    scored = read_jsonl(BASELINE / "scored_predictions.jsonl")
    summary = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    sft_forget = {row["language"]: row for row in summary
                  if row["model"] == "sft" and row["split"] == "forget"}

    # Current notebook source and release gates.
    notebook = json.loads(local_notebook.read_text(encoding="utf-8"))
    code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    for cell in code_cells:
        compile(cell, str(local_notebook), "exec")
    executed_cells = sum(cell.get("execution_count") is not None for cell in notebook["cells"])
    output_count = sum(len(cell.get("outputs", [])) for cell in notebook["cells"])
    run_switches = re.findall(r"^\s*(RUN_[A-Z0-9_]+)\s*=\s*False\s*$", "\n".join(code_cells), re.M)

    assembly = run([sys.executable, str(V5 / "training_release_draft/scripts/assemble_training_release.py")])
    assembler_path = V5 / "training_release_draft/scripts/assemble_training_release.py"
    assembler_spec = importlib.util.spec_from_file_location("r1_assembler_audit", assembler_path)
    assembler_module = importlib.util.module_from_spec(assembler_spec)
    assert assembler_spec and assembler_spec.loader
    assembler_spec.loader.exec_module(assembler_module)
    translation_reviews = assembler_module.checked_translation_reviews()
    # Let the production builder pass its own atomic completeness check using
    # the measured 5/3 coverage, so it can also exercise the 0/44 chain gate.
    excluded = re.search(r"^ValueError: Translation screen excluded required atomic cases: (\{[^}]+\})", assembly.stderr, re.M)
    if not excluded:
        raise RuntimeError("Could not extract the assembler's measured atomic eligible counts")
    assembler_module.EXPECTED_ATOMIC_CASES_BY_SPLIT = ast.literal_eval(excluded.group(1))
    try:
        assembler_module.build_corpus(translation_reviews)
        composition_gate = {"blocked": False, "message": "Unexpectedly passed composition-chain completeness"}
    except ValueError as exc:
        composition_gate = {"blocked": "composition chains" in str(exc), "message": str(exc)}

    review_report = json.loads((V5 / "training_release_draft/translation_reviews_roundtrip_qwen14b/report.json").read_text())
    translation_review_counts = {
        lang: {
            "questions_reviewed": detail["qwen14b_reviewed_count"],
            "passes": detail["final_status_counts"].get("AI_ONLY_14B_ADJUDICATED_PASS", 0)
                     + detail["final_status_counts"].get("AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", 0),
            "flagged": detail["final_status_counts"].get("AI_ONLY_REVIEW_FLAGGED", 0),
        }
        for lang, detail in review_report["languages"].items()
    }
    sft_preflight = run([sys.executable, str(V5 / "scripts/train_sft_lora.py")])
    try:
        preflight_json = json.loads(sft_preflight.stdout)
    except json.JSONDecodeError:
        preflight_json = {"parse_error": sft_preflight.stdout}

    tests = run([sys.executable, str(V5 / "scripts/test_r1_pilot_pipeline.py")])
    compile_check = run([sys.executable, "-m", "py_compile",
                         str(V5 / "scripts/train_sft_lora.py"),
                         str(V5 / "scripts/unlearn_lora.py"),
                         str(V5 / "scripts/evaluate_pilot_three_probe.py"),
                         str(V5 / "training_release_draft/scripts/assemble_training_release.py"),
                         str(V5 / "scripts/test_r1_pilot_pipeline.py")])

    # Record the known stale manifest in the older v4 upload snapshot without
    # rewriting its history. An unrelated error is a genuine audit failure.
    legacy_v4_verify = run([sys.executable, str(ROOT / "data/synthworlds_sm_pilot_v4/scripts/verify_v3.py")])
    legacy_v4_text = legacy_v4_verify.stdout + legacy_v4_verify.stderr
    if legacy_v4_verify.returncode == 0:
        legacy_v4_status = "PASS"
    elif "AssertionError: kaggle/kernel_update/experiments.ipynb" in legacy_v4_text:
        legacy_v4_status = "KNOWN_STALE_HASH"
    else:
        legacy_v4_status = "UNEXPECTED_FAILURE"

    # Read-only status query for the most recently checked Kaggle kernel.
    kaggle_status = run(["kaggle", "kernels", "status", "bunnoob2005/experiments"])
    (kaggle_evidence / "kaggle_status.txt").write_text(
        "Command: kaggle kernels status bunnoob2005/experiments\n"
        f"Exit code: {kaggle_status.returncode}\n"
        f"{kaggle_status.stdout}{kaggle_status.stderr}", encoding="utf-8")

    remote_meta_path = kaggle_evidence / "kernel-metadata.json"
    remote_metadata = json.loads(remote_meta_path.read_text()) if remote_meta_path.is_file() else {}
    kaggle_log_path = kaggle_evidence / "experiments.log"
    log_records = json.loads(kaggle_log_path.read_text()) if kaggle_log_path.is_file() else []
    log_text = "".join(record.get("data", "") for record in log_records)
    kaggle_preflight_path = ROOT / "reports/hypothesis_diagnostics/r1_kaggle_preflight_v12_sourcebound_2026-11-03/preflight_summary.json"
    kaggle_preflight = json.loads(kaggle_preflight_path.read_text())
    runtime_sources = kaggle_preflight.get("dataset_sources", [])
    configured_sources = remote_metadata.get("dataset_sources", [])
    kaggle_v12_checks = {
        "status_complete": kaggle_preflight.get("kernel_status") == "COMPLETE",
        "preflight_only": kaggle_preflight.get("run_type") == "preflight_only",
        "gpu_available": kaggle_preflight.get("cuda_available") is True,
        "training_package_preflight_ready": kaggle_preflight.get("training_package_preflight", {}).get("training_packages_ready") is True,
        "hf_token_missing": kaggle_preflight.get("hf_token_present") is False,
        "combined_gate_blocked": kaggle_preflight.get("combined_training_gate") == "BLOCKED",
        "no_training_stages_started": all(kaggle_preflight.get(key) is False for key in (
            "sft_started", "retain_only_started", "unlearning_started", "evaluation_started")),
        "log_sha_matches_saved_summary": sha256(kaggle_log_path) == kaggle_preflight.get("log_sha256"),
        "notebook_sha_matches_saved_summary": sha256(kaggle_evidence / "experiments.ipynb") == kaggle_preflight.get("remote_notebook_sha256"),
    }
    log_checks = {
        "secret_lookup_failed": "HF_TOKEN secret retrieval failed" in log_text,
        "training_started_false": '"training_started": false' in log_text,
        "hf_token_present_false": '"hf_token_present": false' in log_text,
        "training_complete_record_absent": '"status": "TRAINING_COMPLETE"' not in log_text,
        "unlearning_complete_record_absent": '"status": "UNLEARNING_COMPLETE"' not in log_text,
        "composition_stage_disabled": "RUN_COMPOSITION_SFT=False" in log_text,
        "post_evaluation_disabled": "RUN_COMPOSITION_POST_EVALUATION=False" in log_text,
    }

    verification = {
        "audit_date": "2026-11-03",
        "claim": "score derivation and fail-closed gates are verified; the v5.1 model experiment has not run",
        "historical_composition_baseline": {
            "prediction_rows": len(predictions),
            "prediction_rows_by_model": dict(Counter(row["model"] for row in predictions)),
            "scored_prediction_rows": len(scored),
            "case_metric_rows": len(cases),
            "generation_failures": sum(bool(row.get("generation_failed")) for row in predictions),
            "output_invalid_flags": sum(bool(row.get("output_invalid")) for row in scored),
            "artifact_index_all_match": all(row["matches"] for row in indexed),
            "artifact_index_entries": indexed,
            "training_corpus_sha256": sha256(corpus),
            "training_corpus_matches_both_manifests": corpus_matches,
            "training_manifest_sha256": sha256(corpus_manifest),
            "training_manifest_matches_both_run_manifests": corpus_manifest_matches,
            "score_only_reproduction_status": "PASS" if score.returncode == 0 else "FAIL",
            "score_only_reproduction_exit_code": score.returncode,
            "score_only_reproduction_stdout": score.stdout,
            "score_only_reproduction_stderr": score.stderr,
            "derived_outputs": rescored,
            "sft_forget_accuracy": {
                lang: {"direct": row["direct_accuracy"], "multi_hop": row["multi_accuracy"]}
                for lang, row in sft_forget.items()
            },
            "bilingual_precompetence_by_language": gate_by_language,
            "gate_threshold": {"cases_per_language_min": 15, "entity_groups_per_language_min": 8},
            "unlearning_contrasts": len(json.loads((BASELINE / "cross_language_contrasts.json").read_text())),
        },
        "historical_corrected_v4_exploratory": {
            "archive_verification": v4_archive_checks,
            "score_only_reproduction_status": "PASS" if v4_score.returncode == 0 and all(row["matches"] for row in v4_rescored) else "FAIL",
            "score_only_reproduction_exit_code": v4_score.returncode,
            "score_only_reproduction_stdout": v4_score.stdout,
            "score_only_reproduction_stderr": v4_score.stderr,
            "derived_outputs": v4_rescored,
        },
        "current_v5_1": {
            "training_translation_review_status": review_report.get("status"),
            # `do_not_train` is a release decision; do not mislabel it as
            # authorization for AI-only review. Human review is separately
            # recorded and was not performed.
            "training_translation_do_not_train": review_report.get("do_not_train"),
            "training_translation_review_mode": review_report.get("review_mode"),
            "training_translation_human_reviewed": review_report.get("human_translation_reviewed"),
            "training_translation_rows_by_language": translation_review_counts,
            "sft_training_script_sha256": sha256(V5 / "scripts/train_sft_lora.py"),
            "release_assembler_sha256": sha256(assembler_path),
            "three_seed_notebook_sha256": sha256(local_notebook),
            "eligible_atomic_cases_by_split_from_assembler": excluded.group(1),
            "composition_chain_gate_probe": composition_gate,
            "local_notebook_cells": len(notebook["cells"]),
            "local_notebook_code_cells": len(code_cells),
            "local_notebook_code_cells_compile": True,
            "local_notebook_executed_cells": executed_cells,
            "local_notebook_outputs": output_count,
            "disabled_run_switches": run_switches,
            "assembler_exit_code_expected_nonzero": assembly.returncode,
            "assembler_stdout": assembly.stdout,
            "assembler_stderr": assembly.stderr,
            "sft_preflight_exit_code": sft_preflight.returncode,
            "sft_preflight": preflight_json,
            "fixture_suite_exit_code": tests.returncode,
            "fixture_suite_summary": tests.stdout + tests.stderr,
            "py_compile_exit_code": compile_check.returncode,
            "approved_corpus_exists": (V5 / "sft_training/corpus.jsonl").is_file(),
            "approved_manifest_exists": (V5 / "sft_training/manifest.json").is_file(),
        },
        "kaggle_kernel_metadata_snapshot": {
            "status_command_exit_code": kaggle_status.returncode,
            "status_command_output": kaggle_status.stdout + kaggle_status.stderr,
            "submitted_code_file": remote_metadata.get("code_file"),
            "dataset_sources": remote_metadata.get("dataset_sources", []),
            "metadata_sha256": sha256(remote_meta_path) if remote_meta_path.is_file() else None,
            "notebook_sha256": sha256(kaggle_evidence / "experiments.ipynb") if (kaggle_evidence / "experiments.ipynb").is_file() else None,
            "log_sha256": sha256(kaggle_log_path) if kaggle_log_path.is_file() else None,
            "log_checks": log_checks,
        },
        "kaggle_kernel_version_12_preflight": {
            "kernel": kaggle_preflight.get("kaggle_kernel"),
            "kernel_version": kaggle_preflight.get("kernel_version"),
            "status": kaggle_preflight.get("kernel_status"),
            "runtime_project_root": kaggle_preflight.get("project_root"),
            "runtime_dataset_sources": runtime_sources,
            "configured_dataset_sources_in_saved_metadata": configured_sources,
            "runtime_vs_configured_source_lists_differ": set(runtime_sources) != set(configured_sources),
            "gpu": kaggle_preflight.get("gpu"),
            "training_package_preflight": kaggle_preflight.get("training_package_preflight"),
            "combined_training_gate": kaggle_preflight.get("combined_training_gate"),
            "sft_started": kaggle_preflight.get("sft_started"),
            "retain_only_started": kaggle_preflight.get("retain_only_started"),
            "unlearning_started": kaggle_preflight.get("unlearning_started"),
            "evaluation_started": kaggle_preflight.get("evaluation_started"),
            "checks": kaggle_v12_checks,
        },
        "legacy_v4_upload_snapshot_verification": {
            "status": legacy_v4_status,
            "exit_code": legacy_v4_verify.returncode,
            "expected_failure_artifact": "kaggle/kernel_update/experiments.ipynb" if legacy_v4_status == "KNOWN_STALE_HASH" else None,
            "stdout": legacy_v4_verify.stdout,
            "stderr": legacy_v4_verify.stderr,
        },
    }

    (OUT / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks_ok = (
        all(row["matches"] for row in indexed)
        and corpus_matches and corpus_manifest_matches
        and score.returncode == 0 and all(row["matches"] for row in rescored)
        and v4_score.returncode == 0 and all(row["matches"] for row in v4_rescored)
        and v4_archive_checks["zip_crc"] == "PASS" and v4_archive_checks["unique_paths"]
        and v4_archive_checks["embedded_index_all_match"]
        and v4_archive_checks["bundle_sha256_matches_saved_verification"]
        and assembly.returncode != 0 and "Translation screen excluded required atomic cases" in assembly.stderr
        and composition_gate["blocked"] is True and "0/44" in composition_gate["message"]
        and review_report.get("status") == "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE"
        and review_report.get("do_not_train") is True
        and review_report.get("human_translation_reviewed") is False
        and preflight_json.get("status") == "BLOCKED"
        and tests.returncode == 0 and compile_check.returncode == 0
        and executed_cells == 0 and output_count == 0
        and all(log_checks.values())
        and all(kaggle_v12_checks.values())
        and legacy_v4_status in {"PASS", "KNOWN_STALE_HASH"}
    )
    print(json.dumps({"verification": "PASS" if checks_ok else "FAIL",
                      "baseline_artifacts_match": all(row["matches"] for row in indexed),
                      "baseline_derived_outputs_match": all(row["matches"] for row in rescored),
                      "corrected_v4_derived_outputs_match": all(row["matches"] for row in v4_rescored),
                      "corrected_v4_archive_verified": v4_archive_checks["zip_crc"] == "PASS" and v4_archive_checks["embedded_index_all_match"],
                      "baseline_gate": gate_by_language,
                      "v5_1_preflight": preflight_json.get("status"),
                      "fixture_suite_exit_code": tests.returncode,
                      "kaggle_v12_preflight_checks": kaggle_v12_checks,
                      "legacy_v4_upload_snapshot": legacy_v4_status,
                      "kaggle_log_checks": log_checks}, indent=2))
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
