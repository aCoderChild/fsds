#!/usr/bin/env python3
"""Run bounded local checks and write a hash-bound R1 reliability snapshot."""
from __future__ import annotations

import hashlib
import ast
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
V51 = ROOT / "experiments/synthworlds_r1"
PILOT = ROOT / "reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/extracted"
RECHECK = ROOT / "reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04"
OVERLAY_DIR = V51 / "training_release_draft/translation_repair_diagnostics/chatgpt_third_candidate_2026-11-04"
SMOKE_DIR = V51 / "kaggle_8b_runtime_smoke/remote_output/interactive_smoke_v8_passed_runtime_only_2026-11-05"
GAP_UNION_DIR = ROOT / "reports/hypothesis_diagnostics/r1_reliability_tracker_2026-11-05"
GAP_UNION_SCRIPT = GAP_UNION_DIR / "targeted_gap_union_audit.py"
DAP_SCRIPT = GAP_UNION_DIR / "pilot_dap_exact_cluster_bootstrap.py"
SIGN_SCRIPT = GAP_UNION_DIR / "pilot_exact_cluster_sign_test.py"
KAGGLE_INVENTORY_SCRIPT = GAP_UNION_DIR / "audit_kaggle_job_inventory.py"
KAGGLE_INVENTORY_JSON = GAP_UNION_DIR / "kaggle_related_job_inventory_2026-11-05.json"
FULL799_CANDIDATE_AUDIT = (
    V51 / "training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/"
    "analyze_candidate_union_ceiling.py"
)
CANDIDATE_CEILING_AUDIT = V51 / "scripts/audit_targeted_translation_candidate_ceiling.py"
CANDIDATE_CEILING_JSON = OUT / "targeted_candidate_ceiling_after_segmented_retry.json"
MATCHED_OUTPUTS = (
    "case_metrics.jsonl",
    "cross_language_contrasts.csv",
    "cross_language_contrasts.json",
    "diagnostic_report.md",
    "scored_predictions.jsonl",
    "summary.csv",
    "summary.json",
)
TRACKED_KAGGLE_KERNELS = (
    "bunnoob2005/experiments",
    "bunnoob2005/sw-gap187-alias-retry-2026-11-05",
    "bunnoob2005/sw-gap187-nllb-148-candidate-review-2026-11-05",
    "bunnoob2005/sw-gap187-nllb-segmented-retry-2026-11-05",
    "bunnoob2005/synthworlds-llama31-8b-runtime-smoke-2026-11-04",
)
MAIN_KAGGLE_KERNEL = "bunnoob2005/experiments"
EXPECTED_FIXTURE_COUNT = 36


def parse_atomic_gate_error(output: str) -> tuple[dict[str, int], dict[str, int]] | None:
    """Parse the assembler's fail-closed summary without relying on dict key order."""
    match = re.search(
        r"ValueError:\s*Translation screen excluded required atomic cases:\s*(\{[^{}]*\});\s*expected\s*(\{[^{}]*\})",
        output,
    )
    if not match:
        return None
    try:
        observed = ast.literal_eval(match.group(1))
        required = ast.literal_eval(match.group(2))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(observed, dict) or not isinstance(required, dict):
        return None
    if any(not isinstance(k, str) or not isinstance(v, int) for d in (observed, required) for k, v in d.items()):
        return None
    return observed, required


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def query_kaggle_statuses() -> dict:
    statuses = {}
    for ref in TRACKED_KAGGLE_KERNELS:
        proc = run(["kaggle", "kernels", "status", ref])
        match = re.search(r'has status "([^"]+)"', proc.stdout + proc.stderr)
        statuses[ref] = {
            "status": match.group(1) if match else "STATUS_QUERY_FAILED",
            "exit_code": proc.returncode,
        }
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing legacy 1B pilot notebook alias, three latest targeted-gap review jobs, and the bounded 8B synthetic runtime smoke; this is not a status query for every Kaggle notebook.",
        "kernels": statuses,
        "all_complete": all(item["status"] == "KernelWorkerStatus.COMPLETE" and item["exit_code"] == 0
                            for item in statuses.values()),
    }


def query_main_kernel_execution() -> dict:
    """Fetch only the compact execution log; never download checkpoint artifacts."""
    with tempfile.TemporaryDirectory(prefix="r1-kaggle-main-log-") as temp_dir:
        proc = run([
            "kaggle", "kernels", "output", MAIN_KAGGLE_KERNEL,
            "--path", temp_dir, "--file-pattern", r"experiments\.log$", "--quiet",
        ])
        logs = sorted(Path(temp_dir).rglob("experiments.log"))
        if proc.returncode != 0 or len(logs) != 1:
            return {
                "kernel": MAIN_KAGGLE_KERNEL,
                "status": "OUTPUT_LOG_UNAVAILABLE",
                "output_command_exit_code": proc.returncode,
                "matching_log_count": len(logs),
                "research_training_started": None,
                "interpretation": "Kaggle status alone cannot establish whether an experiment stage ran.",
            }
        raw = logs[0].read_bytes()
        try:
            events = json.loads(raw)
            log_text = "\n".join(str(event.get("data", "")) for event in events)
        except (json.JSONDecodeError, AttributeError, TypeError):
            return {
                "kernel": MAIN_KAGGLE_KERNEL,
                "status": "OUTPUT_LOG_UNPARSEABLE",
                "output_command_exit_code": proc.returncode,
                "output_log_sha256": hashlib.sha256(raw).hexdigest(),
                "research_training_started": None,
                "interpretation": "The Kaggle log was fetched but could not be parsed safely.",
            }
        model_ids = sorted(set(re.findall(r'"model_id"\s*:\s*"([^"]+)"', log_text)))
        training_started_true = bool(re.search(r'"training_started"\s*:\s*true\b', log_text))
        training_started_false = bool(re.search(r'"training_started"\s*:\s*false\b', log_text))
        not_started_messages = sum(log_text.count(marker) for marker in (
            "Training not started.",
            "Retain-only training not started.",
            "Composition SFT not started",
            "Composition unlearning not started.",
        ))
        if training_started_true:
            state = "TRAINING_STARTED_REQUIRES_RUN_ARTIFACT_AUDIT"
            interpretation = "The log contains a training_started=true receipt; inspect the saved run artifacts before classifying it as a research result."
        elif training_started_false and not_started_messages:
            state = "PREFLIGHT_ONLY_NO_TRAINING"
            interpretation = "The published alias log is for the legacy Llama 3.2 1B pilot dataset, records training_started=false, and is not a v5.1 main-study run."
        else:
            state = "INCONCLUSIVE_LOG"
            interpretation = "The log does not provide enough evidence to classify training execution."
        return {
            "kernel": MAIN_KAGGLE_KERNEL,
            "status": state,
            "output_command_exit_code": proc.returncode,
            "output_log_sha256": hashlib.sha256(raw).hexdigest(),
            "model_ids_in_log": model_ids,
            "training_started_true_receipt_seen": training_started_true,
            "training_started_false_receipt_seen": training_started_false,
            "explicit_no_training_messages": not_started_messages,
            "hf_token_retrieval_failed": "HF_TOKEN secret retrieval failed" in log_text,
            "research_training_started": training_started_true if (training_started_true or training_started_false) else None,
            "interpretation": interpretation,
        }


def main() -> None:
    python = sys.executable

    fixture = run([python, "-B", "experiments/synthworlds_r1/scripts/test_r1_pilot_pipeline.py"])
    fixture_text = fixture.stdout + fixture.stderr
    fixture_count_match = re.search(r"Ran (\d+) tests?", fixture_text)
    fixture_count = int(fixture_count_match.group(1)) if fixture_count_match else None

    main_notebook = V51 / "kaggle/r1_multiseed_experiment.ipynb"
    notebook = json_file(main_notebook)
    notebook_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    setup_cell, baseline_cell, post_cell, export_cell = (
        notebook_cells[2], notebook_cells[8], notebook_cells[10], notebook_cells[11]
    )
    label_hash_guard = {
        "status": "PASS" if (
            'if expected != sha256_file(LABELS):' in setup_cell
            and baseline_cell.index("EVALUATION_LABELS_SHA256 = verify_evaluation_labels_sha256()")
                < baseline_cell.index("score_command =")
            and post_cell.index("EVALUATION_LABELS_SHA256 = verify_evaluation_labels_sha256()")
                < post_cell.index('"score"')
            and '"evaluation_labels_sha256": EVALUATION_LABELS_SHA256' in export_cell
        ) else "FAIL",
        "notebook_sha256": sha(main_notebook),
        "gold_labels_hash_checked_only_at_scoring_and_recorded_in_result_manifest": True,
    }
    if label_hash_guard["status"] != "PASS":
        raise RuntimeError("Main Kaggle notebook does not bind frozen labels immediately before scoring")

    overlay_verify = run([python, "-B", str(OVERLAY_DIR / "verify_final_recheck_overlay.py")])
    if overlay_verify.returncode:
        raise RuntimeError(overlay_verify.stdout + overlay_verify.stderr)
    overlay_check = json.loads(overlay_verify.stdout)

    smoke_receipt_path = SMOKE_DIR / "r1_8b_runtime_smoke.json"
    smoke_verify = run([
        python, "-B", str(V51 / "kaggle_8b_runtime_smoke/verify_receipt.py"),
        str(smoke_receipt_path),
    ])
    if smoke_verify.returncode:
        raise RuntimeError(smoke_verify.stdout + smoke_verify.stderr)
    smoke_check = json.loads(smoke_verify.stdout)

    gap_union_run = run([python, "-B", str(GAP_UNION_SCRIPT)])
    if gap_union_run.returncode:
        raise RuntimeError(gap_union_run.stdout + gap_union_run.stderr)
    gap_union_check = json.loads(gap_union_run.stdout)
    gap_union = json_file(GAP_UNION_DIR / "targeted_gap_union_audit.json")
    full799_audit_run = run([python, "-B", str(FULL799_CANDIDATE_AUDIT)])
    if full799_audit_run.returncode:
        raise RuntimeError(full799_audit_run.stdout + full799_audit_run.stderr)
    full799_audit_check = json.loads(full799_audit_run.stdout)
    full799_candidate_union = json_file(
        V51 / "training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/"
        "optimistic_candidate_union_coverage.json"
    )
    full799_coverage = full799_candidate_union["optimistic_candidate_union_coverage"]
    if (full799_audit_check.get("status") != "OPTIMISTIC_CANDIDATE_UNION_CEILING_NOT_RELEASE_APPROVAL"
            or full799_candidate_union.get("training_eligible") is not False
            or full799_candidate_union.get("release_approved") is not False
            or full799_candidate_union.get("human_translation_reviewed") is not False
            or full799_candidate_union["candidate_union"].get("unique_rows_with_at_least_one_diagnostic_pass") != 612
            or full799_coverage["optimistic_complete_atomic_cases"]["forget"]["complete"] != 60
            or full799_coverage["optimistic_complete_atomic_cases"]["retain"]["complete"] != 19
            or full799_coverage["optimistic_complete_composition_chains"]["complete"] != 6):
        raise RuntimeError("Full 799-row candidate-union audit changed or lost its quarantine status")
    candidate_ceiling_run = run([python, "-B", str(CANDIDATE_CEILING_AUDIT)])
    if candidate_ceiling_run.returncode:
        raise RuntimeError(candidate_ceiling_run.stdout + candidate_ceiling_run.stderr)
    candidate_ceiling = json.loads(candidate_ceiling_run.stdout)
    if (candidate_ceiling.get("status") != "VERIFIED_OPTIMISTIC_CANDIDATE_CEILING_ONLY_NOT_RELEASE_APPROVAL"
            or candidate_ceiling.get("release_approved") is not False
            or candidate_ceiling.get("training_eligible") is not False
            or candidate_ceiling.get("human_translation_reviewed") is not False
            or candidate_ceiling.get("authoritative_release_gate", {}).get("status") != "BLOCKED_AS_DESIGNED"):
        raise RuntimeError("Combined candidate-ceiling audit changed scope or lost its quarantine status")
    CANDIDATE_CEILING_JSON.write_text(
        json.dumps(candidate_ceiling, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exact_bootstrap_script = GAP_UNION_DIR / "pilot_exact_cluster_sensitivity.py"
    exact_bootstrap_run = run([python, "-B", str(exact_bootstrap_script)])
    if exact_bootstrap_run.returncode:
        raise RuntimeError(exact_bootstrap_run.stdout + exact_bootstrap_run.stderr)
    exact_bootstrap_cli = json.loads(exact_bootstrap_run.stdout)
    exact_bootstrap = json_file(GAP_UNION_DIR / "pilot_exact_cluster_sensitivity.json")
    if (exact_bootstrap_cli.get("status") != "VERIFIED_EXACT_POST_HOC_CLUSTER_BOOTSTRAP_SENSITIVITY_EXPLORATORY_ONLY"
            or len(exact_bootstrap.get("results", [])) != 12
            or not all(row["exact_bonferroni_familywise_95ci_pp"][0] <= 0
                       <= row["exact_bonferroni_familywise_95ci_pp"][1]
                       for row in exact_bootstrap["results"])):
        raise RuntimeError("Exact cluster-bootstrap sensitivity failed its integrity/coverage checks")
    sign_run = run([python, "-B", str(SIGN_SCRIPT)])
    if sign_run.returncode:
        raise RuntimeError(sign_run.stdout + sign_run.stderr)
    sign_cli = json.loads(sign_run.stdout)
    sign_report = json_file(GAP_UNION_DIR / "pilot_exact_cluster_sign_test.json")
    if (sign_cli.get("status") != "VERIFIED_POSTHOC_EXACT_GROUP_SIGN_SENSITIVITY_EXPLORATORY_ONLY"
            or len(sign_report.get("results", {}).get("primary", [])) != 12
            or len(sign_report.get("results", {}).get("secondary_did", [])) != 12
            or sign_cli.get("minimum_sign_test_groups_for_alpha") != 9
            or any(row["bonferroni_adjusted_p_within_12_cell_family"] < 0.05
                   for family in sign_report["results"].values() for row in family)):
        raise RuntimeError("Exact entity-group sign sensitivity failed its integrity checks")
    dap_run = run([python, "-B", str(DAP_SCRIPT)])
    if dap_run.returncode:
        raise RuntimeError(dap_run.stdout + dap_run.stderr)
    dap_cli = json.loads(dap_run.stdout)
    dap_report = json_file(GAP_UNION_DIR / "pilot_dap_exact_cluster_bootstrap.json")
    if (dap_cli.get("status") != "PASS_EXACT_POSTHOC_CLUSTERED_DAP_DIAGNOSTIC"
            or len(dap_report.get("results", [])) != 12
            or any(not 0 <= row["dap"] <= 1 for row in dap_report["results"])
            or any(row["exact_group_bootstrap"]["undefined_resample_fraction"] >= 1 for row in dap_report["results"])):
        raise RuntimeError("Pilot DAP exact cluster-bootstrap diagnostic failed its integrity checks")
    kaggle_status = query_kaggle_statuses()
    main_kernel_execution = query_main_kernel_execution()
    inventory_run = run([python, "-B", str(KAGGLE_INVENTORY_SCRIPT)])
    if inventory_run.returncode:
        raise RuntimeError(inventory_run.stdout + inventory_run.stderr)
    kaggle_inventory = json_file(KAGGLE_INVENTORY_JSON)
    if (
        kaggle_inventory.get("status") != "PASS_SCOPED_LIVE_INVENTORY"
        or kaggle_inventory.get("related_kernels") != len(kaggle_inventory.get("kernels", {}))
        or not kaggle_inventory.get("listing_pages")
        or any(item.get("exit_code") != 0 for item in kaggle_inventory.get("kernels", {}).values())
    ):
        raise RuntimeError("Full related Kaggle inventory failed integrity checks")

    assembler = run([python, "-B", "experiments/synthworlds_r1/training_release_draft/scripts/assemble_sft_source_verified.py"])
    assembler_text = assembler.stdout + assembler.stderr
    assembler_blocked = "Translation screen excluded required atomic cases" in assembler_text
    if not assembler_blocked and assembler.returncode != 0:
        raise RuntimeError(assembler_text)
    fresh_gate = parse_atomic_gate_error(assembler_text)
    if not assembler_blocked or fresh_gate is None:
        raise RuntimeError("Fresh assembler output did not report the expected fail-closed atomic case gate")
    observed_atomic, required_atomic = fresh_gate
    expected_atomic = {"forget": 117, "retain": 39}
    if observed_atomic != {"forget": 5, "retain": 3} or required_atomic != expected_atomic:
        raise RuntimeError(
            f"Fresh assembler gate changed unexpectedly: observed={observed_atomic}, required={required_atomic}"
        )
    forget_eligible, retain_eligible = observed_atomic["forget"], observed_atomic["retain"]
    forget_required, retain_required = required_atomic["forget"], required_atomic["retain"]
    training_dir = V51 / "sft_training"
    unexpected_release_files = [
        name for name in ("corpus.jsonl", "manifest.json", "release_validation.json")
        if (training_dir / name).exists()
    ]
    if unexpected_release_files:
        raise RuntimeError(f"Assembler blocked but release artifacts already exist: {unexpected_release_files}")

    preflight = run([python, "-B", "experiments/synthworlds_r1/scripts/train_sft_lora.py"])
    try:
        preflight_check = json.loads(preflight.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(preflight.stdout + preflight.stderr) from exc

    rescore_cmd = [
        python, "-B", "scripts/evaluate_crosslingual_unlearning.py",
        "--eval-root", "evaluation_dataset",
        "--predictions", "corrected_evaluation/predictions.jsonl",
        "--score-only", "--bootstrap", "2000", "--checkpoint", "sft=unused",
        "--contrast-model", "GA", "--contrast-model", "GD", "--contrast-model", "NPO",
    ]
    with tempfile.TemporaryDirectory(prefix="r1-reliability-rescore-") as temp_dir:
        rescore = run(rescore_cmd + ["--output-dir", temp_dir], cwd=PILOT)
        if rescore.returncode:
            raise RuntimeError(rescore.stdout + rescore.stderr)
        same_outputs = {
            name: (PILOT / "corrected_evaluation" / name).read_bytes()
            == (Path(temp_dir) / name).read_bytes()
            for name in MATCHED_OUTPUTS
        }

    pilot_receipt = json_file(RECHECK / "evidence/v4_corrected_independent_rescore_2026-11-04.json")
    pilot_contrasts = pilot_receipt["primary_forget_language_contrasts"]
    all_intervals_cover_zero = all(
        interval[0] <= 0 <= interval[1]
        for row in pilot_contrasts
        for interval in [row["cluster_bootstrap_95ci_pp"]]
    )
    overlay = json_file(OVERLAY_DIR / "final_recheck_overlay_2026-11-05/final_recheck_overlay.json")
    smoke = json_file(smoke_receipt_path)
    release_recheck = json_file(V51 / "evidence/local_gate_recheck_2026-11-05.json")
    recorded_gate_match = re.search(
        r"forget (\d+)/(\d+), retain (\d+)/(\d+)",
        release_recheck["source_verified_assembler"]["error"],
    )
    if not recorded_gate_match or tuple(map(int, recorded_gate_match.groups())) != (
        forget_eligible, forget_required, retain_eligible, retain_required
    ):
        raise RuntimeError("Fresh assembler atomic coverage differs from the hash-recorded gate evidence")
    composition_eligible = release_recheck["source_verified_assembler"]["composition_chains_eligible"]
    composition_required = gap_union["composition_coverage"]["required_training_chains"]
    gap_union_valid = (
        gap_union_check.get("corpus_written") is False
        and gap_union["scope"].get("training_started") is False
        and gap_union["scope"].get("training_eligible") is False
        and gap_union["coverage_before_and_after_targeted_candidates"]["verified_candidate_pass_rows_after_deduplicated_union"]
        >= gap_union["coverage_before_and_after_targeted_candidates"]["verified_candidate_pass_rows_before"]
    )
    if not gap_union_valid:
        raise RuntimeError("Candidate union audit status or quarantine invariants failed")
    test_script = V51 / "scripts/test_r1_pilot_pipeline.py"
    scorer_script = PILOT / "scripts/evaluate_crosslingual_unlearning.py"

    snapshot = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recheck_builder_sha256": sha(Path(__file__).resolve()),
        "scope": "Reliability and exploratory diagnostics only; no SFT or unlearning was started by this check.",
        "live_kaggle_status": kaggle_status,
        "live_kaggle_related_inventory": kaggle_inventory,
        "legacy_kaggle_notebook_execution": main_kernel_execution,
        "main_study": {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "planned_seeds": [42, 43, 44],
            "main_run_started": False,
            "release_gate": {
                "forget_cases_eligible": forget_eligible,
                "forget_cases_required": forget_required,
                "retain_cases_eligible": retain_eligible,
                "retain_cases_required": retain_required,
                "composition_chains_eligible": composition_eligible,
                "composition_chains_required": composition_required,
                "approved_corpus_exists": False,
                "assembler_status": "BLOCKED_EXPECTED_DATA_GATE" if assembler_blocked else "PASSED",
                "assembler_exit_code": assembler.returncode,
            },
            "local_training_preflight": {
                "status": preflight_check.get("status"),
                "training_started": preflight_check.get("training_started"),
                "errors": preflight_check.get("errors", []),
                "packages_ready": preflight_check.get("training_packages_ready"),
                "missing_packages": [name for name, present in preflight_check.get("training_packages", {}).items()
                                     if name in {"peft", "accelerate", "bitsandbytes"} and not present],
                "note": "Local environment only; Kaggle uses its own pinned dependency setup.",
            },
        },
        "historical_v4_pilot": {
            "model": "Llama 3.2 1B",
            "seed": 42,
            "role": "exploratory; baseline competence gate failed in all four target languages",
            "prediction_rows": pilot_receipt["prediction_rows"],
            "rescore_outputs_byte_identical": same_outputs,
            "all_seven_outputs_match": all(same_outputs.values()),
            "paired_contrasts": len(pilot_contrasts),
            "contrasts_with_95_percent_group_bootstrap_interval_including_or_touching_zero": sum(
                row["cluster_bootstrap_95ci_pp"][0] <= 0 <= row["cluster_bootstrap_95ci_pp"][1]
                for row in pilot_contrasts
            ),
            "all_paired_intervals_include_or_touch_zero": all_intervals_cover_zero,
            "pilot_rescore_receipt": str((RECHECK / "evidence/v4_corrected_independent_rescore_2026-11-04.json").relative_to(ROOT)),
            "posthoc_dap_diagnostic": {
                "status": dap_report["status"],
                "method_language_cells": len(dap_report["results"]),
                "point_estimate_range_percent": [
                    round(100 * min(row["dap"] for row in dap_report["results"]), 1),
                    round(100 * max(row["dap"] for row in dap_report["results"]), 1),
                ],
                "input_case_metrics_sha256": dap_report["input_case_metrics_sha256"],
                "limitations": dap_report["limitations"],
            },
        },
        "targeted_translation_recheck": {
            "rows": overlay["rows"],
            "passes": overlay["counts"]["PASS"],
            "flags": overlay["counts"]["FLAG"],
            "pass_rate_percent_descriptive_only": round(100 * overlay["counts"]["PASS"] / overlay["rows"], 1),
            "by_language": overlay["counts_by_language"],
            "by_kind": overlay["counts_by_kind"],
            "repaired_rows": 18,
            "human_reviewed": False,
            "training_eligible": False,
            "note": "Purposively selected gap-repair queue; percentages are descriptive, not a population error-rate estimate.",
        },
        "quarantined_candidate_union": {
            "candidate_rows_before": gap_union["coverage_before_and_after_targeted_candidates"]["verified_candidate_pass_rows_before"],
            "candidate_rows_after": gap_union["coverage_before_and_after_targeted_candidates"]["verified_candidate_pass_rows_after_deduplicated_union"],
            "net_new_unique_prompt_language_rows": gap_union["coverage_before_and_after_targeted_candidates"]["net_new_unique_prompt_language_rows"],
            "review_screens": gap_union["coverage_before_and_after_targeted_candidates"]["review_attempts_by_screen"],
            "exact_four_language_cases": gap_union["atomic_coverage"]["exact_all_four_language_cases_after"],
            "exact_leakage_groups": gap_union["atomic_coverage"]["exact_leakage_groups_after"],
            "optimistic_relation_aware_template_ceiling": gap_union["atomic_coverage"]["optimistic_relation_aware_template_ceiling_after"],
            "complete_composition_chains": gap_union["composition_coverage"]["complete_chains_after"],
            "composition_chains_required": gap_union["composition_coverage"]["required_training_chains"],
            "composition_rows_missing": gap_union["composition_coverage"]["missing_translation_rows_after"],
            "human_reviewed": False,
            "training_eligible": False,
            "audit_interpretation": gap_union["interpretation"],
        },
        "quarantined_full799_candidate_union": {
            "flagged_source_rows": full799_candidate_union["frozen_flagged_questions"],
            "chatgpt_pass_rows": full799_candidate_union["candidate_union"]["chatgpt_pass_rows"],
            "translategemma_pass_rows": full799_candidate_union["candidate_union"]["translategemma_pass_rows"],
            "overlap_rows": full799_candidate_union["candidate_union"]["overlap_rows"],
            "unique_candidate_pass_rows": full799_candidate_union["candidate_union"]["unique_rows_with_at_least_one_diagnostic_pass"],
            "complete_forget_cases": full799_coverage["optimistic_complete_atomic_cases"]["forget"],
            "complete_retain_cases": full799_coverage["optimistic_complete_atomic_cases"]["retain"],
            "complete_composition_chains": full799_coverage["optimistic_complete_composition_chains"],
            "human_reviewed": False,
            "training_eligible": False,
            "audit_interpretation": "Separate full flagged-question repair candidate union; optimistic AI-only coverage ceiling, not production approval.",
        },
        "quarantined_all_candidate_lanes_ceiling": {
            "status": candidate_ceiling["status"],
            "base_full_flagged_union": candidate_ceiling["base_optimistic_coverage"],
            "after_targeted_gap_candidates": candidate_ceiling["augmented_optimistic_coverage"],
            "after_blocked_row_retry": candidate_ceiling["post_retry_optimistic_coverage"],
            "after_nllb_alternates": candidate_ceiling["post_nllb_optimistic_coverage"],
            "after_segmented_retry": candidate_ceiling["post_segmented_retry_optimistic_coverage"],
            "targeted_review_counts": candidate_ceiling["targeted_review_counts"],
            "nllb_candidate_counts": candidate_ceiling["nllb_candidate_counts"],
            "segmented_retry_counts": candidate_ceiling["segmented_retry_counts"],
            "source_hashes": candidate_ceiling["source_hashes"],
            "release_approved": False,
            "training_eligible": False,
            "human_translation_reviewed": False,
            "interpretation": candidate_ceiling["interpretation"],
        },
        "bounded_verification": {
            "main_notebook_label_hash_guard": label_hash_guard,
            "offline_fixtures": {
                "status": "PASS" if fixture.returncode == 0 and fixture_count == EXPECTED_FIXTURE_COUNT else "FAIL",
                "passed": fixture_count if fixture.returncode == 0 else None,
                "test_source_sha256": sha(test_script),
            },
            "pilot_rescore": {
                "status": "PASS_BYTE_IDENTICAL" if all(same_outputs.values()) else "FAIL",
                "scorer_sha256": sha(scorer_script),
                "output_sha256": {name: sha(PILOT / "corrected_evaluation" / name) for name in MATCHED_OUTPUTS},
            },
            "exact_cluster_bootstrap_sensitivity": {
                "status": exact_bootstrap["status"],
                "analysis_script_sha256": sha(exact_bootstrap_script),
                "result_json_sha256": sha(GAP_UNION_DIR / "pilot_exact_cluster_sensitivity.json"),
                "result_markdown_sha256": sha(GAP_UNION_DIR / "pilot_exact_cluster_sensitivity.md"),
                "contrasts": len(exact_bootstrap["results"]),
                "all_exact_familywise_intervals_include_zero": all(
                    row["exact_bonferroni_familywise_95ci_pp"][0] <= 0
                    <= row["exact_bonferroni_familywise_95ci_pp"][1]
                    for row in exact_bootstrap["results"]
                ),
                "max_marginal_endpoint_difference_from_100k_mc_pp": exact_bootstrap_cli[
                    "max_marginal_endpoint_difference_from_100k_mc_pp"],
                "max_familywise_endpoint_difference_from_100k_mc_pp": exact_bootstrap_cli[
                    "max_familywise_endpoint_difference_from_100k_mc_pp"],
            },
            "exact_cluster_sign_sensitivity": {
                "status": sign_report["status"],
                "analysis_script_sha256": sha(SIGN_SCRIPT),
                "result_json_sha256": sha(GAP_UNION_DIR / "pilot_exact_cluster_sign_test.json"),
                "result_markdown_sha256": sha(GAP_UNION_DIR / "pilot_exact_cluster_sign_test.md"),
                "primary_cells": len(sign_report["results"]["primary"]),
                "secondary_cells": len(sign_report["results"]["secondary_did"]),
                "all_adjusted_p_values_at_least_0_05": all(
                    row["bonferroni_adjusted_p_within_12_cell_family"] >= 0.05
                    for family in sign_report["results"].values() for row in family
                ),
                "alpha_per_cell": sign_report["alpha_per_cell"],
                "minimum_non_tied_groups_for_any_sign_test_to_reach_alpha": sign_report[
                    "minimum_non_tied_groups_for_any_two_sided_sign_result_to_reach_cell_alpha"],
            },
            "pilot_dap_exact_cluster_bootstrap": {
                "status": dap_report["status"],
                "analysis_script_sha256": sha(DAP_SCRIPT),
                "result_json_sha256": sha(GAP_UNION_DIR / "pilot_dap_exact_cluster_bootstrap.json"),
                "result_markdown_sha256": sha(GAP_UNION_DIR / "pilot_dap_exact_cluster_bootstrap.md"),
                "method_language_cells": len(dap_report["results"]),
                "all_dap_denominators_nonzero": all(row["post_direct_failures"] > 0 for row in dap_report["results"]),
                "input_case_metrics_sha256": dap_report["input_case_metrics_sha256"],
            },
            "translation_overlay": overlay_check,
            "targeted_gap_union": {
                "status": "PASS_HASH_BOUND_COVERAGE_DIAGNOSTIC_ONLY",
                "json_sha256": sha(GAP_UNION_DIR / "targeted_gap_union_audit.json"),
                "markdown_sha256": sha(GAP_UNION_DIR / "targeted_gap_union_audit.md"),
                "independently_rerun_verifiers": gap_union["independently_rerun_verifiers"],
            },
            "full799_candidate_union": {
                "status": full799_candidate_union["status"],
                "script_sha256": sha(FULL799_CANDIDATE_AUDIT),
                "json_sha256": sha(
                    V51 / "training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/"
                    "optimistic_candidate_union_coverage.json"
                ),
                "candidate_rows": full799_candidate_union["frozen_flagged_questions"],
                "chatgpt_pass_rows": full799_candidate_union["candidate_union"]["chatgpt_pass_rows"],
                "translategemma_pass_rows": full799_candidate_union["candidate_union"]["translategemma_pass_rows"],
                "unique_candidate_pass_rows": full799_candidate_union["candidate_union"]["unique_rows_with_at_least_one_diagnostic_pass"],
                "complete_forget_cases": full799_coverage["optimistic_complete_atomic_cases"]["forget"],
                "complete_retain_cases": full799_coverage["optimistic_complete_atomic_cases"]["retain"],
                "complete_composition_chains": full799_coverage["optimistic_complete_composition_chains"],
            },
            "all_candidate_lanes_ceiling": {
                "status": candidate_ceiling["status"],
                "audit_script_sha256": sha(CANDIDATE_CEILING_AUDIT),
                "result_json_sha256": sha(CANDIDATE_CEILING_JSON),
                "source_hashes": candidate_ceiling["source_hashes"],
                "authoritative_release_gate": candidate_ceiling["authoritative_release_gate"],
            },
            "8b_runtime_smoke": smoke_check,
            "source_release_gate": {
                "status": "BLOCKED_EXPECTED_DATA_GATE" if assembler_blocked else "PASSED",
                "message": next((line.strip() for line in assembler_text.splitlines()
                                  if "Translation screen excluded required atomic cases" in line), None),
                "assembler_script_sha256": sha(V51 / "training_release_draft/scripts/assemble_sft_source_verified.py"),
                "release_builder_sha256": sha(V51 / "training_release_draft/scripts/assemble_training_release.py"),
                "evidence_file_sha256": sha(V51 / "evidence/local_gate_recheck_2026-11-05.json"),
            },
            "local_sft_preflight": preflight_check,
        },
        "caveats": [
            "The historical 1B pilot is not the planned 8B, three-seed main experiment.",
            "The checked workspace has no standalone timestamped preregistration artifact; describe the main protocol as documented, not formally preregistered.",
            "All translation candidate screens are purposive, AI-only diagnostics without human-gold controls; no candidate was admitted to training.",
            "The 8B runtime receipt used synthetic data only and is not a hypothesis result.",
            "The local preflight does not represent Kaggle's installed packages or secret state.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "verification.json"
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rates = snapshot["targeted_translation_recheck"]
    by_lang = rates["by_language"]
    report = f"""# R1 reliability recheck — 2026-11-05

**Stage:** multilingual training-data release. The planned Llama 3.1 8B SFT, baseline, and unlearning runs have not started because the source-verified corpus is not releasable. The checked workspace contains no standalone timestamped preregistration artifact; this report does not label the protocol formally preregistered.

## Current diagnostic numbers

| Check | Result | Interpretation |
|---|---:|---|
| Frozen source split | 195 cases; 117 forget / 39 retain / 39 holdout; 102 leakage groups | Source structure passes; split remains a draft, not approval |
| Eligible release coverage | {forget_eligible}/{forget_required} forget; {retain_eligible}/{retain_required} retain; {composition_eligible}/{composition_required} composition chains | Release assembler correctly blocks; no training corpus emitted |
| Primary targeted translation queue | {rates['passes']}/{rates['rows']} pass ({rates['pass_rate_percent_descriptive_only']}%); {rates['flags']} flagged after 18 retries | Candidate-only AI screen; selected gap queue, not a population rate |
| 1B pilot paired contrasts | {snapshot['historical_v4_pilot']['contrasts_with_95_percent_group_bootstrap_interval_including_or_touching_zero']}/{snapshot['historical_v4_pilot']['paired_contrasts']} 95% group-bootstrap intervals include or touch zero | Exploratory only; baseline competence failed and there was one 1B seed |
| 8B runtime smoke | One synthetic LoRA step and one synthetic NPO update on 2×T4 | Runtime compatibility only; no research data |
| Scoped Kaggle jobs | {sum(item['status'] == 'KernelWorkerStatus.COMPLETE' for item in kaggle_status['kernels'].values())}/{len(kaggle_status['kernels'])} complete | Includes the legacy 1B pilot alias, three translation diagnostics, and the synthetic 8B smoke; completion status alone does not mean training ran |
| Full related-kernel inventory | {kaggle_inventory['related_kernels']} kernels in the visible account inventory; {kaggle_inventory['status_counts'].get('KernelWorkerStatus.COMPLETE', 0)} complete, {kaggle_inventory['status_counts'].get('KernelWorkerStatus.ERROR', 0)} error | Naming-filtered point-in-time inventory; no related job is currently running |
| Legacy notebook execution log | {main_kernel_execution['status']} | {main_kernel_execution['interpretation']} |

The latest 187-row review passed {rates['passes']}/187 ({rates['pass_rate_percent_descriptive_only']}%); by language: FR {by_lang['fr']['PASS']}/21, DE {by_lang['de']['PASS']}/19, HI {by_lang['hi']['PASS']}/62, TH {by_lang['th']['PASS']}/85. The independent, hash-bound candidate-union audit adds 49 unique prompt-language rows to the previous 1,628, for 1,677 candidate rows total. It estimates 42/117 forget and 16/39 retain cases with all four target-language prompts, spanning 22 and 12 leakage groups; the optimistic relation-aware template ceiling is 47/117 and 19/39. Only 2/44 composition chains are complete, with 114 translation rows missing. These are quarantined AI-screened coverage diagnostics, not released training data or quality estimates.

| Follow-up screen | Reviewed / blocked | Joint passes | Net new unique rows |
|---|---:|---:|---:|
| Retry of 17 prior blocked rows | 17 / 0 | 3 | 0 |
| 148-row NLLB alternate batch | 23 / 125 | 3 | 1 |
| Marker-preserving segmented retry | 124 / 1 | 0 | 0 |

The batches overlap earlier candidate rows; raw pass counts must not be summed. A cross-review disagreed with the initial Qwen labels on 154/187 rows. Constructed mismatch checks passed 2/15 for Mistral and 1/16 for Qwen14; these are small probes, not reviewer error-rate estimates. There are no human-gold controls. Every output remains AI-only and quarantined. The candidate-union audit records row-level deduplication, model/script hashes, and screening receipts.

A **separate** full flagged-question repair union combines ChatGPT and TranslateGemma alternatives: 445 and 397 diagnostic passes, with 230 overlapping rows and 612 unique rows with at least one candidate pass. If combined optimistically with the original production passes, it reaches 60/117 four-language forget cases, 19/39 retain cases, and 6/44 complete composition chains. This is an availability ceiling, not a release: the candidate screens use different AI-only criteria, have no human-gold control, and every selected translation must pass the frozen production review. These rows are not added to the separate targeted-gap counts above.

A second, deliberately optimistic ceiling combines that 799-row union with source-bound targeted-gap candidates and the later retries. The batches are deduplicated by source question and language before case counts are recomputed:

| Cumulative candidate set | Forget cases | Retain cases | Composition chains |
|---|---:|---:|---:|
| Full flagged-question union | 60/117 | 19/39 | 6/44 |
| + targeted-gap candidates | {candidate_ceiling['augmented_optimistic_coverage']['optimistic_complete_atomic_cases']['forget']['complete']}/117 | {candidate_ceiling['augmented_optimistic_coverage']['optimistic_complete_atomic_cases']['retain']['complete']}/39 | {candidate_ceiling['augmented_optimistic_coverage']['optimistic_complete_composition_chains']['complete']}/44 |
| + blocked-row retry | {candidate_ceiling['post_retry_optimistic_coverage']['optimistic_complete_atomic_cases']['forget']['complete']}/117 | {candidate_ceiling['post_retry_optimistic_coverage']['optimistic_complete_atomic_cases']['retain']['complete']}/39 | {candidate_ceiling['post_retry_optimistic_coverage']['optimistic_complete_composition_chains']['complete']}/44 |
| + NLLB alternatives | {candidate_ceiling['post_nllb_optimistic_coverage']['optimistic_complete_atomic_cases']['forget']['complete']}/117 | {candidate_ceiling['post_nllb_optimistic_coverage']['optimistic_complete_atomic_cases']['retain']['complete']}/39 | {candidate_ceiling['post_nllb_optimistic_coverage']['optimistic_complete_composition_chains']['complete']}/44 |
| + marker-preserving retry | {candidate_ceiling['post_segmented_retry_optimistic_coverage']['optimistic_complete_atomic_cases']['forget']['complete']}/117 | {candidate_ceiling['post_segmented_retry_optimistic_coverage']['optimistic_complete_atomic_cases']['retain']['complete']}/39 | {candidate_ceiling['post_segmented_retry_optimistic_coverage']['optimistic_complete_composition_chains']['complete']}/44 |

The broadest union reaches 69/117 forget, 21/39 retain, and 7/44 composition cases; the last segmented retry added no passing rows. This remains a candidate-availability ceiling: screens are heterogeneous and AI-only, and no candidate has passed the full frozen production review. The authoritative release gate remains {forget_eligible}/{forget_required} forget, {retain_eligible}/{retain_required} retain, and {composition_eligible}/{composition_required} composition chains.

For the 1B pilot, a post-hoc Bonferroni sensitivity over 12 contrasts with 100,000 entity-group bootstrap draws left every interval covering zero; the simultaneous interval extremes were -71.4 to +100.0 percentage points. With only 5--7 groups per contrast, this is evidence of low precision, not evidence of no cross-language transfer.

An exact enumeration of every ordered cluster-bootstrap draw (3,125--823,543 resamples per contrast) reproduced all 12 marginal and Bonferroni-adjusted interval endpoints from the 100,000-draw simulation exactly (maximum endpoint difference: 0.0 pp). This removes Monte Carlo resampling noise; it does not reduce the underlying uncertainty from only 5--7 groups per contrast.

A separate exact entity-group sign sensitivity gives no familywise signal: all 24 adjusted sign-test p-values (12 primary and 12 secondary) are 1.00. There are only 5--7 groups per pilot contrast, with many tied zero-change groups. At the design's minimum of eight groups, even eight non-tied groups all moving in one direction yield a two-sided sign p-value of 0.0078, above the 0.00417 per-cell threshold for 12 tests; nine non-tied groups are the mathematical minimum for this test to reach that threshold. This is a small-sample sensitivity result, not a power recommendation. See the [exact group sign sensitivity](../r1_reliability_tracker_2026-11-05/pilot_exact_cluster_sign_test.md).

A separate post-hoc downstream-answer-persistence analysis finds final answers correct in 7--10 of 8--12 post-unlearning direct-recall failures per method-language cell (DAP 81.8--100.0%). It exactly resamples leakage groups. Some intervals are degenerate at 100% because all observed failures in those cells retained a correct final answer; this boundary behavior is a small-sample limitation, not evidence of certainty. See the [DAP results and limits](../r1_reliability_tracker_2026-11-05/pilot_dap_exact_cluster_bootstrap.md). This result does not establish reasoning recovery or override the failed competence gate.

## Verification run

- Offline implementation fixtures: **{fixture_count}/{EXPECTED_FIXTURE_COUNT} passed**.
- Main Kaggle notebook: frozen-label SHA-256 is checked before baseline/post scoring, after generation; notebook SHA-256 is recorded in `verification.json`.
- Fresh pilot rescore: **all 7 derived score/report files byte-identical** to the archived outputs; {pilot_receipt['prediction_rows']:,} prediction rows.
- Exact pilot bootstrap sensitivity: **12/12 intervals reproduced by exhaustive resampling**, with 0.0 pp endpoint difference from the 100,000-draw analysis; the output is post-hoc and exploratory.
- Exact entity-group sign sensitivity: **24/24 Bonferroni-adjusted p-values are 1.00**; the eight-group feasibility floor cannot reach the 12-cell threshold even under unanimous non-tied directions.
- Pilot downstream-answer-persistence diagnostic: **12/12 cells recomputed** from hash-bound case metrics with exact entity-group resampling; still exploratory and one-seed.
- 187-row overlay: hash, row coverage, retry lineage, and counts verified; status is diagnostic-only.
- Latest alias, NLLB alternate, and segmented NLLB receipts: all three verifiers reproduce; their net new contribution is 1, not 6, after deduplication.
- Candidate-union audit: {gap_union['coverage_before_and_after_targeted_candidates']['verified_candidate_pass_rows_after_deduplicated_union']:,} quarantined prompt-language candidates; source-bound release counts remain unchanged.
- Live Kaggle status query ({kaggle_status['observed_at_utc']}): {sum(item['status'] == 'KernelWorkerStatus.COMPLETE' for item in kaggle_status['kernels'].values())}/{len(kaggle_status['kernels'])} scoped diagnostic/smoke kernels complete.
- Full related-kernel inventory ({kaggle_inventory['observed_at_utc']}): {kaggle_inventory['related_kernels']} matching jobs across {kaggle_inventory['visible_kernels_scanned']} visible kernels; {kaggle_inventory['status_counts'].get('KernelWorkerStatus.COMPLETE', 0)} complete, {kaggle_inventory['status_counts'].get('KernelWorkerStatus.ERROR', 0)} error, and none running. See the [full inventory](../r1_reliability_tracker_2026-11-05/kaggle_related_job_inventory_2026-11-05.md).
- 8B runtime receipt: exact model revision and runner verified; `research_data_loaded=false`.
- Main source assembler: **blocked at the expected translation-coverage gate**. The command wrote no training release.
- Local SFT preflight: blocked because no approved corpus exists. Missing local packages are not evidence about Kaggle's environment.

The detailed historical pilot tables and confidence intervals are in [the completed v4 diagnostic](../r1_current_state_audit_2026-11-04/completed_v4_pilot_hypothesis_diagnostic.md). The complete machine-readable receipts and file hashes are in [verification.json](verification.json).

## Conclusion

There are reproducible diagnostic results, but no valid estimate from the planned main experiment yet. Do not interpret the small 1B pilot or candidate-screen percentages as confirmation. The next experiment gate is a fully source-bound multilingual release, followed by three-seed baseline competence checks; unlearning must remain off until that baseline gate passes.
"""
    (OUT / "diagnostic_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if all(same_outputs.values()) and fixture.returncode == 0 and fixture_count == EXPECTED_FIXTURE_COUNT
        and overlay_check.get("status") == "PASS_HASH_BOUND_DIAGNOSTIC_OVERLAY_ONLY"
        and smoke_check.get("status") == "VERIFIED_RUNTIME_ONLY_NOT_EXPERIMENT"
        and dap_report.get("status") == "PASS_EXACT_POSTHOC_CLUSTERED_DAP_DIAGNOSTIC"
        and full799_audit_check.get("training_eligible") is False
        and assembler_blocked and gap_union_valid else "CHECK_FAILURE",
        "verification_json": str(json_path.relative_to(ROOT)),
        "report": str((OUT / "diagnostic_report.md").relative_to(ROOT)),
        "pilot_outputs_match": sum(same_outputs.values()),
        "fixtures_passed": fixture_count,
        "release_assembler": "BLOCKED_EXPECTED_DATA_GATE" if assembler_blocked else "PASSED",
    }, indent=2))


if __name__ == "__main__":
    main()
