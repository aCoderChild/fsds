#!/usr/bin/env python3
"""Independently verify canonical TOFU baseline outputs and derived metrics."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

from rouge_score import rouge_scorer
from scipy.stats import ks_2samp


EXPECTED = {
    "llama_3_2_3b_full": ("open-unlearning/tofu_Llama-3.2-3B-Instruct_full", "24f31ca19f6966dcb6f6b29abc511cce71222d4a", "full"),
    "llama_3_2_3b_retain90": ("open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90", "2c35819e9a731d096df9abd6efbcc361870c5333", "retain90_oracle"),
    "llama_3_1_8b_full": ("open-unlearning/tofu_Llama-3.1-8B-Instruct_full", "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e", "full"),
    "llama_3_1_8b_retain90": ("open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90", "f02956a9f25a18f38ad7808fbefb2f957347273e", "retain90_oracle"),
}
SPLIT_COUNTS = {
    "forget10": (400, 20),
    "retain90": (400, 20),
    "holdout10": (400, 20),
    "real_authors": (100, None),
    "world_facts": (117, None),
}
EPS = 1e-10
N_BOOTSTRAP = 5000
SEED = 42
EVAL_DATA_FILES = {
    "forget10.json",
    "forget10_perturbed.json",
    "retain90.json",
    "retain_perturbed.json",
    "holdout10.json",
    "real_authors_perturbed.json",
    "world_facts_perturbed.json",
}
METRIC_FIELDS = (
    "answer_probability",
    "rougeL_recall",
    "truth_ratio",
    "truth_ratio_score",
    "normalized_option_probability",
)


def bootstrap_ci(rows: list[dict], field: str) -> list[float] | None:
    grouped = collections.defaultdict(list)
    for row in rows:
        if row.get("profile_block") is not None and row.get(field) is not None:
            grouped[int(row["profile_block"])].append(float(row[field]))
    if not grouped:
        return None
    groups = list(grouped.values())
    rng = random.Random(SEED)
    estimates = []
    for _ in range(N_BOOTSTRAP):
        sample = [rng.choice(groups) for _ in groups]
        values = [x for block in sample for x in block]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return [estimates[int(.025 * (N_BOOTSTRAP - 1))], estimates[int(.975 * (N_BOOTSTRAP - 1))]]


def mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def equal(a, b, field: str, problems: list[str], tolerance: float = 1e-10) -> None:
    if a is None or b is None:
        problems.append(f"{field}: missing numeric value (expected {b}, found {a})")
        return
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        problems.append(f"{field}: nonnumeric value (expected {b!r}, found {a!r})")
        return
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        problems.append(f"{field}: non-finite value (expected {b!r}, found {a!r})")
        return
    if not math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance):
        problems.append(f"{field}: expected {b}, found {a}")


def verify_rows(key: str, result: dict, audit: dict) -> tuple[list[str], dict]:
    problems = []
    repo, revision, role = EXPECTED[key]
    model = result.get("model", {})
    if result.get("status") != "PASS_OFFICIAL_TOFU_BASELINE":
        problems.append("unexpected result status")
    if (model.get("repo_id"), model.get("revision"), model.get("role")) != (repo, revision, role):
        problems.append("model repository/revision/role does not match pinned protocol")
    runtime = result.get("runtime", {})
    if runtime.get("training_started") is not False or runtime.get("research_adapter_created") is not False:
        problems.append("runtime receipt does not confirm inference-only execution")
    if runtime.get("gpu_names", []).count("Tesla T4") < 2:
        problems.append("receipt does not show two Tesla T4 GPUs")
    if key.startswith("llama_3_1_8b"):
        if runtime.get("device_map_strategy") != "balanced":
            problems.append("8B receipt does not use the pinned balanced device map")
        if runtime.get("cpu_offload_used") is not False:
            problems.append("8B receipt does not confirm GPU-only model placement")
        if runtime.get("placement_smoke", {}).get("status") != "PASS":
            problems.append("8B receipt lacks a passing pre-evaluation generation smoke")
        if not runtime.get("hf_device_map"):
            problems.append("8B receipt lacks the resolved Hugging Face device map")
    if result.get("evaluation", {}).get("input_truncation") is not False:
        problems.append("unexpected input truncation policy")
    expected_files = audit["data_audit"]["files"]
    recorded_hashes = result.get("evaluation", {}).get("dataset_sha256_by_file", {})
    if set(recorded_hashes) != EVAL_DATA_FILES:
        problems.append("dataset checksum file set does not match the pinned evaluation inputs")
    for filename, digest in recorded_hashes.items():
        if filename not in expected_files or expected_files[filename]["sha256"] != digest:
            problems.append(f"dataset checksum mismatch for {filename}")
    if result.get("evaluation", {}).get("open_unlearning_revision") != "2d98027677f3468d52b6baa3fc4f2b9f2dc08820":
        problems.append("Open-Unlearning source revision does not match the pinned protocol")

    expected_splits = ("forget10",) if role == "retain90_oracle" else tuple(SPLIT_COUNTS)
    all_examples = result.get("per_example", {})
    if set(all_examples) != set(expected_splits):
        problems.append(f"unexpected evaluated split keys: {sorted(all_examples)}")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    recomputed_summary = {}
    for split in expected_splits:
        rows = all_examples.get(split, [])
        expected_n, expected_groups = SPLIT_COUNTS[split]
        if len(rows) != expected_n:
            problems.append(f"{split}: expected {expected_n} examples, found {len(rows)}")
        source_indices = [row.get("source_row_index") for row in rows]
        if len(set(source_indices)) != len(source_indices):
            problems.append(f"{split}: duplicate source row index")
        if split in {"forget10", "holdout10", "retain90"} and source_indices != list(range(expected_n)):
            problems.append(f"{split}: source row indices do not exactly cover the pinned ordered evaluation rows")
        if split in {"real_authors", "world_facts"} and source_indices != list(range(expected_n)):
            problems.append(f"{split}: source row indices do not exactly cover the pinned ordered utility rows")
        groups = {row.get("profile_block") for row in rows if row.get("profile_block") is not None}
        if expected_groups is not None and len(groups) != expected_groups:
            problems.append(f"{split}: expected {expected_groups} profile groups, found {len(groups)}")
        if expected_groups is not None and groups != set(range(expected_groups)):
            problems.append(f"{split}: profile group IDs are not the expected contiguous blocks")
        for i, row in enumerate(rows):
            p = row.get("answer_probability")
            loss = row.get("answer_token_loss")
            if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                problems.append(f"{split} row {i}: invalid answer probability")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
                problems.append(f"{split} row {i}: invalid answer loss")
            else:
                equal(p, math.exp(-float(loss)), f"{split} row {i} exp(-answer loss)", problems)
            if role == "full":
                generation = row.get("generation")
                ref = row.get("reference_answer")
                if not isinstance(generation, str) or not isinstance(ref, str):
                    problems.append(f"{split} row {i}: missing generated text/reference")
                else:
                    expected_rouge = rouge.score(ref, generation)["rougeL"].recall
                    equal(row.get("rougeL_recall"), expected_rouge, f"{split} row {i} ROUGE-L recall", problems)
            if "paraphrased_answer_token_loss" in row:
                para_loss = float(row["paraphrased_answer_token_loss"])
                wrong_losses = row.get("perturbed_answer_token_losses", [])
                if not wrong_losses:
                    problems.append(f"{split} row {i}: truth ratio has no perturbed-answer losses")
                else:
                    wrong_geo = math.exp(-sum(map(float, wrong_losses)) / len(wrong_losses))
                    correct_p = math.exp(-para_loss)
                    ratio = wrong_geo / (correct_p + EPS)
                    equal(row.get("truth_ratio"), ratio, f"{split} row {i} truth ratio", problems)
                    if split == "forget10":
                        score = min(ratio, 1.0 / (ratio + EPS))
                    else:
                        score = max(0.0, 1.0 - ratio)
                    equal(row.get("truth_ratio_score"), score, f"{split} row {i} truth-ratio score", problems)
            elif "perturbed_answer_token_losses" in row:
                wrong_losses = row["perturbed_answer_token_losses"]
                wrong_probs = row.get("perturbed_answer_probabilities", [])
                if len(wrong_losses) != len(wrong_probs) or not wrong_losses:
                    problems.append(f"{split} row {i}: wrong-option losses/probabilities are inconsistent")
                else:
                    for j, (wrong_loss, wrong_prob) in enumerate(zip(wrong_losses, wrong_probs)):
                        equal(wrong_prob, math.exp(-float(wrong_loss)), f"{split} row {i} wrong option {j} probability", problems)
                    wrong_geo = math.exp(-sum(map(float, wrong_losses)) / len(wrong_losses))
                    ratio = wrong_geo / (float(p) + EPS)
                    equal(row.get("truth_ratio"), ratio, f"{split} row {i} truth ratio", problems)
                    equal(row.get("truth_ratio_score"), max(0, 1.0 - ratio), f"{split} row {i} truth-ratio score", problems)
                    normalized = float(p) / (float(p) + sum(map(float, wrong_probs)) + EPS)
                    equal(row.get("normalized_option_probability"), normalized, f"{split} row {i} normalized option probability", problems)
            for field in ("truth_ratio", "truth_ratio_score", "normalized_option_probability", "rougeL_recall"):
                value = row.get(field)
                if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                    problems.append(f"{split} row {i}: invalid {field}")
                if field == "rougeL_recall" and value is not None and float(value) > 1:
                    problems.append(f"{split} row {i}: ROUGE-L recall is outside [0,1]")

        summary = {"n_questions": len(rows), "n_profile_blocks": len(groups)}
        for field in METRIC_FIELDS:
            value = mean(rows, field)
            if value is None:
                continue
            summary[f"mean_{field}"] = value
            ci = bootstrap_ci(rows, field)
            if ci is not None:
                summary[f"profile_cluster_bootstrap_95pct_ci_{field}"] = ci
        recorded = result.get("summary", {}).get(split, {})
        if summary.keys() != recorded.keys():
            problems.append(f"{split}: summary fields mismatch; recomputed={sorted(summary)}, recorded={sorted(recorded)}")
        for field, value in summary.items():
            if isinstance(value, list):
                if len(recorded.get(field, [])) != len(value):
                    problems.append(f"{split}: {field} length mismatch")
                else:
                    for j, (a, b) in enumerate(zip(recorded[field], value)):
                        equal(a, b, f"{split} {field}[{j}]", problems)
            elif isinstance(value, float):
                equal(recorded.get(field), value, f"{split} {field}", problems)
            elif recorded.get(field) != value:
                problems.append(f"{split}: {field} mismatch")
        recomputed_summary[split] = summary

    if role == "full":
        retain = all_examples["retain90"]
        if [row["source_row_index"] for row in retain] != list(range(400)):
            problems.append("canonical retain evaluation rows are not the pinned first 20 complete profiles")
        collision_rows = [row for row in retain if row.get("ambiguous_question_collision")]
        if len(collision_rows) != 1 or collision_rows[0].get("source_row_index") != 100:
            problems.append("canonical retain evaluation must contain exactly one side of the known collision")
        utility = result.get("model_utility", {})
        component_values = {
            "retain_probability": mean(all_examples["retain90"], "answer_probability"),
            "retain_rougeL_recall": mean(all_examples["retain90"], "rougeL_recall"),
            "retain_truth_ratio_score": mean(all_examples["retain90"], "truth_ratio_score"),
            "real_authors_normalized_option_probability": mean(all_examples["real_authors"], "normalized_option_probability"),
            "real_authors_rougeL_recall": mean(all_examples["real_authors"], "rougeL_recall"),
            "real_authors_truth_ratio_score": mean(all_examples["real_authors"], "truth_ratio_score"),
            "world_facts_normalized_option_probability": mean(all_examples["world_facts"], "normalized_option_probability"),
            "world_facts_rougeL_recall": mean(all_examples["world_facts"], "rougeL_recall"),
            "world_facts_truth_ratio_score": mean(all_examples["world_facts"], "truth_ratio_score"),
        }
        if utility.get("components", {}).keys() != component_values.keys():
            problems.append("model-utility component keys do not match official nine metrics")
        for name, value in component_values.items():
            equal(utility.get("components", {}).get(name), value, f"model utility {name}", problems)
        expected_hm = 0.0 if any(value <= 0 for value in component_values.values()) else 9 / sum(1 / x for x in component_values.values())
        equal(utility.get("harmonic_mean"), expected_hm, "TOFU model-utility harmonic mean", problems)

    return problems, recomputed_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Verify available checkpoints and report PARTIAL when other pinned checkpoints are absent.",
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    problems = []
    run_status = None
    raw_run_status = None
    run_error = None
    if not audit.get("data_audit", {}).get("checks", {}).get("structural_checks_pass"):
        problems.append("pinned dataset audit did not pass")
    status_file = args.result_dir / "official_eval_status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text(encoding="utf-8"))
        run_status = status.get("status")
        raw_run_status = status.get("raw_kaggle_terminal_status", status.get("raw_run_status"))
        run_error = status.get("error") or status.get("raw_kaggle_error")
        acceptable_statuses = {"COMPLETE", "PARTIAL", "ERROR"} if args.allow_partial else {"COMPLETE"}
        if status.get("status") not in acceptable_statuses:
            problems.append(f"run status is {status.get('status')}, expected COMPLETE")
    else:
        problems.append("missing official_eval_status.json")

    model_results = {}
    recomputed = {}
    for key in EXPECTED:
        path = args.result_dir / f"{key}_official_eval.json"
        if not path.exists():
            model_results[key] = {"status": "MISSING", "problems": [f"missing {path.name}"]}
            if not args.allow_partial:
                problems.append(f"{key}: missing {path.name}")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        local_problems, local_summary = verify_rows(key, result, audit)
        model_results[key] = {
            "status": "PASS" if not local_problems else "FAIL",
            "problems": local_problems,
            "model": result.get("model"),
            "summary": local_summary,
            "runtime": result.get("runtime"),
        }
        recomputed[key] = result
        problems.extend(f"{key}: {item}" for item in local_problems)

    for full_key, oracle_key in (
        ("llama_3_2_3b_full", "llama_3_2_3b_retain90"),
        ("llama_3_1_8b_full", "llama_3_1_8b_retain90"),
    ):
        if full_key in recomputed and oracle_key in recomputed:
            full = recomputed[full_key]
            oracle = recomputed[oracle_key]
            full_rows = full["per_example"]["forget10"]
            oracle_rows = oracle["per_example"]["forget10"]
            if [(r["source_row_index"], r["question_sha256"]) for r in full_rows] != [(r["source_row_index"], r["question_sha256"]) for r in oracle_rows]:
                problems.append(f"{full_key}: forget rows do not align with retain oracle")
            else:
                stat = ks_2samp(
                    [1 / (r["truth_ratio"] + EPS) for r in full_rows],
                    [1 / (r["truth_ratio"] + EPS) for r in oracle_rows],
                )
                recorded = full.get("forget_quality_vs_retain90_oracle", {})
                equal(recorded.get("ks_statistic"), stat.statistic, f"{full_key} forget-quality KS statistic", problems)
                equal(recorded.get("p_value"), stat.pvalue, f"{full_key} forget-quality p-value", problems)
    present_statuses = [x["status"] for x in model_results.values() if x["status"] != "MISSING"]
    missing_models = [key for key, value in model_results.items() if value["status"] == "MISSING"]
    if problems or any(value != "PASS" for value in present_statuses):
        overall = "FAIL"
    elif missing_models and args.allow_partial:
        overall = "PARTIAL"
    elif missing_models:
        overall = "FAIL"
    else:
        overall = "PASS"
    verification = {
        "overall_status": overall,
        "run_status": run_status,
        "raw_run_status": raw_run_status,
        "run_error": run_error,
        "audit_status": "PASS" if audit.get("data_audit", {}).get("checks", {}).get("structural_checks_pass") else "FAIL",
        "missing_models": missing_models,
        "problems": problems,
        "models": model_results,
    }
    (args.result_dir / "official_eval_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# TOFU official-metric baseline verification", "", f"Overall: **{overall}**", f"Reconciled run status: **{run_status}**"]
    if raw_run_status:
        lines.append(f"Raw Kaggle terminal status: **{raw_run_status}**")
    if run_error:
        lines.append(f"Raw run error: `{run_error}`")
    lines.append("")
    for key, item in model_results.items():
        lines.append(f"## {key}: {item['status']}")
        lines.append("")
        for message in item.get("problems", []):
            lines.append(f"- {message}")
        for split, summary in item.get("summary", {}).items():
            lines.append(f"- {split}: {summary.get('n_questions')} questions, {summary.get('n_profile_blocks')} profile blocks")
        lines.append("")
    for message in problems:
        if not any(message in item.get("problems", []) for item in model_results.values()):
            lines.append(f"- {message}")
    (args.result_dir / "official_eval_verification.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2))
    if overall == "FAIL":
        raise SystemExit(1)
    if overall == "PARTIAL" and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
