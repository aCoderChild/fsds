#!/usr/bin/env python3
"""Evaluate the IDK+AP adapter on locked profile blocks and apply fixed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate_npo_profile_cohort import expand, make_rows
from tofu_official_eval import CHECKPOINTS, DATASET_REVISION, evaluate_checkpoint, load_data


IDK_MARKER = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't)\s+know|i\s+(?:am\s+not|m\s+not)\s+(?:sure|certain)|"
    r"i\s+(?:do\s+not|don't)\s+have\s+(?:the\s+)?(?:answer|information)|"
    r"i\s+cannot\s+answer|i\s+can't\s+answer|not\s+enough\s+information|unable\s+to\s+answer)\b",
    flags=re.IGNORECASE,
)
BOOTSTRAP_DRAWS = 5000
EXPECTED_PROTOCOL_SHA256 = "650af06eed1d254cfdd2b47714299686925cea47b5b2675edb7c07674b01c049"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip()


def select_eval_sets(data: dict[str, list[dict]], protocol: dict, cohort: str) -> dict[str, list[dict]]:
    assignments = protocol["data"]["profile_assignment_ranges_inclusive"]
    if cohort == "dev":
        forget_key = "forget10_dev_train_and_eval"
        retain_key = "retain90_dev_utility_eval"
        holdout_key = "holdout10_dev_utility_eval"
    else:
        forget_key = "forget10_final_train_and_eval"
        retain_key = "retain90_final_utility_eval"
        holdout_key = "holdout10_final_utility_eval"
    return {
        "forget10": make_rows(data, "forget10", expand(assignments[forget_key])),
        "retain90": make_rows(data, "retain90", expand(assignments[retain_key])),
        "holdout10": make_rows(data, "holdout10", expand(assignments[holdout_key])),
    }


def paired_cluster_delta(base_rows: list[dict], candidate_rows: list[dict], field: str, seed: int) -> dict:
    base_by_index = {int(row["source_row_index"]): row for row in base_rows}
    groups: dict[int, list[float]] = {}
    for row in candidate_rows:
        source_index = int(row["source_row_index"])
        base = base_by_index.get(source_index)
        if base is None:
            raise ValueError(f"Unmatched source index {source_index} in {field}")
        groups.setdefault(int(row["profile_block"]), []).append(float(row[field]) - float(base[field]))
    if not groups:
        raise ValueError(f"No matched examples for {field}")
    block_ids = sorted(groups)
    point = sum(sum(values) for values in groups.values()) / sum(map(len, groups.values()))
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_blocks = [rng.choice(block_ids) for _ in block_ids]
        values = [value for block in sampled_blocks for value in groups[block]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {
        "n_questions": sum(map(len, groups.values())),
        "n_profile_blocks": len(groups),
        "mean_delta_candidate_minus_full": point,
        "ci95": [draws[int(0.025 * (BOOTSTRAP_DRAWS - 1))], draws[int(0.975 * (BOOTSTRAP_DRAWS - 1))]],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
    }


def summarize_and_gate(reference: dict, candidate: dict, protocol: dict) -> dict:
    margin = float(protocol["evaluation"]["noninferiority_margin"])
    rules = {
        "forget10.truth_ratio_score": ("forget10", "truth_ratio_score", "lower_zero"),
        "forget10.answer_probability": ("forget10", "answer_probability", "upper_zero"),
        "retain90.answer_probability": ("retain90", "answer_probability", "lower_margin"),
        "retain90.rougeL_recall": ("retain90", "rougeL_recall", "lower_margin"),
        "holdout10.answer_probability": ("holdout10", "answer_probability", "lower_margin"),
    }
    gates = {}
    for name, (split, metric, rule) in rules.items():
        delta = paired_cluster_delta(reference["per_example"][split], candidate["per_example"][split], metric, 42)
        low, high = delta["ci95"]
        passed = low > 0 if rule == "lower_zero" else high < 0 if rule == "upper_zero" else low > -margin
        gates[name] = {
            **delta,
            "metric": metric,
            "rule": rule,
            "threshold": 0.0 if rule.endswith("zero") else -margin,
            "pass": bool(passed),
        }
    all_pass = all(gate["pass"] for gate in gates.values())
    return {
        "status": "PASS_DIAGNOSTIC_GATES" if all_pass else "FAIL_DIAGNOSTIC_GATES",
        "all_gates_pass": all_pass,
        "interpretation": protocol["evaluation"]["research_claim_limit"],
        "gates": gates,
        "bootstrap": {"unit": "profile block", "draws": BOOTSTRAP_DRAWS, "seed": 42},
        "noninferiority_margin": margin,
    }


def add_idk_descriptives(result: dict, idk_path: Path) -> dict:
    responses = [line.strip() for line in idk_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = {normalize_text(answer) for answer in responses}
    rows = result["per_example"].get("forget10", [])
    hits = 0
    exact = 0
    for row in rows:
        text = str(row.get("generation", ""))
        row["idk_marker_match"] = bool(IDK_MARKER.search(text))
        row["exact_official_idk_match"] = normalize_text(text) in normalized
        hits += int(row["idk_marker_match"])
        exact += int(row["exact_official_idk_match"])
    result["idk_generation_diagnostics"] = {
        "n_forget_questions": len(rows),
        "idk_marker_matches": hits,
        "idk_marker_rate": hits / len(rows) if rows else None,
        "exact_official_idk_matches": exact,
        "exact_official_idk_rate": exact / len(rows) if rows else None,
        "marker_rule": IDK_MARKER.pattern,
        "limitation": "Descriptive string-matching signals only; not semantic refusal classification or a research gate.",
    }
    return result


def verify_reference_support(support_dir: Path, protocol: dict) -> Path:
    manifest_path = support_dir / "reference_support_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = protocol["reference_support"]
    result_path = support_dir / expected["reference_file"]
    if manifest.get("status") != "PASS_SOURCE_ARCHIVE_AND_PRIOR_VERIFICATION":
        raise ValueError("Reference-support manifest status did not pass")
    if manifest.get("source_archive_sha256") != expected["source_archive_sha256"]:
        raise ValueError("Reference source archive differs from the locked protocol")
    entry = manifest.get("files", {}).get("reference_eval/full_reference_profile_eval.json", {})
    if entry.get("sha256") != expected["reference_file_sha256"] or sha256(result_path.read_bytes()) != expected["reference_file_sha256"]:
        raise ValueError("Full-reference evaluation hash mismatch")
    reference = json.loads(result_path.read_text(encoding="utf-8"))
    if reference.get("status") != "PASS_PROFILE_COHORT_EVAL":
        raise ValueError("Frozen-checkpoint reference evaluation did not pass")
    model = reference.get("model", {})
    if (model.get("repo_id"), model.get("revision")) != (expected["reference_checkpoint_repo"], expected["reference_checkpoint_revision"]):
        raise ValueError("Frozen reference checkpoint does not match the protocol")
    if reference.get("evaluation", {}).get("dataset_revision") != protocol["data"]["revision"]:
        raise ValueError("Frozen reference was scored on a different TOFU revision")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort", choices=("dev", "final"), required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--reference-support-dir", type=Path, required=True)
    parser.add_argument("--idk", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    receipt = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    if sha256(args.protocol.read_bytes()) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Locked protocol SHA-256 mismatch")
    if receipt.get("status") != "TRAINING_COMPLETE" or not receipt.get("adapter_saved"):
        raise ValueError("No completed training receipt is available")
    if receipt.get("protocol_sha256") != sha256(args.protocol.read_bytes()) or receipt.get("cohort") != args.cohort:
        raise ValueError("Training receipt does not match the locked protocol/cohort")
    expected_adapter_hashes = receipt.get("adapter_files_sha256", {})
    actual_adapter_hashes = {path.name: sha256(path.read_bytes()) for path in sorted(args.adapter_dir.iterdir()) if path.is_file()}
    if not actual_adapter_hashes or expected_adapter_hashes != actual_adapter_hashes:
        raise ValueError("Adapter files do not match their training receipt")

    data, hashes = load_data()
    for filename, expected_hash in protocol["data"]["sha256"].items():
        if hashes.get(filename) != expected_hash:
            raise ValueError(f"Pinned TOFU evaluation source hash mismatch: {filename}")
    eval_sets = select_eval_sets(data, protocol, args.cohort)
    reference_path = verify_reference_support(args.reference_support_dir, protocol)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    selected_indices = {split: {int(row["source_row_index"]) for row in rows} for split, rows in eval_sets.items()}
    ref_subset = {
        split: [row for row in reference["per_example"][split] if int(row["source_row_index"]) in selected_indices[split]]
        for split in eval_sets
    }
    for split, rows in ref_subset.items():
        if {int(row["source_row_index"]) for row in rows} != selected_indices[split]:
            raise ValueError(f"Frozen reference does not cover the exact {args.cohort} {split} rows")
        by_index = {int(row["source_row_index"]): row for row in rows}
        for current in eval_sets[split]:
            prior = by_index[int(current["source_row_index"])]
            if hashlib.sha256(current["question"].encode()).hexdigest() != prior.get("question_sha256"):
                raise ValueError(f"Frozen reference question mismatch at {split}:{current['source_row_index']}")
    reference_subset = {**reference, "per_example": ref_subset}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOFU_KEEP_MODEL_CACHE"] = "1"
    result = evaluate_checkpoint(
        "llama_3_2_3b_full", args.output_dir, eval_sets, hashes,
        score_batch_size=1, generation_batch_size=1, max_memory_per_gpu_gib=12.5,
        adapter_path=args.adapter_dir, result_key=f"{args.cohort}_idk_ap",
        adapter_seed=int(protocol["training"]["seed"]), profile_cohort_mode=True,
    )
    if result.get("status") != "PASS_PROFILE_COHORT_EVAL":
        raise RuntimeError("IDK+AP adapter evaluation did not complete")
    result["model"]["label"] = "Llama 3.2 3B full IDK+AP LoRA diagnostic seed 42"
    if result["model"].get("adapter"):
        result["model"]["adapter"]["method"] = "IDK+AP LoRA adaptation"
    add_idk_descriptives(result, args.idk)
    result_path = args.output_dir / f"{args.cohort}_idk_ap_profile_eval.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    comparison = summarize_and_gate(reference_subset, result, protocol)
    comparison.update({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol.read_bytes()),
        "cohort": args.cohort,
        "model_revision": protocol["model"]["revision"],
        "dataset_revision": DATASET_REVISION,
        "training_receipt_sha256": sha256(args.training_receipt.read_bytes()),
        "evaluation_receipt": result_path.name,
        "candidate_idk_diagnostics": result["idk_generation_diagnostics"],
    })
    comparison_path = args.output_dir / f"{args.cohort}_idk_ap_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
