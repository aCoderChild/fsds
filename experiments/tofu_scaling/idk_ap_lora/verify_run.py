#!/usr/bin/env python3
"""Independent artifact, training-trace, coverage, and gate recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import urllib.request
import zipfile
from pathlib import Path


EXPECTED_PROTOCOL_SHA256 = "650af06eed1d254cfdd2b47714299686925cea47b5b2675edb7c07674b01c049"
PROFILE_SIZE = 20
DRAW_COUNT = 5000


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_source_files(protocol: dict, protocol_path: Path, root: Path) -> list[str]:
    problems = []
    files = protocol["data"]["sha256"]
    for filename, expected in files.items():
        url = f"https://huggingface.co/datasets/locuslab/TOFU/resolve/{protocol['data']['revision']}/{filename}"
        request = urllib.request.Request(url, headers={"User-Agent": "fsds-idk-ap-independent-verifier/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        if sha256(raw) != expected:
            problems.append(f"Pinned source hash mismatch: {filename}")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != int(protocol["data"]["expected_rows"][filename]):
            problems.append(f"Pinned row count mismatch: {filename}")
    idk = root / "source_snapshot" / "idontknow.jsonl"
    if not idk.is_file() or sha256(idk.read_bytes()) != protocol["objective"]["idk_response_sha256"]:
        problems.append("Pinned IDK-response file is missing or has the wrong hash")
    if sha256(protocol_path.read_bytes()) != EXPECTED_PROTOCOL_SHA256:
        problems.append("Protocol digest differs from the locked preregistration")
    provenance = read_json(root / "source_provenance.json")
    for filename, expected_hash in provenance.get("source_snapshot_sha256", {}).items():
        path = root / "source_snapshot" / filename
        if not path.is_file() or sha256(path.read_bytes()) != expected_hash:
            problems.append(f"Source snapshot hash mismatch: {filename}")
    return problems


def expected_indices(ranges: list[list[int]]) -> set[int]:
    blocks = [block for first, last in ranges for block in range(first, last + 1)]
    return {block * PROFILE_SIZE + offset for block in blocks for offset in range(PROFILE_SIZE)}


def verify_rows(rows: list[dict], split: str, ranges: list[list[int]]) -> list[str]:
    expected = expected_indices(ranges)
    actual = {int(row["source_row_index"]) for row in rows}
    problems = []
    if len(rows) != len(expected) or actual != expected:
        problems.append(f"{split}: source rows do not match the locked profile assignment")
    if len(actual) != len(rows):
        problems.append(f"{split}: duplicate source-row IDs")
    expected_blocks = {index // PROFILE_SIZE for index in expected}
    actual_blocks = {int(row["profile_block"]) for row in rows}
    if expected_blocks != actual_blocks:
        problems.append(f"{split}: profile blocks do not match the locked assignment")
    for row in rows:
        for field in ("answer_probability", "answer_token_loss", "rougeL_recall"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                problems.append(f"{split}: missing/non-numeric {field}")
                continue
            if not math.isfinite(value):
                problems.append(f"{split}: non-finite {field}")
        if not 0 <= float(row["answer_probability"]) <= 1:
            problems.append(f"{split}: answer_probability outside [0,1]")
        if not 0 <= float(row["rougeL_recall"]) <= 1:
            problems.append(f"{split}: ROUGE-L outside [0,1]")
        if not isinstance(row.get("generation"), str) or not isinstance(row.get("reference_answer"), str):
            problems.append(f"{split}: generation/reference text missing")
        if abs(float(row["answer_probability"]) - math.exp(-float(row["answer_token_loss"]))) > 2e-8:
            problems.append(f"{split}: answer_probability does not match token loss")
        if split == "forget10":
            wrong = row.get("perturbed_answer_token_losses")
            if not wrong or any(not math.isfinite(float(value)) for value in wrong):
                problems.append("forget10: perturbed-answer NLLs missing or non-finite")
            else:
                wrong_probability = math.exp(-sum(map(float, wrong)) / len(wrong))
                if row.get("paraphrased_answer_token_loss") is not None:
                    correct_probability = math.exp(-float(row["paraphrased_answer_token_loss"]))
                    ratio = wrong_probability / (correct_probability + 1e-10)
                    score = min(ratio, 1.0 / (ratio + 1e-10))
                else:
                    ratio = wrong_probability / (float(row["answer_probability"]) + 1e-10)
                    score = max(0.0, 1.0 - ratio)
                if abs(ratio - float(row.get("truth_ratio", math.nan))) > 2e-8:
                    problems.append("forget10: Truth Ratio recomputation mismatch")
                if abs(score - float(row.get("truth_ratio_score", math.nan))) > 2e-8:
                    problems.append("forget10: Truth Ratio score recomputation mismatch")
    return problems


def paired_bootstrap(base_rows: list[dict], candidate_rows: list[dict], metric: str) -> dict:
    base = {int(row["source_row_index"]): row for row in base_rows}
    grouped: dict[int, list[float]] = {}
    for row in candidate_rows:
        index = int(row["source_row_index"])
        if index not in base:
            raise ValueError(f"Unmatched source index {index} in {metric}")
        grouped.setdefault(int(row["profile_block"]), []).append(float(row[metric]) - float(base[index][metric]))
    block_ids = sorted(grouped)
    point = sum(sum(v) for v in grouped.values()) / sum(map(len, grouped.values()))
    rng = random.Random(42)
    samples = []
    for _ in range(DRAW_COUNT):
        selected = [rng.choice(block_ids) for _ in block_ids]
        values = [x for block in selected for x in grouped[block]]
        samples.append(sum(values) / len(values))
    samples.sort()
    return {
        "mean_delta_candidate_minus_full": point,
        "ci95": [samples[int(0.025 * (DRAW_COUNT - 1))], samples[int(0.975 * (DRAW_COUNT - 1))]],
        "n_questions": sum(map(len, grouped.values())),
        "n_profile_blocks": len(grouped),
    }


def verify_comparison(reference: dict, candidate: dict, comparison: dict, protocol: dict, cohort: str) -> list[str]:
    ranges = protocol["data"]["profile_assignment_ranges_inclusive"]
    split_ranges = {
        "forget10": ranges[f"forget10_{cohort}_train_and_eval"],
        "retain90": ranges[f"retain90_{cohort}_utility_eval"],
        "holdout10": ranges[f"holdout10_{cohort}_utility_eval"],
    }
    problems = []
    for split in split_ranges:
        problems += verify_rows(candidate["per_example"].get(split, []), split, split_ranges[split])
    expected_indices = {split: expected_indices_for_ranges(value) for split, value in split_ranges.items()}
    reference_by_split = {}
    for split in split_ranges:
        reference_rows = reference["per_example"].get(split, [])
        reference_by_split[split] = [row for row in reference_rows if int(row["source_row_index"]) in expected_indices[split]]
        if {int(row["source_row_index"]) for row in reference_by_split[split]} != expected_indices[split]:
            problems.append(f"Reference coverage mismatch: {split}")
        candidate_by_index = {int(row["source_row_index"]): row for row in candidate["per_example"][split]}
        for row in reference_by_split[split]:
            candidate_row = candidate_by_index.get(int(row["source_row_index"]))
            if candidate_row is None or candidate_row.get("question_sha256") != row.get("question_sha256"):
                problems.append(f"Paired question mismatch: {split}:{row.get('source_row_index')}")

    gate_specs = {
        "forget10.truth_ratio_score": ("forget10", "truth_ratio_score", "low_zero"),
        "forget10.answer_probability": ("forget10", "answer_probability", "high_zero"),
        "retain90.answer_probability": ("retain90", "answer_probability", "low_margin"),
        "retain90.rougeL_recall": ("retain90", "rougeL_recall", "low_margin"),
        "holdout10.answer_probability": ("holdout10", "answer_probability", "low_margin"),
    }
    margin = float(protocol["evaluation"]["noninferiority_margin"])
    expected_gates = {}
    for name, (split, metric, rule) in gate_specs.items():
        stats = paired_bootstrap(reference_by_split[split], candidate["per_example"][split], metric)
        low, high = stats["ci95"]
        passed = low > 0 if rule == "low_zero" else high < 0 if rule == "high_zero" else low > -margin
        expected_gates[name] = {**stats, "pass": passed}
    recorded = comparison.get("gates", {})
    if set(recorded) != set(expected_gates):
        problems.append("Recorded metric gate set mismatch")
    for name, expected in expected_gates.items():
        observed = recorded.get(name, {})
        for key in ("n_questions", "n_profile_blocks", "pass"):
            if observed.get(key) != expected[key]:
                problems.append(f"Gate mismatch for {name}: {key}")
        for key in ("mean_delta_candidate_minus_full", "ci95"):
            got = observed.get(key)
            want = expected[key]
            if isinstance(want, list):
                if not isinstance(got, list) or len(got) != len(want) or any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(got, want)):
                    problems.append(f"Gate mismatch for {name}: {key}")
            elif got is None or abs(float(got) - float(want)) > 1e-12:
                problems.append(f"Gate mismatch for {name}: {key}")
    all_pass = all(item["pass"] for item in expected_gates.values())
    expected_status = "PASS_DIAGNOSTIC_GATES" if all_pass else "FAIL_DIAGNOSTIC_GATES"
    if comparison.get("all_gates_pass") is not all_pass or comparison.get("status") != expected_status:
        problems.append("Overall gate status does not match independent recomputation")
    return problems


def expected_indices_for_ranges(ranges: list[list[int]]) -> set[int]:
    blocks = [block for first, last in ranges for block in range(first, last + 1)]
    return {block * PROFILE_SIZE + offset for block in blocks for offset in range(PROFILE_SIZE)}


def verify_training(cohort_dir: Path, protocol: dict, protocol_hash: str, source_provenance: dict) -> list[str]:
    problems = []
    receipt_path = cohort_dir / "training" / "training_receipt.json"
    trace_path = cohort_dir / "training" / "loss_trace.jsonl"
    receipt = read_json(receipt_path)
    training = protocol["training"]
    if receipt.get("status") != "TRAINING_COMPLETE" or receipt.get("protocol_sha256") != protocol_hash:
        problems.append(f"{cohort_dir.name}: incomplete/mismatched training receipt")
    if receipt.get("cohort") != cohort_dir.name or receipt.get("optimizer_steps") != int(training["expected_optimizer_steps"]):
        problems.append(f"{cohort_dir.name}: cohort/optimizer step mismatch")
    if receipt.get("training_script_sha256") != source_provenance["source_snapshot_sha256"].get("train.py"):
        problems.append(f"{cohort_dir.name}: training source hash mismatch")
    if receipt.get("runtime", {}).get("cpu_or_disk_offload") is not False:
        problems.append(f"{cohort_dir.name}: CPU/disk offload was not explicitly false")
    smoke = receipt.get("runtime", {}).get("real_data_smoke", {})
    if smoke.get("status") != "PASS_BOUNDED_REAL_DATA_SMOKE" or not smoke.get("initial_lora_restored_before_research_training"):
        problems.append(f"{cohort_dir.name}: real-data smoke/initialization reset did not pass")
    if receipt.get("dataset", {}).get("forget_rows") != 200 or receipt.get("dataset", {}).get("retain_optimizer_rows") != 2760:
        problems.append(f"{cohort_dir.name}: training row count mismatch")
    assignments = receipt.get("dataset", {}).get("pairing_assignments", [])
    if len(assignments) != 200:
        problems.append(f"{cohort_dir.name}: retain/IDK pairing assignment count mismatch")
    else:
        sequence = [[int(x["forget_source_index"]), int(x["retain_source_index"]), int(x["idk_response_index"])] for x in assignments]
        digest = sha256(json.dumps(sequence, separators=(",", ":")).encode())
        if digest != receipt.get("dataset", {}).get("pairing_sha256"):
            problems.append(f"{cohort_dir.name}: pairing assignment digest mismatch")
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(trace) != int(training["expected_loss_trace_rows"]):
        problems.append(f"{cohort_dir.name}: loss trace row count mismatch")
    step_counts: dict[int, int] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    for row in trace:
        step = int(row["global_step"])
        step_counts[step] = step_counts.get(step, 0) + 1
        key = (int(row["epoch"]), int(row["pair_index"]))
        pair_counts[key] = pair_counts.get(key, 0) + 1
        for field in ("forget_idk_loss", "retain_ap_loss", "combined_loss", "gradient_norm_preclip", "learning_rate"):
            if not math.isfinite(float(row[field])):
                problems.append(f"{cohort_dir.name}: non-finite loss trace value {field}")
    if sorted(step_counts) != list(range(1, int(training["expected_optimizer_steps"]) + 1)) or set(step_counts.values()) != {8}:
        problems.append(f"{cohort_dir.name}: optimizer step/microbatch accounting mismatch")
    if len(pair_counts) != 200 * int(training["epochs"]) or set(pair_counts.values()) != {1}:
        problems.append(f"{cohort_dir.name}: each training pair must appear once in each epoch")
    expected_pairs = {int(x["forget_source_index"]): x for x in assignments}
    for row in trace:
        pair = expected_pairs.get(int(row["forget_source_index"]))
        if pair is None or int(row["retain_source_index"]) != int(pair["retain_source_index"]) or int(row["idk_index"]) != int(pair["idk_response_index"]):
            problems.append(f"{cohort_dir.name}: trace pair differs from locked pairing assignment")
            break
    adapter_dir = cohort_dir / "training" / "adapter"
    actual = {path.name: sha256(path.read_bytes()) for path in sorted(adapter_dir.iterdir()) if path.is_file()}
    if not actual or actual != receipt.get("adapter_files_sha256"):
        problems.append(f"{cohort_dir.name}: adapter file hash mismatch")
    if sha256(trace_path.read_bytes()) != receipt.get("loss_trace_sha256"):
        problems.append(f"{cohort_dir.name}: loss trace hash mismatch")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--artifact-zip", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root
    protocol_path = args.protocol or root / "idk_ap_lora_protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = sha256(protocol_path.read_bytes())
    problems = []
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        problems.append("Protocol digest mismatch")
    problems += check_source_files(protocol, protocol_path, root)
    audit = read_json(root / "idk_ap_data_audit.json")
    if audit.get("status") != "PASS_IDK_AP_PROTOCOL_DATA_AUDIT" or audit.get("protocol_sha256") != protocol_hash:
        problems.append("Pinned data/profile audit receipt failed")
    objective = read_json(root / "idk_ap_objective_equivalence.json")
    if objective.get("status") != "PASS" or objective.get("protocol_sha256") != protocol_hash:
        problems.append("Official objective equivalence receipt failed")
    reference = read_json(root / "reference_support" / "full_reference_profile_eval.json")
    if reference.get("status") != "PASS_PROFILE_COHORT_EVAL":
        problems.append("Archived full-checkpoint reference evaluation did not pass")
    reference_manifest = read_json(root / "reference_support" / "reference_support_manifest.json")
    reference_spec = protocol["reference_support"]
    reference_file_hash = sha256((root / "reference_support" / "full_reference_profile_eval.json").read_bytes())
    manifest_entry = reference_manifest.get("files", {}).get("reference_eval/full_reference_profile_eval.json", {})
    if reference_manifest.get("status") != "PASS_SOURCE_ARCHIVE_AND_PRIOR_VERIFICATION":
        problems.append("Archived reference-support manifest did not pass")
    if reference_manifest.get("source_archive_sha256") != reference_spec["source_archive_sha256"]:
        problems.append("Reference-support source archive hash mismatch")
    if manifest_entry.get("sha256") != reference_spec["reference_file_sha256"] or reference_file_hash != reference_spec["reference_file_sha256"]:
        problems.append("Archived frozen-reference result hash mismatch")
    reference_model = reference.get("model", {})
    if (reference_model.get("repo_id"), reference_model.get("revision")) != (
        reference_spec["reference_checkpoint_repo"], reference_spec["reference_checkpoint_revision"]
    ):
        problems.append("Archived frozen-reference checkpoint mismatch")
    if reference.get("evaluation", {}).get("dataset_revision") != protocol["data"]["revision"]:
        problems.append("Archived frozen-reference TOFU revision mismatch")
    receipt = read_json(root / "run_receipt.json")
    if receipt.get("protocol_sha256") != protocol_hash:
        problems.append("Run receipt protocol hash mismatch")

    for cohort in ("dev", "final"):
        cohort_dir = root / cohort
        if not cohort_dir.is_dir():
            continue
        problems += verify_training(cohort_dir, protocol, protocol_hash, read_json(root / "source_provenance.json"))
        comparison_path = cohort_dir / "evaluation" / f"{cohort}_idk_ap_comparison.json"
        eval_path = cohort_dir / "evaluation" / f"{cohort}_idk_ap_profile_eval.json"
        if not comparison_path.is_file() or not eval_path.is_file():
            problems.append(f"{cohort}: missing evaluation or gate artifact")
            continue
        candidate = read_json(eval_path)
        comparison = read_json(comparison_path)
        if candidate.get("status") != "PASS_PROFILE_COHORT_EVAL":
            problems.append(f"{cohort}: profile evaluation status failed")
        if comparison.get("protocol_sha256") != protocol_hash or comparison.get("cohort") != cohort:
            problems.append(f"{cohort}: comparison protocol/cohort mismatch")
        problems += verify_comparison(reference, candidate, comparison, protocol, cohort)

    expected_run_status = "DEV_GATES_FAILED_STOP"
    dev_comparison_path = root / "dev" / "evaluation" / "dev_idk_ap_comparison.json"
    if dev_comparison_path.is_file():
        dev_comparison = read_json(dev_comparison_path)
        if dev_comparison.get("all_gates_pass"):
            if not (root / "final").is_dir():
                problems.append("Development passed all gates but no final cohort was executed")
            else:
                expected_run_status = read_json(root / "final" / "evaluation" / "final_idk_ap_comparison.json").get("status")
    if receipt.get("status") != expected_run_status:
        problems.append(f"Run stop status mismatch: expected {expected_run_status}, got {receipt.get('status')}")
    if expected_run_status == "DEV_GATES_FAILED_STOP":
        if receipt.get("final_training_started") is not False or (root / "final").exists():
            problems.append("Final cohort was started or created despite failed development gates")
        if receipt.get("development_gate_status") != "FAIL_DIAGNOSTIC_GATES":
            problems.append("Development stop receipt does not record failed gates")
    elif receipt.get("final_training_started") is not True:
        problems.append("Development gates passed but the final training-start receipt is missing")

    archive_summary = None
    if args.artifact_zip:
        with zipfile.ZipFile(args.artifact_zip) as archive:
            bad = archive.testzip()
            if bad:
                problems.append(f"Artifact archive CRC failure: {bad}")
            archive_summary = {"bytes": args.artifact_zip.stat().st_size, "sha256": sha256(args.artifact_zip.read_bytes()), "crc": "PASS" if not bad else "FAIL"}

    result = {
        "status": "PASS_ARTIFACT_AND_RESULT_RECOMPUTATION" if not problems else "FAIL_ARTIFACT_OR_RESULT_RECOMPUTATION",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_hash,
        "run_status": receipt.get("status"),
        "cohorts_verified": [cohort for cohort in ("dev", "final") if (root / cohort).is_dir()],
        "artifact_archive": archive_summary,
        "problems": problems,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
