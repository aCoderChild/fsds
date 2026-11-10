#!/usr/bin/env python3
"""Independent integrity and metric checks for the one-seed TOFU NPO-LoRA pilot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


DRAWS = 5000
SEED = 42
EPS = 1e-10
SPLIT_COUNTS = {"forget10": 400, "retain90": 400, "holdout10": 400, "real_authors": 100, "world_facts": 117}
MODEL_KEYS = {
    "3b": ("llama_3_2_3b_full", "llama_3_2_3b_retain90"),
    "8b": ("llama_3_1_8b_full", "llama_3_1_8b_retain90"),
}


def adapter_result_key(model: str, seed: int) -> str:
    family = "llama_3_2_3b" if model == "3b" else "llama_3_1_8b"
    return f"{family}_npo_lora_seed{seed}"


def fail_if(condition: bool, message: str, problems: list[str]) -> None:
    if condition:
        problems.append(message)


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trainer_batching(source_path: Path, model_key: str) -> tuple[int, int]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODEL_FAMILIES" for target in node.targets)
    )
    families = ast.literal_eval(assignment.value)
    matches = [value for value in families.values() if value[0] == model_key]
    if len(matches) != 1:
        raise ValueError(f"Expected one batching entry for {model_key}, found {len(matches)}")
    return int(matches[0][1]), int(matches[0][2])


def paired_cluster_delta(rows_a: list[dict], rows_b: list[dict], field: str) -> dict:
    b_by_key = {(r["split"], r["source_row_index"]): r for r in rows_b}
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows_a:
        other = b_by_key.get((row["split"], row["source_row_index"]))
        if other is None or field not in row or field not in other:
            continue
        grouped[row["profile_block"]].append(float(other[field]) - float(row[field]))
    if not grouped:
        raise ValueError(f"No matched profile blocks for {field}")
    blocks = sorted(grouped)
    values = [value for group in grouped.values() for value in group]
    point = sum(values) / len(values)
    rng = random.Random(SEED)
    draws = []
    for _ in range(DRAWS):
        sample_groups = [rng.choice(blocks) for _ in blocks]
        sample = [value for group in sample_groups for value in grouped[group]]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return {
        "n_questions": len(values),
        "n_profile_blocks": len(grouped),
        "mean_delta_b_minus_a": point,
        "profile_cluster_bootstrap_95pct_ci": [draws[int(.025 * (DRAWS - 1))], draws[int(.975 * (DRAWS - 1))]],
        "bootstrap_draws": DRAWS,
        "bootstrap_seed": SEED,
    }


def validate_metrics(result: dict, baseline: dict, protocol: dict, problems: list[str]) -> None:
    per_example = result.get("per_example", {})
    base_examples = baseline.get("per_example", {})
    for split, expected_count in SPLIT_COUNTS.items():
        rows = per_example.get(split, [])
        base_rows = base_examples.get(split, [])
        fail_if(len(rows) != expected_count, f"{split}: expected {expected_count} adapter rows, found {len(rows)}", problems)
        fail_if(len(base_rows) != expected_count, f"{split}: expected {expected_count} baseline rows, found {len(base_rows)}", problems)
        if len(rows) != len(base_rows):
            continue
        for index, (row, base_row) in enumerate(zip(rows, base_rows)):
            for field in ("source_row_index", "profile_block", "question_sha256", "source_answer"):
                fail_if(row.get(field) != base_row.get(field), f"{split}[{index}]: baseline alignment mismatch in {field}", problems)
            for field in ("answer_probability", "answer_token_loss", "answer_scored_tokens"):
                fail_if(field not in row, f"{split}[{index}]: missing {field}", problems)
            if not all(field in row for field in ("answer_probability", "answer_token_loss", "answer_scored_tokens")):
                continue
            probability = float(row["answer_probability"])
            token_loss = float(row["answer_token_loss"])
            fail_if(not (0 < probability <= 1) or not math.isfinite(token_loss) or token_loss < 0,
                    f"{split}[{index}]: invalid answer probability/loss", problems)
            fail_if(int(row["answer_scored_tokens"]) <= 0 or not close(probability, math.exp(-token_loss), 1e-7),
                    f"{split}[{index}]: answer probability does not match token loss", problems)
            if "rougeL_recall" in row:
                score = float(row["rougeL_recall"])
                fail_if(not math.isfinite(score) or score < 0 or score > 1,
                        f"{split}[{index}]: ROUGE-L outside [0,1]", problems)
            if "paraphrased_answer_token_loss" in row:
                para_loss = float(row["paraphrased_answer_token_loss"])
                para_probability = float(row.get("paraphrased_answer_probability", float("nan")))
                fail_if(not math.isfinite(para_loss) or para_loss < 0 or not close(para_probability, math.exp(-para_loss), 1e-7),
                        f"{split}[{index}]: paraphrased probability does not match token loss", problems)

            perturbed_losses = row.get("perturbed_answer_token_losses")
            if perturbed_losses is None:
                continue
            perturbed_losses = [float(x) for x in perturbed_losses]
            perturbed_probabilities = [float(x) for x in row.get("perturbed_answer_probabilities", [])]
            fail_if(not perturbed_losses or len(perturbed_losses) != len(perturbed_probabilities),
                    f"{split}[{index}]: perturbed answer loss/probability count mismatch", problems)
            if not perturbed_losses or len(perturbed_losses) != len(perturbed_probabilities):
                continue
            fail_if(any(not math.isfinite(x) or x < 0 for x in perturbed_losses),
                    f"{split}[{index}]: invalid perturbed token loss", problems)
            fail_if(any(not close(p, math.exp(-loss), 1e-7) for p, loss in zip(perturbed_probabilities, perturbed_losses)),
                    f"{split}[{index}]: perturbed probability does not match token loss", problems)
            denominator_field = "paraphrased_answer_probability" if "paraphrased_answer_probability" in row else "answer_probability"
            denominator = float(row[denominator_field])
            ratio = math.exp(-sum(perturbed_losses) / len(perturbed_losses)) / (denominator + EPS)
            expected_score = min(ratio, 1.0 / (ratio + EPS)) if split == "forget10" and denominator_field == "paraphrased_answer_probability" else max(0.0, 1.0 - ratio)
            fail_if(not close(float(row.get("truth_ratio", float("nan"))), ratio, 1e-7),
                    f"{split}[{index}]: Truth Ratio does not match saved token losses", problems)
            fail_if(not close(float(row.get("truth_ratio_score", float("nan"))), expected_score, 1e-7),
                    f"{split}[{index}]: Truth Ratio score does not match formula", problems)
            if "normalized_option_probability" in row:
                expected_option = probability / (probability + sum(perturbed_probabilities) + EPS)
                fail_if(not close(float(row["normalized_option_probability"]), expected_option, 1e-7),
                        f"{split}[{index}]: normalized option probability mismatch", problems)

    ev = result.get("evaluation", {})
    expected_hashes = protocol["dataset"]["evaluation_file_sha256"]
    fail_if(ev.get("dataset_sha256_by_file") != expected_hashes, "evaluation file hashes differ from locked protocol", problems)
    fail_if(ev.get("dataset_revision") != protocol["dataset"]["revision"], "evaluation dataset revision differs from locked protocol", problems)


def recompute_comparison(base: dict, adapter: dict, oracle: dict) -> dict:
    deltas = {}
    for split in ("forget10", "retain90", "holdout10"):
        deltas[split] = {}
        metrics = ("answer_probability", "truth_ratio_score", "rougeL_recall")
        for metric in metrics:
            if all(metric in row for row in base["per_example"][split]) and all(metric in row for row in adapter["per_example"][split]):
                deltas[split][f"npo_minus_full.{metric}"] = paired_cluster_delta(base["per_example"][split], adapter["per_example"][split], metric)
            if split == "forget10" and all(metric in row for row in oracle["per_example"][split]):
                deltas[split][f"npo_minus_oracle.{metric}"] = paired_cluster_delta(oracle["per_example"][split], adapter["per_example"][split], metric)
    primary = deltas["forget10"]["npo_minus_full.truth_ratio_score"]
    primary_pass = primary["profile_cluster_bootstrap_95pct_ci"][0] > 0
    gates = {}
    for path in ("retain90.answer_probability", "retain90.rougeL_recall", "holdout10.answer_probability"):
        split, field = path.split(".")
        delta = deltas[split][f"npo_minus_full.{field}"]
        low = delta["profile_cluster_bootstrap_95pct_ci"][0]
        gates[path] = {
            "delta": delta["mean_delta_b_minus_a"],
            "ci95": delta["profile_cluster_bootstrap_95pct_ci"],
            "noninferiority_margin": -0.05,
            "pass": low > -0.05,
        }
    utility_delta = float(adapter["model_utility"]["harmonic_mean"]) - float(base["model_utility"]["harmonic_mean"])
    gates["model_utility.harmonic_mean"] = {
        "baseline": float(base["model_utility"]["harmonic_mean"]),
        "npo": float(adapter["model_utility"]["harmonic_mean"]),
        "delta": utility_delta,
        "noninferiority_margin": -0.05,
        "pass": utility_delta > -0.05,
        "interval": None,
        "note": "Official composite includes utility rows without profile IDs; point-difference gate only.",
    }
    return {
        "primary_npo_minus_full": primary,
        "cluster_bootstrap_deltas": deltas,
        "utility_noninferiority_gates": gates,
        "primary_forget_gate_pass": primary_pass,
        "all_utility_gates_pass": all(item["pass"] for item in gates.values()),
        "status": "PASS_DIAGNOSTIC_GATES" if primary_pass and all(item["pass"] for item in gates.values()) else "FAIL_DIAGNOSTIC_GATES",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_KEYS, default="3b")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--objective-check", type=Path, required=True)
    args = parser.parse_args()

    problems: list[str] = []
    model_key, oracle_key = MODEL_KEYS[args.model]
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    objective = json.loads(args.objective_check.read_text(encoding="utf-8"))
    receipt_path = args.training_dir / "training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_protocol_hash = sha256(args.protocol)
    fail_if(protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN", "NPO protocol not preregistered", problems)
    fail_if(protocol.get("model_family") != args.model, "requested verifier model differs from locked protocol", problems)
    fail_if(objective.get("status") != "PASS", "pinned objective value/gradient check failed", problems)
    fail_if(objective.get("protocol_sha256") != expected_protocol_hash, "objective check used a different protocol", problems)
    fail_if(receipt.get("status") != "TRAINING_COMPLETE", "full training receipt is not complete", problems)
    fail_if(receipt.get("run_type") != "exploratory_single_seed", "run is not the locked one-seed pilot", problems)
    fail_if(receipt.get("protocol_sha256") != expected_protocol_hash, "training protocol hash mismatch", problems)
    fail_if(receipt.get("objective_equivalence") != objective, "training receipt objective check differs from supplied check", problems)
    trainer_source = args.training_dir.parent / "npo_lora_train.py"
    fail_if(not trainer_source.is_file(), "trainer source snapshot missing beside training receipt", problems)
    source_batching = None
    if trainer_source.is_file():
        fail_if(receipt.get("training_script_sha256") != sha256(trainer_source), "trainer source hash mismatch", problems)
        try:
            source_batching = trainer_batching(trainer_source, model_key)
        except (SyntaxError, ValueError, StopIteration, TypeError, IndexError) as exc:
            problems.append(f"could not verify trainer microbatch/accumulation settings: {exc}")

    if source_batching is None:
        microbatch_size, accumulation_steps = 1, 1
    else:
        microbatch_size, accumulation_steps = source_batching
    schedule = receipt.get("training_schedule", {})
    if schedule:
        fail_if(schedule.get("forget_examples_per_microbatch") != microbatch_size,
                "training receipt forget microbatch size differs from trainer source", problems)
        fail_if(schedule.get("retain_examples_per_microbatch") != microbatch_size,
                "training receipt retain microbatch size differs from trainer source", problems)
        fail_if(schedule.get("gradient_accumulation_steps") != accumulation_steps,
                "training receipt accumulation steps differ from trainer source", problems)

    model = receipt.get("model", {})
    expected_model = protocol["base_checkpoint"]
    fail_if(model.get("repo_id") != expected_model["repo_id"] or model.get("revision") != expected_model["revision"],
            "training checkpoint/revision mismatch", problems)
    fail_if(receipt.get("dataset", {}).get("repo") != protocol["dataset"]["repo_id"]
            or receipt.get("dataset", {}).get("revision") != protocol["dataset"]["revision"],
            "training dataset ID/revision mismatch", problems)
    fail_if(receipt.get("method", {}).get("beta") != protocol["objective"]["beta"]
            or receipt.get("method", {}).get("epochs") != protocol["adaptation"]["epochs"]
            or receipt.get("method", {}).get("lora_rank") != protocol["adaptation"]["lora_rank"],
            "training objective/hyperparameters differ from the locked protocol", problems)
    fail_if(receipt.get("seed") != protocol["seed"], "training seed mismatch", problems)
    expected_packages = {"transformers": "5.22.1", "accelerate": "1.15.0", "peft": "0.21.2", "scipy": "1.16.3"}
    for package, version in expected_packages.items():
        fail_if(receipt.get("package_versions", {}).get(package) != version,
                f"training package version mismatch for {package}", problems)
    fail_if(receipt.get("package_versions", {}).get("torchao") is not None,
            "optional torchao must be absent for this unquantized LoRA run", problems)
    fail_if(not str(receipt.get("package_versions", {}).get("torch", "")), "missing torch version", problems)
    fail_if(receipt.get("global_steps") != 500, f"expected 500 completed optimizer steps, found {receipt.get('global_steps')}", problems)
    fail_if(not receipt.get("training_started") or not receipt.get("adapter_saved"), "training or adapter save receipt missing", problems)
    fail_if(receipt.get("runtime", {}).get("cpu_or_disk_offload") is not False, "training used or failed to rule out CPU/disk offload", problems)
    fail_if(len(receipt.get("gpu_names", [])) < 2 or receipt.get("gpu_names", []).count("Tesla T4") < 2,
            "training receipt does not show Kaggle T4 x2", problems)
    fail_if(receipt.get("tokenization", {}).get("truncated_examples") != 0, "training data was truncated", problems)
    fail_if(receipt.get("tokenization", {}).get("max_length_observed", 999999) > 512, "training sequence exceeded locked maximum length", problems)
    train_data = receipt.get("dataset", {})
    locked_training_files = protocol["dataset"]["training"]
    fail_if(train_data.get("forget10_sha256") != locked_training_files["forget_file"]["sha256"], "forget10 training data hash mismatch", problems)
    fail_if(train_data.get("retain90_sha256") != locked_training_files["retain_file"]["sha256"], "retain90 training data hash mismatch", problems)
    fail_if(train_data.get("forget10_rows") != 400 or train_data.get("retain90_rows") != 3600,
            "training data row counts differ from protocol", problems)
    epochs = int(protocol["adaptation"]["epochs"])
    forget_count = int(train_data.get("forget10_rows", 0))
    microbatches_per_epoch = math.ceil(forget_count / microbatch_size)
    steps_per_epoch = math.ceil(microbatches_per_epoch / accumulation_steps)
    expected_steps = steps_per_epoch * epochs
    expected_trace_rows = microbatches_per_epoch * epochs
    fail_if(microbatch_size * accumulation_steps != protocol["adaptation"]["effective_paired_batch_size"],
            "trainer batching does not match the preregistered effective batch size", problems)
    fail_if(expected_steps != 500, f"protocol and trainer imply {expected_steps} optimizer steps, expected 500", problems)
    if schedule:
        fail_if(schedule.get("epochs") != epochs, "training receipt epoch count differs from protocol", problems)
        fail_if(schedule.get("microbatches_per_epoch") != microbatches_per_epoch,
                "training receipt microbatch count differs from trainer/data", problems)
        fail_if(schedule.get("optimizer_steps_per_epoch") != steps_per_epoch,
                "training receipt optimizer steps per epoch differ from trainer/data", problems)
        fail_if(schedule.get("planned_optimizer_steps") != expected_steps,
                "training receipt planned optimizer step count differs from trainer/data", problems)
        fail_if(schedule.get("planned_loss_trace_rows") != expected_trace_rows,
                "training receipt planned loss trace row count differs from trainer/data", problems)
    device_map = receipt.get("runtime", {}).get("device_map", {})
    fail_if(not device_map or any(str(device).lower() in {"cpu", "disk"} for device in device_map.values()),
            "training device map is missing or contains CPU/disk placement", problems)

    adapter_dir = args.training_dir / "adapter"
    recorded_adapter_hashes = receipt.get("adapter_files_sha256", {})
    actual_adapter_hashes = {path.name: sha256(path) for path in sorted(adapter_dir.iterdir()) if path.is_file()} if adapter_dir.is_dir() else {}
    fail_if(not actual_adapter_hashes or actual_adapter_hashes != recorded_adapter_hashes, "adapter file hashes do not match training receipt", problems)
    trace_path = args.training_dir / "loss_trace.jsonl"
    trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trace_problems: list[str] = []
    fail_if(len(trace_rows) != expected_trace_rows,
            f"expected {expected_trace_rows} paired-microbatch loss trace rows "
            f"({forget_count} forget examples x {epochs} epochs, microbatch={microbatch_size}), "
            f"found {len(trace_rows)}", trace_problems)
    expected_by_epoch = {epoch: microbatches_per_epoch for epoch in range(1, epochs + 1)}
    observed_by_epoch = Counter(row.get("epoch") for row in trace_rows)
    fail_if(dict(observed_by_epoch) != expected_by_epoch,
            f"loss trace rows by epoch differ from schedule: {dict(observed_by_epoch)}", trace_problems)
    expected_by_step = {}
    global_step = 0
    for _ in range(epochs):
        for start in range(0, microbatches_per_epoch, accumulation_steps):
            global_step += 1
            expected_by_step[global_step] = min(accumulation_steps, microbatches_per_epoch - start)
    observed_by_step = Counter(row.get("global_step") for row in trace_rows)
    fail_if(dict(observed_by_step) != expected_by_step,
            "loss trace rows by optimizer step differ from gradient-accumulation schedule", trace_problems)
    for index, row in enumerate(trace_rows):
        for name in ("forget_npo_loss", "retain_token_nll", "learning_rate", "gradient_norm"):
            fail_if(name not in row or not math.isfinite(float(row[name])), f"loss trace row {index} has invalid {name}", trace_problems)
        for name in ("forget_npo_loss", "retain_token_nll", "combined_loss", "gradient_norm"):
            fail_if(name in row and float(row[name]) < 0, f"loss trace row {index} has negative {name}", trace_problems)
        if "forget_examples_in_microbatch" in row:
            fail_if(row["forget_examples_in_microbatch"] < 1 or row["forget_examples_in_microbatch"] > microbatch_size,
                    f"loss trace row {index} has invalid forget microbatch size", trace_problems)
            fail_if(row.get("retain_examples_in_microbatch") != row["forget_examples_in_microbatch"],
                    f"loss trace row {index} has mismatched forget/retain microbatch sizes", trace_problems)
    if trace_rows and all("forget_examples_in_microbatch" in row for row in trace_rows):
        examples_by_epoch = Counter()
        for row in trace_rows:
            examples_by_epoch[row["epoch"]] += row["forget_examples_in_microbatch"]
        fail_if(dict(examples_by_epoch) != {epoch: forget_count for epoch in range(1, epochs + 1)},
                f"loss trace example coverage by epoch is incorrect: {dict(examples_by_epoch)}", trace_problems)
    problems.extend(trace_problems)
    trace_audit = {
        "status": "PASS" if not trace_problems else "FAIL",
        "unit": schedule.get("loss_trace_unit", "one mean-loss record per paired microbatch (derived from hashed trainer source)"),
        "forget_examples_per_microbatch": microbatch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "expected_optimizer_steps": expected_steps,
        "observed_optimizer_steps": receipt.get("global_steps"),
        "expected_rows": expected_trace_rows,
        "observed_rows": len(trace_rows),
        "rows_by_epoch": dict(observed_by_epoch),
        "unique_optimizer_steps": len(observed_by_step),
        "problems": trace_problems,
    }

    base_path = args.eval_dir / f"{model_key}_official_eval.json"
    oracle_path = args.eval_dir / f"{oracle_key}_official_eval.json"
    adapter_path = args.eval_dir / f"{adapter_result_key(args.model, int(protocol['seed']))}_official_eval.json"
    comparison_path = args.eval_dir / "npo_comparison.json"
    base, oracle, adapter, comparison = [json.loads(path.read_text(encoding="utf-8")) for path in (base_path, oracle_path, adapter_path, comparison_path)]
    fail_if(base.get("status") != "PASS_OFFICIAL_TOFU_BASELINE" or oracle.get("status") != "PASS_OFFICIAL_TOFU_BASELINE",
            f"{args.model.upper()} full/oracle baseline receipts do not pass", problems)
    fail_if(adapter.get("status") != "PASS_OFFICIAL_TOFU_ADAPTER_EVAL", "NPO adapter evaluation did not pass", problems)
    fail_if(adapter.get("model", {}).get("adapter", {}).get("files_sha256") != actual_adapter_hashes,
            "evaluated adapter hashes differ from trained adapter", problems)
    fail_if(adapter.get("model", {}).get("repo_id") != expected_model["repo_id"] or adapter.get("model", {}).get("revision") != expected_model["revision"],
            "adapter evaluation checkpoint/revision mismatch", problems)
    fail_if(adapter.get("model", {}).get("adapter", {}).get("seed") != protocol["seed"],
            "adapter evaluation seed differs from locked protocol", problems)
    fail_if(adapter.get("runtime", {}).get("cpu_offload_used") is not False or adapter.get("runtime", {}).get("research_adapter_created") is not True,
            "adapter evaluation placement or adapter receipt invalid", problems)
    validate_metrics(adapter, base, protocol, problems)

    recomputed = recompute_comparison(base, adapter, oracle)
    for name in ("primary_npo_minus_full", "cluster_bootstrap_deltas", "utility_noninferiority_gates"):
        if comparison.get(name) != recomputed[name]:
            problems.append(f"recorded comparison does not reproduce for {name}")
    if comparison.get("status") != recomputed["status"]:
        problems.append("recorded diagnostic-gate status does not reproduce")
    fail_if(comparison.get("training_receipt_sha256") != sha256(receipt_path), "comparison refers to a different training receipt", problems)

    verification = {
        "overall_status": "PASS" if not problems else "FAIL",
        "research_diagnostic_gate_status": recomputed["status"],
        "research_interpretation": "Single-seed exploratory NPO-LoRA diagnostic. Artifact integrity may pass even when the prespecified scientific/utility gates fail.",
        "problems": problems,
        "training_receipt_sha256": sha256(receipt_path),
        "protocol_sha256": expected_protocol_hash,
        "objective_equivalence": objective,
        "loss_trace_audit": trace_audit,
        "verified_splits": SPLIT_COUNTS,
        "primary_result": recomputed["primary_npo_minus_full"],
        "utility_gates": recomputed["utility_noninferiority_gates"],
    }
    out = args.eval_dir / "npo_verification.json"
    out.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    report = [
        "# TOFU NPO-LoRA pilot verification",
        "",
        f"Artifact verification: **{verification['overall_status']}**",
        f"Prespecified diagnostic gates: **{verification['research_diagnostic_gate_status']}**",
        f"Loss trace audit: **{trace_audit['status']}** ({trace_audit['observed_rows']} paired-microbatch records; "
        f"{microbatch_size} forget examples per microbatch; {trace_audit['unique_optimizer_steps']} optimizer steps).",
        "",
        "A PASS here means provenance, saved outputs, official per-row formulas, paired profile-bootstrap results, and artifact hashes reproduce. It does not by itself mean the NPO method passed its scientific or utility gates.",
        "",
        "## Primary outcome",
        "",
        f"NPO minus full Truth Ratio score: {recomputed['primary_npo_minus_full']['mean_delta_b_minus_a']:.6f} "
        f"[{recomputed['primary_npo_minus_full']['profile_cluster_bootstrap_95pct_ci'][0]:.6f}, "
        f"{recomputed['primary_npo_minus_full']['profile_cluster_bootstrap_95pct_ci'][1]:.6f}], "
        f"{recomputed['primary_npo_minus_full']['n_questions']} rows across "
        f"{recomputed['primary_npo_minus_full']['n_profile_blocks']} profile blocks.",
        "",
        "## Utility gates",
        "",
    ]
    for name, gate in recomputed["utility_noninferiority_gates"].items():
        report.append(f"- `{name}`: delta {gate['delta']:.6f}, pass={gate['pass']}.")
    if problems:
        report.extend(["", "## Verification problems", ""])
        report.extend(f"- {problem}" for problem in problems)
    (args.eval_dir / "npo_verification.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
