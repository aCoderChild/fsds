#!/usr/bin/env python3
"""Check whether the locked 8B forget Truth Ratio score points toward the oracle.

This is a baseline-only directional sanity check. It does not use or change the
preregistered NPO outcome, thresholds, or interpretation of the eventual run.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
BASELINE_DIR = ROOT / "experiments/tofu_scaling/kaggle/npo_lora_8b_pilot/baseline_input"
PROTOCOL_PATH = ROOT / "experiments/tofu_scaling/npo_lora_8b_pilot_protocol.json"
OUTPUT_PATH = HERE / "baseline_direction_sanity.json"
FULL_FILE = "llama_3_1_8b_full_official_eval.json"
ORACLE_FILE = "llama_3_1_8b_retain90_official_eval.json"
DRAWS = 5_000
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected a nonempty finite metric vector")
    return sum(values) / len(values)


def main() -> None:
    manifest_path = BASELINE_DIR / "npo_8b_baseline_manifest.json"
    source_path = BASELINE_DIR / "source_verification_receipt.json"
    manifest = load_json(manifest_path)
    source = load_json(source_path)
    protocol = load_json(PROTOCOL_PATH)
    if manifest.get("status") != "PASS" or manifest.get("source_independent_verification_status") != "PASS":
        raise RuntimeError("8B baseline-input manifest did not pass")
    if source.get("overall_status") != "PASS":
        raise RuntimeError("Independent source verification did not pass")
    for key in ("llama_3_1_8b_full", "llama_3_1_8b_retain90"):
        if source.get("models", {}).get(key, {}).get("status") != "PASS":
            raise RuntimeError(f"Independent verification failed for {key}")
    for relative, expected in manifest["files"].items():
        path = BASELINE_DIR / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Baseline input is missing or hash-mismatched: {relative}")

    full = load_json(BASELINE_DIR / FULL_FILE)
    oracle = load_json(BASELINE_DIR / ORACLE_FILE)
    if full.get("status") != "PASS_OFFICIAL_TOFU_BASELINE" or oracle.get("status") != "PASS_OFFICIAL_TOFU_BASELINE":
        raise RuntimeError("Full/oracle official baseline evaluation did not pass")
    if full.get("evaluation", {}).get("dataset_revision") != protocol["dataset"]["revision"]:
        raise RuntimeError("Full baseline dataset revision differs from the locked protocol")
    if oracle.get("evaluation", {}).get("dataset_revision") != protocol["dataset"]["revision"]:
        raise RuntimeError("Oracle baseline dataset revision differs from the locked protocol")

    full_rows = full["per_example"]["forget10"]
    oracle_rows = oracle["per_example"]["forget10"]
    if len(full_rows) != 400 or len(oracle_rows) != 400:
        raise RuntimeError("Expected 400 forget10 rows for both checkpoints")
    row_keys = lambda rows: [(row["source_row_index"], row["question_sha256"]) for row in rows]
    if row_keys(full_rows) != row_keys(oracle_rows):
        raise RuntimeError("Full and oracle rows are not aligned by source row and question")

    deltas_by_block: dict[int, list[float]] = defaultdict(list)
    full_scores: list[float] = []
    oracle_scores: list[float] = []
    for full_row, oracle_row in zip(full_rows, oracle_rows):
        full_score = float(full_row["truth_ratio_score"])
        oracle_score = float(oracle_row["truth_ratio_score"])
        block = int(full_row["profile_block"])
        if not (0.0 <= full_score <= 1.0 and 0.0 <= oracle_score <= 1.0):
            raise ValueError("Forget Truth Ratio scores must be in [0, 1]")
        if int(oracle_row["profile_block"]) != block:
            raise RuntimeError("Full/oracle profile-block assignments differ")
        full_scores.append(full_score)
        oracle_scores.append(oracle_score)
        deltas_by_block[block].append(oracle_score - full_score)

    if len(deltas_by_block) != 20 or any(len(rows) != 20 for rows in deltas_by_block.values()):
        raise RuntimeError("Expected 20 profile blocks with 20 forget rows each")
    point = mean([value for rows in deltas_by_block.values() for value in rows])
    rng = random.Random(SEED)
    blocks = sorted(deltas_by_block)
    bootstrap_means = []
    for _ in range(DRAWS):
        sampled = [rng.choice(blocks) for _ in blocks]
        values = [delta for block in sampled for delta in deltas_by_block[block]]
        bootstrap_means.append(mean(values))
    bootstrap_means.sort()

    result = {
        "status": "PASS_BASELINE_DIRECTION_SANITY",
        "scope": "Supplemental baseline-only direction check; not the preregistered NPO effect test.",
        "not_preregistered_primary_test": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "baseline_manifest_sha256": sha256(manifest_path),
        "source_verification_sha256": sha256(source_path),
        "full_baseline_sha256": sha256(BASELINE_DIR / FULL_FILE),
        "oracle_baseline_sha256": sha256(BASELINE_DIR / ORACLE_FILE),
        "metric": "forget10 truth_ratio_score; larger values indicate less distinction between paraphrased and perturbed answer likelihoods under the locked formula.",
        "full_mean": mean(full_scores),
        "retain90_oracle_mean": mean(oracle_scores),
        "oracle_minus_full_mean_delta": point,
        "paired_profile_block_bootstrap_95pct_ci": [
            bootstrap_means[int(0.025 * (DRAWS - 1))],
            bootstrap_means[int(0.975 * (DRAWS - 1))],
        ],
        "n_questions": len(full_scores),
        "n_profile_blocks": len(deltas_by_block),
        "bootstrap_draws": DRAWS,
        "bootstrap_seed": SEED,
        "interpretation": (
            "The oracle baseline has a higher mean forget10 score than the full checkpoint. "
            "A positive NPO-minus-full change is directionally consistent with this oracle contrast, "
            "but alone does not establish that the NPO adapter moved closer to the oracle."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
