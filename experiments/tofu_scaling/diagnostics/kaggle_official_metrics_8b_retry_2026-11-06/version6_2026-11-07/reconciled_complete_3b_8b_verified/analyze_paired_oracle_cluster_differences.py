#!/usr/bin/env python3
"""Paired profile-block bootstrap for full-vs-retain90-oracle forget scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


FIELDS = ("answer_probability", "truth_ratio", "truth_ratio_score")
PAIRS = {
    "llama_3_2_3b": ("llama_3_2_3b_full", "llama_3_2_3b_retain90"),
    "llama_3_1_8b": ("llama_3_1_8b_full", "llama_3_1_8b_retain90"),
}
N_DRAWS = 5000
SEED = 42


def load_rows(root: Path, key: str) -> tuple[list[dict], str]:
    path = root / f"{key}_official_eval.json"
    raw = path.read_bytes()
    data = json.loads(raw)
    return data["per_example"]["forget10"], hashlib.sha256(raw).hexdigest()


def summarize(full: list[dict], oracle: list[dict]) -> dict:
    if len(full) != 400 or len(oracle) != 400:
        raise ValueError("Expected 400 matched forget10 rows in each checkpoint")
    identity = lambda row: (row["source_row_index"], row["question_sha256"], row["profile_block"])
    if [identity(row) for row in full] != [identity(row) for row in oracle]:
        raise ValueError("Full/oracle source questions or profile groups do not align")
    groups = sorted({int(row["profile_block"]) for row in full})
    if len(groups) != 20:
        raise ValueError(f"Expected 20 profile blocks, found {len(groups)}")

    output = {}
    for field in FIELDS:
        diffs_by_group = {
            group: [
                float(left[field]) - float(right[field])
                for left, right in zip(full, oracle)
                if int(left["profile_block"]) == group
            ]
            for group in groups
        }
        if any(len(values) != 20 for values in diffs_by_group.values()):
            raise ValueError(f"{field}: expected 20 paired rows in each profile block")
        group_means = {g: sum(v) / len(v) for g, v in diffs_by_group.items()}
        estimate = sum(group_means.values()) / len(groups)
        rng = random.Random(SEED)
        draws = []
        for _ in range(N_DRAWS):
            sample = [rng.choice(groups) for _ in groups]
            draws.append(sum(group_means[g] for g in sample) / len(groups))
        draws.sort()
        output[field] = {
            "full_minus_oracle_mean": estimate,
            "profile_cluster_bootstrap_95pct_ci": [
                draws[int(0.025 * (N_DRAWS - 1))],
                draws[int(0.975 * (N_DRAWS - 1))],
            ],
            "n_questions": 400,
            "n_matched_profile_blocks": 20,
            "bootstrap_draws": N_DRAWS,
            "bootstrap_seed": SEED,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Directory with the four verified result JSON files")
    args = parser.parse_args()
    result = {
        "analysis": "paired full-minus-retain90-oracle profile-block bootstrap on forget10",
        "interpretation": "Positive answer-probability differences mean the full checkpoint assigns greater likelihood to the target reference answers than its matching retain-only oracle. Intervals resample matched profile blocks, not individual questions.",
        "pairs": {},
    }
    for family, (full_key, oracle_key) in PAIRS.items():
        full, full_hash = load_rows(args.result_dir, full_key)
        oracle, oracle_hash = load_rows(args.result_dir, oracle_key)
        result["pairs"][family] = {
            "full_result_sha256": full_hash,
            "oracle_result_sha256": oracle_hash,
            "metrics": summarize(full, oracle),
        }
    out = args.result_dir / "paired_oracle_cluster_differences.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
