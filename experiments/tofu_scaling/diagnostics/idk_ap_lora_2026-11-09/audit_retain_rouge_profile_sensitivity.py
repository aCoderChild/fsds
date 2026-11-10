#!/usr/bin/env python3
"""Post-hoc leave-one-profile-out audit for final-cohort retain ROUGE-L.

This is descriptive sensitivity analysis only. It does not change the
preregistered bootstrap gate or authorize method selection on the final cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path


EXPECTED_ARCHIVE_SHA256 = "0eb6f7ee9f34fef1aa5497ba068d3f691ed92ada9ea04105174c02654995bdb3"
RUN_ROOT = "tofu_3b_idk_ap_lora_diagnostic_run/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual_hash = sha256(args.archive)
    if actual_hash != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit(f"Archive hash mismatch: expected {EXPECTED_ARCHIVE_SHA256}, got {actual_hash}")

    with zipfile.ZipFile(args.archive) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SystemExit(f"ZIP CRC failure in {bad_member}")
        reference = load_json(archive, RUN_ROOT + "reference_support/full_reference_profile_eval.json")
        candidate = load_json(archive, RUN_ROOT + "final/evaluation/final_idk_ap_profile_eval.json")
        comparison = load_json(archive, RUN_ROOT + "final/evaluation/final_idk_ap_comparison.json")

    base_rows = reference["per_example"]["retain90"]
    candidate_rows = candidate["per_example"]["retain90"]
    base = {int(row["source_row_index"]): row for row in base_rows}
    if len(base) != len(base_rows):
        raise SystemExit("Duplicate source row IDs in baseline retain rows")

    grouped: dict[int, list[float]] = defaultdict(list)
    seen: set[int] = set()
    for row in candidate_rows:
        row_id = int(row["source_row_index"])
        if row_id in seen or row_id not in base:
            raise SystemExit(f"Duplicate or unmatched candidate retain row {row_id}")
        seen.add(row_id)
        block = int(row["profile_block"])
        delta = float(row["rougeL_recall"]) - float(base[row_id]["rougeL_recall"])
        if not math.isfinite(delta):
            raise SystemExit(f"Non-finite ROUGE-L difference at row {row_id}")
        grouped[block].append(delta)

    if len(candidate_rows) != 400 or len(grouped) != 20 or set(map(len, grouped.values())) != {20}:
        raise SystemExit("Expected 400 matched retain rows across 20 blocks of 20")

    profile_means = {block: sum(values) / len(values) for block, values in grouped.items()}
    pooled = sum(sum(values) for values in grouped.values()) / len(candidate_rows)
    leave_one_out = {
        block: sum(value for other, value in profile_means.items() if other != block) / (len(profile_means) - 1)
        for block in profile_means
    }

    recorded = comparison["gates"]["retain90.rougeL_recall"]
    if abs(pooled - float(recorded["mean_delta_candidate_minus_full"])) > 1e-12:
        raise SystemExit("Independent pooled ROUGE-L point estimate does not match the recorded gate")

    result = {
        "status": "PASS_POSTHOC_RECOMPUTATION",
        "analysis": "descriptive final-cohort profile influence; not preregistered and not a replacement gate",
        "archive_sha256": actual_hash,
        "cohort": "final",
        "metric": "retain90.rougeL_recall",
        "n_questions": len(candidate_rows),
        "n_profile_blocks": len(profile_means),
        "pooled_delta_candidate_minus_full_pp": 100 * pooled,
        "preregistered_gate_ci95_pp": [100 * float(x) for x in recorded["ci95"]],
        "profile_blocks_with_negative_mean": sum(value < 0 for value in profile_means.values()),
        "profile_block_delta_pp_min": 100 * min(profile_means.values()),
        "profile_block_delta_pp_max": 100 * max(profile_means.values()),
        "leave_one_profile_out_pooled_mean_pp_min": 100 * min(leave_one_out.values()),
        "leave_one_profile_out_pooled_mean_pp_max": 100 * max(leave_one_out.values()),
        "profile_block_means_pp": {str(k): 100 * v for k, v in sorted(profile_means.items())},
        "interpretation": "The point estimate remains negative after omitting any single profile block. This does not make the failed preregistered non-inferiority gate pass.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
