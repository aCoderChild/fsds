#!/usr/bin/env python3
"""Validate the locked TOFU profile-cohort recovery design and source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


DATASET_ID = "locuslab/TOFU"
DATASET_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"


def get_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "fsds-tofu-cohort-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def fetch(protocol: dict, filename: str) -> tuple[list[dict], str]:
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{filename}"
    raw = get_bytes(url)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, hashlib.sha256(raw).hexdigest()


def expand(ranges: list[list[int]]) -> list[int]:
    blocks = [block for first, last in ranges for block in range(first, last + 1)]
    if len(blocks) != len(set(blocks)):
        raise ValueError("A profile assignment range contains duplicate blocks")
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("npo_lora_recovery_protocol.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        raise ValueError("Recovery protocol is not in its required locked-before-GPU status")
    if protocol["data"]["repo_id"] != DATASET_ID or protocol["data"]["revision"] != DATASET_REVISION:
        raise ValueError("Protocol data source does not match the pinned TOFU evaluator")
    size = int(protocol["data"]["profile_block_size"])
    rows_by_file = {}
    observed_hashes = {}
    for filename, expected_count in protocol["data"]["expected_rows"].items():
        rows, digest = fetch(protocol, filename)
        if len(rows) != expected_count:
            raise ValueError(f"{filename}: expected {expected_count} rows, got {len(rows)}")
        if digest != protocol["data"]["sha256"][filename]:
            raise ValueError(f"{filename}: SHA-256 mismatch")
        if len(rows) % size:
            raise ValueError(f"{filename}: rows do not form complete {size}-QA profiles")
        rows_by_file[filename] = rows
        observed_hashes[filename] = digest

    assignments = protocol["data"]["profile_assignment_ranges_inclusive"]
    forget_dev = expand(assignments["forget10_dev_train_and_eval"])
    forget_final = expand(assignments["forget10_final_train_and_eval"])
    retain_train = expand(assignments["retain90_optimizer_train"])
    retain_dev = expand(assignments["retain90_dev_utility_eval"])
    retain_final = expand(assignments["retain90_final_utility_eval"])
    holdout_dev = expand(assignments["holdout10_dev_utility_eval"])
    holdout_final = expand(assignments["holdout10_final_utility_eval"])
    retain_excluded = set(assignments["retain90_excluded_conflicting_question_profiles"])

    if set(forget_dev) & set(forget_final) or sorted(forget_dev + forget_final) != list(range(20)):
        raise ValueError("Forget10 dev/final assignment must partition profile blocks 0..19 exactly")
    if set(retain_train) & set(retain_dev) or set(retain_train) & set(retain_final) or set(retain_dev) & set(retain_final):
        raise ValueError("Retain optimizer, dev, and final profile assignments overlap")
    if retain_excluded & (set(retain_train) | set(retain_dev) | set(retain_final)):
        raise ValueError("Conflicting-question retain profiles must be excluded from all pools")
    if set(retain_train) | set(retain_dev) | set(retain_final) | retain_excluded != set(range(180)):
        raise ValueError("Retain profile assignment must account for all 180 profiles")
    if set(holdout_dev) & set(holdout_final) or sorted(holdout_dev + holdout_final) != list(range(20)):
        raise ValueError("Holdout dev/final assignment must partition profile blocks 0..19 exactly")

    retain_rows = rows_by_file["retain90.json"]
    answers_by_question: dict[str, set[str]] = {}
    for row in retain_rows:
        answers_by_question.setdefault(row["question"], set()).add(row["answer"])
    collision_indices = [i for i, row in enumerate(retain_rows) if len(answers_by_question[row["question"]]) > 1]
    collision_profiles = sorted({i // size for i in collision_indices})
    if collision_profiles != sorted(retain_excluded):
        raise ValueError(f"Conflicting-question profiles differ from protocol: {collision_profiles}")

    forget = rows_by_file["forget10.json"]
    forget_perturbed = rows_by_file["forget10_perturbed.json"]
    if any((a["question"], a["answer"]) != (b["question"], b["answer"]) for a, b in zip(forget, forget_perturbed)):
        raise ValueError("Forget10 perturbed rows do not align with forget10 in source order")

    counts = {
        "dev": {
            "forget10": len(forget_dev) * size,
            "retain90": len(retain_dev) * size,
            "holdout10": len(holdout_dev) * size,
            "retain_optimizer": len(retain_train) * size,
        },
        "final": {
            "forget10": len(forget_final) * size,
            "retain90": len(retain_final) * size,
            "holdout10": len(holdout_final) * size,
            "retain_optimizer": len(retain_train) * size,
        },
    }
    if counts != {"dev": {"forget10": 200, "retain90": 400, "holdout10": 200, "retain_optimizer": 2760},
                  "final": {"forget10": 200, "retain90": 400, "holdout10": 200, "retain_optimizer": 2760}}:
        raise ValueError(f"Unexpected source-cohort sizes: {counts}")

    receipt = {
        "status": "PASS_PROFILE_COHORT_DATA_AUDIT",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "file_sha256": observed_hashes,
        "profile_block_size": size,
        "profile_counts": {
            "forget10": len(forget) // size,
            "retain90": len(retain_rows) // size,
            "holdout10": len(rows_by_file["holdout10.json"]) // size,
        },
        "assignments": assignments,
        "conflicting_question_source_rows_excluded": collision_indices,
        "counts": counts,
        "training_started": False,
        "model_weights_loaded": False,
    }
    serialized = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
