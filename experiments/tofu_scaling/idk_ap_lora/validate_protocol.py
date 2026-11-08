#!/usr/bin/env python3
"""Fail-closed validator for the locked TOFU IDK+AP profile diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


EXPECTED_PROTOCOL_SHA256 = "650af06eed1d254cfdd2b47714299686925cea47b5b2675edb7c07674b01c049"

def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fetch(filename: str, revision: str) -> tuple[bytes, list[dict]]:
    url = f"https://huggingface.co/datasets/locuslab/TOFU/resolve/{revision}/{filename}"
    request = Request(url, headers={"User-Agent": "fsds-idk-ap-protocol-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{filename} contains a non-object row")
    if any(not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str) for row in rows):
        raise ValueError(f"{filename} has invalid question/answer fields")
    return raw, rows


def expand(ranges: list[list[int]]) -> list[int]:
    blocks = [block for first, last in ranges for block in range(first, last + 1)]
    if len(blocks) != len(set(blocks)):
        raise ValueError("A profile assignment contains duplicate blocks")
    return blocks


def validate(protocol_path: Path, idk_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_GPU_RUN":
        raise ValueError("Protocol is not marked preregistered before GPU work")
    if protocol.get("protocol_id") != "tofu_3b_idk_ap_lora_profile_diagnostic_v1":
        raise ValueError("Unexpected protocol ID")
    if sha256(protocol_path.read_bytes()) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Locked protocol SHA-256 mismatch; create a new protocol version for any design changes")
    data = protocol["data"]
    if data["repo_id"] != "locuslab/TOFU" or data["revision"] != "324592d84ae4f482ac7249b9285c2ecdb53e3a68":
        raise ValueError("TOFU source does not match the locked revision")
    observed_hashes: dict[str, str] = {}
    rows_by_file: dict[str, list[dict]] = {}
    for filename, expected_count in data["expected_rows"].items():
        raw, rows = fetch(filename, data["revision"])
        observed_hashes[filename] = sha256(raw)
        if observed_hashes[filename] != data["sha256"][filename]:
            raise ValueError(f"TOFU file hash mismatch: {filename}")
        if len(rows) != int(expected_count):
            raise ValueError(f"TOFU row count mismatch: {filename}: {len(rows)}")
        if len(rows) % int(data["profile_block_size"]):
            raise ValueError(f"{filename} does not contain complete profile blocks")
        rows_by_file[filename] = rows

    assignments = data["profile_assignment_ranges_inclusive"]
    forget_dev = expand(assignments["forget10_dev_train_and_eval"])
    forget_final = expand(assignments["forget10_final_train_and_eval"])
    retain_train = expand(assignments["retain90_optimizer_train"])
    retain_dev = expand(assignments["retain90_dev_utility_eval"])
    retain_final = expand(assignments["retain90_final_utility_eval"])
    holdout_dev = expand(assignments["holdout10_dev_utility_eval"])
    holdout_final = expand(assignments["holdout10_final_utility_eval"])
    excluded = set(assignments["retain90_excluded_conflicting_question_profiles"])
    if set(forget_dev) & set(forget_final) or sorted(forget_dev + forget_final) != list(range(20)):
        raise ValueError("Forget development/final blocks do not partition 0..19")
    if (set(retain_train) & set(retain_dev) or set(retain_train) & set(retain_final)
            or set(retain_dev) & set(retain_final) or excluded & (set(retain_train) | set(retain_dev) | set(retain_final))):
        raise ValueError("Retain training/evaluation blocks overlap or include excluded blocks")
    if set(retain_train) | set(retain_dev) | set(retain_final) | excluded != set(range(180)):
        raise ValueError("Retain profile assignment does not cover 0..179")
    if set(holdout_dev) & set(holdout_final) or sorted(holdout_dev + holdout_final) != list(range(20)):
        raise ValueError("Holdout development/final blocks do not partition 0..19")

    retain = rows_by_file["retain90.json"]
    answers_by_question: dict[str, set[str]] = {}
    for row in retain:
        answers_by_question.setdefault(row["question"], set()).add(row["answer"])
    collisions = sorted({i // 20 for i, row in enumerate(retain) if len(answers_by_question[row["question"]]) > 1})
    if collisions != sorted(excluded):
        raise ValueError(f"Question/answer collision profile mismatch: found {collisions}")
    forget = rows_by_file["forget10.json"]
    forget_perturbed = rows_by_file["forget10_perturbed.json"]
    if len(forget) != len(forget_perturbed) or any(
        (a["question"], a["answer"]) != (b["question"], b["answer"])
        for a, b in zip(forget, forget_perturbed)
    ):
        raise ValueError("Forget perturbed rows are not aligned with source order")

    idk_raw = idk_path.read_bytes()
    objective = protocol["objective"]
    if sha256(idk_raw) != objective["idk_response_sha256"]:
        raise ValueError("Pinned IDK response file hash mismatch")
    idk_lines = [line for line in idk_raw.decode("utf-8").splitlines() if line.strip()]
    if len(idk_lines) != int(objective["idk_response_count"]):
        raise ValueError("Pinned IDK response count mismatch")
    idk_rows = idk_lines
    if len(idk_rows) != 100 or any(not isinstance(x, str) or not x.strip() for x in idk_rows):
        raise ValueError("Official IDK response file must contain exactly 100 nonempty newline-delimited strings")

    counts = {
        "dev": {"forget10": len(forget_dev) * 20, "retain90_eval": len(retain_dev) * 20,
                "holdout10_eval": len(holdout_dev) * 20, "retain90_optimizer": len(retain_train) * 20},
        "final": {"forget10": len(forget_final) * 20, "retain90_eval": len(retain_final) * 20,
                  "holdout10_eval": len(holdout_final) * 20, "retain90_optimizer": len(retain_train) * 20},
    }
    expected = {
        "dev": {"forget10": 200, "retain90_eval": 400, "holdout10_eval": 200, "retain90_optimizer": 2760},
        "final": {"forget10": 200, "retain90_eval": 400, "holdout10_eval": 200, "retain90_optimizer": 2760},
    }
    if counts != expected:
        raise ValueError(f"Unexpected cohort sizes: {counts}")
    return {
        "status": "PASS_IDK_AP_PROTOCOL_DATA_AUDIT",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path.read_bytes()),
        "tofu_sha256": observed_hashes,
        "idk_responses_sha256": sha256(idk_raw),
        "idk_response_count": len(idk_rows),
        "profile_assignments": assignments,
        "excluded_retain_conflict_blocks": collisions,
        "cohort_counts": counts,
        "training_started": False,
        "model_weights_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("idk_ap_lora_protocol.json"))
    parser.add_argument("--idk", type=Path, default=Path(__file__).with_name("idontknow.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.protocol, args.idk)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
