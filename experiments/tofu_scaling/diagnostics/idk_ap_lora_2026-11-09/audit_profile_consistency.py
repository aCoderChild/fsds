#!/usr/bin/env python3
"""Post-hoc, no-GPU profile consistency audit for the completed 3B IDK+AP run.

This audit does not change preregistered gates. It checks whether paired metric
changes are directionally consistent across the independent TOFU profile blocks
and reproduces each pooled final-cohort difference from the archived rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "kaggle_run2" / "tofu_3b_idk_ap_lora_diagnostic.zip"
VERIFICATION = HERE / "kaggle_run2" / "LOCAL_VERIFICATION_RECHECK_CURRENT.json"
EXPECTED_ARCHIVE_SHA256 = "0eb6f7ee9f34fef1aa5497ba068d3f691ed92ada9ea04105174c02654995bdb3"
RUN_ROOT = "tofu_3b_idk_ap_lora_diagnostic_run/"
PROFILE_SIZE = 20


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_two_sided_sign_p(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def read_json(zf: zipfile.ZipFile, name: str) -> dict:
    return json.loads(zf.read(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--verification", type=Path, default=VERIFICATION)
    parser.add_argument("--output", type=Path, default=HERE / "profile_consistency_posthoc.json")
    args = parser.parse_args()

    archive_bytes = args.archive.read_bytes()
    archive_hash = sha256(archive_bytes)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"Archive is not the locked completed run: {archive_hash}")
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS_ARTIFACT_AND_RESULT_RECOMPUTATION":
        raise ValueError("Independent artifact/result verification is not PASS")
    if verification.get("artifact_archive", {}).get("sha256") != archive_hash:
        raise ValueError("Verification receipt belongs to another archive")

    specs = [
        ("forget10.truth_ratio_score", "forget10", "truth_ratio_score", "increase"),
        ("forget10.answer_probability", "forget10", "answer_probability", "decrease"),
        ("retain90.answer_probability", "retain90", "answer_probability", "utility"),
        ("retain90.rougeL_recall", "retain90", "rougeL_recall", "utility"),
        ("holdout10.answer_probability", "holdout10", "answer_probability", "utility"),
    ]
    result_metrics = {}

    with zipfile.ZipFile(args.archive) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed at {bad}")
        candidate = read_json(zf, RUN_ROOT + "final/evaluation/final_idk_ap_profile_eval.json")
        reference = read_json(zf, RUN_ROOT + "reference_support/full_reference_profile_eval.json")
        comparison = read_json(zf, RUN_ROOT + "final/evaluation/final_idk_ap_comparison.json")
        if candidate.get("status") != "PASS_PROFILE_COHORT_EVAL" or reference.get("status") != "PASS_PROFILE_COHORT_EVAL":
            raise ValueError("Candidate or reference evaluation status is not PASS")

        recorded_gates = comparison["gates"]
        for gate_name, split, metric, direction in specs:
            ref_rows = reference["per_example"].get(split, [])
            cand_rows = candidate["per_example"].get(split, [])
            ref_by_index = {int(row["source_row_index"]): row for row in ref_rows}
            cand_by_index = {int(row["source_row_index"]): row for row in cand_rows}
            if len(ref_by_index) != len(ref_rows) or len(cand_by_index) != len(cand_rows):
                raise ValueError(f"Duplicate source row in {gate_name}")
            if not set(cand_by_index).issubset(ref_by_index):
                raise ValueError(f"Candidate contains source rows absent from reference in {gate_name}")

            per_profile: dict[int, list[float]] = defaultdict(list)
            for index, cand_row in cand_by_index.items():
                ref_row = ref_by_index[index]
                if cand_row.get("question_sha256") != ref_row.get("question_sha256"):
                    raise ValueError(f"Question hash mismatch for {gate_name}, row {index}")
                profile = int(cand_row["profile_block"])
                if profile != int(ref_row["profile_block"]) or index // PROFILE_SIZE != profile:
                    raise ValueError(f"Profile assignment mismatch for {gate_name}, row {index}")
                delta = float(cand_row[metric]) - float(ref_row[metric])
                if not math.isfinite(delta):
                    raise ValueError(f"Non-finite paired delta in {gate_name}, row {index}")
                per_profile[profile].append(delta)

            pooled = [value for block in sorted(per_profile) for value in per_profile[block]]
            pooled_delta = statistics.mean(pooled)
            gate_delta = float(recorded_gates[gate_name]["mean_delta_candidate_minus_full"])
            if abs(pooled_delta - gate_delta) > 1e-12:
                raise ValueError(f"Pooled delta differs from independent verifier for {gate_name}")
            profile_deltas = {
                str(profile): {
                    "n_questions": len(values),
                    "mean_delta_pp": 100 * statistics.mean(values),
                }
                for profile, values in sorted(per_profile.items())
            }
            signs = [entry["mean_delta_pp"] for entry in profile_deltas.values()]
            positive = sum(value > 0 for value in signs)
            negative = sum(value < 0 for value in signs)
            tied = len(signs) - positive - negative
            if direction == "increase":
                expected_direction_count = positive
            elif direction == "decrease":
                expected_direction_count = negative
            else:
                expected_direction_count = None

            leave_one_out = []
            for omitted in per_profile:
                remaining = [value for profile, values in per_profile.items() if profile != omitted for value in values]
                leave_one_out.append(100 * statistics.mean(remaining))
            result_metrics[gate_name] = {
                "n_questions": len(pooled),
                "n_profiles": len(per_profile),
                "pooled_delta_pp": 100 * pooled_delta,
                "profile_mean_delta_pp_min": min(signs),
                "profile_mean_delta_pp_max": max(signs),
                "profiles_positive": positive,
                "profiles_negative": negative,
                "profiles_tied": tied,
                "profiles_in_preregistered_direction": expected_direction_count,
                "exact_two_sided_profile_sign_test_p": exact_two_sided_sign_p(positive, negative),
                "leave_one_profile_out_pooled_delta_pp_min": min(leave_one_out),
                "leave_one_profile_out_pooled_delta_pp_max": max(leave_one_out),
                "profile_deltas": profile_deltas,
            }

    result = {
        "status": "PASS_POSTHOC_PROFILE_RECOMPUTATION",
        "analysis": "Exploratory profile-level consistency audit; no GPU used; does not replace preregistered gates or support confirmation.",
        "archive_sha256": archive_hash,
        "verification_sha256": sha256(args.verification.read_bytes()),
        "profile_unit": "TOFU profile block; 20 questions per block",
        "sign_test": "Two-sided exact binomial sign test over profile mean changes; descriptive and not multiplicity-adjusted.",
        "metrics": result_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "metrics": {
        name: {key: value for key, value in details.items() if key != "profile_deltas"}
        for name, details in result_metrics.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
