#!/usr/bin/env python3
"""Independently verify downloaded TOFU baseline receipts and score files."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import zipfile
from pathlib import Path


EXPECTED = {
    "llama_3_2_3b": {
        "repo": "open-unlearning/tofu_Llama-3.2-3B-Instruct_full",
        "revision": "24f31ca19f6966dcb6f6b29abc511cce71222d4a",
    },
    "llama_3_1_8b": {
        "repo": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
        "revision": "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e",
    },
}
EXPECTED_ROWS = {"forget10": 400, "retain90": 400, "holdout10": 400}


def read_scores(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{i}: invalid JSON: {error}") from error
    return rows


def bootstrap_ci(profile_means: dict[int, float], seed: int = 42, draws: int = 5000) -> list[float]:
    values = list(profile_means.values())
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return [estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]]


def recompute(rows: list[dict], split: str) -> dict:
    subset = [row for row in rows if row["split"] == split]
    grouped: dict[int, list[float]] = collections.defaultdict(list)
    for row in subset:
        grouped[int(row["profile_block"])].append(float(row["answer_probability"]))
    profile_means = {group: sum(values) / len(values) for group, values in grouped.items()}
    return {
        "n_questions": len(subset),
        "n_profile_blocks": len(profile_means),
        "mean_answer_probability": sum(row["answer_probability"] for row in subset) / len(subset),
        "mean_answer_token_loss": sum(row["avg_answer_token_loss"] for row in subset) / len(subset),
        "profile_cluster_bootstrap_95pct_ci_answer_probability": bootstrap_ci(profile_means),
    }


def verify_model(result_dir: Path, key: str, audit: dict) -> dict:
    result_path = result_dir / f"{key}_baseline.json"
    score_path = result_dir / f"{key}_scores.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = read_scores(score_path)
    problems = []
    expected = EXPECTED[key]
    if result.get("status") != "PASS_BASELINE_DIAGNOSTIC":
        problems.append(f"unexpected result status: {result.get('status')}")
    if result.get("model", {}).get("repo_id") != expected["repo"]:
        problems.append("model repository does not match the pinned protocol")
    if result.get("model", {}).get("revision") != expected["revision"]:
        problems.append("model revision does not match the pinned protocol")
    if result.get("runtime", {}).get("training_started") is not False:
        problems.append("receipt does not affirm that training was not started")
    if result.get("runtime", {}).get("research_adapter_created") is not False:
        problems.append("receipt does not affirm that no research adapter was created")

    for split, n in EXPECTED_ROWS.items():
        subset = [row for row in rows if row.get("split") == split]
        if len(subset) != n:
            problems.append(f"{split}: expected {n} rows, found {len(subset)}")
        if len({row.get("row_index") for row in subset}) != len(subset):
            problems.append(f"{split}: duplicate row indices")
        groups = {int(row["profile_block"]) for row in subset}
        if len(groups) != 20:
            problems.append(f"{split}: expected 20 profile blocks, found {len(groups)}")
        for i, row in enumerate(subset):
            prob = row.get("answer_probability")
            loss = row.get("avg_answer_token_loss")
            tokens = row.get("scored_answer_tokens")
            if not isinstance(prob, (int, float)) or not math.isfinite(prob) or not 0 <= prob <= 1:
                problems.append(f"{split} row {i}: invalid probability")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
                problems.append(f"{split} row {i}: invalid token loss")
            if not isinstance(tokens, int) or tokens <= 0:
                problems.append(f"{split} row {i}: invalid answer-token count")
        recomputed = recompute(rows, split)
        recorded = result.get("summary", {}).get(split, {})
        for field in ("n_questions", "n_profile_blocks"):
            if recorded.get(field) != recomputed[field]:
                problems.append(f"{split}: recorded {field} does not match score rows")
        for field in ("mean_answer_probability", "mean_answer_token_loss"):
            if not math.isclose(float(recorded.get(field, float("nan"))), recomputed[field], rel_tol=1e-10, abs_tol=1e-12):
                problems.append(f"{split}: recorded {field} does not reproduce")
        ci = recorded.get("profile_cluster_bootstrap_95pct_ci_answer_probability", [])
        if len(ci) != 2 or any(not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(ci, recomputed["profile_cluster_bootstrap_95pct_ci_answer_probability"])):
            problems.append(f"{split}: cluster-bootstrap interval does not reproduce")
        expected_hash = audit["data_audit"]["files"][f"{split}.json"]["sha256"]
        actual_hash = result.get("scoring", {}).get("dataset_sha256_by_split", {}).get(split)
        if actual_hash != expected_hash:
            problems.append(f"{split}: dataset checksum mismatch")

    clean = result.get("summary", {}).get("retain90_ambiguous_questions_excluded")
    ambiguous = [row for row in rows if row["split"] == "retain90" and row.get("ambiguous_question_collision")]
    if len(ambiguous) != 2:
        problems.append("ambiguous-question rows/sensitivity sample do not match the expected two-row exclusion")
    if (clean or {}).get("n_questions") != 398:
        problems.append("retain collision sensitivity must contain 398 questions")
    else:
        retain_clean_rows = [
            row for row in rows
            if row["split"] == "retain90" and not row.get("ambiguous_question_collision")
        ]
        recomputed_clean = recompute(retain_clean_rows, "retain90")
        for field in ("n_questions", "n_profile_blocks"):
            if clean.get(field) != recomputed_clean[field]:
                problems.append(f"retain collision sensitivity: recorded {field} does not match score rows")
        for field in ("mean_answer_probability",):
            if not math.isclose(float(clean.get(field, float("nan"))), recomputed_clean[field], rel_tol=1e-10, abs_tol=1e-12):
                problems.append(f"retain collision sensitivity: recorded {field} does not reproduce")
        ci = clean.get("profile_cluster_bootstrap_95pct_ci_answer_probability", [])
        expected_ci = recomputed_clean["profile_cluster_bootstrap_95pct_ci_answer_probability"]
        if len(ci) != 2 or any(not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(ci, expected_ci)):
            problems.append("retain collision sensitivity: cluster-bootstrap interval does not reproduce")

    return {
        "model_key": key,
        "model_repo": expected["repo"],
        "model_revision": expected["revision"],
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "n_scored_questions": len(rows),
        "summary": result.get("summary", {}),
        "runtime": result.get("runtime", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Unzipped Kaggle output directory")
    parser.add_argument("--allow-incomplete", action="store_true", help="Verify any completed model receipt")
    args = parser.parse_args()
    result_dir = args.result_dir
    audit_path = result_dir / "audit" / "tofu_readiness_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    model_results = {}
    for key in EXPECTED:
        path = result_dir / f"{key}_baseline.json"
        if not path.exists():
            model_results[key] = {"status": "MISSING", "problems": [f"missing {path.name}"]}
        else:
            model_results[key] = verify_model(result_dir, key, audit)
    all_pass = all(item["status"] == "PASS" for item in model_results.values())
    if not args.allow_incomplete and not all_pass:
        overall = "FAIL_OR_INCOMPLETE"
    elif all(item["status"] in ("PASS", "MISSING") for item in model_results.values()):
        overall = "PASS_PARTIAL" if not all_pass else "PASS"
    else:
        overall = "FAIL"

    report = {
        "overall_status": overall,
        "data_audit_status": "PASS" if audit["data_audit"]["checks"]["structural_checks_pass"] else "FAIL",
        "models": model_results,
    }
    out_json = result_dir / "baseline_diagnostic_verification.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# TOFU baseline diagnostic verification", "", f"Overall: **{overall}**", ""]
    for key, item in model_results.items():
        lines.append(f"## {key}: {item['status']}")
        lines.append("")
        for message in item.get("problems", []):
            lines.append(f"- {message}")
        if item.get("summary"):
            lines += ["", "| Split | Questions | Profile blocks | Mean answer probability | Profile-bootstrap 95% CI |", "|---|---:|---:|---:|---:|"]
            for split in ("forget10", "retain90", "holdout10"):
                s = item["summary"][split]
                ci = s["profile_cluster_bootstrap_95pct_ci_answer_probability"]
                lines.append(f"| {split} | {s['n_questions']} | {s['n_profile_blocks']} | {s['mean_answer_probability']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
            clean = item["summary"].get("retain90_ambiguous_questions_excluded")
            if clean:
                ci = clean["profile_cluster_bootstrap_95pct_ci_answer_probability"]
                lines.append(f"| retain90, conflicting question rows removed | {clean['n_questions']} | {clean['n_profile_blocks']} | {clean['mean_answer_probability']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
        lines.append("")
    (result_dir / "baseline_diagnostic_verification.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if overall in ("FAIL", "FAIL_OR_INCOMPLETE"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
