#!/usr/bin/env python3
"""Recompute source-bound training-translation coverage without assembling data."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "experiments/synthworlds_r1"
RELEASE = PROJECT / "training_release_draft"
REPAIR_ROOT = RELEASE / "translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04"
REPAIR_INPUTS = REPAIR_ROOT / "independent_review_inputs_2026-11-04"
REPAIR_PACKAGE = REPAIR_ROOT / "candidate_review_kaggle"
OUTPUT = Path(__file__).resolve().parent / "training_translation_coverage_audit.json"
sys.path.insert(0, str(RELEASE / "scripts"))
import assemble_training_release as assembler  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_live_repair_input(flagged_by_language: Counter) -> dict:
    input_path = REPAIR_INPUTS / "full799.jsonl"
    map_path = REPAIR_INPUTS / "full799.private_map.jsonl"
    input_manifest_path = REPAIR_INPUTS / "full799.manifest.json"
    launch_path = REPAIR_INPUTS / "full799.kaggle_launch_manifest.json"
    review_script_path = REPAIR_PACKAGE / "review.py"
    input_manifest = read_json(input_manifest_path)
    launch = read_json(launch_path)
    rows = read_jsonl(input_path)
    mapping = read_jsonl(map_path)
    expected_hashes = {lang: sha(RELEASE / "translation_reviews_roundtrip_qwen14b" / f"{lang}.jsonl")
                       for lang in assembler.TARGET_LANGS}

    if (input_manifest.get("status") != "MASKED_AI_REVIEW_INPUT_ONLY_NOT_APPROVED"
            or input_manifest.get("rows") != len(rows) or len(rows) != len(mapping)
            or input_manifest.get("input_sha256") != sha(input_path)
            or input_manifest.get("private_map_sha256") != sha(map_path)
            or input_manifest.get("source_review_file_sha256_by_language") != expected_hashes
            or launch.get("mode") != "full799" or launch.get("rows") != len(rows)
            or launch.get("input_jsonl_sha256") != sha(input_path)
            or launch.get("private_map_sha256") != sha(map_path)
            or launch.get("review_script_sha256") != sha(review_script_path)):
        raise ValueError("Full repair-review input, row map, source receipts, or submitted script failed integrity checks")

    assignments = {}
    for node in ast.parse(review_script_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"RAW_INPUT", "INPUT_SHA256"}:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
    raw_input = assignments.get("RAW_INPUT")
    embedded_rows = json.loads(raw_input) if isinstance(raw_input, str) else None
    compact_hash = hashlib.sha256(raw_input.encode("utf-8")).hexdigest() if isinstance(raw_input, str) else None
    flagged_rows_by_language = {lang: flagged_by_language[lang] for lang in assembler.TARGET_LANGS}
    if (embedded_rows != rows or compact_hash != assignments.get("INPUT_SHA256")
            or compact_hash != launch.get("input_compact_array_sha256")
            or flagged_rows_by_language != input_manifest.get("language_counts")
            or sum(flagged_rows_by_language.values()) != len(rows)):
        raise ValueError("Submitted Kaggle input does not exactly match the complete source-bound flagged-row queue")

    return {
        "status": "PASS_EXACT_FULL799_SUBMITTED_INPUT_AND_SOURCE_RECEIPTS",
        "rows": len(rows),
        "flagged_rows_by_language": flagged_rows_by_language,
        "input_manifest_sha256": sha(input_manifest_path),
        "input_jsonl_sha256": sha(input_path),
        "private_map_sha256": sha(map_path),
        "launch_manifest_sha256": sha(launch_path),
        "review_script_sha256": sha(review_script_path),
        "embedded_input_compact_array_sha256": compact_hash,
    }


def main() -> None:
    reviews = assembler.checked_translation_reviews()
    if len(reviews) != 4 * 488:
        raise ValueError(f"Expected 1,952 fully verified review rows, found {len(reviews)}")

    row_counts = Counter(lang for lang, _, _ in reviews)
    pass_counts = Counter(
        lang for (lang, _, _), row in reviews.items()
        if row.get("final_review_status") in assembler.PASS_STATUSES
    )
    statuses_by_language = {
        lang: dict(sorted(Counter(
            row["final_review_status"] for (row_lang, _, _), row in reviews.items() if row_lang == lang
        ).items()))
        for lang in assembler.TARGET_LANGS
    }
    flagged_by_language = Counter(
        lang for (lang, _, _), row in reviews.items()
        if row.get("final_review_status") not in assembler.PASS_STATUSES
    )

    atomic_reuse = read_jsonl(RELEASE / "atomic_reuse.jsonl")
    atomic_requests = read_jsonl(RELEASE / "atomic_translation_requests.jsonl")
    atomic_ids_by_case: dict[str, set[str]] = defaultdict(set)
    split_by_case: dict[str, str] = {}
    for row in atomic_reuse + atomic_requests:
        case_id, sft_id = row["source_qa_id"], row["sft_id"]
        atomic_ids_by_case[case_id].add(sft_id)
        split = row["split"]
        if case_id in split_by_case and split_by_case[case_id] != split:
            raise ValueError(f"Atomic split mismatch in {case_id}")
        split_by_case[case_id] = split
    if len(atomic_ids_by_case) != 156 or any(len(ids) != 2 for ids in atomic_ids_by_case.values()):
        raise ValueError("Atomic source universe differs from the frozen 117/39 case set")
    atomic_eligible = {
        case_id for case_id, ids in atomic_ids_by_case.items()
        if all(reviews.get((lang, sft_id, "atomic"), {}).get("final_review_status") in assembler.PASS_STATUSES
               for lang in assembler.TARGET_LANGS for sft_id in ids)
    }
    atomic_eligible_by_split = Counter(split_by_case[case_id] for case_id in atomic_eligible)
    atomic_total_by_split = Counter(split_by_case.values())
    atomic_repair_rows_needed = {
        case_id: sum(
            reviews[(lang, sft_id, "atomic")].get("final_review_status") not in assembler.PASS_STATUSES
            for lang in assembler.TARGET_LANGS for sft_id in ids
        )
        for case_id, ids in atomic_ids_by_case.items()
    }
    atomic_repair_need_histogram = {
        split: dict(sorted(Counter(
            atomic_repair_rows_needed[case_id]
            for case_id in atomic_ids_by_case if split_by_case[case_id] == split
        ).items()))
        for split in ("forget", "retain", "holdout")
    }

    composition = read_jsonl(RELEASE / "composition_reuse.jsonl")
    composition_ids_by_chain: dict[str, set[str]] = defaultdict(set)
    for row in composition:
        composition_ids_by_chain[row["source_qa_id"]].add(row["sft_id"])
    if len(composition_ids_by_chain) != 44 or any(len(ids) != 4 for ids in composition_ids_by_chain.values()):
        raise ValueError("Composition source universe differs from the frozen 44 four-prompt chains")
    composition_eligible = {
        chain_id for chain_id, ids in composition_ids_by_chain.items()
        if all(reviews.get((lang, sft_id, "composition"), {}).get("final_review_status") in assembler.PASS_STATUSES
               for lang in assembler.TARGET_LANGS for sft_id in ids)
    }
    composition_repair_rows_needed = {
        chain_id: sum(
            reviews[(lang, sft_id, "composition")].get("final_review_status") not in assembler.PASS_STATUSES
            for lang in assembler.TARGET_LANGS for sft_id in ids
        )
        for chain_id, ids in composition_ids_by_chain.items()
    }
    repair_input = verify_live_repair_input(flagged_by_language)
    repair_rows_required = (
        sum(atomic_repair_rows_needed.values()) + sum(composition_repair_rows_needed.values())
    )
    if repair_rows_required != repair_input["rows"]:
        raise ValueError("Whole-case/chain release coverage does not require exactly the complete flagged repair queue")

    report_path = assembler.REVIEW_DIR / "report.json"
    report = read_json(report_path)
    provenance = assembler.adjudication_script_provenance(report["adjudication_script_sha256"])
    result = {
        "status": "TRANSLATION_COVERAGE_FAIL_SOURCE_BOUND_REVIEW_VERIFIED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_approved": False,
        "training_eligible": False,
        "translation_review_mode": "AI_ONLY",
        "human_translation_reviewed": False,
        "review_rows_verified": len(reviews),
        "review_rows_by_language": dict(sorted(row_counts.items())),
        "final_pass_rows_by_language": {lang: pass_counts[lang] for lang in assembler.TARGET_LANGS},
        "final_review_statuses_by_language": statuses_by_language,
        "atomic_cases": {
            "source_total_by_split": {split: atomic_total_by_split[split] for split in ("forget", "retain")},
            "eligible_by_split": {split: atomic_eligible_by_split[split] for split in ("forget", "retain")},
            "required_by_split": dict(assembler.EXPECTED_ATOMIC_CASES_BY_SPLIT),
            "eligible_total": len(atomic_eligible),
            "currently_flagged_candidate_rows_needed_per_case_histogram": atomic_repair_need_histogram,
        },
        "composition_chains": {
            "source_total": len(composition_ids_by_chain),
            "eligible": len(composition_eligible),
            "required": assembler.EXPECTED_COMPOSITION_CHAINS,
            "currently_flagged_candidate_rows_needed_per_chain_histogram": dict(sorted(
                Counter(composition_repair_rows_needed.values()).items()
            )),
        },
        "repair_coverage_requirement": {
            "source_bound_pending_review_input": repair_input,
            "all_candidate_rows_required_for_full_frozen_coverage": repair_rows_required,
            "candidate_rows_available": repair_input["rows"],
            "all_rows_required": repair_rows_required == repair_input["rows"],
        },
        "reviewer_script_provenance": provenance,
        "input_receipts": {
            "adjudication_report_sha256": sha(report_path),
            "adjudication_rows_sha256_by_language": {
                lang: sha(assembler.REVIEW_DIR / f"{lang}.jsonl") for lang in assembler.TARGET_LANGS
            },
            "source_files_sha256": {
                name: sha(RELEASE / name) for name in (
                    "atomic_reuse.jsonl", "atomic_translation_requests.jsonl",
                    "composition_reuse.jsonl", "composition_translation_review_queue.jsonl",
                )
            },
            "audit_script_sha256": sha(Path(__file__).resolve()),
        },
        "limitations": [
            "AI-only translation review is not fluent-human certification.",
            "Coverage is computed at whole-case/whole-chain granularity across all four target languages.",
            "A failed translation-coverage gate does not estimate model competence or unlearning effects.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
