#!/usr/bin/env python3
"""Extend the translation-coverage diagnostic with the matched direct-31 screen.

Read-only with respect to the canonical data and training release. The direct
screen is AI-only and remains quarantined. This script estimates candidate
coverage; it does not approve translations, write a corpus, or start training.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
TRACKER = Path(__file__).resolve().parent
RELEASE = REPO / "experiments/synthworlds_r1/training_release_draft"
DIRECT = RELEASE / "translation_repair_diagnostics/blocked_marker_alias_probe_2026-11-04/direct_same31_2026-11-04"
GROUPS = REPO / "experiments/synthworlds_r1/leakage_groups.draft.jsonl"
LANGUAGES = ("fr", "de", "hi", "th")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_previous_audit_module():
    path = TRACKER / "template_reuse_ceiling_audit.py"
    spec = importlib.util.spec_from_file_location("template_reuse_ceiling_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the verified base coverage audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_direct_screen() -> tuple[set[tuple[str, str, str]], dict[str, Any], list[Path]]:
    """Re-run the source/script/output verifier and bind its 31 rows to prompts."""
    pulled_script = DIRECT / "remote_source/synthworlds-r1-direct-same31-review-2026-11-04.py"
    command = [
        sys.executable,
        "-B",
        str(DIRECT / "verify_direct.py"),
        "--remote-dir",
        str(DIRECT / "remote_output"),
        "--pulled-script",
        str(pulled_script),
    ]
    proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"Direct-31 verifier failed ({proc.returncode}): {proc.stderr[-2000:]}")

    manifest_path = DIRECT / "manifest.json"
    input_path = DIRECT / "review_input.json"
    overlay_path = DIRECT.parent / "final_diagnostic_overlay_2026-11-04/rechecked_blocked_31_rows.jsonl"
    parent_receipt_path = DIRECT.parent / "final_diagnostic_overlay_2026-11-04/verification.json"
    output_path = DIRECT / "remote_output/direct_recheck.json"
    verifier_receipt_path = DIRECT / "remote_output/verification.json"
    manifest = read_json(manifest_path)
    inputs = read_json(input_path)
    overlay = read_jsonl(overlay_path)
    output = read_json(output_path)
    receipt = read_json(verifier_receipt_path)

    expected_projection = [
        {key: row[key] for key in ("review_key", "language", "kind", "source_question_en", "translated_question")}
        for row in overlay
    ]
    if (
        manifest.get("status") != "SOURCE_BOUND_BLINDED_DIRECT_REVIEW_READY_NOT_APPROVAL"
        or manifest.get("release_approved") is not False
        or manifest.get("training_eligible") is not False
        or manifest.get("human_translation_reviewed") is not False
        or manifest.get("answers_or_splits_sent") is not False
        or len(inputs) != 31
        or len(overlay) != 31
        or inputs != expected_projection
        or sha_bytes(input_path.read_bytes().rstrip(b"\n")) != manifest.get("input_sha256")
        or sha_file(overlay_path) != manifest.get("parent_overlay_sha256")
        or read_json(parent_receipt_path).get("result_rows_sha256") != sha_file(overlay_path)
    ):
        raise RuntimeError("Direct-31 source/input/approval-state binding failed")

    expected = {row["review_key"]: row for row in inputs}
    parent_by_key = {row["review_key"]: row for row in overlay}
    if len(expected) != 31 or len(parent_by_key) != 31:
        raise RuntimeError("Direct-31 input keys are not unique")
    if (
        output.get("status") != "DIAGNOSTIC_COMPLETE"
        or output.get("model_id") != "Qwen/Qwen2.5-14B-Instruct"
        or output.get("model_revision") != "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
        or output.get("input_sha256") != manifest.get("input_sha256")
        or output.get("parent_overlay_sha256") != manifest.get("parent_overlay_sha256")
        or output.get("human_translation_reviewed") is not False
        or output.get("answers_or_splits_sent") is not False
        or output.get("backtranslation_provided") is not False
        or receipt.get("status") != "VERIFIED_MATCHED_DIRECT_REVIEW_DIAGNOSTIC_ONLY"
        or receipt.get("release_approved") is not False
        or receipt.get("training_eligible") is not False
        or receipt.get("human_translation_reviewed") is not False
        or receipt.get("answers_or_splits_sent") is not False
        or receipt.get("rows") != 31
        or receipt.get("output_sha256") != sha_file(output_path)
    ):
        raise RuntimeError("Direct-31 output is not a verified quarantined diagnostic")

    if not isinstance(output.get("records"), list) or len(output["records"]) != 31:
        raise RuntimeError("Expected 31 direct-screen records")
    direct_passes: set[tuple[str, str, str]] = set()
    seen: set[str] = set()
    for record in output["records"]:
        key = record.get("review_key")
        if key in seen or key not in expected or key not in parent_by_key:
            raise RuntimeError(f"Duplicate or unknown direct-review key: {key}")
        seen.add(key)
        candidate = expected[key]
        parent = parent_by_key[key]
        decision = record.get("decision")
        try:
            raw_decision = json.loads(record["raw_output"])["decisions"][0]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid raw direct-review decision: {key}") from exc
        if (
            candidate["review_key"] != parent["review_key"]
            or candidate["language"] != parent["language"]
            or candidate["kind"] != parent["kind"]
            or candidate["source_question_en"] != parent["source_question_en"]
            or candidate["translated_question"] != parent["translated_question"]
            or record.get("language") != candidate["language"]
            or record.get("kind") != candidate["kind"]
            or record.get("source_sha256") != sha_bytes(candidate["source_question_en"].encode("utf-8"))
            or record.get("translation_sha256") != sha_bytes(candidate["translated_question"].encode("utf-8"))
            or decision != raw_decision
            or not isinstance(decision, dict)
            or type(decision.get("semantic_pass")) is not bool
            or type(decision.get("fluency_pass")) is not bool
        ):
            raise RuntimeError(f"Direct-screen row is not exactly bound to source candidate: {key}")
        if decision["semantic_pass"] and decision["fluency_pass"]:
            direct_passes.add((candidate["kind"], parent["prompt_id"], candidate["language"]))

    if seen != set(expected):
        raise RuntimeError("Direct-screen output does not cover the exact 31-row input set")
    direct_paths = [
        manifest_path,
        input_path,
        overlay_path,
        parent_receipt_path,
        DIRECT / "kaggle_kernel/direct_review.py",
        pulled_script,
        output_path,
        verifier_receipt_path,
    ]
    return direct_passes, receipt, direct_paths


def main() -> None:
    base = load_previous_audit_module()
    source_meta, structure = base.build_source_meta()
    base_reviews, base_passes, original_integrity = base.verify_original_review_inputs(source_meta)
    repair_passes, repair_integrity = base.verify_repair_reviews(source_meta)
    direct_passes, direct_receipt, direct_paths = verify_direct_screen()
    if direct_passes & (base_passes | repair_passes):
        raise RuntimeError("Direct-31 PASS rows overlap already accepted review candidates")
    union_passes = base_passes | repair_passes | direct_passes

    atomic_rows = {
        row["sft_id"]: row
        for row in base.read_jsonl(RELEASE / "atomic_reuse.jsonl")
        if row["language"] == "en"
    }
    composition_rows = {
        row["sft_id"]: row
        for row in base.read_jsonl(RELEASE / "composition_reuse.jsonl")
        if row["language"] == "en"
    }
    atomic_cases: dict[str, list[str]] = collections.defaultdict(list)
    comp_chains: dict[str, list[str]] = collections.defaultdict(list)
    split_by_case: dict[str, str] = {}
    for prompt_id, row in atomic_rows.items():
        atomic_cases[row["source_qa_id"]].append(prompt_id)
        case_id = row["source_qa_id"]
        previous_split = split_by_case.setdefault(case_id, row["split"])
        if previous_split != row["split"]:
            raise RuntimeError(f"Atomic prompts for one source case disagree on split: {case_id}")
    for prompt_id, row in composition_rows.items():
        comp_chains[row["source_qa_id"]].append(prompt_id)

    group_rows = read_jsonl(GROUPS)
    group_by_case: dict[str, str] = {}
    for group in group_rows:
        for case_id in group["source_qa_ids"]:
            if case_id in group_by_case:
                raise RuntimeError(f"QA case belongs to multiple leakage groups: {case_id}")
            group_by_case[case_id] = group["group_id"]
    if any(case_id not in group_by_case for case_id in atomic_cases):
        raise RuntimeError("An atomic source case is absent from the frozen leakage groups")

    exact_complete: dict[str, list[str]] = {"forget": [], "retain": []}
    for split in exact_complete:
        for case_id, prompt_ids in atomic_cases.items():
            if split_by_case[case_id] != split or len(prompt_ids) != 2:
                continue
            if all(("atomic", prompt_id, language) in union_passes for prompt_id in prompt_ids for language in LANGUAGES):
                exact_complete[split].append(case_id)
        exact_complete[split].sort()
    exact_group_counts = {
        split: len({group_by_case[case_id] for case_id in case_ids})
        for split, case_ids in exact_complete.items()
    }

    # Optimistic relation-aware reuse ceiling, recalculated after including direct-31 passes.
    template_languages = {
        (source_meta[(kind, prompt_id)][1]["template"], language)
        for kind, prompt_id, language in union_passes
    }
    template_complete: dict[str, list[str]] = {"forget": [], "retain": []}
    for split in template_complete:
        for case_id, prompt_ids in atomic_cases.items():
            if split_by_case[case_id] != split or len(prompt_ids) != 2:
                continue
            if all((source_meta[("atomic", prompt_id)][1]["template"], language) in template_languages for prompt_id in prompt_ids for language in LANGUAGES):
                template_complete[split].append(case_id)
        template_complete[split].sort()

    chain_missing_rows: dict[str, list[dict[str, str]]] = {}
    for chain_id, prompt_ids in comp_chains.items():
        if len(prompt_ids) != 4:
            raise RuntimeError(f"Expected four source prompts per composition chain: {chain_id}")
        gaps = [
            {"prompt_id": prompt_id, "language": language}
            for prompt_id in sorted(prompt_ids)
            for language in LANGUAGES
            if ("composition", prompt_id, language) not in union_passes
        ]
        chain_missing_rows[chain_id] = gaps
    missing_histogram = collections.Counter(map(len, chain_missing_rows.values()))
    complete_chains = sorted(chain_id for chain_id, gaps in chain_missing_rows.items() if not gaps)
    near_complete = {
        chain_id: gaps for chain_id, gaps in sorted(chain_missing_rows.items()) if len(gaps) == 1
    }

    row_counts: dict[str, dict[str, int]] = {}
    for language in LANGUAGES:
        row_counts[language] = {}
        for kind, pool in (("atomic", atomic_rows), ("composition", composition_rows)):
            row_counts[language][kind] = sum((kind, prompt_id, language) in union_passes for prompt_id in pool)

    all_paths = set(original_integrity["input_paths"]) | set(repair_integrity["input_paths"]) | set(structure["input_paths"])
    all_paths.update(direct_paths)
    all_paths.add(GROUPS)
    input_hashes = {
        str(path.relative_to(REPO)): sha_file(path)
        for path in sorted(all_paths)
    }
    result = {
        "audit": "Extended candidate-coverage diagnostic including hash-bound matched direct-31 screen",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "scope": {
            "canonical_source_modified": False,
            "canonical_reviews_modified": False,
            "corpus_written": False,
            "release_approved": False,
            "training_eligible": False,
            "training_started": False,
            "human_translation_reviewed": False,
            "all_candidate_passes_quarantined": True,
            "release_gate": "unchanged: 117 forget, 39 retain, and all 44 training-only composition chains",
        },
        "verification": {
            "base_review_rows": len(base_reviews),
            "base_review_pass_rows": len(base_passes),
            "repair_review_rows": repair_integrity["review_rows"],
            "repair_review_hash_source_binding_failures": repair_integrity["hash_and_source_binding_failures"],
            "repair_review_pass_rows": len(repair_passes),
            "direct_review_verifier_status": direct_receipt["status"],
            "direct_review_rows": direct_receipt["rows"],
            "direct_review_pass_rows": len(direct_passes),
            "direct_review_pass_flag_agreement_with_roundtrip": direct_receipt["overall_pass_flag_agreement"],
            "direct_review_model": f"{direct_receipt['model_id']}@{direct_receipt['model_revision']}",
            "direct_review_human_gold_controls": 0,
            "unique_candidate_pass_rows_after_union": len(union_passes),
            "source_hashes": input_hashes,
        },
        "candidate_row_coverage_by_language": row_counts,
        "exact_all_four_language_complete_atomic_cases": {
            "required_full_release": {"forget": 117, "retain": 39},
            "diagnostic_candidate_union": {split: len(ids) for split, ids in exact_complete.items()},
            "diagnostic_union_leakage_groups": exact_group_counts,
            "case_ids": exact_complete,
            "note": "Case-count comparison is descriptive only; it does not establish pre-unlearning competence, power, or release eligibility.",
        },
        "optimistic_relation_aware_template_reuse_ceiling": {
            "forget_cases": len(template_complete["forget"]),
            "retain_cases": len(template_complete["retain"]),
            "forget_case_ids": template_complete["forget"],
            "retain_case_ids": template_complete["retain"],
            "method_caveat": "A passing translation is assumed reusable for all same-template prompts without independent rendering review; this is an upper-bound diagnostic, never training data.",
        },
        "composition_training_chains": {
            "required": 44,
            "complete_candidate_union_chains": len(complete_chains),
            "complete_chain_ids": complete_chains,
            "unpassed_translation_rows_across_chains": sum(map(len, chain_missing_rows.values())),
            "chain_gap_histogram_missing_rows_out_of_16": {str(k): missing_histogram[k] for k in sorted(missing_histogram)},
            "chains_with_exactly_one_unpassed_translation_row": near_complete,
        },
        "interpretation": (
            "Adding the matched direct-31 AI screen increases the quarantined candidate union and may justify prioritizing near-complete composition rows for further review. "
            "However, this direct screen disagreed with its round-trip screen on 18/31 overall labels (including 18 former flags now passes), had no human-gold controls, "
            "and the earlier independent calibration accepted one known error. These statuses cannot establish translation correctness. "
            "The frozen full-release thresholds remain unmet, and no training corpus or experiment is authorized by this audit."
        ),
    }
    output_path = TRACKER / "direct31_extended_coverage_audit.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path.relative_to(REPO)),
        "direct_pass_rows": len(direct_passes),
        "candidate_union_pass_rows": len(union_passes),
        "atomic_case_counts": result["exact_all_four_language_complete_atomic_cases"]["diagnostic_candidate_union"],
        "atomic_group_counts": exact_group_counts,
        "composition_complete": len(complete_chains),
        "composition_missing_rows": sum(map(len, chain_missing_rows.values())),
        "one_gap_chains": len(near_complete),
        "corpus_written": False,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
