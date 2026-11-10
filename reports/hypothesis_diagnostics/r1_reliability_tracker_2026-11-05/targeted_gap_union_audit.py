#!/usr/bin/env python3
"""Recompute quarantined translation coverage after the targeted-gap screens.

This is a source-bound coverage diagnostic only. It does not promote any
AI-reviewed translation, write a training corpus, or start model training.
All verifier reruns write to temporary directories to preserve archived receipts.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
TRACKER = Path(__file__).resolve().parent
RELEASE = REPO / "experiments/synthworlds_r1/training_release_draft"
GAP = RELEASE / "translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05"
THIRD = RELEASE / "translation_repair_diagnostics/chatgpt_third_candidate_2026-11-04"
FULL_INPUTS = RELEASE / "translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/independent_review_inputs_2026-11-04"
MISTRAL_CONTROL = REPO / "reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03"
QWEN_CONTROL = REPO / "reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls"
LANGUAGES = ("fr", "de", "hi", "th")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audit module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rerun_verifier(label: str, verifier: Path, remote_dir: Path, pulled_script: Path) -> dict[str, Any]:
    """Run a Kaggle artifact verifier against a temporary output copy."""
    with tempfile.TemporaryDirectory(prefix=f"sw-{label}-verify-") as temporary:
        copied_output = Path(temporary) / "remote_output"
        shutil.copytree(remote_dir, copied_output)
        proc = subprocess.run(
            [
                sys.executable,
                str(verifier),
                "--remote-dir",
                str(copied_output),
                "--pulled-script",
                str(pulled_script),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode:
            raise RuntimeError(f"{label} verifier failed: {proc.stderr[-2000:]}")
        receipt = json.loads(proc.stdout)
        archived_receipt = remote_dir / "verification.json"
        if archived_receipt.exists() and read_json(archived_receipt) != receipt:
            raise RuntimeError(f"{label} verifier differs from archived receipt")
        return receipt


def run_json_script(label: str, script: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"{label} verification failed: {proc.stderr[-2000:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} did not return a JSON receipt") from exc


def main() -> None:
    base = load_module(
        "template_reuse_ceiling_audit",
        TRACKER / "template_reuse_ceiling_audit.py",
    )
    extended = load_module(
        "direct31_extended_coverage_audit",
        TRACKER / "direct31_extended_coverage_audit.py",
    )
    source_meta, structure = base.build_source_meta()
    base_reviews, original_passes, original_integrity = base.verify_original_review_inputs(source_meta)
    repair_passes, repair_integrity = base.verify_repair_reviews(source_meta)
    original_direct = extended.DIRECT
    original_overlay_dir = original_direct.parent / "final_diagnostic_overlay_2026-11-04"
    with tempfile.TemporaryDirectory(prefix="sw-direct31-verify-") as temporary:
        temporary_parent = Path(temporary)
        temporary_direct = temporary_parent / original_direct.name
        shutil.copytree(original_direct, temporary_direct)
        shutil.copytree(original_overlay_dir, temporary_parent / original_overlay_dir.name)
        extended.DIRECT = temporary_direct
        try:
            direct_passes, direct_receipt, _temporary_paths = extended.verify_direct_screen()
        finally:
            extended.DIRECT = original_direct
    direct_paths = [
        original_direct / "manifest.json",
        original_direct / "review_input.json",
        original_overlay_dir / "rechecked_blocked_31_rows.jsonl",
        original_overlay_dir / "verification.json",
        original_direct / "kaggle_kernel/direct_review.py",
        original_direct / "remote_source/synthworlds-r1-direct-same31-review-2026-11-04.py",
        original_direct / "remote_output/direct_review.json",
        original_direct / "remote_output/direct_recheck.json",
        original_direct / "remote_output/verification.json",
    ]
    pre_target_union = original_passes | repair_passes | direct_passes

    primary_dir = GAP / "candidate_review_kaggle"
    primary_artifact = primary_dir / "remote_output/targeted_gap_review.json"
    # The primary verifier has a positional artifact argument, so run it separately.
    with tempfile.TemporaryDirectory(prefix="sw-targeted-gap-primary-") as temporary:
        receipt_path = Path(temporary) / "verification.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(primary_dir / "verify_targeted_gap_review.py"),
                str(primary_artifact),
                "--output",
                str(receipt_path),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode:
            raise RuntimeError(f"Primary targeted-gap verifier failed: {proc.stderr[-2000:]}")
        primary_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    retry_dir = GAP / "targeted_gap_retry_2026-11-05"
    retry_receipt = rerun_verifier(
        "targeted-gap-retry",
        retry_dir / "verify_retry.py",
        retry_dir / "remote_output",
        retry_dir / "remote_source/sw-gap187-alias-retry-2026-11-05.py",
    )
    nllb_dir = GAP / "targeted_gap_nllb_candidates_2026-11-05"
    nllb_receipt = rerun_verifier(
        "targeted-gap-nllb-candidates",
        nllb_dir / "verify_candidate_review.py",
        nllb_dir / "remote_output",
        nllb_dir / "remote_source/sw-gap187-nllb-148-candidate-review-2026-11-05.py",
    )
    segmented_dir = GAP / "targeted_gap_nllb_segmented_retry_2026-11-05"
    segmented_receipt = rerun_verifier(
        "targeted-gap-segmented-retry",
        segmented_dir / "verify_segmented_retry.py",
        segmented_dir / "remote_output",
        segmented_dir / "remote_source/sw-gap187-nllb-segmented-retry-2026-11-05.py",
    )

    third_overlay_verifier = THIRD / "verify_final_recheck_overlay.py"
    mistral_direct_verifier = THIRD / "mistral_direct_review/verify.py"
    mistral_direct_receipt = run_json_script("Third-candidate Mistral direct review", mistral_direct_verifier)
    reviewer_comparison = run_json_script("Qwen/Mistral row comparison", THIRD / "mistral_direct_review/compare_qwen_mistral.py")
    rationale_audit = run_json_script("Mistral rationale consistency", THIRD / "mistral_direct_review/audit_decision_rationale_consistency.py")
    mistral_control_receipt = run_json_script("Mistral hard-negative control", MISTRAL_CONTROL / "verify_mistral_negative_control.py")
    qwen_control_receipt = run_json_script("Qwen14B hard-negative control", QWEN_CONTROL / "verify.py")
    if (mistral_direct_receipt.get("status") != "VERIFIED_MISTRAL_DIRECT_187_DIAGNOSTIC_ONLY"
            or reviewer_comparison.get("status") != "VERIFIED_REVIEWER_DISAGREEMENT_DIAGNOSTIC_ONLY"
            or rationale_audit.get("status") != "VERIFIED_INTERNAL_CONTRADICTION_DIAGNOSTIC_ONLY"
            or mistral_control_receipt.get("status") != "CONTROLLED_MISMATCH_DIAGNOSTIC_ONLY"
            or qwen_control_receipt.get("status") != "VERIFIED_DIAGNOSTIC_SCREEN_FAILED"):
        raise RuntimeError("A reviewer diagnostic did not reproduce its expected hash-bound status")
    private_map = read_jsonl(GAP / "private_row_map.jsonl")
    gap_rows = read_jsonl(GAP / "local_gap_rows.jsonl")
    if len(private_map) != 187 or len(gap_rows) != 187:
        raise RuntimeError("Expected complete 187-row private map and gap queue")
    by_key = {row["row_id"]: row for row in private_map}
    gap_by_identity = {(row["kind"], row["prompt_id"], row["language"]): row for row in gap_rows}
    if len(by_key) != 187 or len(gap_by_identity) != 187:
        raise RuntimeError("Duplicate row key or prompt identity in targeted-gap inputs")
    third_overlay_proc = subprocess.run(
        [sys.executable, str(third_overlay_verifier)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if third_overlay_proc.returncode:
        raise RuntimeError(f"Third-candidate overlay verifier failed: {third_overlay_proc.stderr[-2000:]}")
    third_overlay_receipt = json.loads(third_overlay_proc.stdout)
    third_overlay = read_json(THIRD / "final_recheck_overlay_2026-11-05/final_recheck_overlay.json")
    third_requests = read_jsonl(THIRD / "requests.jsonl")
    third_input = read_json(THIRD / "kaggle_review/review_input.json")
    full_masked_rows = read_jsonl(FULL_INPUTS / "full799.jsonl")
    full_private_map = read_jsonl(FULL_INPUTS / "full799.private_map.jsonl")
    if (
        third_overlay_receipt.get("status") != "PASS_HASH_BOUND_DIAGNOSTIC_OVERLAY_ONLY"
        or len(third_requests) != 187
        or len(third_input) != 187
        or third_overlay.get("rows") != 187
        or third_overlay.get("training_eligible") is not False
    ):
        raise RuntimeError("Third-candidate overlay is incomplete or not explicitly quarantined")
    request_by_key = {row["source_review_key"]: row for row in third_requests}
    if len(request_by_key) != 187 or set(request_by_key) != {row["review_key"] for row in third_input}:
        raise RuntimeError("Third-candidate request identities do not match the verified review inputs")
    masked_by_key = {row["review_key"]: row for row in full_masked_rows}
    private_by_key = {row["review_key"]: row for row in full_private_map}
    if len(masked_by_key) != 799 or len(private_by_key) != 799:
        raise RuntimeError("The full-review source map needed to reconcile masked third candidates is incomplete")
    third_passes: set[tuple[str, str, str]] = set()
    third_overlay_by_key = {row["review_key"]: row for row in third_overlay["records"]}
    if len(third_overlay_by_key) != 187 or set(third_overlay_by_key) != set(request_by_key):
        raise RuntimeError("Third-candidate overlay keys do not match its exact request universe")
    for row in third_input:
        key = row["review_key"]
        request = request_by_key[key]
        masked_source = masked_by_key.get(key)
        private = private_by_key.get(key)
        if masked_source is None or private is None:
            raise RuntimeError(f"Third-candidate review key is absent from the frozen 799-row source map: {key}")
        identity = (row["kind"], private["prompt_id"], row["language"])
        source_row = source_meta.get((identity[0], identity[1]))
        source_hash = sha_bytes(row["source_question_en"].encode("utf-8"))
        if (
            source_row is None
            or masked_source["source_question_en"] != row["source_question_en"]
            or masked_source["language"] != row["language"]
            or masked_source["kind"] != row["kind"]
            or source_hash != request["source_question_sha256"]
            or sha_bytes(source_row[0]["question_en"].encode("utf-8")) != private["source_text_sha256"]
            or identity[1] != request["prompt_id"]
            or identity not in gap_by_identity
        ):
            raise RuntimeError(f"Third-candidate source identity/hash mismatch: {key}")
        final_row = third_overlay_by_key[key]
        if (final_row.get("language"), final_row.get("kind"), final_row.get("source_sha256")) != (
            row["language"], row["kind"], source_hash
        ):
            raise RuntimeError(f"Third-candidate output is not bound to its source row: {key}")
        decision = final_row.get("decision") or {}
        if final_row.get("diagnostic_label") == "PASS" and decision.get("semantic_pass") is True and decision.get("fluency_pass") is True:
            third_passes.add(identity)

    if set(primary_receipt["model_revisions"].values()) != {
        "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    }:
        raise RuntimeError("Unexpected reviewer model revision")

    primary = read_json(primary_artifact)
    primary_by_key = {row["review_key"]: row for row in primary["records"]}
    if len(primary_by_key) != 187 or set(primary_by_key) != set(by_key):
        raise RuntimeError("Primary output keys do not match the private source map")

    def identity_for(record: dict[str, Any], require_primary_source: bool = True) -> tuple[str, str, str]:
        key = record.get("review_key")
        mapped = by_key.get(key)
        if mapped is None:
            raise RuntimeError(f"Unknown targeted-gap row key: {key}")
        identity = (mapped["kind"], mapped["prompt_id"], mapped["language"])
        if (
            record.get("language") != mapped["language"]
            or record.get("kind") != mapped["kind"]
            or identity not in gap_by_identity
        ):
            raise RuntimeError(f"Candidate identity does not match private map: {key}")
        source_row = source_meta.get((mapped["kind"], mapped["prompt_id"]))
        if source_row is None or sha_bytes(source_row[0]["question_en"].encode("utf-8")) != mapped["source_text_sha256"]:
            raise RuntimeError(f"Private source hash does not match canonical English prompt: {key}")
        if require_primary_source and record.get("source_sha256") != primary_by_key[key].get("source_sha256"):
            raise RuntimeError(f"Follow-up source hash differs from primary source: {key}")
        return identity

    additions_by_screen: dict[str, set[tuple[str, str, str]]] = {}
    initial_passes: set[tuple[str, str, str]] = set()
    for record in primary["records"]:
        identity = identity_for(record, require_primary_source=False)
        decision = record.get("decision") or {}
        if decision.get("semantic_pass") is True and decision.get("fluency_pass") is True:
            initial_passes.add(identity)
    additions_by_screen["initial_187"] = initial_passes

    retry = read_json(retry_dir / "remote_output/retry_output.json")
    retry_passes: set[tuple[str, str, str]] = set()
    for record in retry["records"]:
        identity = identity_for(record)
        decision = record.get("decision") or {}
        if decision.get("semantic_pass") is True and decision.get("fluency_pass") is True:
            retry_passes.add(identity)
    additions_by_screen["alias_retry_17"] = retry_passes

    nllb = read_json(nllb_dir / "remote_output/candidate_review.json")
    nllb_passes: set[tuple[str, str, str]] = set()
    for record in nllb["records"]:
        identity = identity_for(record)
        decision = record.get("decision") or {}
        if (
            record.get("candidate_translation")
            and not record.get("candidate_error")
            and decision.get("semantic_pass") is True
            and decision.get("fluency_pass") is True
        ):
            nllb_passes.add(identity)
    additions_by_screen["nllb_alternates"] = nllb_passes

    segmented = read_json(segmented_dir / "remote_output/candidate_review.json")
    segmented_passes: set[tuple[str, str, str]] = set()
    for record in segmented["records"]:
        identity = identity_for(record)
        decision = record.get("decision") or {}
        if (
            record.get("candidate_translation")
            and not record.get("candidate_error")
            and decision.get("semantic_pass") is True
            and decision.get("fluency_pass") is True
        ):
            segmented_passes.add(identity)
    additions_by_screen["segmented_nllb_retry"] = segmented_passes
    additions_by_screen["third_candidate_overlay"] = third_passes

    candidate_passes = set().union(*additions_by_screen.values())
    new_passes = candidate_passes - pre_target_union
    union_passes = pre_target_union | candidate_passes

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
    composition_chains: dict[str, list[str]] = collections.defaultdict(list)
    split_by_case: dict[str, str] = {}
    for prompt_id, row in atomic_rows.items():
        atomic_cases[row["source_qa_id"]].append(prompt_id)
        split = split_by_case.setdefault(row["source_qa_id"], row["split"])
        if split != row["split"]:
            raise RuntimeError(f"Inconsistent split for case {row['source_qa_id']}")
    for prompt_id, row in composition_rows.items():
        composition_chains[row["source_qa_id"]].append(prompt_id)

    groups = base.read_jsonl(REPO / "experiments/synthworlds_r1/leakage_groups.draft.jsonl")
    group_by_case: dict[str, str] = {}
    for group in groups:
        for case_id in group["source_qa_ids"]:
            if case_id in group_by_case:
                raise RuntimeError(f"Case appears in multiple leakage groups: {case_id}")
            group_by_case[case_id] = group["group_id"]

    def exact_atomic_counts(passes: set[tuple[str, str, str]]) -> tuple[dict[str, int], dict[str, int]]:
        cases: dict[str, list[str]] = {"forget": [], "retain": []}
        for split in cases:
            for case_id, prompts in atomic_cases.items():
                if split_by_case[case_id] != split or len(prompts) != 2:
                    continue
                if all(("atomic", prompt, lang) in passes for prompt in prompts for lang in LANGUAGES):
                    cases[split].append(case_id)
        group_counts = {
            split: len({group_by_case[case_id] for case_id in case_ids})
            for split, case_ids in cases.items()
        }
        return {split: len(ids) for split, ids in cases.items()}, group_counts

    def template_atomic_counts(passes: set[tuple[str, str, str]]) -> dict[str, int]:
        template_languages = {
            (source_meta[(kind, prompt)][1]["template"], lang)
            for kind, prompt, lang in passes
        }
        result = {"forget": 0, "retain": 0}
        for split in result:
            for case_id, prompts in atomic_cases.items():
                if split_by_case[case_id] != split or len(prompts) != 2:
                    continue
                if all(
                    (source_meta[("atomic", prompt)][1]["template"], lang) in template_languages
                    for prompt in prompts
                    for lang in LANGUAGES
                ):
                    result[split] += 1
        return result

    def composition_count(passes: set[tuple[str, str, str]]) -> tuple[int, int, dict[str, int]]:
        missing_by_chain: dict[str, int] = {}
        for chain_id, prompts in composition_chains.items():
            if len(prompts) != 4:
                raise RuntimeError(f"Expected four prompts in composition chain {chain_id}")
            missing_by_chain[chain_id] = sum(
                ("composition", prompt, lang) not in passes
                for prompt in prompts
                for lang in LANGUAGES
            )
        histogram = collections.Counter(missing_by_chain.values())
        return (
            sum(missing == 0 for missing in missing_by_chain.values()),
            sum(missing_by_chain.values()),
            {str(k): histogram[k] for k in sorted(histogram)},
        )

    before_exact, before_groups = exact_atomic_counts(pre_target_union)
    after_exact, after_groups = exact_atomic_counts(union_passes)
    before_composition = composition_count(pre_target_union)
    after_composition = composition_count(union_passes)
    added_counts = {
        lang: {
            kind: sum((kind, prompt, lang) in new_passes for prompt in source_pool)
            for kind, source_pool in (("atomic", atomic_rows), ("composition", composition_rows))
        }
        for lang in LANGUAGES
    }
    net_by_screen: dict[str, int] = {}
    overlap_by_screen: dict[str, int] = {}
    seen_passes = set(pre_target_union)
    for name, passes in additions_by_screen.items():
        net_by_screen[name] = len(passes - seen_passes)
        overlap_by_screen[name] = len(passes & seen_passes)
        seen_passes |= passes
    if seen_passes != union_passes or sum(net_by_screen.values()) != len(new_passes):
        raise RuntimeError("Screen-by-screen deduplication does not reproduce the unique union")

    tracked_paths = [
        GAP / "private_row_map.jsonl",
        GAP / "local_gap_rows.jsonl",
        primary_dir / "review_input_manifest.json",
        primary_dir / "review.py",
        primary_dir / "verify_targeted_gap_review.py",
        primary_artifact,
        primary_dir / "remote_output/kernel.log",
        retry_dir / "retry_input_manifest.json",
        retry_dir / "retry.py",
        retry_dir / "verify_retry.py",
        retry_dir / "remote_source/sw-gap187-alias-retry-2026-11-05.py",
        retry_dir / "remote_output/retry_output.json",
        retry_dir / "remote_output/retry_verification.json",
        nllb_dir / "candidate_input_manifest.json",
        nllb_dir / "prepare_candidates.py",
        nllb_dir / "verify_candidate_review.py",
        nllb_dir / "remote_source/sw-gap187-nllb-148-candidate-review-2026-11-05.py",
        nllb_dir / "remote_output/candidate_review.json",
        nllb_dir / "remote_output/candidate_verification.json",
        segmented_dir / "segmented_retry_input_manifest.json",
        segmented_dir / "verify_segmented_retry.py",
        segmented_dir / "remote_source/sw-gap187-nllb-segmented-retry-2026-11-05.py",
        segmented_dir / "remote_output/candidate_review.json",
        segmented_dir / "remote_output/segmented_retry_verification.json",
        THIRD / "manifest.json",
        THIRD / "requests.jsonl",
        THIRD / "proposed_translations.jsonl",
        THIRD / "queue_coverage_audit.json",
        THIRD / "kaggle_review/review_input.json",
        THIRD / "kaggle_review/build_manifest.json",
        THIRD / "kaggle_review/remote_output/repair_review.json",
        THIRD / "kaggle_review/verification.json",
        THIRD / "kaggle_review/verify_output.py",
        THIRD / "marker_safe_retry_2026-11-05/remote_output/marker_safe_retry.json",
        THIRD / "marker_safe_retry_2026-11-05/verify_retry.py",
        THIRD / "qwen_only_retry_2026-11-05/remote_output/qwen_only_retry.json",
        THIRD / "qwen_only_retry_2026-11-05/verify_retry.py",
        THIRD / "key_constrained_retry_2026-11-05/remote_output/key_constrained_retry.json",
        THIRD / "key_constrained_retry_2026-11-05/verify_retry.py",
        THIRD / "final_recheck_overlay_2026-11-05/final_recheck_overlay.json",
        third_overlay_verifier,
        THIRD / "mistral_direct_review/cross_review_comparison.json",
        THIRD / "mistral_direct_review/review_input.json",
        THIRD / "mistral_direct_review/remote_output/mistral_direct_review.json",
        THIRD / "mistral_direct_review/verification.json",
        THIRD / "mistral_direct_review/review.py",
        THIRD / "mistral_direct_review/remote_kernel_source.py",
        THIRD / "mistral_direct_review/verify.py",
        THIRD / "mistral_direct_review/compare_qwen_mistral.py",
        THIRD / "mistral_direct_review/decision_rationale_audit.json",
        THIRD / "mistral_direct_review/audit_decision_rationale_consistency.py",
        MISTRAL_CONTROL / "kaggle_mistral_review_probe/review.py",
        MISTRAL_CONTROL / "kaggle_mistral_negative_control/review.py",
        MISTRAL_CONTROL / "evidence/mistral_negative_control/review.json",
        MISTRAL_CONTROL / "verify_mistral_negative_control.py",
        QWEN_CONTROL / "control_manifest.json",
        QWEN_CONTROL / "input_array.json",
        QWEN_CONTROL / "review.py",
        QWEN_CONTROL / "remote_output/review.json",
        QWEN_CONTROL / "remote_output/verification.json",
        QWEN_CONTROL / "remote_output/submitted_source.py",
        QWEN_CONTROL / "remote_output/synthworlds-qwen14b-hard-negative-2026-11-04.log",
        QWEN_CONTROL / "verify.py",
    ] + original_integrity["input_paths"] + repair_integrity["input_paths"] + direct_paths + structure["input_paths"]
    primary_manifest = read_json(primary_dir / "review_input_manifest.json")
    tracked_paths.extend(GAP / rel for rel in primary_manifest.get("input_source_hashes", {}) if rel != "candidate_audit")
    if "candidate_audit" in primary_manifest.get("input_source_hashes", {}):
        tracked_paths.append(GAP / "candidates/candidate_audit.json")
    tracked_paths.extend(GAP / rel for rel in primary_manifest.get("candidate_file_hashes", {}))
    tracked_paths.extend([
        GAP / "targeted_gap_retry_2026-11-05/retry_input.json",
        GAP / "targeted_gap_nllb_candidates_2026-11-05/candidate_input.json",
        GAP / "targeted_gap_nllb_segmented_retry_2026-11-05/segmented_retry_input.json",
    ])
    tracked_paths = sorted({p.resolve() for p in tracked_paths if p.exists()})
    input_hashes = {str(path.relative_to(REPO)): sha_file(path) for path in tracked_paths}

    result = {
        "audit": "Quarantined targeted-gap candidate union coverage; source-bound AI-only diagnostic",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "scope": {
            "source_or_canonical_reviews_modified": False,
            "training_corpus_written": False,
            "release_approved": False,
            "training_eligible": False,
            "training_started": False,
            "human_translation_reviewed": False,
            "all_new_candidate_passes_quarantined": True,
            "frozen_release_gate": "117 forget, 39 retain, and all 44 training-only composition chains",
        },
        "independently_rerun_verifiers": {
            "primary": primary_receipt,
            "alias_retry": retry_receipt,
            "nllb_alternate_candidates": nllb_receipt,
            "segmented_nllb_retry": segmented_receipt,
            "third_candidate_overlay": third_overlay_receipt,
            "third_candidate_mistral_direct_review": mistral_direct_receipt,
            "third_candidate_reviewer_comparison": reviewer_comparison["overall"],
            "third_candidate_mistral_rationale_audit": {
                "mistral_semantic_false_with_affirmative_rationale": len(rationale_audit["mistral_semantic_false_with_affirmative_rationale"]),
                "mistral_fluency_false_with_affirmative_rationale": len(rationale_audit["mistral_fluency_false_with_affirmative_rationale"]),
            },
            "mistral_hard_negative_control": mistral_control_receipt,
            "qwen14_hard_negative_control": qwen_control_receipt,
        },
        "coverage_before_and_after_targeted_candidates": {
            "verified_candidate_pass_rows_before": len(pre_target_union),
            "verified_candidate_pass_rows_after_deduplicated_union": len(union_passes),
            "net_new_unique_prompt_language_rows": len(new_passes),
            "net_new_by_language_and_kind": added_counts,
            "review_attempts_by_screen": {
                name: {
                    "joint_pass_rows": len(passes),
                    "net_unique_rows": net_by_screen[name],
                    "already_covered_before_screen": overlap_by_screen[name],
                }
                for name, passes in additions_by_screen.items()
            },
        },
        "atomic_coverage": {
            "exact_all_four_language_cases_before": before_exact,
            "exact_all_four_language_cases_after": after_exact,
            "exact_leakage_groups_before": before_groups,
            "exact_leakage_groups_after": after_groups,
            "optimistic_relation_aware_template_ceiling_before": template_atomic_counts(pre_target_union),
            "optimistic_relation_aware_template_ceiling_after": template_atomic_counts(union_passes),
            "required_cases": {"forget": 117, "retain": 39},
            "note": "These are translation-coverage counts, not model competence, statistical power, or an approved training split.",
        },
        "composition_coverage": {
            "complete_chains_before": before_composition[0],
            "complete_chains_after": after_composition[0],
            "required_training_chains": 44,
            "missing_translation_rows_before": before_composition[1],
            "missing_translation_rows_after": after_composition[1],
            "missing_row_histogram_after": after_composition[2],
        },
        "provenance": {
            "targeted_gap_candidate_rows": 187,
            "targeted_gap_joint_pass_rows": 36,
            "third_candidate_rows": 187,
            "third_candidate_joint_pass_rows": len(third_passes),
            "reviewers": primary_receipt["model_revisions"],
            "third_candidate_reviewer_disagreement_rows": read_json(THIRD / "mistral_direct_review/decision_rationale_audit.json")["reviewer_disagreement_rows"],
            "third_candidate_mistral_hard_negative_passes": "2/15",
            "gpu": read_json(primary_dir / "run_receipt.json").get("accelerator"),
            "audit_script_sha256": sha_file(Path(__file__).resolve()),
            "all_input_and_code_hashes": input_hashes,
        },
        "interpretation": (
            "The added rows are candidate translations screened by AI-only semantic and fluency checks, with no human-gold controls. "
            "They remain quarantined and are not admitted to the multilingual SFT corpus. Coverage changes are feasibility diagnostics only; "
            "they cannot show that a trained model will meet the pre-unlearning competence gate or support a causal unlearning comparison. "
            "The candidate review runtime also logged an SDPA sliding-window attention caveat. A direct Mistral cross-review disagreed with the initial Qwen screen on 154/187 rows; its deliberate-mismatch control passed 2/15, and the Qwen14 deliberate-mismatch screen passed 1/16. These small controls are not error-rate estimates, but they show that neither AI reviewer can certify the candidates."
        ),
        "reviewer_reliability_diagnostic": {
            "human_gold_controls": 0,
            "human_translation_reviewed": False,
            "initial_qwen_screen": {"rows": 187, "pass": 29, "flag": 140, "blocked": 18},
            "qwen_final_overlay_after_retries": {"rows": 187, "pass": 36, "flag": 151, "blocked": 0, "repaired_rows": 18},
            "mistral_direct_screen": {"rows": 187, "pass": 178, "flag": 9, "blocked": 0},
            "mistral_vs_initial_qwen": {
                "agreement": reviewer_comparison["overall"]["agreement"],
                "disagreement": reviewer_comparison["overall"]["disagreement"],
                "qwen_flag_mistral_pass": reviewer_comparison["overall"]["qwen_flag_mistral_pass"],
            },
            "mistral_controlled_mismatches": {
                "rows": mistral_control_receipt["rows"],
                "both_pass": mistral_control_receipt["counts"]["both_pass"],
                "semantic_pass": mistral_control_receipt["counts"]["semantic_pass"],
            },
            "qwen14_controlled_mismatches": {
                "rows": qwen_control_receipt["rows"],
                "both_pass": qwen_control_receipt["counts"]["both_pass"],
                "semantic_pass": qwen_control_receipt["counts"]["semantic_pass"],
                "predeclared_screen_passed": qwen_control_receipt["predeclared_rule_passed"],
            },
            "mistral_rationale_contradictions": {
                "semantic_false_with_affirmative_rationale": len(rationale_audit["mistral_semantic_false_with_affirmative_rationale"]),
                "fluency_false_with_affirmative_rationale": len(rationale_audit["mistral_fluency_false_with_affirmative_rationale"]),
            },
            "interpretation": "Constructed small mismatch probes are diagnostic only, not estimates of reviewer accuracy. They do not change the frozen translation gate.",
        },
    }

    json_path = TRACKER / "targeted_gap_union_audit.json"
    md_path = TRACKER / "targeted_gap_union_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Targeted translation-candidate union audit",
        "",
        f"Generated: {result['created_at_utc']} (UTC)",
        "",
        "This is an AI-only, source-bound coverage diagnostic. It does not approve translations or create training data.",
        "",
        "| Screen | Joint-pass review rows | New unique rows after ordered deduplication |",
        "|---|---:|---:|",
    ]
    for name, values in result["coverage_before_and_after_targeted_candidates"]["review_attempts_by_screen"].items():
        md.append(f"| {name} | {values['joint_pass_rows']} | {values['net_unique_rows']} |")
    md += [
        "",
        "| Coverage measure | Before added candidates | After all deduplicated candidates | Required |",
        "|---|---:|---:|---:|",
        f"| Exact four-language forget cases | {before_exact['forget']} | {after_exact['forget']} | 117 |",
        f"| Exact four-language retain cases | {before_exact['retain']} | {after_exact['retain']} | 39 |",
        f"| Optimistic relation-aware forget ceiling | {template_atomic_counts(pre_target_union)['forget']} | {template_atomic_counts(union_passes)['forget']} | 117 |",
        f"| Optimistic relation-aware retain ceiling | {template_atomic_counts(pre_target_union)['retain']} | {template_atomic_counts(union_passes)['retain']} | 39 |",
        f"| Complete composition training chains | {before_composition[0]} | {after_composition[0]} | 44 |",
        f"| Unpassed composition translation rows | {before_composition[1]} | {after_composition[1]} | 0 |",
        "",
        "| Language | New atomic rows | New composition rows |",
        "|---|---:|---:|",
    ]
    for lang in LANGUAGES:
        md.append(f"| {lang.upper()} | {added_counts[lang]['atomic']} | {added_counts[lang]['composition']} |")
    md += [
        "",
        "All rows above remain quarantined. The AI screen had no human-gold controls and logged a sliding-window/SDPA attention caveat. Translation coverage is not model competence or evidence for the unlearning hypothesis.",
        "",
        f"The third-candidate overlay ended at {len(third_passes)} pass rows and {187 - len(third_passes)} flags after 18 retries; {net_by_screen['third_candidate_overlay']} rows were new to the existing candidate union. Direct Mistral review marked 178/187 pass but disagreed with the initial Qwen labels on {reviewer_comparison['overall']['disagreement']}/187 rows. On constructed mismatch probes Mistral passed {mistral_control_receipt['counts']['both_pass']}/{mistral_control_receipt['rows']} and Qwen14 passed {qwen_control_receipt['counts']['both_pass']}/{qwen_control_receipt['rows']}; neither small control estimates real-world error rates. No fluent-human or human-gold review exists, so all rows remain quarantined.",
        "",
        "The frozen full-release gate remains unchanged. No training corpus was written and no training or unlearning run started.",
        "",
        f"Machine-readable provenance: `{json_path.name}`.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "json": str(json_path.relative_to(REPO)),
        "markdown": str(md_path.relative_to(REPO)),
        "before_exact_cases": before_exact,
        "after_exact_cases": after_exact,
        "before_exact_groups": before_groups,
        "after_exact_groups": after_groups,
        "new_unique_prompt_language_rows": len(new_passes),
        "new_rows_by_language_and_kind": added_counts,
        "before_complete_composition_chains": before_composition[0],
        "after_complete_composition_chains": after_composition[0],
        "corpus_written": False,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
