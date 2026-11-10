#!/usr/bin/env python3
"""Estimate whether candidate passes could justify a full production re-screen.

This is a read-only diagnostic. A candidate-review pass is counted only as an
optimistic possibility; it is never treated as production approval and this
script never changes translations, review receipts, or training artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
PROJECT = HERE.parents[2]
SCRIPTS = RELEASE / "scripts"
INPUTS = HERE / "independent_review_inputs_2026-11-04"
REMOTE = HERE / "remote_output/full799/version_5"
LANGS = ("fr", "de", "hi", "th")
PASS_STATUSES = {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}
EXPECTED_FLAGGED = {"fr": 164, "de": 179, "hi": 200, "th": 256}
EXPECTED_ATOMIC = {"forget": 117, "retain": 39}
EXPECTED_COMPOSITION = 44

sys.path.insert(0, str(SCRIPTS))
from review_training_translations_qwen import collect_rows, read_jsonl  # noqa: E402


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_json_array_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return sha_bytes(payload.encode("utf-8"))


def read_bound_verified_rows(verification: dict, path: Path, expected_rows: int) -> list[dict]:
    rows = read_jsonl(path)
    if verification.get("verified_rows_sha256") != sha_file(path):
        raise ValueError("The verified local decisions are not bound to the verifier receipt")
    counts = Counter(row.get("review_status") for row in rows)
    if dict(counts) != verification.get("counts"):
        raise ValueError("Verifier receipt counts differ from the bound local review decisions")
    if len(rows) != expected_rows or sum(counts.values()) != verification.get("rows"):
        raise ValueError("Verifier receipt row count differs from its local decisions")
    return rows


def unique_index(rows: list[dict], key_fn, label: str) -> dict:
    result = {}
    for row in rows:
        key = key_fn(row)
        if key in result:
            raise ValueError(f"Duplicate {label} key: {key}")
        result[key] = row
    return result


def compute_coverage(pass_by_language: dict[str, set[tuple[str, str]]],
                     atomic_rows: list[dict], composition_rows: list[dict]) -> dict:
    """Count cases/chains complete across all four target languages."""
    cases: dict[str, list[dict]] = defaultdict(list)
    for row in atomic_rows:
        cases[row["source_qa_id"]].append(row)
    atomic_total = Counter()
    atomic_complete = Counter()
    for case_id, rows in cases.items():
        splits = {row.get("split") for row in rows}
        hops = {row.get("hop") for row in rows}
        ids = {row.get("sft_id") for row in rows}
        if len(splits) != 1 or len(hops) != 2 or hops != {"r1", "r2"} or len(ids) != 2:
            raise ValueError(f"Atomic case is not exactly two same-split hops: {case_id}")
        split = next(iter(splits))
        atomic_total[split] += 1
        if all((sft_id, "atomic") in pass_by_language[lang]
               for sft_id in ids for lang in LANGS):
            atomic_complete[split] += 1
    if dict(atomic_total) != EXPECTED_ATOMIC:
        raise ValueError(f"Atomic source cases differ from the frozen universe: {dict(atomic_total)}")

    chains: dict[str, list[dict]] = defaultdict(list)
    for row in composition_rows:
        if row.get("language") not in LANGS:
            raise ValueError(f"Unexpected composition language: {row.get('language')}")
        chains[row["source_qa_id"]].append(row)
    composition_complete = 0
    for chain_id, rows in chains.items():
        if len(rows) != 16 or len({(r["language"], r["prompt_id"]) for r in rows}) != 16:
            raise ValueError(f"Composition chain does not contain 4 prompts x 4 languages: {chain_id}")
        if any(sum(r["language"] == lang for r in rows) != 4 for lang in LANGS):
            raise ValueError(f"Composition language coverage is incomplete: {chain_id}")
        if all((row["prompt_id"], "composition") in pass_by_language[row["language"]]
               for row in rows):
            composition_complete += 1

    return {
        "atomic_cases_total": dict(sorted(atomic_total.items())),
        "optimistic_complete_atomic_cases": {
            split: {"complete": atomic_complete.get(split, 0), "required": expected}
            for split, expected in EXPECTED_ATOMIC.items()
        },
        "composition_chains_total": len(chains),
        "optimistic_complete_composition_chains": {
            "complete": composition_complete, "required": EXPECTED_COMPOSITION
        },
    }


def analyze() -> dict:
    queue_manifest_path = HERE / "manifest.json"
    queue_manifest = read_json(queue_manifest_path)
    candidate_report_path = HERE / "parsed_repair_candidates_report.json"
    candidate_path = HERE / "parsed_repair_candidates.jsonl"
    candidate_report = read_json(candidate_report_path)
    input_manifest_path = INPUTS / "full799.manifest.json"
    input_path = INPUTS / "full799.jsonl"
    map_path = INPUTS / "full799.private_map.jsonl"
    verified_path = INPUTS / "full799.verified_repair_reviews.jsonl"
    verification_path = REMOTE / "verification.json"
    result_path = REMOTE / "repair_review.json"
    remote_source_path = REMOTE / "source/synthworlds-ai-only-translation-repair-review.py"
    launch_path = INPUTS / "full799.kaggle_launch_manifest.json"

    if (queue_manifest.get("status") != "TRANSLATION_REQUEST_QUEUE_ONLY_NOT_APPROVED"
            or queue_manifest.get("total_flagged_rows") != 799
            or candidate_report.get("status") != "CHATGPT_REPAIR_CANDIDATES_COMPLETE_PENDING_INDEPENDENT_REVIEW"
            or candidate_report.get("do_not_train") is not True
            or candidate_report.get("candidate_file_sha256") != sha_file(candidate_path)
            or candidate_report.get("candidate_rows") != 799
            or candidate_report.get("deterministic_failures_by_language")):
        raise ValueError("The source repair queue or parsed candidate receipt is incomplete or changed")

    input_manifest = read_json(input_manifest_path)
    launch = read_json(launch_path)
    verification = read_json(verification_path)
    input_rows = read_jsonl(input_path)
    compact_input_sha256 = compact_json_array_sha256(input_rows)
    if (input_manifest.get("status") != "MASKED_AI_REVIEW_INPUT_ONLY_NOT_APPROVED"
            or input_manifest.get("mode") != "full799" or input_manifest.get("rows") != 799
            or input_manifest.get("answers_or_splits_in_input") is not False
            or input_manifest.get("human_translation_reviewed") is not False
            or input_manifest.get("input_sha256") != sha_file(input_path)
            or input_manifest.get("candidate_file_sha256") != sha_file(candidate_path)
            or input_manifest.get("candidate_report_sha256") != sha_file(candidate_report_path)
            or input_manifest.get("private_map_sha256") != sha_file(map_path)
            or input_manifest.get("source_review_file_sha256_by_language") != queue_manifest.get("source_review_rows_sha256_by_language")):
        raise ValueError("The independent review input manifest is not bound to the current queue/candidates")
    if (launch.get("mode") != "full799" or launch.get("rows") != 799
            or launch.get("input_jsonl_sha256") != sha_file(input_path)
            or launch.get("input_manifest_sha256") != sha_file(input_manifest_path)
            or launch.get("input_compact_array_sha256") != compact_input_sha256
            or launch.get("private_map_sha256") != sha_file(map_path)
            or verification.get("status") != "VERIFIED_AI_ONLY_REPAIR_REVIEW_DIAGNOSTIC"
            or verification.get("release_approved") is not False
            or verification.get("training_eligible") is not False
            or verification.get("human_translation_reviewed") is not False
            or verification.get("answers_or_splits_sent") is not False
            or verification.get("rows") != 799
            or verification.get("input_sha256") != compact_input_sha256
            or verification.get("review_script_sha256") != launch.get("review_script_sha256")
            or verification.get("submitted_script_matches_launch_manifest") is not True
            or verification.get("output_sha256") != sha_file(result_path)
            or verification.get("review_script_sha256") != sha_file(remote_source_path)):
        raise ValueError("The private Kaggle repair review lacks a complete, hash-bound diagnostic receipt")

    candidates = read_jsonl(candidate_path)
    candidate_by_key = unique_index(candidates,
        lambda r: (r.get("language"), r.get("prompt_id"), r.get("kind")), "candidate")
    private_map = read_jsonl(map_path)
    map_by_key = unique_index(private_map,
        lambda r: (r.get("language"), r.get("prompt_id"), r.get("kind")), "private map")
    verified = read_bound_verified_rows(verification, verified_path, 799)
    verified_by_key = unique_index(verified,
        lambda r: (r.get("language"), r.get("prompt_id"), r.get("kind")), "verified review")
    if set(candidate_by_key) != set(map_by_key) or set(candidate_by_key) != set(verified_by_key):
        raise ValueError("Candidate, private-map, and verified-review row coverage differs")
    if len(candidates) != 799 or len(verified) != 799:
        raise ValueError("Expected exactly 799 unique repair candidates and review decisions")

    source_by_language = {}
    original_by_language = {}
    review_report_path = RELEASE / "translation_reviews_roundtrip_qwen14b/report.json"
    review_report = read_json(review_report_path)
    if (review_report.get("status") != "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE"
            or review_report.get("release_ready_for_assembly") is not True
            or review_report.get("human_translation_reviewed") is not False
            or review_report.get("answers_sent_to_reviewer") is not False):
        raise ValueError("The original full training translation review is incomplete or mislabelled")

    pass_by_language = {lang: set() for lang in LANGS}
    candidate_counts = {lang: Counter() for lang in LANGS}
    row_counts = {}
    for lang in LANGS:
        source_rows, source_hashes = collect_rows(lang)
        source_by_key = unique_index(source_rows,
            lambda r: (r["prompt_id"], r["kind"]), f"{lang} source")
        if len(source_by_key) != 488:
            raise ValueError(f"{lang}: expected 488 current source questions")
        if source_hashes != queue_manifest["reconstructed_source_hashes_by_language"][lang]:
            raise ValueError(f"{lang}: reconstructed translation inputs differ from the frozen repair queue")
        original_path = RELEASE / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl"
        if sha_file(original_path) != input_manifest["source_review_file_sha256_by_language"][lang]:
            raise ValueError(f"{lang}: original 14B review file differs from the frozen repair input")
        detail = review_report.get("languages", {}).get(lang, {})
        original_rows = read_jsonl(original_path)
        original_by_key = unique_index(original_rows,
            lambda r: (r.get("prompt_id"), r.get("kind")), f"{lang} original review")
        if len(original_by_key) != 488 or detail.get("adjudication_sha256") != sha_file(original_path):
            raise ValueError(f"{lang}: original review coverage/hash mismatch")

        source_by_language[lang] = source_by_key
        original_by_language[lang] = original_by_key
        for key, source in source_by_key.items():
            original = original_by_key.get(key)
            if (not original or original.get("text_en") != source.get("text_en")
                    or original.get("text_translation") != source.get("text_translation")
                    or original.get("human_translation_reviewed") is not False
                    or original.get("answers_sent_to_reviewer") is not False):
                raise ValueError(f"{lang}/{key}: original review no longer matches its source text")
            if original.get("final_review_status") in PASS_STATUSES:
                pass_by_language[lang].add(key)

    for key, candidate in candidate_by_key.items():
        lang, prompt_id, kind = key
        source = source_by_language[lang].get((prompt_id, kind))
        old_review = original_by_language[lang].get((prompt_id, kind))
        local_map = map_by_key[key]
        result_row = verified_by_key[key]
        if (lang not in LANGS or source is None or old_review is None
                or old_review.get("final_review_status") != "AI_ONLY_REVIEW_FLAGGED"
                or candidate.get("translation_status") != "CHATGPT_UI_REPAIR_CANDIDATE_PENDING_INDEPENDENT_REVIEW"
                or candidate.get("human_translation_reviewed") is not False
                or candidate.get("answers_sent_to_translator") is not False
                or candidate.get("text_en") != source.get("text_en")
                or candidate.get("source_text_sha256") != sha_bytes(source["text_en"].encode("utf-8"))
                or candidate.get("old_text_translation_sha256") != sha_bytes(source["text_translation"].encode("utf-8"))
                or local_map.get("original_review_status") != "AI_ONLY_REVIEW_FLAGGED"
                or local_map.get("source_text_sha256") != candidate.get("source_text_sha256")
                or local_map.get("old_translation_sha256") != candidate.get("old_text_translation_sha256")
                or local_map.get("candidate_translation_sha256") != sha_bytes(candidate["text_translation"].encode("utf-8"))
                or result_row.get("candidate_translation_sha256") != local_map.get("candidate_translation_sha256")
                or result_row.get("human_translation_reviewed") is not False):
            raise ValueError(f"{key}: candidate or review decision is not bound to the reviewed source")
        status = result_row.get("review_status")
        if status not in {"AI_ONLY_REPAIR_REVIEW_PASS", "AI_ONLY_REPAIR_REVIEW_FLAGGED", "AI_ONLY_REPAIR_REVIEW_BLOCKED"}:
            raise ValueError(f"{key}: unexpected diagnostic review status {status}")
        candidate_counts[lang][status] += 1
        if status == "AI_ONLY_REPAIR_REVIEW_PASS":
            if (result_row.get("backtranslation_markers_pass") is not True
                    or result_row.get("decision", {}).get("semantic_pass") is not True
                    or result_row.get("decision", {}).get("fluency_pass") is not True):
                raise ValueError(f"{key}: diagnostic pass is missing marker, semantic, or fluency evidence")
            pass_by_language[lang].add((prompt_id, kind))

    if {lang: sum(candidate_counts[lang].values()) for lang in LANGS} != EXPECTED_FLAGGED:
        raise ValueError("Repair review counts by language differ from the frozen queue")
    row_counts = {}
    for lang in LANGS:
        source_count = len(source_by_language[lang])
        if len(pass_by_language[lang]) > source_count:
            raise ValueError(f"{lang}: optimistic pass coverage exceeds source row count")
        original_passes = sum(r.get("final_review_status") in PASS_STATUSES
                              for r in original_by_language[lang].values())
        row_counts[lang] = {
            "questions": source_count,
            "original_final_review_pass": original_passes,
            "candidate_diagnostic_pass": candidate_counts[lang]["AI_ONLY_REPAIR_REVIEW_PASS"],
            "candidate_flagged": candidate_counts[lang]["AI_ONLY_REPAIR_REVIEW_FLAGGED"],
            "candidate_blocked": candidate_counts[lang]["AI_ONLY_REPAIR_REVIEW_BLOCKED"],
            "optimistic_pass_after_repair": len(pass_by_language[lang]),
        }

    coverage = compute_coverage(
        pass_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    sufficient = (
        all(coverage["optimistic_complete_atomic_cases"][split]["complete"] == expected
            for split, expected in EXPECTED_ATOMIC.items())
        and coverage["composition_chains_total"] == EXPECTED_COMPOSITION
        and coverage["optimistic_complete_composition_chains"]["complete"] == EXPECTED_COMPOSITION
    )
    report = {
        "status": "OPTIMISTIC_REPAIR_COVERAGE_DIAGNOSTIC_NOT_RELEASE_APPROVAL",
        "decision": "FULL_RESCREEN_MAY_BE_WORTH_EVALUATING" if sufficient else "THIS_REPAIR_ROUTE_ALONE_CANNOT_MEET_RELEASE_COUNTS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_review_status": verification["status"],
        "candidate_rows": 799,
        "candidate_review_sha256": sha_file(verified_path),
        "candidate_review_receipt_sha256": sha_file(verification_path),
        "source_queue_sha256": sha_file(queue_manifest_path),
        "candidate_file_sha256": sha_file(candidate_path),
        "original_review_receipt_sha256": sha_file(review_report_path),
        "rows_by_language": row_counts,
        **coverage,
        "optimistic_only": True,
        "release_approved": False,
        "training_eligible": False,
        "human_translation_reviewed": False,
        "limitations": [
            "Candidate diagnostic passes are only optimistic possibilities; every changed translation still needs the full production review.",
            "This analysis does not change source translations, reviewer files, release manifests, or SFT data.",
            "AI-only screening does not certify human fluency; the candidate reviewer shares a Qwen model family with the original screen.",
        ],
    }
    output = HERE / "optimistic_coverage_after_full799.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    report["output"] = str(output)
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
