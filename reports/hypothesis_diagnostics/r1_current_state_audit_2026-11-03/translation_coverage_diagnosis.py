#!/usr/bin/env python3
"""Recompute translation-repair coverage without changing release inputs.

This is a read-only audit. It does not call a model, edit translations, or
approve a training release. Repair passes are counted as optimistic candidates
only because they have not received the frozen Qwen 14B round-trip review.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "experiments/synthworlds_r1"
TRAINING = PROJECT / "training_release_draft"
OUT = Path(__file__).resolve().parent / "translation_coverage_diagnosis.json"
LANGS = ("fr", "de", "hi", "th")
PASS_STATUSES = {
    "AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS",
    "AI_ONLY_14B_ADJUDICATED_PASS",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> dict:
    final_report_path = TRAINING / "translation_reviews_roundtrip_qwen14b/report.json"
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    final: dict[tuple[str, str, str], dict] = {}
    for lang in LANGS:
        path = TRAINING / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl"
        rows = jsonl(path)
        detail = final_report["languages"][lang]
        expected_hash = detail.get("adjudication_sha256")
        if len(rows) != 488 or sha256(path) != expected_hash:
            raise RuntimeError(f"Final review file failed count/hash validation: {lang}")
        for row in rows:
            key = (lang, row["prompt_id"], row["kind"])
            if key in final:
                raise RuntimeError(f"Duplicate final review key: {key}")
            final[key] = row

    repair_dir = TRAINING / "translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03"
    repair_report_path = repair_dir / "report.json"
    repair_rows_path = repair_dir / "repair_candidates.jsonl"
    repair_report = json.loads(repair_report_path.read_text(encoding="utf-8"))
    repair_rows = jsonl(repair_rows_path)
    if (repair_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or repair_report.get("processed_rows") != len(repair_rows)
            or sha256(repair_rows_path) != repair_report.get("decisions_sha256")):
        raise RuntimeError("Full repair diagnostic failed status/count/hash validation")
    repair = {}
    for row in repair_rows:
        key = (row["language"], row["prompt_id"], row["kind"])
        if key in repair:
            raise RuntimeError(f"Duplicate repair key: {key}")
        repair[key] = row

    gemma_dir = Path(__file__).resolve().parent / "evidence/gemma3_all_repair_candidates"
    gemma_report_path = gemma_dir / "report.json"
    gemma_rows_path = gemma_dir / "decisions.jsonl"
    gemma_report = json.loads(gemma_report_path.read_text(encoding="utf-8"))
    gemma_rows = jsonl(gemma_rows_path)
    if (gemma_report.get("status") != "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or gemma_report.get("completed_rows") != len(gemma_rows)
            or sha256(gemma_rows_path) != gemma_report.get("decision_sha256")
            or gemma_report.get("candidate_rows_sha256") != sha256(repair_rows_path)
            or gemma_report.get("candidate_report_sha256") != sha256(repair_report_path)
            or gemma_report.get("final_review_report_sha256") != sha256(final_report_path)
            or len(gemma_rows) != len(repair_rows)):
        raise RuntimeError("Gemma3 repair review failed status/count/hash validation")
    gemma = {}
    for row in gemma_rows:
        key = (row["language"], row["prompt_id"], row["kind"])
        if key in gemma:
            raise RuntimeError(f"Duplicate Gemma3 review key: {key}")
        candidate = repair.get(key)
        if (candidate is None
                or row.get("source_en_sha256") != candidate.get("source_en_sha256")
                or row.get("candidate_translation_sha256") != sha256_text(candidate.get("new_translation", ""))
                or candidate.get("new_translation_sha256") != row.get("candidate_translation_sha256")
                or row.get("qwen7b_preliminary_pass") != (candidate.get("status") == "DIAGNOSTIC_QWEN_PASS")
                or row.get("gemma_pass") != all(row.get(field) is True for field in (
                    "semantic_equivalent", "relation_direction_preserved", "answer_type_preserved", "fluent"))):
            raise RuntimeError(f"Gemma3 decision does not match candidate or review fields: {key}")
        gemma[key] = row
    if set(gemma) != set(repair):
        raise RuntimeError("Gemma3 review and repair candidate keys differ")

    # The saved repair review rows do not carry protected_entity_names. Rejoin
    # them to the hash-bound source set and recompute integrity from actual text.
    sys.path.insert(0, str(TRAINING / "scripts"))
    from review_training_translations_qwen import collect_rows, deterministic_checks
    source_rows = {}
    for lang in LANGS:
        rows, _ = collect_rows(lang)
        for row in rows:
            key = (lang, row["prompt_id"], row["kind"])
            if key in source_rows or key not in final or row["text_en"] != final[key]["text_en"]:
                raise RuntimeError(f"Source/review key or text mismatch: {key}")
            source_rows[key] = row
    if set(source_rows) != set(final):
        raise RuntimeError("Source and final review keys differ")
    check_names = ("nonempty", "target_script_present", "protected_names_preserved",
                   "protected_numbers_preserved", "no_unrestored_marker")
    repair_checks = {}
    strict_failures = Counter()
    for key, source in source_rows.items():
        checks = deterministic_checks(key[0], {**source, "text_translation": final[key]["text_translation"]})
        if final[key].get("final_review_status") in PASS_STATUSES:
            for name in check_names:
                if checks[name] is not True:
                    strict_failures[name] += 1
    for key, candidate in repair.items():
        repair_checks[key] = deterministic_checks(
            key[0], {**source_rows[key], "text_translation": candidate.get("new_translation", "")}
        )

    def repair_integrity_ok(lang: str, prompt_id: str, kind: str) -> bool:
        checks = repair_checks.get((lang, prompt_id, kind), {})
        return all(checks.get(name) is True for name in check_names)

    def strict_ok(lang: str, prompt_id: str, kind: str) -> bool:
        return final[(lang, prompt_id, kind)].get("final_review_status") in PASS_STATUSES

    def optimistic_ok(lang: str, prompt_id: str, kind: str) -> bool:
        candidate = repair.get((lang, prompt_id, kind), {})
        return strict_ok(lang, prompt_id, kind) or candidate.get("status") == "DIAGNOSTIC_QWEN_PASS"

    def gemma_candidate_ok(lang: str, prompt_id: str, kind: str) -> bool:
        row = gemma.get((lang, prompt_id, kind), {})
        return row.get("review_status") == "AI_REVIEW_COMPLETE" and row.get("gemma_pass") is True

    def qwen_gemma_candidate_ok(lang: str, prompt_id: str, kind: str) -> bool:
        candidate = repair.get((lang, prompt_id, kind), {})
        return (candidate.get("status") == "DIAGNOSTIC_QWEN_PASS"
                and gemma_candidate_ok(lang, prompt_id, kind))

    def combined_ok(strategy):
        return lambda lang, prompt_id, kind: strict_ok(lang, prompt_id, kind) or strategy(lang, prompt_id, kind)

    queue = jsonl(TRAINING / "composition_translation_review_queue.jsonl")
    prompt_meta: dict[str, dict] = {}
    for row in queue:
        prompt_id = row["prompt_id"]
        value = {"source_qa_id": row["source_qa_id"], "kind": prompt_id.rsplit(":", 1)[-1]}
        if prompt_id in prompt_meta and prompt_meta[prompt_id] != value:
            raise RuntimeError(f"Composition prompt has inconsistent metadata: {prompt_id}")
        prompt_meta[prompt_id] = value

    composition = {}
    composition_strategies = (
        ("strict_final_review", strict_ok),
        ("qwen7b_only_unreviewed_repairs", combined_ok(lambda l, p, k: repair.get((l, p, k), {}).get("status") == "DIAGNOSTIC_QWEN_PASS")),
        ("gemma3_only_unreviewed_repairs", combined_ok(gemma_candidate_ok)),
        ("qwen7b_and_gemma3_unreviewed_repairs", combined_ok(qwen_gemma_candidate_ok)),
        ("qwen7b_gemma3_integrity_unreviewed_repairs", combined_ok(lambda l, p, k: qwen_gemma_candidate_ok(l, p, k) and repair_integrity_ok(l, p, k))),
    )
    for label, passes in composition_strategies:
        common_prompts = {
            prompt_id for prompt_id, meta in prompt_meta.items()
            if all(passes(lang, prompt_id, "composition") for lang in LANGS)
        }
        by_chain: dict[str, set[str]] = defaultdict(set)
        for prompt_id in common_prompts:
            by_chain[prompt_meta[prompt_id]["source_qa_id"]].add(prompt_id)
        full_chains = 0
        source_chains = {row["source_qa_id"] for row in queue}
        for source_qa_id in source_chains:
            expected = {
                row["prompt_id"] for row in queue
                if row["source_qa_id"] == source_qa_id
            }
            if len(expected) == 4 and expected.issubset(common_prompts):
                full_chains += 1
        type_counts = Counter(prompt_meta[prompt_id]["kind"] for prompt_id in common_prompts)
        composition[label] = {
            "common_prompt_ids_across_all_four_target_languages": len(common_prompts),
            "common_multi_hop_prompt_ids": type_counts.get("multi", 0),
            "common_support_prompt_ids_by_type": {key: type_counts.get(key, 0) for key in ("q1", "q2", "q3")},
            "chains_with_at_least_one_common_prompt": len(by_chain),
            "chains_with_all_four_common_prompts": full_chains,
        }

    atomic_rows = jsonl(TRAINING / "atomic_reuse.jsonl")
    request_rows = jsonl(TRAINING / "atomic_translation_requests.jsonl")
    sft_ids_by_case: dict[str, set[str]] = defaultdict(set)
    split_by_case: dict[str, str] = {}
    for row in atomic_rows:
        case_id = row["source_qa_id"]
        if row["language"] == "en":
            sft_ids_by_case[case_id].add(row["sft_id"])
            split_by_case[case_id] = row["split"]
    for row in request_rows:
        sft_ids_by_case[row["source_qa_id"]].add(row["sft_id"])
        split_by_case[row["source_qa_id"]] = row["split"]

    def atomic_case_passes(case_id: str, checker) -> bool:
        sft_ids = sft_ids_by_case.get(case_id, set())
        return len(sft_ids) == 2 and all(
            checker(lang, sft_id, "atomic")
            for lang in LANGS for sft_id in sft_ids
        )

    all_atomic = {}
    atomic_strategies = (
        ("strict_final_review", strict_ok),
        ("qwen7b_only_unreviewed_repairs", combined_ok(lambda l, p, k: repair.get((l, p, k), {}).get("status") == "DIAGNOSTIC_QWEN_PASS")),
        ("gemma3_only_unreviewed_repairs", combined_ok(gemma_candidate_ok)),
        ("qwen7b_and_gemma3_unreviewed_repairs", combined_ok(qwen_gemma_candidate_ok)),
        ("qwen7b_gemma3_integrity_unreviewed_repairs", combined_ok(lambda l, p, k: qwen_gemma_candidate_ok(l, p, k) and repair_integrity_ok(l, p, k))),
    )
    for label, checker in atomic_strategies:
        counts = Counter()
        for case_id in sft_ids_by_case:
            if atomic_case_passes(case_id, checker):
                counts[split_by_case[case_id]] += 1
        all_atomic[label] = dict(counts)

    eval_cases = jsonl(TRAINING / "evaluation/cases.jsonl")
    eval_coverage = {}
    for label, checker in atomic_strategies:
        by_split = {}
        for split in ("forget", "retain", "holdout"):
            selected = [
                row for row in eval_cases
                if row["split"] == split and atomic_case_passes(row["source_qa_id"], checker)
            ]
            by_split[split] = {
                "eligible_cases": len(selected),
                "eligible_entity_groups": len({row["group_id"] for row in selected}),
                "evaluation_cases": sum(row["split"] == split for row in eval_cases),
            }
        eval_coverage[label] = by_split

    calibration_path = TRAINING / "translation_repair_diagnostics/nllb13_cal64_roundtrip_qwen14_2026-11-03/report.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_rows_path = calibration_path.parent / "selected_candidates.jsonl"
    if sha256(calibration_rows_path) != calibration.get("selected_rows_sha256"):
        raise RuntimeError("NLLB 1.3B calibration file hash mismatch")
    calibration_passes = sum(
        cell.get("full_screen_pass", 0)
        for per_language in calibration.get("per_language_kind", {}).values()
        for cell in per_language.values()
    )

    gemma_review_counts = {}
    for language in LANGS:
        gemma_review_counts[language] = {}
        for kind in ("atomic", "composition"):
            rows = [row for row in gemma_rows if row["language"] == language and row["kind"] == kind]
            gemma_review_counts[language][kind] = {
                "reviewed": len(rows),
                "gemma3_passes": sum(row.get("gemma_pass") is True and row.get("review_status") == "AI_REVIEW_COMPLETE" for row in rows),
                "qwen7b_preliminary_passes": sum(row.get("qwen7b_preliminary_pass") is True for row in rows),
                "qwen7b_gemma3_agreements": sum(row.get("qwen7b_preliminary_pass") is True and row.get("gemma_pass") is True for row in rows),
                "review_errors": sum(row.get("review_status") != "AI_REVIEW_COMPLETE" for row in rows),
            }

    result = {
        "status": "READ_ONLY_DIAGNOSTIC_NOT_RELEASE_APPROVAL",
        "source_hashes": {
            "final_review_report": sha256(final_report_path),
            "repair_report": sha256(repair_report_path),
            "repair_candidates": sha256(repair_rows_path),
            "nllb_calibration_report": sha256(calibration_path),
            "nllb_calibration_candidates": sha256(calibration_rows_path),
        },
        "final_translation_review": {
            "rows_per_language": {lang: final_report["languages"][lang]["questions"] for lang in LANGS},
            "passes_per_language": {
                lang: sum(strict_ok(lang, row["prompt_id"], row["kind"]) for row in final.values() if row["language"] == lang)
                for lang in LANGS
            },
            "human_reviewed": False,
        },
        "full_flagged_repair_diagnostic": {
            "candidate_rows": len(repair_rows),
            "translation_successes": repair_report.get("diagnostic_translation_successes"),
            "deterministic_passes": repair_report.get("deterministic_passes"),
            "preliminary_qwen7b_passes": repair_report.get("qwen_passes"),
            "recomputed_integrity": {
                "required_checks": list(check_names),
                "passes": sum(all(checks[name] is True for name in check_names) for checks in repair_checks.values()),
                "failures_by_check": {name: sum(checks[name] is not True for checks in repair_checks.values()) for name in check_names},
                "stored_pass_but_recomputed_fail": sum(repair[key].get("deterministic_pass") is True and not all(checks[name] is True for name in check_names) for key, checks in repair_checks.items()),
                "gemma_source_with_unmasked_protected_name": sum(
                    any(name and name in source_rows[key]["text_en"] and name in row.get("source_question_masked", "")
                        for name in source_rows[key].get("protected_entity_names", []))
                    for key, row in gemma.items()
                ),
                "strict_original_final_pass_failures_by_check": dict(strict_failures),
            },
            "preliminary_qwen7b_passes_by_language": {
                lang: sum(row.get("language") == lang and row.get("status") == "DIAGNOSTIC_QWEN_PASS" for row in repair_rows)
                for lang in LANGS
            },
            "qwen14b_roundtrip_release_review_completed": False,
            "gemma3_second_opinion": {
                "review_report_sha256": sha256(gemma_report_path),
                "decision_rows": len(gemma_rows),
                "decision_sha256": sha256(gemma_rows_path),
                "passes_by_language_and_kind": gemma_review_counts,
                "total_gemma3_passes": sum(row.get("gemma_pass") is True for row in gemma_rows),
                "total_qwen7b_gemma3_agreements": sum(row.get("qwen7b_preliminary_pass") is True and row.get("gemma_pass") is True for row in gemma_rows),
                "same_model_family_as_repair_translator": True,
                "different_model_family_from_qwen_reviewers": True,
            },
        },
        "nllb_1_3b_calibration": {
            "sample_size": calibration.get("sample_size"),
            "full_screen_passes": calibration_passes,
            "status": calibration.get("status"),
            "do_not_train": calibration.get("do_not_train"),
        },
        "atomic_training_case_coverage_by_split": all_atomic,
        "evaluation_case_atomic_coverage": eval_coverage,
        "matched_composition_training_prompts": composition,
        "limits": [
            "All repaired rows remain unapproved: Qwen7B and Gemma3 checks are diagnostics, and the frozen Qwen14B/back-translation release review has not been run on replacements.",
            "The NLLB 1.3B calibration has only 64 rows and does not estimate full-corpus translation accuracy.",
            "TranslateGemma and Gemma3 share the Gemma family; Qwen7B is a different family. AI-only agreement is not fluent-human certification.",
            "Repair-generation and Gemma-review inputs omitted protected entity lists from saved review rows; recomputed checks use the original protected lists. The saved review did not consistently mask entities.",
            "No source translation, split, training file, or model artifact is changed by this script.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
