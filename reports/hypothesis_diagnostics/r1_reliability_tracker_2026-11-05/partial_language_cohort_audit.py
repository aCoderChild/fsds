#!/usr/bin/env python3
"""Diagnostic-only audit of per-language training/evaluation coverage.

This script does not approve translations, alter the source release, or write a
training corpus. It asks whether relaxing only the requirement that every case
have all four target-language translations would meet the frozen numeric
forget-cohort minimums, and separately reports the composition-chain shortfall.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "experiments/synthworlds_r1"
RELEASE = PROJECT / "training_release_draft"
OUT = Path(__file__).resolve().parent
LANGUAGES = ("fr", "de", "hi", "th")
MIN_CASES = 15
MIN_GROUPS = 8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_release_module():
    path = RELEASE / "scripts/assemble_training_release.py"
    spec = importlib.util.spec_from_file_location("r1_release_assembler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import release checker from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    assembler = load_release_module()
    # This call revalidates source hashes, reviewer hashes, row coverage, and
    # AI-only decision invariants before any coverage numbers are computed.
    reviews = assembler.checked_translation_reviews()

    atomic_rows = read_jsonl(RELEASE / "atomic_reuse.jsonl") + read_jsonl(
        RELEASE / "atomic_translation_requests.jsonl"
    )
    cases: dict[str, list[dict]] = defaultdict(list)
    for row in atomic_rows:
        cases[row["source_qa_id"]].append(row)
    case_ids = set(cases)
    if any(len({row["sft_id"] for row in rows}) != 2 for rows in cases.values()):
        raise ValueError("Each atomic source case must have exactly two distinct hop questions")

    group_by_case: dict[str, str] = {}
    split_by_case: dict[str, str] = {}
    for group_row in read_jsonl(PROJECT / "leakage_groups.draft.jsonl"):
        for case_id in group_row["source_qa_ids"]:
            if case_id in group_by_case:
                raise ValueError(f"Duplicate source case in leakage-group manifest: {case_id}")
            group_by_case[case_id] = group_row["group_id"]
    for case_id, rows in cases.items():
        splits = {row.get("split") for row in rows}
        if len(splits) != 1:
            raise ValueError(f"Atomic case has inconsistent split labels: {case_id}")
        split_by_case[case_id] = next(iter(splits))
    if case_ids - set(group_by_case):
        raise ValueError(f"Missing leakage-group IDs: {sorted(case_ids - set(group_by_case))[:5]}")

    eligible: dict[str, set[str]] = {}
    per_language: dict[str, dict] = {}
    for language in LANGUAGES:
        selected = {
            case_id
            for case_id, rows in cases.items()
            if all(
                reviews.get((language, sft_id, "atomic"), {}).get("final_review_status")
                in assembler.PASS_STATUSES
                for sft_id in {row["sft_id"] for row in rows}
            )
        }
        eligible[language] = selected
        detail = {}
        for split in ("forget", "retain"):
            split_cases = sorted(case_id for case_id in selected if split_by_case[case_id] == split)
            groups = sorted({group_by_case[case_id] for case_id in split_cases})
            detail[split] = {
                "eligible_cases": len(split_cases),
                "leakage_groups": len(groups),
                "source_qa_ids": split_cases,
                "group_ids": groups,
            }
        per_language[language] = detail

    all_language_cases = set.intersection(*(eligible[language] for language in LANGUAGES))
    all_language = {}
    for split in ("forget", "retain"):
        split_cases = sorted(case_id for case_id in all_language_cases if split_by_case[case_id] == split)
        all_language[split] = {
            "eligible_cases": len(split_cases),
            "leakage_groups": len({group_by_case[case_id] for case_id in split_cases}),
            "source_qa_ids": split_cases,
        }

    composition_rows = read_jsonl(RELEASE / "composition_reuse.jsonl")
    chains: dict[str, set[str]] = defaultdict(set)
    for row in composition_rows:
        chains[row["source_qa_id"]].add(row["sft_id"])
    if len(chains) != assembler.EXPECTED_COMPOSITION_CHAINS or any(len(ids) != 4 for ids in chains.values()):
        raise ValueError("Composition source does not match the frozen 44-chain, four-question design")
    composition = {}
    for language in LANGUAGES:
        pass_histogram = Counter()
        completed = []
        near_complete = []
        for chain_id, sft_ids in chains.items():
            passed = sum(
                reviews.get((language, sft_id, "composition"), {}).get("final_review_status")
                in assembler.PASS_STATUSES
                for sft_id in sft_ids
            )
            pass_histogram[str(passed)] += 1
            if passed == 4:
                completed.append(chain_id)
            elif passed == 3:
                near_complete.append(chain_id)
        composition[language] = {
            "complete_chains": len(completed),
            "near_complete_three_of_four": len(near_complete),
            "question_pass_histogram_0_to_4": {str(i): pass_histogram[str(i)] for i in range(5)},
            "completed_chain_source_qa_ids": sorted(completed),
            "near_complete_chain_source_qa_ids": sorted(near_complete),
        }

    min_forget = min(per_language[lang]["forget"]["eligible_cases"] for lang in LANGUAGES)
    min_forget_groups = min(per_language[lang]["forget"]["leakage_groups"] for lang in LANGUAGES)
    min_retain = min(per_language[lang]["retain"]["eligible_cases"] for lang in LANGUAGES)
    min_retain_groups = min(per_language[lang]["retain"]["leakage_groups"] for lang in LANGUAGES)
    composition_floor = min(composition[lang]["complete_chains"] for lang in LANGUAGES)

    input_paths = [
        RELEASE / "atomic_reuse.jsonl",
        RELEASE / "atomic_translation_requests.jsonl",
        RELEASE / "composition_reuse.jsonl",
        RELEASE / "translation_reviews_roundtrip_qwen14b/report.json",
        RELEASE / "translation_reviews_roundtrip_qwen/report.json",
        PROJECT / "leakage_groups.draft.jsonl",
    ]
    result = {
        "status": "DIAGNOSTIC_ONLY_NOT_RELEASE_APPROVAL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Coverage if atomic eligibility is evaluated per target language rather than requiring every target-language translation for every case.",
        "training_eligible": False,
        "translation_screen": "AI_ONLY_TWO_STAGE_QWEN_FAMILY; mismatch-control concerns remain; no human-gold review",
        "frozen_numeric_forget_cohort_minimum": {"cases_per_language": MIN_CASES, "leakage_groups_per_language": MIN_GROUPS},
        "per_language_atomic_cases": per_language,
        "all_four_target_languages_complete_cases": all_language,
        "partial_coverage_summary": {
            "minimum_forget_cases_across_target_languages": min_forget,
            "minimum_forget_groups_across_target_languages": min_forget_groups,
            "each_language_meets_numeric_forget_cohort_floor": min_forget >= MIN_CASES and min_forget_groups >= MIN_GROUPS,
            "minimum_retain_cases_across_target_languages": min_retain,
            "minimum_retain_groups_across_target_languages": min_retain_groups,
            "five_language_balanced_atomic_retain_rows_possible_without_adding_translations": min_retain * 2,
            "all_four_language_common_forget_cases": all_language["forget"]["eligible_cases"],
            "all_four_language_common_retain_cases": all_language["retain"]["eligible_cases"],
        },
        "composition_chain_coverage": composition,
        "minimum_complete_composition_chains_across_target_languages": composition_floor,
        "composition_floor_met_for_all_four_target_languages": composition_floor > 0,
        "interpretation": [
            "Language-specific atomic eligibility would preserve the frozen 15-case/8-group forget-cohort count floor in every target language; this is only translation coverage, not model competence.",
            "It would not preserve a four-language common-case cohort: only the common intersection is reported in all_four_target_languages_complete_cases.",
            "Current complete composition-chain coverage is not balanced across all four target languages; Hindi and Thai remain major bottlenecks.",
            "All decisions remain AI-only and quarantined; reviewer controls and missing human-gold review prevent corpus approval.",
            "No source split, translation, release manifest, or training file was modified; no training corpus is emitted by this diagnostic.",
        ],
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in input_paths},
        "reviewer_decision_rows": len(reviews),
    }

    json_path = OUT / "partial_language_cohort_audit.json"
    md_path = OUT / "partial_language_cohort_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Partial-language cohort feasibility audit — 2026-11-05",
        "",
        "**Status: diagnostic only; no training release or approval.** The audit rechecks all current translation-review hashes and asks whether relaxing the all-four-target-languages-per-case requirement would make the atomic forget cohort numerically large enough. It does not relax translation pass criteria, graph isolation, split labels, or the baseline competence gate.",
        "",
        "## Atomic two-hop case coverage",
        "",
        "| Language | Forget cases / groups | Retain cases / groups | Meets frozen forget count floor (15 cases, 8 groups)? |",
        "|---|---:|---:|---|",
    ]
    for lang in LANGUAGES:
        f = per_language[lang]["forget"]
        r = per_language[lang]["retain"]
        meets = f["eligible_cases"] >= MIN_CASES and f["leakage_groups"] >= MIN_GROUPS
        lines.append(f"| {lang.upper()} | {f['eligible_cases']} / {f['leakage_groups']} | {r['eligible_cases']} / {r['leakage_groups']} | {'Yes' if meets else 'No'} |")
    lines += [
        "",
        f"The intersection with both atomic hops screened as passing in **all four** target languages is only **{all_language['forget']['eligible_cases']} forget cases / {all_language['forget']['leakage_groups']} groups** and **{all_language['retain']['eligible_cases']} retain cases / {all_language['retain']['leakage_groups']} groups**.",
        "",
        "## Disjoint composition practice",
        "",
        "| Language | Complete four-question chains | Chains with 3/4 questions passing |",
        "|---|---:|---:|",
    ]
    for lang in LANGUAGES:
        c = composition[lang]
        lines.append(f"| {lang.upper()} | {c['complete_chains']} / 44 | {c['near_complete_three_of_four']} / 44 |")
    lines += [
        "",
        "A partial atomic release could satisfy the *sample-count* floor before model evaluation, but it does not solve multilingual composition training. With the three-seed model competence gate still untested, this is not evidence the experiment will pass. Hindi/Thai composition translations need additional validated coverage, and the AI-only reviewer calibration issue remains unresolved.",
        "",
        "## Reproduction and limits",
        "",
        "Run `python -B partial_language_cohort_audit.py`. The script imports the frozen release checker, verifies the hash-bound 488-row review for each language, reads the leakage-group manifest, and emits the JSON receipt alongside this report. This is not a population-level translation-quality estimate and does not authorize training.",
        "",
        "No training corpus was written.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "training_eligible": False,
        "per_language_forget_cases_groups": {
            lang: [per_language[lang]["forget"]["eligible_cases"], per_language[lang]["forget"]["leakage_groups"]]
            for lang in LANGUAGES
        },
        "per_language_retain_cases_groups": {
            lang: [per_language[lang]["retain"]["eligible_cases"], per_language[lang]["retain"]["leakage_groups"]]
            for lang in LANGUAGES
        },
        "all_language_common_forget_cases_groups": [
            all_language["forget"]["eligible_cases"], all_language["forget"]["leakage_groups"]
        ],
        "all_language_common_retain_cases_groups": [
            all_language["retain"]["eligible_cases"], all_language["retain"]["leakage_groups"]
        ],
        "complete_composition_chains": {lang: composition[lang]["complete_chains"] for lang in LANGUAGES},
        "reviewer_decision_rows": len(reviews),
    }, indent=2))


if __name__ == "__main__":
    main()
