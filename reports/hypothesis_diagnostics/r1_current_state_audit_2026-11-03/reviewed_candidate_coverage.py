#!/usr/bin/env python3
"""Compute complete-case coverage if quarantined calibration passes were reusable.

This is an optimistic read-only diagnostic. It does not approve, merge, or
create a training release. The alternate-mask decisions are not the frozen
production review protocol.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from verify_source_bound_calibration import PROJECT, TRAINING, sha
from verify_nllb_matched32 import main as verify_matched, status
from verify_prior_nllb_calibration import main as verify_prior

HERE = Path(__file__).resolve().parent
TG = HERE / "evidence/source_bound_calibration_roundtrip_retry"
TG_ALT = HERE / "evidence/source_bound_calibration_masking_comparison"
NLLB = HERE / "evidence/nllb13_matched_roundtrip"
OLD = TRAINING / "translation_repair_diagnostics/nllb13_cal64_roundtrip_qwen14_2026-11-03"
OUT = HERE / "reviewed_candidate_coverage.json"
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(TRAINING / "scripts"))
import assemble_training_release as release
import validate_sft_source_integrity as source_integrity
from review_training_translations_qwen import collect_rows, deterministic_checks

LANGS = ("fr", "de", "hi", "th")
REQUIRED = ("nonempty", "target_script_present", "protected_names_preserved",
            "protected_numbers_preserved", "no_unrestored_marker")


def main() -> None:
    # The verifiers recompute source/decision hashes without loading models.
    with contextlib.redirect_stdout(io.StringIO()):
        verify_matched()
        verify_prior()
    canonical, sources = source_integrity.validate_prepared_inputs()
    final = release.checked_translation_reviews()
    review_sources = {}
    for lang in LANGS:
        for row in collect_rows(lang)[0]:
            review_sources[(lang, row["prompt_id"], row["kind"])] = row
    rows_tg = json.loads((TG / "checkpoint.json").read_text(encoding="utf-8"))["records"]
    rows_nllb = json.loads((NLLB / "checkpoint.json").read_text(encoding="utf-8"))["records"]
    alt = json.loads((TG_ALT / "checkpoint.json").read_text(encoding="utf-8"))["decisions"]
    if len(rows_tg) != 32 or len(rows_nllb) != 32:
        raise ValueError("Matched calibration must contain 32 rows per translator")
    provisional = set()
    paired = Counter()
    for tg, nllb in zip(rows_tg, rows_nllb):
        key = (tg["language"], tg["prompt_id"], tg["kind"])
        if key != (nllb["language"], nllb["prompt_id"], nllb["kind"]):
            raise ValueError("Matched calibration rows are misaligned")
        tg_judge = alt[f"{key[0]}|{key[2]}|{key[1]}"]
        tg_status, nllb_status = status(tg, tg_judge), status(nllb, nllb.get("qwen14", {}))
        if "BLOCKED_OR_ERROR" in (tg_status, nllb_status):
            raise ValueError(f"Unresolved review error: {key}")
        tg_pass = tg_status in {"TWO_MODEL_PASS", "14B_ADJUDICATED_PASS"}
        nllb_pass = nllb_status in {"TWO_MODEL_PASS", "14B_ADJUDICATED_PASS"}
        paired[("pass" if tg_pass else "flag") + "/" + ("pass" if nllb_pass else "flag")] += 1
        if tg_pass or nllb_pass:
            if final[key].get("final_review_status") != "AI_ONLY_REVIEW_FLAGGED":
                raise ValueError(f"Sample key does not replace a final-review flag: {key}")
            for passed, candidate in ((tg_pass, tg), (nllb_pass, nllb)):
                if passed and release.norm(canonical[key[1]]["answer_en"]) in release.norm(candidate["text_translation"]):
                    raise ValueError(f"Accepted diagnostic candidate discloses its exact answer: {key}")
            provisional.add(key)

    old_rows = release.read_jsonl(OLD / "selected_candidates.jsonl")
    old_passes = set()
    old_excluded = Counter()
    for row in old_rows:
        if row.get("final_calibration_status") != "AI_ONLY_FULL_SCREEN_CALIBRATION_PASS":
            continue
        key = (row["language"], row["prompt_id"], row["kind"])
        if (row.get("selected_model") != "facebook/nllb-200-distilled-1.3B"
                or row.get("nllb_qwen_pass") is not True or row.get("backtranslation_status") != "PASS"
                or row.get("qwen14b_semantic_pass") is not True or row.get("qwen14b_fluency_pass") is not True
                or final[key].get("final_review_status") != "AI_ONLY_REVIEW_FLAGGED"):
            raise ValueError(f"Prior NLLB full-screen pass is inconsistent: {key}")
        original = canonical[row["prompt_id"]]
        review_source = review_sources[key]
        translation = row.get("selected_translation", "")
        if row.get("nllb_translation_sha256") != hashlib.sha256(translation.encode()).hexdigest():
            raise ValueError(f"Prior selected translation hash mismatch: {key}")
        # Rejoin the source name list; do not trust stored integrity results.
        checked = deterministic_checks(key[0], {"text_en": row["text_en"],
            "text_translation": translation, "protected_entity_names": review_source["protected_entity_names"]})
        if not all(checked[name] is True for name in REQUIRED):
            old_excluded["source_integrity_failure"] += 1
            continue
        back = row.get("backtranslation_en", "")
        names = [name for name in review_source["protected_entity_names"] if name in row["text_en"]]
        number = re.compile(r"\d+(?:[-/.]\d+)*")
        numbers = number.findall(row["text_en"])
        if (not back.strip() or not any(c.isascii() and c.isalpha() for c in back)
                or any(row["text_en"].count(name) != back.count(name) for name in names)
                or any(numbers.count(value) != number.findall(back).count(value) for value in set(numbers))
                or re.search(r"ZXQ[A-Z]+QXZ", back)):
            old_excluded["backtranslation_integrity_failure"] += 1
            continue
        if release.norm(original["answer_en"]) in release.norm(translation):
            old_excluded["exact_answer_in_question"] += 1
            continue
        old_passes.add(key)

    atomic_by_case = defaultdict(set)
    split_by_case = {}
    for row in release.read_jsonl(TRAINING / "atomic_reuse.jsonl"):
        if row["language"] == "en":
            atomic_by_case[row["source_qa_id"]].add(row["sft_id"])
            split_by_case[row["source_qa_id"]] = row["split"]
    composition_by_chain = defaultdict(set)
    for row in release.read_jsonl(TRAINING / "composition_reuse.jsonl"):
        if row["language"] == "en":
            composition_by_chain[row["source_qa_id"]].add(row["sft_id"])
    eval_cases = release.read_jsonl(TRAINING / "evaluation/cases.jsonl")
    flagged = {key for key, row in final.items() if row.get("final_review_status") == "AI_ONLY_REVIEW_FLAGGED"}
    if len(final) != 1952 or len(flagged) != 799 or not (provisional | old_passes) <= flagged:
        raise ValueError("The frozen review population or provisional candidate selection changed")
    bundle_keys = [(language, prompt_id, kind)
                   for kind, bundles in (("atomic", atomic_by_case), ("composition", composition_by_chain))
                   for ids in bundles.values() for prompt_id in ids for language in LANGS]
    if (len(atomic_by_case) != 156 or len(composition_by_chain) != 44
            or any(len(ids) != 2 for ids in atomic_by_case.values())
            or any(len(ids) != 4 for ids in composition_by_chain.values())
            or len(bundle_keys) != len(set(bundle_keys)) or set(bundle_keys) != set(final)):
        raise ValueError("Atomic/composition bundles do not partition the complete review population")

    def missing_distribution(bundles: dict[str, set[str]], kind: str, additions: set[tuple[str, str, str]]) -> dict:
        counts = Counter()
        unresolved = flagged - additions
        for ids in bundles.values():
            missing = sum((language, prompt_id, kind) in unresolved
                          for language in LANGS for prompt_id in ids)
            counts[str(missing)] += 1
        return dict(sorted(counts.items(), key=lambda item: int(item[0])))

    def coverage(additions: set[tuple[str, str, str]]) -> dict:
        def passed(language: str, prompt_id: str, kind: str) -> bool:
            key = (language, prompt_id, kind)
            return final[key].get("final_review_status") in release.PASS_STATUSES or key in additions
        cases = {case_id for case_id, ids in atomic_by_case.items()
                 if len(ids) == 2 and all(passed(lang, prompt_id, "atomic") for lang in LANGS for prompt_id in ids)}
        chains = {case_id for case_id, ids in composition_by_chain.items()
                  if len(ids) == 4 and all(passed(lang, prompt_id, "composition") for lang in LANGS for prompt_id in ids)}
        counts = Counter(split_by_case[case_id] for case_id in cases)
        eval_counts = {}
        for split in ("forget", "retain"):
            selected = [row for row in eval_cases if row["split"] == split and row["source_qa_id"] in cases]
            eval_counts[split] = {"cases": len(selected), "groups": len({row["group_id"] for row in selected})}
        return {"atomic_cases_by_split": {split: counts[split] for split in ("forget", "retain")},
                "complete_composition_chains": len(chains), "eval_cases_with_complete_atomic_training": eval_counts}

    result = {"status": "PROVISIONAL_COVERAGE_ONLY_NOT_RELEASE_APPROVAL",
              "frozen_review_population": {"candidate_question_language_pairs": len(final),
                  "flagged_pairs": len(flagged),
                  "flags_by_language_and_kind": {lang: {kind: sum(key[0] == lang and key[2] == kind for key in flagged)
                      for kind in ("atomic", "composition")} for lang in LANGS},
                  "atomic_cases_by_number_of_flagged_pairs": missing_distribution(atomic_by_case, "atomic", set()),
                  "composition_chains_by_number_of_flagged_pairs": missing_distribution(composition_by_chain, "composition", set())},
              "strict_original": coverage(set()),
              "matched_32_union": {"provisional_prompt_language_keys": len(provisional), **coverage(provisional)},
              "matched_plus_prior_nllb": {"prior_full_screen_keys": len(old_passes),
                  "new_unique_prompt_language_keys": len(provisional | old_passes),
                  "overlap_between_calibrations": len(provisional & old_passes),
                  "remaining_flagged_pairs_if_all_were_accepted": len(flagged - (provisional | old_passes)),
                  "atomic_cases_by_remaining_flagged_pairs": missing_distribution(atomic_by_case, "atomic", provisional | old_passes),
                  "composition_chains_by_remaining_flagged_pairs": missing_distribution(composition_by_chain, "composition", provisional | old_passes),
                  **coverage(provisional | old_passes)},
              "old_candidate_exclusions": dict(old_excluded), "matched_status_pairs_tg_nllb": dict(paired),
              "source_manifest_sha256": sources["source_manifest_sha256"],
              "final_review_report_sha256": sha(TRAINING / "translation_reviews_roundtrip_qwen14b/report.json"),
              "matched_tg_review_sha256": sha(TG_ALT / "checkpoint.json"),
              "matched_nllb_review_sha256": sha(NLLB / "checkpoint.json"),
              "prior_nllb_rows_sha256": sha(OLD / "selected_candidates.jsonl"),
              "limitations": ["Calibration outputs remain quarantined and have not passed the production release assembler.",
                  "This calculation mixes historical and alternative-mask review decisions and is an optimistic feasibility bound, not a release policy.",
                  "AI-only screening is not human fluency certification; no model competence or unlearning effect is measured."]}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
