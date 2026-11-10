#!/usr/bin/env python3
"""Verify and summarize the v5.1 blinded AI-only translation screen."""
from __future__ import annotations

import collections
import argparse
import hashlib
import json
from pathlib import Path

from draft_eval_translations_nllb import LANG_CODES, read_jsonl, sha256
from review_eval_translations_qwen import blinded_review_item


ROOT = Path(__file__).resolve().parents[1]
PROBES = ("direct_positive", "direct_negative_control", "multi_hop")
SPLITS = ("forget", "retain", "holdout")
PASS = "AI_ONLY_REVIEW_PASS"
FLAG = "AI_ONLY_REVIEW_FLAGGED"
TRANSLATION_FAILED = "TRANSLATION_DRAFT_FAILED"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--draft-dir", type=Path, default=Path("translation_drafts/merged_v1"))
    parser.add_argument("--review-dir", type=Path, default=Path("translation_reviews/qwen_blinded_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    draft_root = args.draft_dir if args.draft_dir.is_absolute() else root / args.draft_dir
    review_root = args.review_dir if args.review_dir.is_absolute() else root / args.review_dir
    request_path = root / "translation_requests/en.draft.jsonl"
    requests = read_jsonl(request_path)
    request_by_id = {row["prompt_id"]: row for row in requests}
    check(len(requests) == 585 and len(request_by_id) == 585, "Expected 585 unique English prompts")

    report_path = review_root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    check(report.get("source_requests_sha256") == sha256(request_path), "Request source hash differs from review manifest")
    check(report.get("duplicate_pair_decision_conflicts") == 0, "Review manifest reports duplicate-decision conflicts")
    check(report.get("unresolved_total") == 0, "AI reviewer has unresolved call rows")

    drafts_by_lang: dict[str, dict] = {}
    reviews_by_lang: dict[str, dict] = {}
    hashes: dict[str, dict] = {}
    language_counts = {}
    total_pass = total_flag = total_failed_translation = total_reviewed = 0
    pair_decisions: dict[tuple[str, str], bool] = {}
    pair_conflicts = []

    for lang in LANG_CODES:
        draft_path = draft_root / f"{lang}.jsonl"
        review_path = review_root / f"{lang}.jsonl"
        drafts = read_jsonl(draft_path)
        reviews = read_jsonl(review_path)
        draft_by_id = {row["prompt_id"]: row for row in drafts}
        review_by_id = {row["prompt_id"]: row for row in reviews}
        check(len(drafts) == 585 and len(draft_by_id) == 585, f"{lang}: draft coverage is not 585 unique rows")
        check(len(reviews) == 585 and len(review_by_id) == 585, f"{lang}: review coverage is not 585 unique rows")
        check(set(draft_by_id) == set(request_by_id) == set(review_by_id), f"{lang}: request/draft/review IDs differ")
        check(sha256(draft_path) == report["languages"][lang]["translation_sha256"], f"{lang}: translation hash mismatch")
        check(sha256(review_path) == report["languages"][lang]["review_sha256"], f"{lang}: review hash mismatch")

        for prompt_id, draft in draft_by_id.items():
            request = request_by_id[prompt_id]
            check(draft.get("language") == lang, f"{lang}/{prompt_id}: wrong language label")
            for key in ("source_qa_id", "group_id", "split", "probe", "text_en", "protected_entity_names"):
                check(draft.get(key) == request.get(key), f"{lang}/{prompt_id}: source provenance mismatch in {key}")
            review = review_by_id[prompt_id]
            if draft.get("translation_status") == "MACHINE_DRAFT":
                check(review.get("review_status") in {PASS, FLAG}, f"{lang}/{prompt_id}: machine draft lacks a reviewer verdict")
                key, _ = blinded_review_item(draft)
                digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
                check(review.get("review_pair_sha256") == digest, f"{lang}/{prompt_id}: masked-pair hash mismatch")
                verdict = review.get("independent_model_checks", {}).get("blinded_translation_model_pass")
                check(isinstance(verdict, bool), f"{lang}/{prompt_id}: reviewer verdict is not boolean")
                pair_key = (lang, digest)
                if pair_key in pair_decisions and pair_decisions[pair_key] != verdict:
                    pair_conflicts.append({"language": lang, "review_pair_sha256": digest})
                pair_decisions[pair_key] = verdict
            else:
                check(draft.get("translation_status") == "FAILED", f"{lang}/{prompt_id}: unsupported translation status")
                check(review.get("review_status") == TRANSLATION_FAILED, f"{lang}/{prompt_id}: failed translation not blocked in review")

        statuses = collections.Counter(row["review_status"] for row in reviews)
        passed = statuses[PASS]
        flagged = statuses[FLAG]
        failed = statuses[TRANSLATION_FAILED]
        reviewed = passed + flagged
        check(statuses.total() == 585, f"{lang}: review status total mismatch")
        manifest_lang = report["languages"][lang]
        check(manifest_lang["pass_rows"] == passed, f"{lang}: pass count differs from report")
        check(manifest_lang["flagged_rows"] == flagged, f"{lang}: flagged count differs from report")
        check(manifest_lang["translation_failed_rows"] == failed, f"{lang}: failed count differs from report")
        language_counts[lang] = {
            "input_rows": 585,
            "reviewed_rows": reviewed,
            "pass_rows": passed,
            "flagged_rows": flagged,
            "translation_failed_rows": failed,
            "pass_rate_all_rows": passed / 585,
            "cross_model_review_rows": manifest_lang["cross_model_review_rows"],
            "same_model_self_check_rows": manifest_lang["same_model_self_check_rows"],
            "translation_sha256": sha256(draft_path),
            "review_sha256": sha256(review_path),
        }
        drafts_by_lang[lang] = draft_by_id
        reviews_by_lang[lang] = review_by_id
        hashes[lang] = {"translation_sha256": sha256(draft_path), "review_sha256": sha256(review_path)}
        total_pass += passed
        total_flag += flagged
        total_failed_translation += failed
        total_reviewed += reviewed

    check(not pair_conflicts, f"Duplicate masked-pair verdict conflicts: {pair_conflicts[:3]}")
    check(total_pass == report["ai_review_pass_total"], "Overall pass count differs from report")
    check(total_flag == report["ai_review_flagged_total"], "Overall flag count differs from report")
    check(total_failed_translation == report["translation_failed_total"], "Overall translation failure count differs from report")
    check(total_pass + total_flag == total_reviewed, "Reviewed row accounting mismatch")

    probe_counts = {}
    for probe in PROBES:
        counts = {"input_rows": 0, "pass_rows": 0, "flagged_rows": 0, "translation_failed_rows": 0}
        for lang in LANG_CODES:
            for row in reviews_by_lang[lang].values():
                if row["probe"] != probe:
                    continue
                counts["input_rows"] += 1
                status = row["review_status"]
                if status == PASS:
                    counts["pass_rows"] += 1
                elif status == FLAG:
                    counts["flagged_rows"] += 1
                elif status == TRANSLATION_FAILED:
                    counts["translation_failed_rows"] += 1
        counts["pass_rate"] = counts["pass_rows"] / counts["input_rows"]
        probe_counts[probe] = counts

    case_ids = sorted({row["source_qa_id"] for row in requests})
    case_meta = {row["source_qa_id"]: row for row in requests if row["probe"] == "multi_hop"}
    all_language_pass_cases: dict[str, set[str]] = {}
    for probe in PROBES:
        all_language_pass_cases[probe] = {
            case_id for case_id in case_ids
            if all(reviews_by_lang[lang].get(f"{case_id}:{probe}", {}).get("review_status") == PASS for lang in LANG_CODES)
        }

    joint_primary = all_language_pass_cases["direct_positive"] & all_language_pass_cases["multi_hop"]
    joint_all_probes = set.intersection(*(all_language_pass_cases[probe] for probe in PROBES))
    intersection_counts = {}
    for probe, cases in all_language_pass_cases.items():
        intersection_counts[probe] = {
            "cases_all_four_languages": len(cases),
            "by_split": {
                split: {
                    "cases": sum(case_meta[case]["split"] == split for case in cases),
                    "entity_groups": len({case_meta[case]["group_id"] for case in cases if case_meta[case]["split"] == split}),
                }
                for split in SPLITS
            },
        }

    joint_by_split = {
        split: {
            "cases": sum(case_meta[case]["split"] == split for case in joint_primary),
            "entity_groups": len({case_meta[case]["group_id"] for case in joint_primary if case_meta[case]["split"] == split}),
        }
        for split in SPLITS
    }
    all_probe_by_split = {
        split: {
            "cases": sum(case_meta[case]["split"] == split for case in joint_all_probes),
            "entity_groups": len({case_meta[case]["group_id"] for case in joint_all_probes if case_meta[case]["split"] == split}),
        }
        for split in SPLITS
    }

    result = {
        "status": report["status"],
        "reviewer_model": report["reviewer_model"],
        "translation_models": report["translation_models"],
        "translation_reviewed_by_human": False,
        "translation_requests_sha256": sha256(request_path),
        "review_report_sha256": sha256(report_path),
        "prompt_rows_total": len(requests) * len(LANG_CODES),
        "ai_reviewed_rows": total_reviewed,
        "ai_review_pass_rows": total_pass,
        "ai_review_flagged_rows": total_flag,
        "translation_failed_rows": total_failed_translation,
        "unresolved_reviewer_rows": report["unresolved_total"],
        "unique_masked_pair_payloads_reviewed": report["unique_masked_pair_payloads_reviewed"],
        "duplicate_pair_decision_conflicts": len(pair_conflicts),
        "by_language": language_counts,
        "by_probe": probe_counts,
        "all_four_language_pass_intersections": intersection_counts,
        "direct_positive_and_multi_hop_all_four_by_split": joint_by_split,
        "all_three_probes_all_four_by_split": all_probe_by_split,
    }

    lines = [
        "# v5.1 evaluation translation audit",
        "",
        "## Status",
        "",
        f"This is a blinded AI-only translation screen, not fluent-human certification. Draft models: {', '.join(report['translation_models'])}. Reviewer: {report['reviewer_model']}. Rows translated and screened by the same model count as self-checks, not independent-model review.",
        "",
        f"Audit status: **{result['status']}**. Coverage: {result['ai_reviewed_rows']:,} AI-reviewed rows, {result['translation_failed_rows']} translation failures, {result['unresolved_reviewer_rows']} unresolved reviewer rows, and {result['duplicate_pair_decision_conflicts']} duplicate-verdict conflicts.",
        "",
        "## Results by language",
        "",
        "| Language | Rows | AI reviewed | Pass | Flagged | Translation failed | Pass rate / all rows | Cross-model | Same-model self-check |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lang in LANG_CODES:
        row = language_counts[lang]
        lines.append(
            f"| {lang.upper()} | {row['input_rows']} | {row['reviewed_rows']} | {row['pass_rows']} | {row['flagged_rows']} | {row['translation_failed_rows']} | {row['pass_rate_all_rows']:.1%} | {row['cross_model_review_rows']} | {row['same_model_self_check_rows']} |"
        )
    lines.extend([
        f"| **Total** | **{result['prompt_rows_total']}** | **{total_reviewed}** | **{total_pass}** | **{total_flag}** | **{total_failed_translation}** | **{total_pass / result['prompt_rows_total']:.1%}** | **{sum(v['cross_model_review_rows'] for v in language_counts.values())}** | **{sum(v['same_model_self_check_rows'] for v in language_counts.values())}** |",
        "",
        "Rows marked as translation failures are not included in AI-reviewed counts. They remain blocked and are not interpreted as review passes or flags.",
        "",
        "## Results by probe",
        "",
        "| Probe | Rows | AI pass | AI flagged | Translation failed | Pass rate / all rows |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    probe_titles = {
        "direct_positive": "Direct positive",
        "direct_negative_control": "Direct negative control",
        "multi_hop": "Multi-hop",
    }
    for probe in PROBES:
        row = probe_counts[probe]
        lines.append(f"| {probe_titles[probe]} | {row['input_rows']} | {row['pass_rows']} | {row['flagged_rows']} | {row['translation_failed_rows']} | {row['pass_rate']:.1%} |")
    lines.extend([
        "",
        "## All-four-language case intersections",
        "",
        "A case counts only if every translation for that probe passes in French, German, Hindi, and Thai. Counts are prompt-screen results, not model competence results.",
        "",
        "| Probe | All-four cases | Forget cases / groups | Retain cases / groups | Holdout cases / groups |",
        "|---|---:|---:|---:|---:|",
    ])
    for probe in PROBES:
        row = intersection_counts[probe]
        cells = [f"{row['by_split'][split]['cases']} / {row['by_split'][split]['entity_groups']}" for split in SPLITS]
        lines.append(f"| {probe_titles[probe]} | {row['cases_all_four_languages']} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend([
        "",
        "### Primary-probe joint eligibility",
        "",
        "| Joint screen | Forget cases / groups | Retain cases / groups | Holdout cases / groups |",
        "|---|---:|---:|---:|",
        f"| Direct positive + multi-hop pass in all four target languages | {joint_by_split['forget']['cases']} / {joint_by_split['forget']['entity_groups']} | {joint_by_split['retain']['cases']} / {joint_by_split['retain']['entity_groups']} | {joint_by_split['holdout']['cases']} / {joint_by_split['holdout']['entity_groups']} |",
        f"| All three probes pass in all four target languages | {all_probe_by_split['forget']['cases']} / {all_probe_by_split['forget']['entity_groups']} | {all_probe_by_split['retain']['cases']} / {all_probe_by_split['retain']['entity_groups']} | {all_probe_by_split['holdout']['cases']} / {all_probe_by_split['holdout']['entity_groups']} |",
        "",
        "The forget-set primary-probe intersection is below the planned 15-case/8-group floor. The translations are therefore not ready to establish the pre-unlearning competence cohort. No SFT or unlearning was started from this v5.1 draft.",
        "",
        "## Integrity and limits",
        "",
        f"- Request SHA-256: `{result['translation_requests_sha256']}`",
        f"- Review report SHA-256: `{result['review_report_sha256']}`",
        "- Each language draft and review file is SHA-256-bound in `report.json`; all prompt IDs and source metadata matched the English request file.",
        f"- The screen reviewed {report['unique_masked_pair_payloads_reviewed']:,} unique masked source/translation payloads; identical payloads had no conflicting verdicts.",
        "- AI judgments may miss semantic or pragmatic translation errors. No fluent-human review was performed.",
        "- The source question wording itself can be awkward; a screen pass is not a guarantee that a prompt is scientifically well-posed.",
        "- Failed placeholder recovery and AI flags require repair or retranslation before training/evaluation approval.",
        "",
        "## Reproduction",
        "",
        f"Run `python experiments/synthworlds_r1/scripts/summarize_translation_audit.py --draft-dir {draft_root} --review-dir {review_root}` from the repository root. The script verifies request, translation, review, provenance, coverage, and masked-pair hashes before writing this report and `summary.json`.",
        "",
    ])
    (review_root / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    (review_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "prompt_rows_total": result["prompt_rows_total"], "ai_review_pass_rows": total_pass, "ai_review_flagged_rows": total_flag, "translation_failed_rows": total_failed_translation, "primary_forget_cases_and_groups": joint_by_split["forget"], "diagnostic_report": str(review_root / "diagnostic_report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
