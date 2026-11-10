#!/usr/bin/env python3
"""Reconcile current final training-translation flags into reviewer/integrity gates."""
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TRAINING = REPO / "experiments/synthworlds_r1/training_release_draft"
REVIEWS = TRAINING / "translation_reviews_roundtrip_qwen14b"
OUT = Path(__file__).resolve().parent / "training_translation_flag_reason_audit.json"
LANGS = ("fr", "de", "hi", "th")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    report = json.loads((REVIEWS / "report.json").read_text(encoding="utf-8"))
    output = {
        "status": "VERIFIED_FLAG_REASON_COUNTS_ONLY_NOT_TRANSLATION_TRUTH",
        "source_report_sha256": sha(REVIEWS / "report.json"),
        "languages": {},
        "limitations": [
            "Reviewer pass/fail labels are not gold translation judgments.",
            "Flag-reason counts overlap; do not sum them as mutually exclusive rows.",
            "A missing back-translation is an integrity/review blocker, while a semantic or fluency rejection may be a false positive or a true translation defect.",
        ],
    }
    all_rows = []
    for lang in LANGS:
        path = REVIEWS / f"{lang}.jsonl"
        rows = read_jsonl(path)
        expected = report["languages"][lang]
        if len(rows) != 488 or sha(path) != expected["adjudication_sha256"]:
            raise ValueError(f"{lang}: source review count/hash does not match report")
        keys = [(r.get("prompt_id"), r.get("kind")) for r in rows]
        if len(set(keys)) != 488 or any(not k[0] or not k[1] for k in keys):
            raise ValueError(f"{lang}: duplicate or empty source keys")
        flags = [r for r in rows if r.get("final_review_status") == "AI_ONLY_REVIEW_FLAGGED"]
        statuses = Counter(r.get("final_review_status") for r in rows)
        target_deterministic_failures = [r for r in flags if not r.get("target_deterministic_checks")
                                         or not all(r["target_deterministic_checks"].values())]
        backtranslation_failures = [r for r in flags if not r.get("backtranslation_checks")
                                    or not all(r["backtranslation_checks"].values())]
        semantic_failures = [r for r in flags if r.get("qwen14b_semantic_pass") is False]
        fluency_failures = [r for r in flags if r.get("qwen14b_fluency_pass") is False]
        both_model_fields_pass = [r for r in flags if r.get("qwen14b_semantic_pass") is True
                                  and r.get("qwen14b_fluency_pass") is True]
        output["languages"][lang] = {
            "review_file_sha256": sha(path),
            "rows": len(rows),
            "final_status_counts": dict(statuses),
            "flagged_rows": len(flags),
            "qwen14_semantic_rejection_among_flags": len(semantic_failures),
            "qwen14_fluency_rejection_among_flags": len(fluency_failures),
            "missing_or_failed_backtranslation_checks_among_flags": len(backtranslation_failures),
            "target_deterministic_integrity_failures_among_flags": len(target_deterministic_failures),
            "qwen14_both_fields_pass_but_row_flagged": [
                {"prompt_id": r["prompt_id"], "kind": r["kind"], "backtranslation_checks": r.get("backtranslation_checks")}
                for r in both_model_fields_pass
            ],
        }
        all_rows.extend(rows)
    flags = [r for r in all_rows if r.get("final_review_status") == "AI_ONLY_REVIEW_FLAGGED"]
    final_statuses = Counter(r.get("final_review_status") for r in all_rows)
    output["totals"] = {
        "rows": len(all_rows),
        "unique_language_prompt_keys": len({(r.get("language"), r.get("prompt_id"), r.get("kind")) for r in all_rows}),
        "final_status_counts": dict(final_statuses),
        "flagged_rows": len(flags),
        "qwen14_semantic_rejections_among_flags": sum(r.get("qwen14b_semantic_pass") is False for r in flags),
        "qwen14_fluency_rejections_among_flags": sum(r.get("qwen14b_fluency_pass") is False for r in flags),
        "missing_or_failed_backtranslation_checks_among_flags": sum(
            not r.get("backtranslation_checks") or not all(r["backtranslation_checks"].values()) for r in flags),
        "target_deterministic_integrity_failures_among_flags": sum(
            not r.get("target_deterministic_checks") or not all(r["target_deterministic_checks"].values()) for r in flags),
        "qwen14_both_fields_pass_but_row_flagged": sum(
            r.get("qwen14b_semantic_pass") is True and r.get("qwen14b_fluency_pass") is True for r in flags),
    }
    if (output["totals"]["rows"] != 1952
            or output["totals"]["unique_language_prompt_keys"] != 1952
            or output["totals"]["flagged_rows"] != 799
            or output["totals"]["target_deterministic_integrity_failures_among_flags"] != 0):
        raise ValueError("Flag coverage or deterministic integrity totals differ from the audited release state")
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(OUT), "totals": output["totals"]}, indent=2))


if __name__ == "__main__":
    main()
