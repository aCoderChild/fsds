#!/usr/bin/env python3
"""Verify the completed diagnostic sample against its current source inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "experiments/synthworlds_r1"
TRAINING = PROJECT / "training_release_draft"
SCRIPT = TRAINING / "scripts/repair_all_flagged_training_translations_v2.py"
RUN = TRAINING / "translation_repair_diagnostics/source_bound_finalflagged_calibration32"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> tuple[list[dict], list[dict], dict]:
    spec = importlib.util.spec_from_file_location("source_bound_repair", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from assemble_training_release import checked_translation_reviews

    report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
    raw = (RUN / "repair_candidates.jsonl").read_bytes()
    decisions = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or report.get("script_sha256") != sha(SCRIPT)
            or report.get("decisions_sha256") != hashlib.sha256(raw).hexdigest()
            or report.get("processed_rows") != len(decisions)
            or report.get("source", {}).get("limit_per_cell") != 4):
        raise ValueError("Calibration status, script, decisions, or sample size is inconsistent")
    source_id = report["source"]
    if source_id["review_report_sha256"] != sha(TRAINING / "translation_reviews_roundtrip_qwen/report.json"):
        raise ValueError("First-stage report hash mismatch")
    if source_id["final_review_report_sha256"] != sha(TRAINING / "translation_reviews_roundtrip_qwen14b/report.json"):
        raise ValueError("Final review report hash mismatch")
    for lang in module.LANGS:
        if (source_id["review_sha256_by_language"][lang] != sha(TRAINING / f"translation_reviews_roundtrip_qwen/{lang}.jsonl")
                or source_id["final_review_sha256_by_language"][lang] != sha(TRAINING / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl")):
            raise ValueError(f"Review hash mismatch: {lang}")
    for name, digest in source_id["helper_scripts_sha256"].items():
        if sha(PROJECT / name) != digest:
            raise ValueError(f"Helper hash mismatch: {name}")

    _, rows_by_lang = module.source_manifest()
    all_flagged = module.select_flagged_rows(rows_by_lang, checked_translation_reviews())
    counts = Counter()
    selected = []
    for item in all_flagged:
        cell = (item["language"], item["row"]["kind"])
        if counts[cell] < 4:
            counts[cell] += 1
            selected.append(item)
    if len(selected) != 32 or len(decisions) != 32:
        raise ValueError("Sample coverage is not exactly 32")
    for item, decision in zip(selected, decisions):
        row, lang = item["row"], item["language"]
        if ((decision.get("language"), decision.get("kind"), decision.get("prompt_id"))
                != (lang, row["kind"], row["prompt_id"])
                or decision.get("source_en_sha256") != module.sha_text(row["text_en"])
                or decision.get("old_translation_sha256") != module.sha_text(row["text_translation"])
                or decision.get("final_review_status") != "AI_ONLY_REVIEW_FLAGGED"):
            raise ValueError("Selection order or source provenance mismatch")
        candidate = decision.get("new_translation")
        if not isinstance(candidate, str) or decision.get("new_translation_sha256") != module.sha_text(candidate):
            raise ValueError("Missing or changed candidate translation")
        checks = module.deterministic_checks(lang, {**row, "text_translation": candidate})
        if checks != decision.get("checks") or not all(checks[name] for name in module.REQUIRED_CHECKS):
            raise ValueError("Recorded candidate integrity differs from recomputed checks")
        if type(decision.get("qwen_pass")) is not bool or not decision.get("qwen_reason", "").strip():
            raise ValueError("Missing Qwen decision or explanation")
        expected_status = "DIAGNOSTIC_QWEN_PASS" if decision["qwen_pass"] else "DIAGNOSTIC_QWEN_FLAGGED"
        if decision.get("status") != expected_status:
            raise ValueError("Qwen decision and status disagree")
    if report.get("qwen_passes") != sum(d["qwen_pass"] for d in decisions):
        raise ValueError("Qwen pass count mismatch")
    return selected, decisions, report


if __name__ == "__main__":
    selected, decisions, report = verify()
    print(json.dumps({"status": "PASS_DIAGNOSTIC_ONLY", "selected": len(selected),
                      "deterministic_passes": len(decisions),
                      "qwen_passes": sum(row["qwen_pass"] for row in decisions),
                      "decisions_sha256": report["decisions_sha256"]}, indent=2))
