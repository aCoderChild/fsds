#!/usr/bin/env python3
"""Verify the small feedback-repair diagnostic and its staged review."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from verify_calibration_masking_comparison import main as verify_comparison
from verify_source_bound_calibration import PROJECT, TRAINING, sha

HERE = Path(__file__).resolve().parent
FEEDBACK = HERE / "evidence/source_bound_feedback_calibration8"
PRIOR = HERE / "evidence/source_bound_calibration_roundtrip_retry"
OUT = HERE / "evidence/source_bound_feedback_roundtrip"
sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
from review_training_translations_qwen import deterministic_checks


def main() -> None:
    verify_comparison()
    fb = json.loads((FEEDBACK / "checkpoint.json").read_text(encoding="utf-8"))
    fb_report = json.loads((FEEDBACK / "report.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    prior_rows = json.loads((PRIOR / "checkpoint.json").read_text(encoding="utf-8"))["records"]
    prior = {f"{r['language']}|{r['kind']}|{r['prompt_id']}": r for r in prior_rows}
    if (fb_report.get("checkpoint_sha256") != sha(FEEDBACK / "checkpoint.json")
            or fb_report.get("identity") != fb.get("identity")
            or fb["identity"].get("script_sha256") != sha(HERE / "calibrate_feedback_repair.py")
            or report.get("checkpoint_sha256") != sha(OUT / "checkpoint.json")
            or report.get("identity") != checkpoint.get("identity")
            or checkpoint["identity"].get("feedback_checkpoint_sha256") != sha(FEEDBACK / "checkpoint.json")
            or checkpoint["identity"].get("feedback_report_sha256") != sha(FEEDBACK / "report.json")
            or checkpoint["identity"].get("source_checkpoint_sha256") != sha(PRIOR / "checkpoint.json")
            or checkpoint["identity"].get("script_sha256") != sha(HERE / "review_feedback_calibration.py")
            or checkpoint["identity"].get("roundtrip_script_sha256") != sha(TRAINING / "scripts/review_training_translations_roundtrip.py")
            or checkpoint["identity"].get("judge_script_sha256") != sha(TRAINING / "scripts/adjudicate_training_translations_qwen14b.py")
            or checkpoint["identity"].get("mask_script_sha256") != sha(TRAINING / "scripts/review_masking_v2.py")
            or checkpoint["identity"].get("model_digests") != json.loads((PRIOR / "report.json").read_text())["identity"]["model_digests"]):
        raise ValueError("Feedback or review provenance mismatch")
    if set(fb["decisions"]) != set(fb["identity"]["selected_keys"]) or len(fb["decisions"]) != 8:
        raise ValueError("Feedback selection coverage mismatch")
    feedback_counts = Counter(row["status"] for row in fb["decisions"].values())
    if dict(feedback_counts) != fb_report.get("counts"):
        raise ValueError("Feedback status counts differ")
    expected_keys = {key for key, row in fb["decisions"].items() if row["status"] in {"PRELIMINARY_PASS", "FLAGGED"}}
    rows = checkpoint["records"]
    if {row["feedback_key"] for row in rows} != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("Feedback review row coverage mismatch")
    counts, by_language = Counter(), {lang: Counter() for lang in ("fr", "de", "hi", "th")}
    for row in rows:
        key = row["feedback_key"]
        source, candidate = prior[key], fb["decisions"][key]
        if (row["text_en"] != source["text_en"] or row["protected_entity_names"] != source["protected_entity_names"]
                or row["text_translation"] != candidate["new_translation"]
                or row["candidate_sha256"] != hashlib.sha256(row["text_translation"].encode()).hexdigest()
                or deterministic_checks(row["language"], row) != candidate["checks"]):
            raise ValueError(f"Feedback candidate source or checks differ: {key}")
        q7, q14 = row.get("qwen7_roundtrip", {}), row.get("qwen14", {})
        sem, flu = q7.get("semantic", {}).get("decision"), q7.get("fluency", {}).get("decision")
        if row.get("backtranslation_status") != "PASS" or type(sem) is not bool or type(flu) is not bool or type(q14.get("semantic_pass")) is not bool or type(q14.get("fluency_pass")) is not bool:
            status = "BLOCKED_OR_ERROR"
        elif q14["semantic_pass"] and q14["fluency_pass"]:
            status = "TWO_MODEL_PASS" if sem and flu else "14B_ADJUDICATED_PASS"
        else:
            status = "FLAGGED"
        counts[status] += 1
        by_language[row["language"]][status] += 1
    if (dict(counts) != report.get("counts")
            or {lang: dict(by_language[lang]) for lang in by_language} != report.get("by_language")
            or report.get("selected") != 8 or report.get("usable") != len(rows)
            or report.get("generation_errors") != feedback_counts.get("ERROR", 0)):
        raise ValueError("Feedback review summary differs from decisions")
    print(json.dumps({"verification": "PASS_DIAGNOSTIC_ONLY", "generation": feedback_counts,
                      "review": counts, "by_language": by_language}, indent=2))


if __name__ == "__main__":
    main()
