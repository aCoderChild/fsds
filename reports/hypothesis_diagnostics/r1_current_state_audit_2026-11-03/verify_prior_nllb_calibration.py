#!/usr/bin/env python3
"""Recheck the prior NLLB/TranslateGemma union against source entity lists."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from verify_source_bound_calibration import PROJECT, TRAINING, sha

sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
from review_training_translations_qwen import collect_rows, deterministic_checks

RUN = TRAINING / "translation_repair_diagnostics/nllb13_cal64_roundtrip_qwen14_2026-11-03"
REQUIRED = ("nonempty", "target_script_present", "protected_names_preserved", "protected_numbers_preserved", "no_unrestored_marker")


def main() -> None:
    report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
    path = RUN / "selected_candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or report.get("selected_rows_sha256") != sha(path) or len(rows) != 64):
        raise ValueError("Prior calibration status, hash or row count mismatch")
    source = {}
    for language in ("fr", "de", "hi", "th"):
        records, _ = collect_rows(language)
        for row in records:
            source[(language, row["prompt_id"], row["kind"])] = row
    counts, failures = Counter(), []
    seen = set()
    for row in rows:
        key = (row["language"], row["prompt_id"], row["kind"])
        if key in seen or key not in source:
            raise ValueError("Duplicate or unknown selected candidate")
        seen.add(key)
        original = source[key]
        if (row.get("text_en") != original["text_en"]
                or row.get("source_en_sha256") != hashlib.sha256(original["text_en"].encode()).hexdigest()):
            raise ValueError(f"Source mismatch: {key}")
        translation = row.get("selected_translation")
        if not translation:
            counts["no_candidate"] += 1
            continue
        checks = deterministic_checks(key[0], {**original, "text_translation": translation})
        passed = all(checks[name] is True for name in REQUIRED)
        counts["selected"] += 1
        counts["integrity_pass"] += passed
        counts["integrity_failure"] += not passed
        counts[f"model:{row['selected_model']}"] += 1
        counts["full_screen_pass_with_integrity"] += passed and row.get("final_calibration_status") == "AI_ONLY_FULL_SCREEN_CALIBRATION_PASS"
        if not passed:
            failures.append({"language": key[0], "prompt_id": key[1], "kind": key[2],
                             "selected_model": row["selected_model"], "missing_names": checks["missing_names"],
                             "final_calibration_status": row["final_calibration_status"]})
    print(json.dumps({"status": "PASS_RECOMPUTATION_NOT_RELEASE_APPROVAL", "counts": counts,
                      "integrity_failures": failures, "selected_rows_sha256": sha(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
