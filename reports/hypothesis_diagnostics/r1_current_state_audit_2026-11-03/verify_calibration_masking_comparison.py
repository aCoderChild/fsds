#!/usr/bin/env python3
"""Recompute the collision-free masking comparison from saved decisions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from verify_calibration_roundtrip import main as verify_previous
from verify_source_bound_calibration import TRAINING, sha

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "evidence/source_bound_calibration_roundtrip_retry"
OUT = HERE / "evidence/source_bound_calibration_masking_comparison"


def main() -> None:
    verify_previous()
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
    prior = json.loads((PRIOR / "checkpoint.json").read_text(encoding="utf-8"))
    identity = report["identity"]
    if (report.get("status") != "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or report.get("checkpoint_sha256") != sha(OUT / "checkpoint.json")
            or checkpoint.get("identity") != identity
            or identity.get("prior_checkpoint_sha256") != sha(PRIOR / "checkpoint.json")
            or identity.get("prior_report_sha256") != sha(PRIOR / "report.json")
            or identity.get("script_sha256") != sha(HERE / "compare_calibration_masking.py")
            or identity.get("masking_script_sha256") != sha(TRAINING / "scripts/review_masking_v2.py")
            or identity.get("judge_script_sha256") != sha(TRAINING / "scripts/adjudicate_training_translations_qwen14b.py")):
        raise ValueError("Comparison provenance or checkpoint hash mismatch")
    rows, decisions = prior["records"], checkpoint["decisions"]
    if len(rows) != 32 or len(decisions) != 32:
        raise ValueError("Comparison coverage differs from 32 rows")
    counts, by_language, changes = Counter(), {lang: Counter() for lang in ("fr", "de", "hi", "th")}, []
    for row in rows:
        key = f"{row['language']}|{row['kind']}|{row['prompt_id']}"
        decision = decisions[key]
        q7 = row["qwen7_roundtrip"]
        if type(decision.get("semantic_pass")) is not bool or type(decision.get("fluency_pass")) is not bool:
            status = "BLOCKED_OR_ERROR"
        elif decision["semantic_pass"] and decision["fluency_pass"]:
            status = "TWO_MODEL_PASS" if q7["semantic"]["decision"] and q7["fluency"]["decision"] else "14B_ADJUDICATED_PASS"
        else:
            status = "FLAGGED"
        counts[status] += 1
        by_language[row["language"]][status] += 1
        if status != row["diagnostic_status"]:
            changes.append({"language": row["language"], "kind": row["kind"], "prompt_id": row["prompt_id"],
                            "previous_status": row["diagnostic_status"], "new_status": status})
    if (report.get("counts") != dict(counts)
            or report.get("by_language") != {lang: dict(by_language[lang]) for lang in by_language}
            or report.get("changed_decisions") != changes):
        raise ValueError("Comparison summary differs from saved decisions")
    print(json.dumps({"verification": "PASS_DIAGNOSTIC_ONLY", "counts": counts,
                      "by_language": by_language, "changes": changes}, indent=2))


if __name__ == "__main__":
    main()
