#!/usr/bin/env python3
"""Check the source-bound calibration review and recompute its summary."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from verify_source_bound_calibration import verify, sha, RUN

HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence/source_bound_calibration_roundtrip_retry"
PRIOR = HERE / "evidence/source_bound_calibration_roundtrip"


def main() -> None:
    selected, candidates, _ = verify()
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
    prior_report = json.loads((PRIOR / "report.json").read_text(encoding="utf-8"))
    prior = json.loads((PRIOR / "checkpoint.json").read_text(encoding="utf-8"))
    identity = report["identity"]
    if (report.get("status") != "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or checkpoint.get("identity") != identity
            or report.get("checkpoint_sha256") != sha(OUT / "checkpoint.json")
            or identity.get("script_sha256") != sha(HERE / "review_source_bound_calibration_retry.py")
            or identity.get("source_decisions_sha256") != sha(RUN / "repair_candidates.jsonl")
            or identity.get("prior_report_sha256") != sha(PRIOR / "report.json")
            or identity.get("prior_checkpoint_sha256") != sha(PRIOR / "checkpoint.json")
            or prior_report.get("checkpoint_sha256") != sha(PRIOR / "checkpoint.json")):
        raise ValueError("Review report, source, or checkpoint hash mismatch")
    rows, old = checkpoint["records"], prior["records"]
    if len(rows) != 32 or len(old) != 32:
        raise ValueError("Review row count mismatch")
    counts = Counter()
    by_lang = {lang: Counter() for lang in ("fr", "de", "hi", "th")}
    for row, previous, item, candidate in zip(rows, old, selected, candidates):
        if ((row["language"], row["kind"], row["prompt_id"]) !=
                (item["language"], item["row"]["kind"], item["row"]["prompt_id"])
                or row["text_en"] != item["row"]["text_en"]
                or row["text_translation"] != candidate["new_translation"]
                or row["protected_entity_names"] != item["row"]["protected_entity_names"]):
            raise ValueError("Review source or selected question mismatch")
        for field in ("backtranslation_en", "backtranslation_checks", "backtranslation_status", "qwen7_roundtrip"):
            if row.get(field) != previous.get(field):
                raise ValueError(f"Prior reviewed data changed: {field}")
        q7, q14 = row.get("qwen7_roundtrip", {}), row.get("qwen14", {})
        sem = q7.get("semantic", {}).get("decision")
        flu = q7.get("fluency", {}).get("decision")
        if row.get("backtranslation_status") != "PASS" or type(sem) is not bool or type(flu) is not bool or type(q14.get("semantic_pass")) is not bool or type(q14.get("fluency_pass")) is not bool:
            status = "BLOCKED_OR_ERROR"
        elif q14["semantic_pass"] and q14["fluency_pass"]:
            status = "TWO_MODEL_PASS" if sem and flu else "14B_ADJUDICATED_PASS"
        else:
            status = "FLAGGED"
        if row.get("diagnostic_status") != status:
            raise ValueError("Stored and recomputed final review decisions disagree")
        counts[status] += 1
        by_lang[row["language"]][status] += 1
    if report.get("counts") != dict(counts) or report.get("by_language") != {k: dict(v) for k, v in by_lang.items()}:
        raise ValueError("Report summary differs from decisions")
    print(json.dumps({"verification": "PASS_DIAGNOSTIC_ONLY", "counts": counts,
                      "by_language": by_lang, "checkpoint_sha256": sha(OUT / "checkpoint.json")}, indent=2))


if __name__ == "__main__":
    main()
