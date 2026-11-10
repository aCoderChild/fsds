#!/usr/bin/env python3
"""Verify matched NLLB calibration and compare its paired review outcomes."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from verify_source_bound_calibration import PROJECT, TRAINING, RUN, sha, verify
from verify_calibration_masking_comparison import main as verify_comparison

HERE = Path(__file__).resolve().parent
NLLB = HERE / "evidence/nllb13_matched_calibration32"
OUT = HERE / "evidence/nllb13_matched_roundtrip"
TG = HERE / "evidence/source_bound_calibration_roundtrip_retry"
COMPARE = HERE / "evidence/source_bound_calibration_masking_comparison"
sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
from review_training_translations_qwen import deterministic_checks


def status(row: dict, q14: dict) -> str:
    q7 = row.get("qwen7_roundtrip", {})
    sem, flu = q7.get("semantic", {}).get("decision"), q7.get("fluency", {}).get("decision")
    if row.get("backtranslation_status") != "PASS" or type(sem) is not bool or type(flu) is not bool or type(q14.get("semantic_pass")) is not bool or type(q14.get("fluency_pass")) is not bool:
        return "BLOCKED_OR_ERROR"
    if q14["semantic_pass"] and q14["fluency_pass"]:
        return "TWO_MODEL_PASS" if sem and flu else "14B_ADJUDICATED_PASS"
    return "FLAGGED"


def main() -> None:
    verify_comparison()
    selected, _, source_report = verify()
    nllb_report = json.loads((NLLB / "report.json").read_text(encoding="utf-8"))
    candidates = [json.loads(line) for line in (NLLB / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    tg_rows = json.loads((TG / "checkpoint.json").read_text(encoding="utf-8"))["records"]
    alt = json.loads((COMPARE / "checkpoint.json").read_text(encoding="utf-8"))["decisions"]
    identity = report["identity"]
    if (nllb_report.get("candidates_sha256") != sha(NLLB / "candidates.jsonl")
            or nllb_report.get("generator_script_sha256") != sha(HERE / "calibrate_nllb_matched32.py")
            or nllb_report.get("nllb_helper_sha256") != sha(TRAINING / "scripts/calibrate_nllb13_flagged_translations.py")
            or nllb_report.get("source_calibration_report_sha256") != sha(RUN / "report.json")
            or nllb_report.get("source_calibration_decisions_sha256") != source_report["decisions_sha256"]
            or report.get("checkpoint_sha256") != sha(OUT / "checkpoint.json")
            or identity != checkpoint.get("identity")
            or identity.get("nllb_candidates_sha256") != sha(NLLB / "candidates.jsonl")
            or identity.get("nllb_report_sha256") != sha(NLLB / "report.json")
            or identity.get("script_sha256") != sha(HERE / "review_nllb_matched32.py")
            or identity.get("roundtrip_script_sha256") != sha(TRAINING / "scripts/review_training_translations_roundtrip.py")
            or identity.get("judge_script_sha256") != sha(TRAINING / "scripts/adjudicate_training_translations_qwen14b.py")
            or identity.get("mask_script_sha256") != sha(TRAINING / "scripts/review_masking_v2.py")
            or identity.get("model_digests") != json.loads((TG / "report.json").read_text())["identity"]["model_digests"]):
        raise ValueError("NLLB sample or review provenance mismatch")
    rows = checkpoint["records"]
    if any(len(x) != 32 for x in (rows, candidates, selected, tg_rows)):
        raise ValueError("Matched sample does not have 32 rows")
    counts, paired = Counter(), Counter()
    by_language = {lang: Counter() for lang in ("fr", "de", "hi", "th")}
    required = ("nonempty", "target_script_present", "protected_names_preserved", "protected_numbers_preserved", "no_unrestored_marker")
    for row, candidate, item, tg in zip(rows, candidates, selected, tg_rows):
        source = item["row"]
        key = (item["language"], source["kind"], source["prompt_id"])
        if ((row["language"], row["kind"], row["prompt_id"]) != key
                or (candidate["language"], candidate["kind"], candidate["prompt_id"]) != key
                or (tg["language"], tg["kind"], tg["prompt_id"]) != key
                or row["text_en"] != source["text_en"]
                or row["protected_entity_names"] != source["protected_entity_names"]
                or row["text_translation"] != candidate["text_translation"]
                or candidate["translation_sha256"] != hashlib.sha256(row["text_translation"].encode()).hexdigest()):
            raise ValueError(f"Matched source or candidate differs: {key}")
        checks = deterministic_checks(row["language"], row)
        if checks != candidate.get("deterministic_checks") or not all(checks[name] is True for name in required):
            raise ValueError(f"NLLB integrity check differs: {key}")
        if candidate.get("source_en_sha256") != hashlib.sha256(source["text_en"].encode()).hexdigest():
            raise ValueError(f"NLLB source hash differs: {key}")
        if row.get("backtranslation_status") == "PASS":
            back = row.get("backtranslation_en", "")
            names = [name for name in source["protected_entity_names"] if name in source["text_en"]]
            number = re.compile(r"\d+(?:[-/.]\d+)*")
            source_numbers = number.findall(source["text_en"])
            if (not back.strip() or any(source["text_en"].count(name) != back.count(name) for name in names)
                    or any(source_numbers.count(n) != number.findall(back).count(n) for n in set(source_numbers))
                    or re.search(r"ZXQ[A-Z]+QXZ", back)):
                raise ValueError(f"NLLB back-translation integrity differs: {key}")
        nllb_status = status(row, row.get("qwen14", {}))
        tg_status = status(tg, alt[f"{key[0]}|{key[1]}|{key[2]}"])
        counts[nllb_status] += 1
        by_language[key[0]][nllb_status] += 1
        if "BLOCKED_OR_ERROR" in (nllb_status, tg_status):
            paired["blocked"] += 1
        else:
            n_pass = nllb_status in {"TWO_MODEL_PASS", "14B_ADJUDICATED_PASS"}
            t_pass = tg_status in {"TWO_MODEL_PASS", "14B_ADJUDICATED_PASS"}
            paired[{(True, True): "both_pass", (True, False): "nllb_only", (False, True): "translategemma_only", (False, False): "neither_pass"}[(n_pass, t_pass)]] += 1
    if report.get("counts") != dict(counts) or report.get("by_language") != {lang: dict(by_language[lang]) for lang in by_language}:
        raise ValueError("NLLB review summary differs from saved decisions")
    print(json.dumps({"verification": "PASS_DIAGNOSTIC_ONLY", "counts": counts,
                      "by_language": by_language, "paired_outcomes": paired}, indent=2))


if __name__ == "__main__":
    main()
