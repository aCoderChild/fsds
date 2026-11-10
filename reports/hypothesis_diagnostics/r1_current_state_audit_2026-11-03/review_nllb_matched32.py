#!/usr/bin/env python3
"""Review matched NLLB translations with source-bound round trips."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from verify_source_bound_calibration import PROJECT, TRAINING, sha, verify, RUN
from verify_calibration_masking_comparison import main as verify_comparison

HERE = Path(__file__).resolve().parent
NLLB = HERE / "evidence/nllb13_matched_calibration32"
PRIOR = HERE / "evidence/source_bound_calibration_roundtrip_retry"
OUT = HERE / "evidence/nllb13_matched_roundtrip"
sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
import review_training_translations_roundtrip as rt
import adjudicate_training_translations_qwen14b as judge
from review_masking_v2 import mask_triplet
from review_training_translations_qwen import deterministic_checks, masked_pair
from repair_flagged_eval_translations_translategemma import protect, restore


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("backtranslate", "qwen7", "qwen14", "summary"), required=True)
    args = parser.parse_args()
    verify_comparison()
    selected, _, original_report = verify()
    nllb_report = json.loads((NLLB / "report.json").read_text(encoding="utf-8"))
    candidates = [json.loads(line) for line in (NLLB / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if (nllb_report.get("status") != "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or nllb_report.get("candidates_sha256") != sha(NLLB / "candidates.jsonl")
            or nllb_report.get("generator_script_sha256") != sha(HERE / "calibrate_nllb_matched32.py")
            or nllb_report.get("nllb_helper_sha256") != sha(TRAINING / "scripts/calibrate_nllb13_flagged_translations.py")
            or nllb_report.get("selection_verifier_sha256") != sha(HERE / "verify_source_bound_calibration.py")
            or nllb_report.get("source_calibration_report_sha256") != sha(RUN / "report.json")
            or nllb_report.get("source_calibration_decisions_sha256") != original_report["decisions_sha256"]
            or len(candidates) != 32):
        raise ValueError("NLLB candidates or provenance mismatch")
    identity = {"nllb_candidates_sha256": sha(NLLB / "candidates.jsonl"),
                "nllb_report_sha256": sha(NLLB / "report.json"),
                "source_calibration_report_sha256": sha(RUN / "report.json"),
                "script_sha256": sha(Path(__file__)),
                "roundtrip_script_sha256": sha(Path(rt.__file__)),
                "judge_script_sha256": sha(Path(judge.__file__)),
                "mask_script_sha256": sha(TRAINING / "scripts/review_masking_v2.py"),
                "model_digests": json.loads((PRIOR / "report.json").read_text())["identity"]["model_digests"]}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "checkpoint.json"
    if path.exists():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint["identity"] != identity:
            raise ValueError("NLLB review inputs changed; refusing resume")
    else:
        records = []
        for item, result in zip(selected, candidates):
            source = item["row"]
            if ((item["language"], source["prompt_id"], source["kind"]) != (result["language"], result["prompt_id"], result["kind"])
                    or result.get("source_en_sha256") != hashlib.sha256(source["text_en"].encode()).hexdigest()
                    or result.get("translation_error") is not None):
                raise ValueError("NLLB candidate does not match source")
            row = {**source, "language": item["language"], "text_translation": result["text_translation"],
                   "candidate_sha256": result["translation_sha256"]}
            checks = deterministic_checks(row["language"], row)
            if checks != result.get("deterministic_checks") or not all(checks[name] for name in ("nonempty", "target_script_present", "protected_names_preserved", "protected_numbers_preserved", "no_unrestored_marker")):
                raise ValueError(f"NLLB candidate integrity mismatch: {row['prompt_id']}")
            records.append(row)
        checkpoint = {"identity": identity, "records": records}
        write(path, checkpoint)
    records = checkpoint["records"]
    if len(records) != 32:
        raise ValueError("Expected 32 usable NLLB translations")
    for row, item, candidate in zip(records, selected, candidates):
        source = item["row"]
        if (row["text_en"] != source["text_en"] or row["protected_entity_names"] != source["protected_entity_names"]
                or row["text_translation"] != candidate["text_translation"]
                or row["candidate_sha256"] != hashlib.sha256(row["text_translation"].encode()).hexdigest()):
            raise ValueError("NLLB review source or candidate text changed")
    model_for_stage = {"backtranslate": "translategemma:4b", "qwen7": "qwen2.5:7b-instruct", "qwen14": "qwen2.5:14b"}
    if args.stage in model_for_stage:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
            models = {item["name"]: item["digest"] for item in json.load(response)["models"]}
        model = model_for_stage[args.stage]
        if models.get(model) != identity["model_digests"][model]:
            raise ValueError(f"Model digest changed: {model}")
    for lang, kind in ((lang, kind) for lang in ("fr", "de", "hi", "th") for kind in ("atomic", "composition")):
        batch = [r for r in records if r["language"] == lang and r["kind"] == kind]
        if args.stage == "backtranslate":
            pending = [r for r in batch if "backtranslation_status" not in r]
            if not pending:
                continue
            prepared = []
            for row in pending:
                protected, mapping, counts = protect(row["text_translation"], row["protected_entity_names"])
                prepared.append({"protected_text": protected, "mapping": mapping, "counts": counts})
            try:
                translated = rt.translate_batch(lang, prepared, 240)
                for i, (row, source) in enumerate(zip(pending, prepared)):
                    try:
                        back = restore(translated[f"ROW_{rt.alpha_label(i)}"], source["mapping"], source["counts"])
                        names = [n for n in row["protected_entity_names"] if n in row["text_en"]]
                        nums = rt.NUMBER.findall(row["text_en"])
                        checks = {"nonempty": bool(back.strip()),
                                  "english_latin_script_present": any("A" <= c <= "Z" or "a" <= c <= "z" for c in back),
                                  "names": all(row["text_en"].count(n) == back.count(n) for n in names),
                                  "numbers": all(nums.count(n) == rt.NUMBER.findall(back).count(n) for n in set(nums)),
                                  "no_marker": not rt.MARKER.search(back)}
                        row.update({"backtranslation_en": back, "backtranslation_checks": checks,
                                    "backtranslation_status": "PASS" if all(checks.values()) else "BLOCKED"})
                    except Exception as exc:
                        row.update({"backtranslation_status": "BLOCKED", "error": type(exc).__name__})
            except Exception as exc:
                for row in pending:
                    row.update({"backtranslation_status": "BLOCKED", "error": type(exc).__name__})
        elif args.stage == "qwen7":
            pending = [r for r in batch if r.get("backtranslation_status") == "PASS" and "qwen7_roundtrip" not in r]
            if not pending:
                continue
            semantic, fluency = [], []
            for row in pending:
                pair = masked_pair({**row, "text_translation": row["backtranslation_en"]})[1]
                semantic.append({"source_english": pair["source_english"], "backtranslation_english": pair["target_translation"]})
                fluency.append({"target_question": masked_pair(row)[1]["target_translation"]})
            sem = rt.ask_qwen("semantic", lang, semantic, 240)
            flu = rt.ask_qwen("fluency", lang, fluency, 240)
            for row, a, b in zip(pending, sem, flu):
                row["qwen7_roundtrip"] = {"semantic": a, "fluency": b}
        elif args.stage == "qwen14":
            pending = [r for r in batch if r.get("backtranslation_status") == "PASS" and "qwen14" not in r]
            if not pending:
                continue
            if any("qwen7_roundtrip" not in row for row in pending):
                raise ValueError("Run the 7B stage first")
            judge.mask_triplet = mask_triplet
            try:
                verdicts = judge.judge_batch(lang, pending, 240)
                for row, result in zip(pending, verdicts):
                    row["qwen14"] = result
            except Exception as exc:
                for row in pending:
                    row["qwen14"] = {"error": type(exc).__name__, "reason": str(exc)[:300]}
        write(path, checkpoint)
        print(f"{args.stage}: {lang}/{kind} complete", flush=True)
    if args.stage == "summary":
        counts = Counter()
        by_lang = {lang: Counter() for lang in ("fr", "de", "hi", "th")}
        for row in records:
            q7, q14 = row.get("qwen7_roundtrip", {}), row.get("qwen14", {})
            s, f = q7.get("semantic", {}).get("decision"), q7.get("fluency", {}).get("decision")
            if row.get("backtranslation_status") != "PASS" or type(s) is not bool or type(f) is not bool or type(q14.get("semantic_pass")) is not bool or type(q14.get("fluency_pass")) is not bool:
                status = "BLOCKED_OR_ERROR"
            elif q14["semantic_pass"] and q14["fluency_pass"]:
                status = "TWO_MODEL_PASS" if s and f else "14B_ADJUDICATED_PASS"
            else:
                status = "FLAGGED"
            counts[status] += 1
            by_lang[row["language"]][status] += 1
        report = {"status": "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW", "identity": identity,
                  "checkpoint_sha256": sha(path), "selected": 32, "usable": len(records),
                  "generation_errors": 0, "counts": dict(counts),
                  "by_language": {lang: dict(by_lang[lang]) for lang in by_lang},
                  "limitations": ["Small fixed matched sample; not a corpus accuracy estimate.",
                                  "The Qwen7 and Qwen14 reviewers share a model family.",
                                  "AI-only review; no human fluency certification or release approval."]}
        write(OUT / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
