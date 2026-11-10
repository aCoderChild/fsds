#!/usr/bin/env python3
"""Review usable feedback-repair samples with source-bound round trips."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from verify_source_bound_calibration import PROJECT, TRAINING, sha
from verify_calibration_masking_comparison import main as verify_comparison

HERE = Path(__file__).resolve().parent
FEEDBACK = HERE / "evidence/source_bound_feedback_calibration8"
PRIOR = HERE / "evidence/source_bound_calibration_roundtrip_retry"
OUT = HERE / "evidence/source_bound_feedback_roundtrip"
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
    prior = json.loads((PRIOR / "checkpoint.json").read_text(encoding="utf-8"))["records"]
    feedback = json.loads((FEEDBACK / "checkpoint.json").read_text(encoding="utf-8"))
    feedback_report = json.loads((FEEDBACK / "report.json").read_text(encoding="utf-8"))
    if (feedback_report.get("status") != "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or feedback_report.get("checkpoint_sha256") != sha(FEEDBACK / "checkpoint.json")
            or feedback_report.get("identity") != feedback.get("identity")
            or feedback["identity"].get("script_sha256") != sha(HERE / "calibrate_feedback_repair.py")
            or len(feedback["decisions"]) != 8):
        raise ValueError("Feedback checkpoint or provenance mismatch")
    by_key = {f"{r['language']}|{r['kind']}|{r['prompt_id']}": r for r in prior}
    identity = {"feedback_checkpoint_sha256": sha(FEEDBACK / "checkpoint.json"),
                "feedback_report_sha256": sha(FEEDBACK / "report.json"),
                "source_checkpoint_sha256": sha(PRIOR / "checkpoint.json"),
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
            raise ValueError("Feedback review inputs changed; refusing resume")
    else:
        records = []
        for key in feedback["identity"]["selected_keys"]:
            result, source = feedback["decisions"][key], by_key[key]
            if result["status"] not in {"PRELIMINARY_PASS", "FLAGGED"}:
                continue
            row = {**source, "text_translation": result["new_translation"],
                   "candidate_sha256": hashlib.sha256(result["new_translation"].encode()).hexdigest(),
                   "feedback_key": key, "feedback_qwen7_pass": result["qwen7_pass"]}
            for field in ("qwen14", "qwen7_roundtrip", "backtranslation_en", "backtranslation_status", "backtranslation_checks", "diagnostic_status", "preliminary_direct_qwen_pass"):
                row.pop(field, None)
            checks = deterministic_checks(row["language"], row)
            if checks != result.get("checks") or not all(checks[name] for name in ("nonempty", "target_script_present", "protected_names_preserved", "protected_numbers_preserved", "no_unrestored_marker")):
                raise ValueError(f"Feedback candidate integrity mismatch: {key}")
            records.append(row)
        checkpoint = {"identity": identity, "records": records}
        write(path, checkpoint)
    records = checkpoint["records"]
    if len(records) != 7:
        raise ValueError("Expected seven usable feedback translations")
    for row in records:
        source = by_key[row["feedback_key"]]
        candidate = feedback["decisions"][row["feedback_key"]]
        if (row["text_en"] != source["text_en"] or row["protected_entity_names"] != source["protected_entity_names"]
                or row["text_translation"] != candidate["new_translation"]
                or row["candidate_sha256"] != hashlib.sha256(row["text_translation"].encode()).hexdigest()):
            raise ValueError("Feedback review source or candidate text changed")
    model_for_stage = {"backtranslate": "translategemma:4b", "qwen7": "qwen2.5:7b-instruct", "qwen14": "qwen2.5:14b"}
    if args.stage in model_for_stage:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
            models = {item["name"]: item["digest"] for item in json.load(response)["models"]}
        model = model_for_stage[args.stage]
        if models.get(model) != identity["model_digests"][model]:
            raise ValueError(f"Model digest changed: {model}")
    for lang in ("fr", "de", "hi", "th"):
        batch = [r for r in records if r["language"] == lang]
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
        print(f"{args.stage}: {lang} complete", flush=True)
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
                  "checkpoint_sha256": sha(path), "selected": 8, "usable": len(records),
                  "generation_errors": 1, "counts": dict(counts),
                  "by_language": {lang: dict(by_lang[lang]) for lang in by_lang},
                  "limitations": ["Tiny reviewer-selected sample; not a corpus accuracy estimate.",
                                  "Qwen14 supplied feedback and reviewed revisions, so this is a same-reviewer rescue.",
                                  "AI-only review; no human fluency certification or release approval."]}
        write(OUT / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
