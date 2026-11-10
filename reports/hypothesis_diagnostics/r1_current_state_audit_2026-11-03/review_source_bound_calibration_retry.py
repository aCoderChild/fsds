#!/usr/bin/env python3
"""Run a source-bound, AI-only round-trip diagnostic on the fixed 32-row sample."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from verify_source_bound_calibration import verify, sha, PROJECT, TRAINING, RUN

sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
import review_training_translations_roundtrip as rt
import adjudicate_training_translations_qwen14b as adjudicator
from review_training_translations_qwen import masked_pair
from repair_flagged_eval_translations_translategemma import protect, restore

OUT = Path(__file__).resolve().parent / "evidence/source_bound_calibration_roundtrip_retry"
PRIOR = Path(__file__).resolve().parent / "evidence/source_bound_calibration_roundtrip"
SCRIPT = Path(__file__).resolve()
LANGS = ("fr", "de", "hi", "th")


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    selected, decisions, source_report = verify()
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        models = {row["name"]: row["digest"] for row in json.load(response)["models"]}
    digests = {name: models[name] for name in ("translategemma:4b", "qwen2.5:7b-instruct", "qwen2.5:14b")}
    for name, digest in source_report["source"]["model_digests"].items():
        if digests[name] != digest:
            raise ValueError(f"Model changed since candidate generation: {name}")
    identity = {
        "source_decisions_sha256": sha(RUN / "repair_candidates.jsonl"),
        "source_report_sha256": sha(RUN / "report.json"),
        "verifier_sha256": sha(Path(__file__).resolve().parent / "verify_source_bound_calibration.py"),
        "script_sha256": sha(SCRIPT),
        "roundtrip_script_sha256": sha(Path(rt.__file__)),
        "adjudication_script_sha256": sha(Path(adjudicator.__file__)),
        "model_digests": digests,
    }
    previous_report_path = PRIOR / "report.json"
    previous_checkpoint_path = PRIOR / "checkpoint.json"
    previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
    previous_checkpoint = json.loads(previous_checkpoint_path.read_text(encoding="utf-8"))
    old_identity = previous_report.get("identity", {})
    if (previous_report.get("checkpoint_sha256") != sha(previous_checkpoint_path)
            or previous_checkpoint.get("identity") != old_identity
            or old_identity.get("script_sha256") != sha(Path(__file__).resolve().parent / "review_source_bound_calibration.py")
            or any(old_identity.get(key) != value for key, value in identity.items() if key != "script_sha256")):
        raise ValueError("Prior checkpoint and source-bound review provenance do not match")
    identity["prior_report_sha256"] = sha(previous_report_path)
    identity["prior_checkpoint_sha256"] = sha(previous_checkpoint_path)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("Checkpoint provenance changed; refusing resume")
    else:
        records = previous_checkpoint["records"]
        for row in records:
            if row.get("qwen14", {}).get("error"):
                del row["qwen14"]
            row.pop("diagnostic_status", None)
            if (row.get("backtranslation_status") != "PASS"
                    or not row.get("backtranslation_checks")
                    or not all(value is True for value in row["backtranslation_checks"].values())
                    or not isinstance(row.get("backtranslation_en"), str)
                    or any(row["text_en"].count(name) != row["backtranslation_en"].count(name)
                           for name in row["protected_entity_names"] if name in row["text_en"])
                    or any(type(row.get("qwen7_roundtrip", {}).get(kind, {}).get("decision")) is not bool
                           for kind in ("semantic", "fluency"))):
                raise ValueError("Prior back-translation or 7B screen is incomplete")
        checkpoint = {"status": "IN_PROGRESS_DIAGNOSTIC_ONLY", "identity": identity,
                      "human_translation_reviewed": False, "answers_sent_to_reviewer": False,
                      "records": records}
        write_json(checkpoint_path, checkpoint)
    records = checkpoint["records"]
    if len(records) != 32 or any(
        (record["language"], record["kind"], record["prompt_id"], record["source_sha256"], record["candidate_sha256"])
        != (item["language"], item["row"]["kind"], item["row"]["prompt_id"], decision["source_en_sha256"], decision["new_translation_sha256"])
        for record, item, decision in zip(records, selected, decisions)
    ):
        raise ValueError("Checkpoint rows differ from the fixed sample")
    if any(hashlib.sha256(r["text_en"].encode()).hexdigest() != r["source_sha256"]
           or hashlib.sha256(r["text_translation"].encode()).hexdigest() != r["candidate_sha256"]
           or r["protected_entity_names"] != item["row"]["protected_entity_names"]
           for r, item in zip(records, selected)):
        raise ValueError("Checkpoint question text or protected names changed")

    for lang in LANGS:
        pending = [r for r in records if r["language"] == lang and "backtranslation_status" not in r]
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            prepared = []
            for row in batch:
                if not all(rt.target_deterministic_checks(lang, row).values()):
                    raise ValueError(f"Candidate integrity failed: {lang}/{row['prompt_id']}")
                protected, mapping, expected = protect(row["text_translation"], row["protected_entity_names"])
                prepared.append({"protected_text": protected, "mapping": mapping, "expected": expected})
            try:
                outputs = rt.translate_batch(lang, prepared, 240)
                for i, (row, source) in enumerate(zip(batch, prepared)):
                    try:
                        back = restore(outputs[f"ROW_{rt.alpha_label(i)}"], source["mapping"], source["expected"])
                        names = [n for n in row["protected_entity_names"] if n in row["text_en"]]
                        nums = rt.NUMBER.findall(row["text_en"])
                        checks = {"nonempty": bool(back.strip()),
                                  "english_latin_script_present": any("A" <= c <= "Z" or "a" <= c <= "z" for c in back),
                                  "source_entity_names_preserved_with_multiplicity": all(row["text_en"].count(n) == back.count(n) for n in names),
                                  "source_numbers_preserved_with_multiplicity": all(nums.count(n) == rt.NUMBER.findall(back).count(n) for n in set(nums)),
                                  "no_marker_remains": not rt.MARKER.search(back)}
                        row.update({"backtranslation_en": back, "backtranslation_checks": checks,
                                    "backtranslation_status": "PASS" if all(checks.values()) else "BLOCKED"})
                    except Exception as exc:
                        row.update({"backtranslation_status": "BLOCKED", "backtranslation_error": type(exc).__name__})
            except Exception as exc:
                for row in batch:
                    row.update({"backtranslation_status": "BLOCKED", "backtranslation_error": type(exc).__name__})
            write_json(checkpoint_path, checkpoint)
            print(f"{lang}: back-translated {min(start + 4, len(pending))}/{len(pending)} pending rows", flush=True)

    for lang in LANGS:
        pending = [r for r in records if r["language"] == lang and r.get("backtranslation_status") == "PASS" and "qwen7_roundtrip" not in r]
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            sem_items, flu_items = [], []
            for row in batch:
                pair = masked_pair({**row, "text_translation": row["backtranslation_en"]})[1]
                sem_items.append({"source_english": pair["source_english"], "backtranslation_english": pair["target_translation"]})
                flu_items.append({"target_question": masked_pair(row)[1]["target_translation"]})
            sem = rt.ask_qwen("semantic", lang, sem_items, 240)
            flu = rt.ask_qwen("fluency", lang, flu_items, 240)
            for row, a, b in zip(batch, sem, flu):
                row["qwen7_roundtrip"] = {"semantic": a, "fluency": b}
            write_json(checkpoint_path, checkpoint)
            print(f"{lang}: Qwen7 round-trip reviewed {min(start + 4, len(pending))}/{len(pending)} pending rows", flush=True)

    for lang in LANGS:
        pending = [r for r in records if r["language"] == lang and r.get("backtranslation_status") == "PASS" and "qwen14" not in r]
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            try:
                results = adjudicator.judge_batch(lang, batch, 240)
                for row, result in zip(batch, results):
                    row["qwen14"] = result
            except Exception as exc:
                for row in batch:
                    row["qwen14"] = {"error": type(exc).__name__, "reason": str(exc)[:500]}
            write_json(checkpoint_path, checkpoint)
            print(f"{lang}: Qwen14 reviewed {min(start + 4, len(pending))}/{len(pending)} pending rows", flush=True)

    statuses = Counter()
    by_language = {lang: Counter() for lang in LANGS}
    for row in records:
        q7 = row.get("qwen7_roundtrip", {})
        q14 = row.get("qwen14", {})
        decisions7 = (q7.get("semantic", {}).get("decision"), q7.get("fluency", {}).get("decision"))
        if row.get("backtranslation_status") != "PASS" or any(type(value) is not bool for value in decisions7) or type(q14.get("semantic_pass")) is not bool or type(q14.get("fluency_pass")) is not bool:
            status = "BLOCKED_OR_ERROR"
        elif q14["semantic_pass"] and q14["fluency_pass"]:
            status = "TWO_MODEL_PASS" if all(decisions7) else "14B_ADJUDICATED_PASS"
        else:
            status = "FLAGGED"
        row["diagnostic_status"] = status
        statuses[status] += 1
        by_language[row["language"]][status] += 1
    checkpoint["status"] = "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
    write_json(checkpoint_path, checkpoint)
    summary = {"status": checkpoint["status"], "identity": identity, "rows": len(records),
               "checkpoint_sha256": sha(checkpoint_path), "counts": dict(statuses),
               "by_language": {lang: dict(by_language[lang]) for lang in LANGS},
               "human_translation_reviewed": False, "answers_sent_to_reviewer": False,
               "limits": ["Fixed sample only; no candidate is approved or merged.",
                          "TranslateGemma is both translator and backtranslator; the Qwen reviewers share a model family.",
                          "AI judgments are not fluent-human certification."]}
    write_json(OUT / "report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
