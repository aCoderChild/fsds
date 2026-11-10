#!/usr/bin/env python3
"""Apply the frozen round-trip + Qwen14B screen to paired calibration winners.

Candidate choice is deterministic and answer-blind: use NLLB when its Qwen7
screen passed; otherwise fall back to the paired TranslateGemma draft when
that draft passed. This remains diagnostic-only and cannot release training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
V51 = SCRIPT.parents[2]
OUT = V51 / "training_release_draft"
CALIBRATION = OUT / "translation_repair_diagnostics/nllb13_strictflagged_cal64_2026-11-03"
REPAIR = OUT / "translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03"
DEFAULT_OUTPUT = OUT / "translation_repair_diagnostics/nllb13_cal64_roundtrip_qwen14_2026-11-03"
LANGS = ("fr", "de", "hi", "th")
KINDS = ("atomic", "composition")

sys.path.insert(0, str(OUT / "scripts"))
sys.path.insert(0, str(V51 / "scripts"))
from adjudicate_training_translations_qwen14b import judge_batch  # noqa: E402
from review_training_translations_qwen import read_jsonl, sha, write_json, write_jsonl  # noqa: E402
from review_training_translations_roundtrip import sha_text, translate_batch  # noqa: E402
from repair_flagged_eval_translations_translategemma import protect, restore  # noqa: E402
from draft_eval_translations_nllb import language_script_ok  # noqa: E402


def selected_candidates() -> tuple[list[dict], dict]:
    report_path = CALIBRATION / "report.json"
    rows_path = CALIBRATION / "candidates.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = read_jsonl(rows_path)
    repair_report_path = REPAIR / "report.json"
    repair_rows_path = REPAIR / "repair_candidates.jsonl"
    repair_report = json.loads(repair_report_path.read_text(encoding="utf-8"))
    repairs = read_jsonl(repair_rows_path)
    expected_source_hashes = report.get("source_hashes", {})
    if (report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or report.get("sample_size") != 64
            or report.get("candidate_rows_sha256") != sha(rows_path)
            or report.get("answers_sent_to_reviewer") is not False
            or report.get("human_translation_reviewed") is not False
            or len(rows) != 64
            or expected_source_hashes.get("translategemma_repair_rows_sha256") != sha(repair_rows_path)
            or expected_source_hashes.get("translategemma_repair_report_sha256") != sha(repair_report_path)
            or repair_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or repair_report.get("human_translation_reviewed") is not False
            or repair_report.get("answers_sent_to_reviewer") is not False):
        raise ValueError("The hash-bound, answer-blind 64-row translation calibration is incomplete")
    repair_by_key = {(r["language"], r["prompt_id"], r["kind"]): r for r in repairs}
    if len(repair_by_key) != len(repairs):
        raise ValueError("The hash-bound TranslateGemma repair file has duplicate prompt/kind keys")
    selected = []
    for row in rows:
        key = (row["language"], row["prompt_id"], row["kind"])
        repair = repair_by_key.get(key)
        if (not repair
                or repair.get("language") != row["language"]
                or repair.get("source_en_sha256") != row.get("source_en_sha256")
                or repair.get("old_translation_sha256") != row.get("base_translation_sha256")
                or repair.get("new_translation_sha256") != sha_text(repair.get("new_translation", ""))
                or (repair.get("qwen_pass") is True) != (row.get("translategemma_qwen_pass") is True)):
            raise ValueError(f"Paired TranslateGemma source mismatch for {row['prompt_id']} / {row['kind']}")
        if row.get("status") == "DIAGNOSTIC_QWEN_PASS" and row.get("nllb_qwen_pass") is True:
            chosen = row["text_translation"]
            chosen_model = row["translation_model"]
        elif row.get("translategemma_qwen_pass") is True:
            if repair.get("status") != "DIAGNOSTIC_QWEN_PASS" or not repair.get("new_translation"):
                raise ValueError(f"Passing TranslateGemma fallback is missing for {row['prompt_id']} / {row['kind']}")
            chosen = repair["new_translation"]
            chosen_model = "ollama:translategemma:4b"
        else:
            selected.append({**row, "selected_translation": "", "selected_model": None,
                             "selection_status": "NO_QWEN7_PASSING_CANDIDATE"})
            continue
        selected.append({**row, "selected_translation": chosen, "selected_model": chosen_model,
                         "selection_status": "QWEN7_PASSING_CANDIDATE_SELECTED"})
    return selected, {
        "calibration_report_sha256": sha(report_path), "calibration_rows_sha256": sha(rows_path),
        "translategemma_repair_report_sha256": sha(repair_report_path),
        "translategemma_repair_rows_sha256": sha(repair_rows_path),
    }


def target_checks(lang: str, row: dict) -> dict:
    source, target = row["text_en"], row["selected_translation"]
    names = [name for name in row.get("protected_entity_names", []) if name and name in source]
    numbers = re.findall(r"\d+(?:[-/.]\d+)*", source)
    target_numbers = re.findall(r"\d+(?:[-/.]\d+)*", target)
    return {
        "nonempty": bool(target.strip()),
        "target_script_present": language_script_ok(lang, target),
        "protected_names_preserved_with_multiplicity": all(source.count(n) == target.count(n) for n in names),
        "protected_numbers_preserved_with_multiplicity": all(numbers.count(n) == target_numbers.count(n) for n in set(numbers)),
        "no_unrestored_marker": "ZXQ" not in target and "QXZ" not in target,
    }


def judge_with_isolation(lang: str, rows: list[dict], timeout: int) -> list[tuple[dict | None, str | None]]:
    """Keep model/review failures row-local; a failed judgment can never pass."""
    try:
        decisions = judge_batch(lang, rows, timeout)
        if len(decisions) != len(rows):
            raise ValueError(f"{lang}: Qwen14B decision count mismatch")
        return [(decision, None) for decision in decisions]
    except Exception as exc:
        if len(rows) == 1:
            return [(None, f"{type(exc).__name__}: {exc}")]
        middle = len(rows) // 2
        return (judge_with_isolation(lang, rows[:middle], timeout)
                + judge_with_isolation(lang, rows[middle:], timeout))


def run(output_dir: Path, batch_size: int, timeout: int) -> dict:
    rows, source_hashes = selected_candidates()
    valid = []
    for row in rows:
        checks = target_checks(row["language"], row) if row["selected_translation"] else {}
        row["target_deterministic_checks"] = checks
        row["target_deterministic_pass"] = bool(checks) and all(checks.values())
        row["backtranslation_status"] = "NOT_RUN"
        row["backtranslation_en"] = ""
        row["backtranslation_checks"] = {}
        row["qwen14b_semantic_pass"] = None
        row["qwen14b_fluency_pass"] = None
        row["qwen14b_reason"] = None
        row["final_calibration_status"] = (
            "NO_QWEN7_PASSING_CANDIDATE" if not row["selected_translation"] else
            ("BLOCKED_TARGET_DETERMINISTIC_CHECK" if not row["target_deterministic_pass"] else "PENDING_BACKTRANSLATION")
        )
        if row["target_deterministic_pass"]:
            valid.append(row)

    for lang in LANGS:
        lang_rows = [r for r in valid if r["language"] == lang]
        prepared = []
        for row in lang_rows:
            protected, mapping, counts = protect(row["selected_translation"], row.get("protected_entity_names", []))
            prepared.append({**row, "text_translation": row["selected_translation"], "protected_text": protected,
                             "restore_mapping": mapping, "restore_counts": counts})
        back_by_key = {}
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start + batch_size]
            try:
                translations = translate_batch(lang, batch, timeout)
                translation_error = None
            except Exception as exc:
                translations = {}
                translation_error = f"{type(exc).__name__}: {exc}"
            for offset, item in enumerate(batch):
                key = f"ROW_{chr(65 + offset)}"
                prompt_key = (item["prompt_id"], item["kind"])
                if key not in translations:
                    back_by_key[prompt_key] = {
                        "text": "", "error": translation_error or "Missing back-translation key",
                    }
                    continue
                try:
                    back_by_key[prompt_key] = {
                        "text": restore(translations[key], item["restore_mapping"], item["restore_counts"]),
                        "error": None,
                    }
                except ValueError as exc:
                    back_by_key[prompt_key] = {"text": "", "error": f"{type(exc).__name__}: {exc}"}
        judge_rows = []
        for row in lang_rows:
            result = back_by_key.get((row["prompt_id"], row["kind"]), {"text": "", "error": "Missing result"})
            row["backtranslation_en"] = result["text"]
            source, back = row["text_en"], result["text"]
            nums, back_nums = re.findall(r"\d+(?:[-/.]\d+)*", source), re.findall(r"\d+(?:[-/.]\d+)*", back)
            checks = {
                "nonempty": bool(back.strip()),
                "english_latin_script_present": any("A" <= c <= "Z" or "a" <= c <= "z" for c in back),
                "protected_names_preserved_with_multiplicity": all(source.count(n) == back.count(n) for n in row.get("protected_entity_names", []) if n in source),
                "protected_numbers_preserved_with_multiplicity": all(nums.count(n) == back_nums.count(n) for n in set(nums)),
                "no_unrestored_marker": "ZXQ" not in back and "QXZ" not in back,
            }
            row["backtranslation_checks"] = checks
            row["backtranslation_error"] = result["error"]
            row["backtranslation_status"] = "PASS" if not result["error"] and all(checks.values()) else "FAILED"
            if row["backtranslation_status"] == "PASS":
                row["final_calibration_status"] = "PENDING_QWEN14B"
                judge_rows.append(row)
            else:
                row["final_calibration_status"] = "BLOCKED_BACKTRANSLATION_CHECK"
        for start in range(0, len(judge_rows), 8):
            batch = judge_rows[start:start + 8]
            decisions = judge_with_isolation(lang, batch, timeout)
            for row, (decision, error) in zip(batch, decisions):
                if error:
                    row["qwen14b_review_error"] = error
                    row["final_calibration_status"] = "AI_ONLY_QWEN14B_REVIEW_ERROR"
                    continue
                row["qwen14b_semantic_pass"] = decision["semantic_pass"]
                row["qwen14b_fluency_pass"] = decision["fluency_pass"]
                row["qwen14b_reason"] = decision["reason"]
                row["qwen14b_review_error"] = None
                row["final_calibration_status"] = (
                    "AI_ONLY_FULL_SCREEN_CALIBRATION_PASS"
                    if decision["semantic_pass"] and decision["fluency_pass"]
                    else "AI_ONLY_QWEN14B_FLAGGED"
                )

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "selected_candidates.jsonl"
    write_jsonl(rows_path, rows)
    summary = {}
    for lang in LANGS:
        summary[lang] = {}
        for kind in KINDS:
            cell = [r for r in rows if r["language"] == lang and r["kind"] == kind]
            summary[lang][kind] = {
                "selected": len(cell),
                "nllb_selected": sum(r.get("selected_model") == "facebook/nllb-200-distilled-1.3B" for r in cell),
                "translategemma_fallback_selected": sum(r.get("selected_model") == "ollama:translategemma:4b" for r in cell),
                "no_qwen7_passing_candidate": sum(r["final_calibration_status"] == "NO_QWEN7_PASSING_CANDIDATE" for r in cell),
                "target_deterministic_pass": sum(r["target_deterministic_pass"] for r in cell),
                "backtranslation_pass": sum(r["backtranslation_status"] == "PASS" for r in cell),
                "qwen14b_pass": sum(r["final_calibration_status"] == "AI_ONLY_FULL_SCREEN_CALIBRATION_PASS" for r in cell),
                "full_screen_pass": sum(r["final_calibration_status"] == "AI_ONLY_FULL_SCREEN_CALIBRATION_PASS" for r in cell),
                "qwen14b_review_error": sum(r["final_calibration_status"] == "AI_ONLY_QWEN14B_REVIEW_ERROR" for r in cell),
            }
    report = {
        "status": "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW",
        "calibration_id": "nllb13_tg4b_qwen7_union_then_roundtrip_qwen14b_cal64_v1",
        "sample_size": len(rows), "per_language_kind": summary,
        "candidate_selection_rule": "NLLB when Qwen7 passes; otherwise paired TranslateGemma when its stored Qwen7 decision passes; otherwise no candidate",
        "backtranslator_model": "ollama:translategemma:4b",
        "adjudicator_model": "ollama:qwen2.5:14b",
        "answers_sent_to_translator": False, "answers_sent_to_reviewers": False,
        "human_translation_reviewed": False, "do_not_train": True,
        "source_hashes": {**source_hashes, "script_sha256": sha(SCRIPT)},
        "selected_rows_sha256": sha(rows_path),
        "limits": [
            "This 64-row stratified calibration is not a representative estimate or a release screen.",
            "All reviewers are AI models; Qwen2.5 7B and 14B share a model family.",
            "No translations, review records, release manifests, or SFT data were modified.",
            "All 1,952 rows still require full hash-bound screening before training.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timeout < 1:
        parser.error("batch-size and timeout must be positive")
    rows, hashes = selected_candidates()
    counts = Counter((r["language"], r["kind"], r["selection_status"]) for r in rows)
    if args.dry_run:
        print(json.dumps({"status": "PREFLIGHT_PASS_NO_MODEL_LOADED", "rows": len(rows),
                          "selected_candidates": {"nllb": sum(r.get("selected_model") == "facebook/nllb-200-distilled-1.3B" for r in rows),
                                                   "translategemma_fallback": sum(r.get("selected_model") == "ollama:translategemma:4b" for r in rows),
                                                   "none": sum(not r.get("selected_model") for r in rows)},
                          "sample_cells": {"/".join(k): v for k, v in counts.items()},
                          "source_hashes": hashes}, indent=2))
        return
    run(args.output_dir, args.batch_size, args.timeout)


if __name__ == "__main__":
    main()
