#!/usr/bin/env python3
"""Create answer-blind, critique-guided diagnostic repairs for flagged rows.

The Qwen critique may be wrong; the English question remains authoritative.
Outputs are separate candidates and never modify training or review inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import repair_all_flagged_training_translations as base

SCRIPT = Path(__file__).resolve()
TRAINING = base.TRAINING
API = base.API
MODEL = base.TRANSLATOR
MODEL_ID = base.TRANSLATOR_ID
REVIEWER_ID = base.REVIEWER_ID
LANGS = base.LANGS
KINDS = base.KINDS
RUN_DIR = TRAINING / "translation_repair_diagnostics" / "translategemma4b_qwen7b_all_flagged_2026-11-03"
REQUIRED_CHECKS = base.REQUIRED_CHECKS


def stable_digest(item: dict) -> str:
    return hashlib.sha256(
        f"r1-feedback-repair-v1\0{item['language']}\0{item['kind']}\0{item['prompt_id']}".encode("utf-8")
    ).hexdigest()


def select_feedback_rows(decisions: list[dict], per_cell: int | None) -> list[dict]:
    """Select flagged candidates evenly and reproducibly by language and QA type."""
    if per_cell is not None and per_cell < 1:
        raise ValueError("per_cell must be positive or None")
    grouped = {(lang, kind): [] for lang in LANGS for kind in KINDS}
    seen = set()
    for row in decisions:
        key = (row.get("language"), row.get("kind"), row.get("prompt_id"))
        if key[0] not in LANGS or key[1] not in KINDS or not isinstance(key[2], str) or not key[2]:
            raise ValueError(f"Invalid repair decision key: {key}")
        if key in seen:
            raise ValueError(f"Duplicate repair decision key: {key}")
        seen.add(key)
        if base.FORBIDDEN_REVIEW_FIELDS.intersection(row):
            raise ValueError(f"answer-bearing data reached the feedback repair input: {key}")
        if row.get("status") != "DIAGNOSTIC_QWEN_FLAGGED":
            continue
        if row.get("qwen_pass") is not False or not isinstance(row.get("qwen_reason"), str) or not row["qwen_reason"].strip():
            raise ValueError(f"Flagged row lacks a usable reviewer critique: {key}")
        if row.get("new_translation") is None:
            raise ValueError(f"Flagged row lacks a candidate translation: {key}")
        grouped[(key[0], key[1])].append(row)
    selected = []
    for lang in LANGS:
        for kind in KINDS:
            cell = sorted(grouped[(lang, kind)], key=stable_digest)
            if not cell:
                raise ValueError(f"No Qwen-flagged rows available for {lang}/{kind}")
            selected.extend(cell if per_cell is None else cell[:per_cell])
    return selected


def protect_old_translation(text: str, mapping: dict[str, str]) -> str:
    masked = text
    for marker, value in sorted(mapping.items(), key=lambda pair: -len(pair[1])):
        masked = re.sub(re.escape(value), marker, masked, flags=re.IGNORECASE)
    return masked


def translate_feedback_once(source: str, old_translation: str, critique: str, lang: str, timeout: int) -> str:
    target = base.LANGUAGE_NAMES[lang]
    prompt = (
        f"You are translating a factual question from English into natural {target}. The English question is "
        "the source of truth. An AI reviewer criticized an earlier translation, but that critique may be wrong. "
        "Use it only to identify a possible translation issue; if it conflicts with the English, follow the English. "
        "Preserve the question's full meaning, relation direction, scope, and answer type. Preserve every ZXQ marker "
        "with identical spelling, order, and number of occurrences. Do not add facts or answer the question. Return only "
        "the corrected translation.\n\n"
        f"ENGLISH SOURCE:\n{source}\n\nEARLIER TARGET-LANGUAGE CANDIDATE:\n{old_translation}\n\n"
        f"AI REVIEWER CRITIQUE (unverified):\n{critique}"
    )
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": 192},
        "keep_alive": "10m",
    }
    request = urllib.request.Request(
        API,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    text = result.get("message", {}).get("content", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("TranslateGemma returned an empty feedback-guided translation")
    return text.strip()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path: Path, report: dict, selected: list[dict], *, allow_tail: bool) -> tuple[list[dict], bytes]:
    count = report.get("processed_rows")
    if not isinstance(count, int) or not 0 <= count <= len(selected):
        raise ValueError("Invalid feedback-repair checkpoint count")
    raw = path.read_bytes() if path.exists() else b""
    lines = raw.splitlines(keepends=True)
    if len(lines) < count or any(not line.endswith(b"\n") for line in lines[:count]):
        raise ValueError("Committed feedback-repair prefix is incomplete")
    committed = b"".join(lines[:count])
    if report.get("decisions_checkpoint_sha256") != hashlib.sha256(committed).hexdigest():
        raise ValueError("Feedback-repair checkpoint hash mismatch")
    if len(lines) - count > 1 or (lines[count:] and not allow_tail):
        raise ValueError("Unexpected uncommitted feedback-repair output")
    decisions = [json.loads(line) for line in lines[:count]]
    for index, row in enumerate(decisions):
        item = selected[index]
        expected_key = (item["language"], item["kind"], item["prompt_id"])
        actual_key = (row.get("language"), row.get("kind"), row.get("prompt_id"))
        if actual_key != expected_key:
            raise ValueError(f"Feedback-repair checkpoint order mismatch at {index}")
        if (row.get("previous_candidate_sha256") != item.get("new_translation_sha256")
                or row.get("critique_sha256") != base.sha_text(item["qwen_reason"])):
            raise ValueError(f"Feedback-repair checkpoint input hash mismatch for {expected_key}")
        if row.get("new_translation") and row.get("new_translation_sha256") != base.sha_text(row["new_translation"]):
            raise ValueError(f"Feedback-repair candidate hash mismatch for {expected_key}")
    if raw != committed:
        with path.open("wb") as stream:
            stream.write(committed)
            stream.flush()
            os.fsync(stream.fileno())
    return decisions, committed


def summarize(decisions: list[dict]) -> dict:
    counts = {lang: {kind: Counter() for kind in KINDS} for lang in LANGS}
    for row in decisions:
        c = counts[row["language"]][row["kind"]]
        c["selected"] += 1
        c["deterministic_pass"] += int(row.get("deterministic_pass") is True)
        c["qwen_pass"] += int(row.get("qwen_pass") is True)
        c["errors"] += int(row.get("status") == "DIAGNOSTIC_ERROR")
        c["deterministic_failed"] += int(row.get("status") == "DIAGNOSTIC_DETERMINISTIC_FAILED")
    return {lang: {kind: dict(counts[lang][kind]) for kind in KINDS} for lang in LANGS}


def run(output_dir: Path, per_cell: int | None, timeout: int, resume: bool) -> dict:
    parent_report_path = RUN_DIR / "report.json"
    parent_rows_path = RUN_DIR / "repair_candidates.jsonl"
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    parent_rows = base.read_jsonl(parent_rows_path)
    if (parent_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or file_sha(parent_rows_path) != parent_report.get("decisions_sha256")):
        raise ValueError("Parent diagnostic is incomplete or hash-mismatched")
    selected = select_feedback_rows(parent_rows, per_cell)
    _, source_rows = base.source_manifest()
    source_by_key = {
        (lang, row["prompt_id"], row["kind"]): row
        for lang, rows in source_rows.items()
        for row in rows
    }
    source_key = {
        "parent_report_sha256": file_sha(parent_report_path),
        "parent_candidates_sha256": file_sha(parent_rows_path),
        "source_review_report_sha256": file_sha(base.REVIEW_ROOT / "report.json"),
        "source_review_sha256_by_language": {
            lang: file_sha(base.REVIEW_ROOT / f"{lang}.jsonl") for lang in LANGS
        },
    }
    by_language = {lang: sum(row["language"] == lang for row in selected) for lang in LANGS}
    by_cell = {lang: {kind: sum(row["language"] == lang and row["kind"] == kind for row in selected)
                      for kind in KINDS} for lang in LANGS}
    decisions_path = output_dir / "feedback_repair_candidates.jsonl"
    report_path = output_dir / "report.json"

    if resume:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") not in {"IN_PROGRESS_DIAGNOSTIC_ONLY", "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"}
                or report.get("source") != source_key
                or report.get("script_sha256") != file_sha(SCRIPT)
                or report.get("selected_rows") != len(selected)
                or report.get("selected_by_cell") != by_cell):
            raise ValueError("Cannot resume feedback repairs: run inputs or script changed")
        decisions, committed = load_checkpoint(
            decisions_path, report, selected,
            allow_tail=report.get("status") == "IN_PROGRESS_DIAGNOSTIC_ONLY",
        )
        if report.get("per_language_kind") != summarize(decisions):
            raise ValueError("Feedback-repair report summary differs from committed decisions")
        if report.get("status") == "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW":
            if len(decisions) != len(selected) or report.get("decisions_sha256") != hashlib.sha256(committed).hexdigest():
                raise ValueError("Completed feedback-repair output failed integrity checks")
            print(json.dumps({"status": report["status"], "processed": len(decisions)}, indent=2))
            return report
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        decisions = []
        committed = b""
        report = {
            "status": "IN_PROGRESS_DIAGNOSTIC_ONLY",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "translator_model": MODEL_ID,
            "reviewer_model": REVIEWER_ID,
            "selection_rule": "stable SHA-256 order; equal per-cell sample from prior Qwen-flagged rows",
            "per_cell_limit": per_cell,
            "selected_rows": len(selected),
            "selected_by_language": by_language,
            "selected_by_cell": by_cell,
            "source": source_key,
            "script_sha256": file_sha(SCRIPT),
            "answers_sent_to_translator": False,
            "answers_sent_to_reviewer": False,
            "human_translation_reviewed": False,
            "limits": [
                "The reviewer critique is AI-generated and may be wrong; the English source is authoritative.",
                "This sample is diagnostic and does not approve candidate translations or training data.",
                "Qwen2.5 7B review is AI-only and does not certify fluent-human quality.",
                "No source translation or release manifest is changed.",
            ],
            "processed_rows": 0,
            "decisions_checkpoint_sha256": hashlib.sha256(b"").hexdigest(),
            "per_language_kind": summarize([]),
        }
        atomic_write_json(report_path, report)
        decisions_path.touch()

    decision_hasher = hashlib.sha256()
    decision_hasher.update(committed)
    pending = selected[len(decisions):]
    with decisions_path.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(pending, 1):
            lang, kind, prompt_id = item["language"], item["kind"], item["prompt_id"]
            source = source_by_key.get((lang, prompt_id, kind))
            if source is None:
                raise ValueError(f"Missing hash-bound source translation row: {(lang, prompt_id, kind)}")
            row = {
                "language": lang,
                "kind": kind,
                "prompt_id": prompt_id,
                "previous_candidate_sha256": item["new_translation_sha256"],
                "critique_sha256": base.sha_text(item["qwen_reason"]),
                "translator_model": MODEL_ID,
                "reviewer_model": REVIEWER_ID,
                "feedback_sent_to_translator": True,
                "answers_sent_to_translator": False,
                "answers_sent_to_reviewer": False,
                "human_translation_reviewed": False,
                "status": "DIAGNOSTIC_ERROR",
            }
            try:
                names = source.get("protected_entity_names", [])
                protected, mapping, expected = base.protect(source["text_en"], names)
                old_masked = protect_old_translation(item["new_translation"], mapping)
                translated = translate_feedback_once(
                    protected, old_masked, item["qwen_reason"], lang, timeout,
                )
                translated = base.restore(translated, mapping, expected)
                row["new_translation"] = translated
                row["new_translation_sha256"] = base.sha_text(translated)
                candidate = {**source, "text_translation": translated}
                checks = base.deterministic_checks(lang, candidate)
                row["checks"] = checks
                row["deterministic_pass"] = all(checks.get(name) is True for name in REQUIRED_CHECKS)
                if row["deterministic_pass"]:
                    _, review_item = base.masked_pair(candidate)
                    row["review_pair_sha256"] = base.sha_text(json.dumps(review_item, ensure_ascii=False, sort_keys=True))
                    verdict = base.ask_qwen(lang, [review_item], timeout)[0]
                    row["qwen_pass"] = verdict.get("decision") is True
                    row["qwen_reason"] = str(verdict.get("reason", ""))
                    row["status"] = "DIAGNOSTIC_QWEN_PASS" if row["qwen_pass"] else "DIAGNOSTIC_QWEN_FLAGGED"
                else:
                    row["status"] = "DIAGNOSTIC_DETERMINISTIC_FAILED"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            serialized = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            stream.write(serialized.decode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
            decision_hasher.update(serialized)
            decisions.append(row)
            report.update({
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "processed_rows": len(decisions),
                "decisions_checkpoint_sha256": decision_hasher.hexdigest(),
                "per_language_kind": summarize(decisions),
            })
            atomic_write_json(report_path, report)
            if index % 8 == 0 or index == len(pending):
                print(f"processed {len(decisions)}/{len(selected)} feedback-guided candidates; latest {lang}/{kind}", flush=True)

    if len(decisions) != len(selected):
        raise RuntimeError(f"Incomplete feedback repair diagnostic: {len(decisions)}/{len(selected)}")
    report.update({
        "status": "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decisions_sha256": decision_hasher.hexdigest(),
        "decisions_checkpoint_sha256": decision_hasher.hexdigest(),
        "per_language_kind": summarize(decisions),
        "translation_successes": sum(bool(row.get("new_translation")) for row in decisions),
        "deterministic_passes": sum(row.get("deterministic_pass") is True for row in decisions),
        "qwen_passes": sum(row.get("qwen_pass") is True for row in decisions),
        "errors": sum(row.get("status") == "DIAGNOSTIC_ERROR" for row in decisions),
    })
    atomic_write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "processed": report["processed_rows"],
        "deterministic_passes": report["deterministic_passes"],
        "qwen_passes": report["qwen_passes"],
        "errors": report["errors"],
        "report": str(report_path),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-cell", type=int, default=8, help="Stable calibration rows per language/kind cell")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("translation_repair_diagnostics/translategemma4b_qwen7b_feedback64_2026-11-03"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.timeout < 1 or args.per_cell < 1:
        parser.error("timeout and per-cell must be positive")
    output_dir = args.output_dir if args.output_dir.is_absolute() else TRAINING / args.output_dir
    run(output_dir, args.per_cell, args.timeout, args.resume)


if __name__ == "__main__":
    main()
