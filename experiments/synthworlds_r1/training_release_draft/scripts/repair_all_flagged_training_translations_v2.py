#!/usr/bin/env python3
"""Create source-bound, diagnostic-only repairs for flagged training translations.

This tool never edits release inputs or approves a training corpus. TranslateGemma
4B drafts a replacement from the English question; deterministic checks and
Qwen2.5 7B screen each candidate. All inputs/outputs are answer-blind and
hash-bound. Human translation quality is not certified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
V51 = SCRIPT.parents[2]
TRAINING = V51 / "training_release_draft"
REVIEW_ROOT = TRAINING / "translation_reviews_roundtrip_qwen"
LANGS = ("fr", "de", "hi", "th")
KINDS = ("atomic", "composition")
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
TRANSLATOR = "translategemma:4b"
TRANSLATOR_ID = "ollama:translategemma:4b"
REVIEWER_ID = "ollama:qwen2.5:7b-instruct"
API = "http://localhost:11434/api/chat"
REQUIRED_CHECKS = (
    "nonempty", "target_script_present", "protected_names_preserved",
    "protected_numbers_preserved", "no_unrestored_marker",
)
FORBIDDEN_REVIEW_FIELDS = {"gold_label", "gold_answers", "final_answer_source_supported_en"}
MARKER = re.compile(r"ZXQ[A-Z]+QXZ")

sys.path.insert(0, str(V51 / "scripts"))
sys.path.insert(0, str(TRAINING / "scripts"))
from draft_eval_translations_nllb import language_script_ok  # noqa: E402
from repair_flagged_eval_translations_translategemma import protect, restore  # noqa: E402
from review_training_translations_qwen import (  # noqa: E402
    ask_qwen,
    collect_rows,
    deterministic_checks,
    masked_pair,
    read_jsonl,
    sha,
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_flagged_rows(rows_by_lang: dict[str, list[dict]], final_reviews: dict) -> list[dict]:
    """Select final-review flags in stable order after checking both review stages."""
    selected = []
    for lang in LANGS:
        rows = rows_by_lang.get(lang, [])
        if len(rows) != 488:
            raise ValueError(f"{lang}: expected 488 hash-bound rows, found {len(rows)}")
        seen = set()
        for row in rows:
            key = (row.get("prompt_id"), row.get("kind"))
            if not all(isinstance(value, str) and value for value in key) or key in seen:
                raise ValueError(f"{lang}: missing or duplicate review key {key}")
            seen.add(key)
            if row.get("kind") not in KINDS:
                raise ValueError(f"{lang}/{key}: unsupported training question kind")
            if FORBIDDEN_REVIEW_FIELDS.intersection(row):
                raise ValueError(f"{lang}/{key}: answer-bearing data must not reach translation review")
            if row.get("review_status") not in {"AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS", "AI_ONLY_REVIEW_FLAGGED"}:
                raise ValueError(f"{lang}/{key}: source review is incomplete or blocked")
            if not isinstance(row.get("text_en"), str) or not row["text_en"].strip():
                raise ValueError(f"{lang}/{key}: missing English source question")
            final = final_reviews.get((lang, row["prompt_id"], row["kind"]))
            if (final is None or final.get("text_en") != row["text_en"]
                    or final.get("text_translation") != row.get("text_translation")
                    or final.get("review_status") != row["review_status"]):
                raise ValueError(f"{lang}/{key}: final review does not match first-stage source")
            if final.get("final_review_status") == "AI_ONLY_REVIEW_FLAGGED":
                row = {**row, "final_review_status": final["final_review_status"]}
                selected.append({"language": lang, "row": row})
    selected.sort(key=lambda item: (
        LANGS.index(item["language"]),
        KINDS.index(item["row"]["kind"]),
        hashlib.sha256(
            f"r1-all-flagged-repair-v1\0{item['language']}\0{item['row']['prompt_id']}".encode("utf-8")
        ).hexdigest(),
        item["row"]["prompt_id"],
    ))
    return selected


def translate_once(text: str, lang: str, timeout: int) -> str:
    target = LANGUAGE_NAMES[lang]
    prompt = (
        f"Translate this English factual question into natural {target}. Preserve its complete meaning, "
        "relation direction, scope, tense, and answer type. Copy every ZXQ marker exactly once, with "
        "identical spelling and capitalization. Do not translate, omit, or duplicate markers. "
        "Return only the translated question.\n\n" + text
    )
    payload = {
        "model": TRANSLATOR,
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
    output = result.get("message", {}).get("content", "")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("TranslateGemma returned an empty translation")
    return output.strip()


def translate_with_retries(text: str, lang: str, timeout: int) -> str:
    last_error = None
    for attempt in range(3):
        try:
            return translate_once(text, lang, timeout)
        except Exception as exc:  # transient local API errors are retried
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"TranslateGemma failed after retries: {type(last_error).__name__}: {last_error}")


def source_manifest() -> tuple[dict, dict[str, list[dict]]]:
    report_path = REVIEW_ROOT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "AI_ONLY_ROUNDTRIP_TRANSLATION_SCREEN_COMPLETE":
        raise ValueError("The hash-bound first-stage review is incomplete")
    if report.get("human_translation_reviewed") is not False or report.get("answers_sent_to_reviewer") is not False:
        raise ValueError("Unexpected human-review or answer-blinding metadata")
    rows_by_lang = {lang: read_jsonl(REVIEW_ROOT / f"{lang}.jsonl") for lang in LANGS}
    for lang in LANGS:
        path = REVIEW_ROOT / f"{lang}.jsonl"
        detail = report.get("languages", {}).get(lang, {})
        if detail.get("review_sha256") != sha(path):
            raise ValueError(f"{lang}: first-stage review file hash mismatch")
        actual_flagged = sum(r.get("review_status") == "AI_ONLY_REVIEW_FLAGGED" for r in rows_by_lang[lang])
        expected_flagged = detail.get("review_status_counts", {}).get("AI_ONLY_REVIEW_FLAGGED")
        if actual_flagged != expected_flagged:
            raise ValueError(f"{lang}: first-stage flagged-row count mismatch")
        source_rows, source_hashes = collect_rows(lang)
        source_hashes.pop("review_protocol", None)
        bound_hashes = detail.get("source_hashes", {})
        if any(bound_hashes.get(name) != value for name, value in source_hashes.items()):
            raise ValueError(f"{lang}: source inputs do not match the first-stage review")
        source_by_key = {(row["prompt_id"], row["kind"]): row for row in source_rows}
        if len(source_by_key) != len(source_rows) or len(source_by_key) != len(rows_by_lang[lang]):
            raise ValueError(f"{lang}: duplicate or incomplete source questions")
        for review_row in rows_by_lang[lang]:
            key = (review_row["prompt_id"], review_row["kind"])
            source = source_by_key.pop(key, None)
            if (source is None or source["text_en"] != review_row.get("text_en")
                    or source["text_translation"] != review_row.get("text_translation")):
                raise ValueError(f"{lang}/{key}: review text does not match the source question")
            names = source.get("protected_entity_names")
            if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
                raise ValueError(f"{lang}/{key}: missing or invalid protected-name list")
            review_row["protected_entity_names"] = names
        if source_by_key:
            raise ValueError(f"{lang}: review does not cover all source questions")
    return report, rows_by_lang


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(
    decisions_path: Path, report: dict, selected: list[dict], *, allow_uncommitted_tail: bool,
) -> tuple[list[dict], bytes]:
    """Validate the committed ordered prefix; discard at most one uncommitted append."""
    count = report.get("processed_rows")
    if not isinstance(count, int) or not 0 <= count <= len(selected):
        raise ValueError("Cannot resume: invalid committed row count")
    raw = decisions_path.read_bytes() if decisions_path.exists() else b""
    lines = raw.splitlines(keepends=True)
    if len(lines) < count or any(not line.endswith(b"\n") for line in lines[:count]):
        raise ValueError("Cannot resume: committed JSONL prefix is incomplete")
    committed = b"".join(lines[:count])
    if report.get("decisions_checkpoint_sha256") != hashlib.sha256(committed).hexdigest():
        raise ValueError("Cannot resume: committed decision checkpoint hash mismatch")
    if len(lines) - count > 1 or (lines[count:] and not allow_uncommitted_tail):
        raise ValueError("Cannot resume: more than one uncommitted decision row")

    decisions = [json.loads(line) for line in lines[:count]]
    for index, decision in enumerate(decisions):
        item = selected[index]
        row = item["row"]
        expected_key = (item["language"], row["prompt_id"], row["kind"])
        actual_key = (decision.get("language"), decision.get("prompt_id"), decision.get("kind"))
        if actual_key != expected_key:
            raise ValueError(f"Cannot resume: decision order/key mismatch at row {index}: {actual_key}")
        if (decision.get("source_en_sha256") != sha_text(row["text_en"])
                or decision.get("old_translation_sha256") != sha_text(row["text_translation"])):
            raise ValueError(f"Cannot resume: source question changed for {expected_key}")
        translated = decision.get("new_translation")
        if translated is not None and decision.get("new_translation_sha256") != sha_text(translated):
            raise ValueError(f"Cannot resume: candidate translation hash mismatch for {expected_key}")

    # A process can stop after appending one row but before atomically advancing
    # report.json. Only the report-hash-bound prefix is committed; discard that tail.
    if raw != committed:
        with decisions_path.open("wb") as stream:
            stream.write(committed)
            stream.flush()
            os.fsync(stream.fileno())
    return decisions, committed


def summarize_decisions(decisions: list[dict]) -> dict:
    counts = defaultdict(Counter)
    for item in decisions:
        key = (item["language"], item["kind"])
        counts[key]["selected"] += 1
        counts[key]["translated"] += int(item.get("new_translation") is not None)
        counts[key]["deterministic_pass"] += int(item.get("deterministic_pass") is True)
        counts[key]["qwen_pass"] += int(item.get("qwen_pass") is True)
        counts[key]["errors"] += int(item.get("status") == "DIAGNOSTIC_ERROR")
    return {
        lang: {
            kind: dict(counts[(lang, kind)])
            for kind in KINDS
        }
        for lang in LANGS
    }


def run(output_dir: Path, timeout: int, resume: bool, limit_per_cell: int | None = None) -> dict:
    source_report, rows_by_lang = source_manifest()
    from assemble_training_release import checked_translation_reviews
    final_reviews = checked_translation_reviews()
    selected = select_flagged_rows(rows_by_lang, final_reviews)
    if limit_per_cell is not None:
        counts = Counter()
        subset = []
        for item in selected:
            cell = (item["language"], item["row"]["kind"])
            if counts[cell] < limit_per_cell:
                subset.append(item)
                counts[cell] += 1
        if any(counts[(lang, kind)] != limit_per_cell for lang in LANGS for kind in KINDS):
            raise ValueError("Not enough flagged candidates for the requested stratified sample")
        selected = subset
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        models = json.load(response).get("models", [])
    model_digests = {}
    for name in (TRANSLATOR, REVIEWER_ID.removeprefix("ollama:")):
        matches = [row for row in models if row.get("name") == name]
        if len(matches) != 1 or not matches[0].get("digest"):
            raise ValueError(f"Cannot resolve a unique local model digest: {name}")
        model_digests[name] = matches[0]["digest"]
    source_report_path = REVIEW_ROOT / "report.json"
    source_hashes = {lang: file_sha(REVIEW_ROOT / f"{lang}.jsonl") for lang in LANGS}
    decisions_path = output_dir / "repair_candidates.jsonl"
    report_path = output_dir / "report.json"
    source_id = {
        "review_report_sha256": file_sha(source_report_path),
        "review_sha256_by_language": source_hashes,
        "final_review_report_sha256": file_sha(TRAINING / "translation_reviews_roundtrip_qwen14b/report.json"),
        "final_review_sha256_by_language": {
            lang: file_sha(TRAINING / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl") for lang in LANGS
        },
        "model_digests": model_digests,
        "limit_per_cell": limit_per_cell,
        "helper_scripts_sha256": {
            str(path.relative_to(V51)): file_sha(path)
            for path in (
                TRAINING / "scripts/review_training_translations_qwen.py",
                TRAINING / "scripts/assemble_training_release.py",
                V51 / "scripts/draft_eval_translations_nllb.py",
                V51 / "scripts/repair_flagged_eval_translations_translategemma.py",
            )
        },
    }
    selected_by_language = {
        lang: sum(item["language"] == lang for item in selected) for lang in LANGS
    }

    if resume:
        if not output_dir.is_dir() or not report_path.is_file():
            raise FileNotFoundError("--resume requires an existing diagnostic run and report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") not in {"IN_PROGRESS_DIAGNOSTIC_ONLY", "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"} or report.get("source") != source_id:
            raise ValueError("Cannot resume: run state or hash-bound source inputs differ")
        if report.get("script_sha256") != file_sha(SCRIPT):
            raise ValueError("Cannot resume: repair script changed since this run began")
        if (report.get("selected_rows") != len(selected)
                or report.get("selected_by_language") != selected_by_language):
            raise ValueError("Cannot resume: selected cohort size or language counts changed")
        decisions, committed_bytes = load_checkpoint(
            decisions_path, report, selected,
            allow_uncommitted_tail=report.get("status") == "IN_PROGRESS_DIAGNOSTIC_ONLY",
        )
        if report.get("per_language_kind") != summarize_decisions(decisions):
            raise ValueError("Cannot resume: report summary differs from committed decisions")
        if report.get("status") == "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW":
            if (len(decisions) != len(selected)
                    or report.get("decisions_sha256") != hashlib.sha256(committed_bytes).hexdigest()):
                raise ValueError("Cannot resume: completed diagnostic output failed integrity checks")
            print(json.dumps({"status": report["status"], "processed": report.get("processed_rows")}, indent=2))
            return report
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        decisions = []
        committed_bytes = b""
        report = {
            "status": "IN_PROGRESS_DIAGNOSTIC_ONLY",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "translator_model": TRANSLATOR_ID,
            "reviewer_model": REVIEWER_ID,
            "selection_rule": (f"first {limit_per_cell} final-review flagged rows per language/kind in stable hash order"
                               if limit_per_cell is not None else
                               "all final-review flagged training rows; stable language/kind/hash ordering"),
            "selected_rows": len(selected),
            "selected_by_language": selected_by_language,
            "source": source_id,
            "script_sha256": file_sha(SCRIPT),
            "human_translation_reviewed": False,
            "answers_sent_to_reviewer": False,
            "limits": [
                "This run creates repair candidates only; it does not modify source translations or release manifests.",
                "Qwen2.5 7B review is AI-only and does not certify fluent-human translation quality.",
                "A passing repair candidate is not automatically eligible for training; the frozen release review must be rerun.",
            ],
            "processed_rows": 0,
            "decisions_checkpoint_sha256": hashlib.sha256(b"").hexdigest(),
            "per_language_kind": summarize_decisions([]),
        }
        atomic_write_json(report_path, report)
        decisions_path.touch()

    decision_hasher = hashlib.sha256()
    decision_hasher.update(committed_bytes)
    pending = selected[len(decisions):]
    with decisions_path.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(pending, 1):
            lang, row = item["language"], item["row"]
            decision = {
                "language": lang,
                "kind": row["kind"],
                "prompt_id": row["prompt_id"],
                "source_en_sha256": sha_text(row["text_en"]),
                "old_translation_sha256": sha_text(row["text_translation"]),
                "first_stage_status": row["review_status"],
                "final_review_status": row["final_review_status"],
                "translator_model": TRANSLATOR_ID,
                "reviewer_model": REVIEWER_ID,
                "human_translation_reviewed": False,
                "answers_sent_to_reviewer": False,
                "status": "DIAGNOSTIC_ERROR",
            }
            try:
                protected, mapping, expected = protect(row["text_en"], row.get("protected_entity_names", []))
                translated = restore(translate_with_retries(protected, lang, timeout), mapping, expected)
                decision["new_translation"] = translated
                decision["new_translation_sha256"] = sha_text(translated)
                candidate = {**row, "text_translation": translated}
                checks = deterministic_checks(lang, candidate)
                decision["checks"] = checks
                passed = all(checks.get(name) is True for name in REQUIRED_CHECKS)
                decision["deterministic_pass"] = passed
                if not passed:
                    decision["status"] = "DIAGNOSTIC_FAILED_DETERMINISTIC_CHECK"
                else:
                    _, review_item = masked_pair(candidate)
                    verdict = ask_qwen(lang, [review_item], timeout)[0]
                    decision["qwen_pass"] = verdict.get("decision") is True
                    decision["qwen_reason"] = str(verdict.get("reason", ""))
                    decision["status"] = "DIAGNOSTIC_QWEN_PASS" if decision["qwen_pass"] else "DIAGNOSTIC_QWEN_FLAGGED"
            except Exception as exc:
                decision["error"] = f"{type(exc).__name__}: {exc}"
            serialized = (json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            stream.write(serialized.decode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
            decision_hasher.update(serialized)
            decisions.append(decision)
            report.update({
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "processed_rows": len(decisions),
                "decisions_checkpoint_sha256": decision_hasher.hexdigest(),
                "per_language_kind": summarize_decisions(decisions),
            })
            atomic_write_json(report_path, report)
            if index % 25 == 0 or index == len(pending):
                print(f"processed {len(decisions)}/{len(selected)} flagged candidates; latest {lang}/{row['kind']}", flush=True)

    if len(decisions) != len(selected):
        raise RuntimeError(f"Incomplete repair diagnostic: {len(decisions)}/{len(selected)} rows")
    report.update({
        "status": "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decisions_sha256": decision_hasher.hexdigest(),
        "decisions_checkpoint_sha256": decision_hasher.hexdigest(),
        "per_language_kind": summarize_decisions(decisions),
        "diagnostic_translation_successes": sum(d.get("new_translation") is not None for d in decisions),
        "deterministic_passes": sum(d.get("deterministic_pass") is True for d in decisions),
        "qwen_passes": sum(d.get("qwen_pass") is True for d in decisions),
        "errors": sum(d.get("status") == "DIAGNOSTIC_ERROR" for d in decisions),
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
    parser.add_argument("--output-dir", type=Path, default=Path("translation_repair_diagnostics/source_bound_repair_v2"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit-per-cell", type=int, help="Diagnostic sample size for each language and question kind")
    parser.add_argument("--resume", action="store_true", help="Resume the same hash-bound run after interruption")
    args = parser.parse_args()
    if args.timeout < 1 or (args.limit_per_cell is not None and args.limit_per_cell < 1):
        parser.error("--timeout and --limit-per-cell must be positive")
    output = args.output_dir if args.output_dir.is_absolute() else TRAINING / args.output_dir
    run(output, args.timeout, args.resume, args.limit_per_cell)


if __name__ == "__main__":
    main()
