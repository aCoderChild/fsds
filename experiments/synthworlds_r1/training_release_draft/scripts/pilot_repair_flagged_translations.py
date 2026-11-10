#!/usr/bin/env python3
"""Small, diagnostic-only test of cross-model repairs for flagged training translations.

This script reads the frozen first-stage review files, translates a deterministic
sample with TranslateGemma 4B, and screens the new text with Qwen 7B. It writes
only to a separate diagnostic directory; it never changes release inputs or
approves training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
V51 = SCRIPT.parents[2]
TRAINING = V51 / "training_release_draft"
LANGS = ("fr", "de", "hi", "th")
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
MODEL = "translategemma:4b"
MODEL_ID = "ollama:translategemma:4b"
API = "http://localhost:11434/api/chat"
MARKER = re.compile(r"ZXQ[A-Z]+QXZ")

sys.path.insert(0, str(V51 / "scripts"))
sys.path.insert(0, str(TRAINING / "scripts"))
from draft_eval_translations_nllb import language_script_ok  # noqa: E402
from repair_flagged_eval_translations_translategemma import protect, restore  # noqa: E402
from review_training_translations_qwen import (  # noqa: E402
    ask_qwen,
    deterministic_checks,
    masked_pair,
    read_jsonl,
    sha,
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def translate(text: str, lang: str, timeout: int) -> str:
    target = LANGUAGE_NAMES[lang]
    prompt = (
        f"Translate this English factual question into natural {target}. Preserve the full meaning, "
        "relation direction, scope, and answer type. Copy every ZXQ marker exactly once, with "
        "identical spelling and capitalization. Do not translate, omit, or duplicate markers. "
        "Return only the translated question.\n\n" + text
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
    output = result.get("message", {}).get("content", "")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("TranslateGemma returned an empty translation")
    return output.strip()


def select_sample(rows: list[dict], lang: str, per_kind: int) -> list[dict]:
    selected = []
    for kind in ("atomic", "composition"):
        candidates = [
            row for row in rows
            if row.get("kind") == kind and row.get("review_status") == "AI_ONLY_REVIEW_FLAGGED"
        ]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"r1-training-repair-pilot-v1\0{lang}\0{row['prompt_id']}".encode("utf-8")
        ).hexdigest())
        selected.extend(candidates[:per_kind])
        if len(candidates) < per_kind:
            raise ValueError(f"{lang}: only {len(candidates)} flagged {kind} rows; need {per_kind}")
    return selected


def run(per_kind: int, timeout: int, output_dir: Path) -> dict:
    review_root = TRAINING / "translation_reviews_roundtrip_qwen"
    review_manifest_path = review_root / "report.json"
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review_manifest.get("status") != "AI_ONLY_ROUNDTRIP_TRANSLATION_SCREEN_COMPLETE":
        raise ValueError("Unexpected source review status; refusing to sample")

    output_dir.mkdir(parents=True, exist_ok=False)
    decisions = []
    report = {
        "status": "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "reviewer_model": "ollama:qwen2.5:7b-instruct",
        "sample_rule": f"{per_kind} flagged atomic and {per_kind} flagged composition rows per language; stable SHA-256 ordering",
        "sample_n": per_kind * 2 * len(LANGS),
        "per_language": {},
        "source_review_report_sha256": sha(review_manifest_path),
        "source_review_sha256_by_language": {},
        "limits": [
            "A small selected sample is calibration only and cannot approve the training corpus.",
            "Qwen review is AI-only and does not certify fluent-human translation quality.",
            "No labels, gold answers, or model predictions are included in reviewer inputs.",
            "The base translations, review records, manifest, and release state are not modified.",
        ],
    }

    for lang in LANGS:
        source_path = review_root / f"{lang}.jsonl"
        rows = read_jsonl(source_path)
        if len(rows) != 488:
            raise ValueError(f"{lang}: expected 488 hash-bound review rows, found {len(rows)}")
        if any({"gold_label", "gold_answers", "final_answer_source_supported_en"} & set(row) for row in rows):
            raise ValueError(f"{lang}: source review rows unexpectedly contain answer labels")
        sample = select_sample(rows, lang, per_kind)
        counts = {"selected": len(sample), "translation_success": 0, "deterministic_pass": 0,
                  "qwen_pass": 0, "errors": 0}
        for row in sample:
            item = {
                "language": lang,
                "kind": row["kind"],
                "prompt_id": row["prompt_id"],
                "old_translation_sha256": sha_text(row["text_translation"]),
                "old_qwen_status": row["review_status"],
                "source_en_sha256": sha_text(row["text_en"]),
                "translation_model": MODEL_ID,
                "reviewer_model": "ollama:qwen2.5:7b-instruct",
                "status": "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW",
            }
            try:
                protected, mapping, expected = protect(row["text_en"], row.get("protected_entity_names", []))
                translated = restore(translate(protected, lang, timeout), mapping, expected)
                counts["translation_success"] += 1
                candidate = {**row, "text_translation": translated}
                checks = deterministic_checks(lang, candidate)
                item["checks"] = checks
                item["new_translation"] = translated
                item["new_translation_sha256"] = sha_text(translated)
                if not all(checks[key] for key in (
                    "nonempty", "target_script_present", "protected_names_preserved",
                    "protected_numbers_preserved", "no_unrestored_marker",
                )):
                    item["status"] = "DIAGNOSTIC_FAILED_DETERMINISTIC_CHECK"
                    counts["errors"] += 1
                else:
                    counts["deterministic_pass"] += 1
                    _, review_item = masked_pair(candidate)
                    verdict = ask_qwen(lang, [review_item], timeout)[0]
                    item["qwen_pass"] = verdict["decision"]
                    item["qwen_reason"] = verdict["reason"]
                    item["status"] = "DIAGNOSTIC_QWEN_PASS" if verdict["decision"] else "DIAGNOSTIC_QWEN_FLAGGED"
                    counts["qwen_pass"] += int(verdict["decision"])
            except Exception as exc:
                item["status"] = "DIAGNOSTIC_ERROR"
                item["error"] = f"{type(exc).__name__}: {exc}"
                counts["errors"] += 1
            decisions.append(item)
            (output_dir / "sample_decisions.jsonl").write_text(
                "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in decisions),
                encoding="utf-8",
            )
        report["per_language"][lang] = counts
        report["source_review_sha256_by_language"][lang] = sha(source_path)
        report["sample_decisions_sha256"] = sha(output_dir / "sample_decisions.jsonl")
        (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{lang}: {counts}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-kind", type=int, default=2, help="Flagged rows per kind and language; default 2")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("translation_repair_diagnostics/translategemma4b_qwen7b_2026-11-03"))
    args = parser.parse_args()
    if args.per_kind < 1:
        raise ValueError("--per-kind must be positive")
    output = args.output_dir if args.output_dir.is_absolute() else TRAINING / args.output_dir
    run(args.per_kind, args.timeout, output)


if __name__ == "__main__":
    main()
