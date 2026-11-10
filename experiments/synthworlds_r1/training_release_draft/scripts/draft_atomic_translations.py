#!/usr/bin/env python3
"""Translate missing v5.1 atomic questions; output remains a draft.

Answers are canonical source labels and are deliberately not translated.
Protected names/numbers must survive exactly; any repair or failed check is
blocked from corpus assembly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
V51 = REPO / "experiments/synthworlds_r1"
OUT = V51 / "training_release_draft"
sys.path.insert(0, str(REPO / "data/synthworlds_sm_pilot_v4/scripts"))
sys.path.insert(0, str(V51 / "scripts"))

from draft_sft_translations_nllb import LANG_CODES, MODEL_ID, translate_batch  # noqa: E402
from draft_eval_translations_nllb import language_script_ok  # noqa: E402
from repair_flagged_eval_translations_translategemma import protect  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def strict_restore(output: str, mapping: dict[str, str], expected: dict[str, int]) -> tuple[str, list[dict]]:
    found = {marker: 0 for marker in mapping}
    repairs = []

    def replace(match: re.Match[str]) -> str:
        marker = match.group(0)
        if marker not in mapping:
            raise ValueError(f"Unexpected protected marker: {marker}")
        found[marker] += 1
        return mapping[marker]

    marker_pattern = re.compile(r"ZXQ[A-Z]+QXZ")
    markers_found = marker_pattern.findall(output)
    for marker, count in expected.items():
        actual = sum(found_marker == marker for found_marker in markers_found)
        if actual < count:
            raise ValueError(f"Protected marker omitted: {marker} expected {count}, found {actual}")
        if actual > count:
            excess = actual - count
            for _ in range(excess):
                output = output.replace(marker, "", 1)
            repairs.append({"type": "duplicate_marker_removed", "marker": marker, "count": excess})
    restored = marker_pattern.sub(replace, output).strip()
    if any(found[marker] != count for marker, count in expected.items()):
        raise ValueError(f"Protected marker restoration mismatch: expected={expected}, found={found}")
    if re.search(r"ZXQ[A-Z]+QXZ", restored):
        raise ValueError("Unrestored protected marker remains")
    return restored, repairs


def run(batch_size: int = 8, only_language: str | None = None) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    requests_path = OUT / "atomic_translation_requests.jsonl"
    requests = read_jsonl(requests_path)
    if len(requests) != 592 or len({(r["sft_id"], r["language"]) for r in requests}) != 592:
        raise ValueError("Expected 592 unique atomic translation requests (148 facts x four target languages)")
    if any(r.get("language") not in LANG_CODES for r in requests):
        raise ValueError("An atomic request has an unsupported target language")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    report = {
        "status": "MACHINE_TRANSLATION_DRAFTS_PENDING_QWEN_CROSS_MODEL_SCREEN",
        "model_id": MODEL_ID, "device": device, "batch_size": batch_size,
        "source_requests_sha256": sha(requests_path), "human_translation_reviewed": False,
        "answers_translated": False, "languages": {},
    }
    langs = [only_language] if only_language else list(LANG_CODES)
    for lang in langs:
        if lang not in LANG_CODES:
            raise ValueError(f"Unsupported language {lang}")
        rows = [r for r in requests if r["language"] == lang]
        if len(rows) != 148:
            raise ValueError(f"{lang}: expected 148 atomic question requests, found {len(rows)}")
        protected = [protect(row["text_en"], row["protected_entity_names"]) for row in rows]
        outputs = translate_batch(model, tokenizer, torch, [p[0] for p in protected], LANG_CODES[lang], device, batch_size)
        translated_rows = []
        repairs = []
        failures = []
        for row, (_, mapping, counts), output in zip(rows, protected, outputs):
            try:
                try:
                    question, repair_log = strict_restore(output, mapping, counts)
                except ValueError:
                    # Retry once after a protective-marker omission. The
                    # second output must restore every exact marker.
                    retry_text = protect(row["text_en"], row["protected_entity_names"])[0]
                    retry = translate_batch(model, tokenizer, torch, [retry_text], LANG_CODES[lang], device, 1)[0]
                    question, repair_log = strict_restore(retry, mapping, counts)
                if not question:
                    raise ValueError("Translator returned empty text")
                if not all(name in question for name in row["protected_entity_names"] if name in row["text_en"]):
                    raise ValueError("A source entity name is missing after marker restoration")
                checks = {
                    "nonempty": bool(question.strip()),
                    "target_script_present": language_script_ok(lang, question),
                    "all_source_names_preserved": all(n in question for n in row["protected_entity_names"] if n in row["text_en"]),
                    "source_numbers_preserved": all(n in question for n in __import__("re").findall(r"\d+(?:[-/.]\d+)*", row["text_en"])),
                    "no_entity_marker": not re.search(r"ZXQ[A-Z]+QXZ", question),
                }
                if not all(checks.values()):
                    raise ValueError(f"Translation checks failed: {checks}")
                translated_rows.append({
                    **row, "question_translation": question,
                    "translation_model": MODEL_ID,
                    "translation_status": "MACHINE_DRAFT_PENDING_QWEN_REVIEW",
                    "review_status": "PENDING_INDEPENDENT_AI_REVIEW",
                    "protected_text_repairs": repair_log, "deterministic_checks": checks,
                    "human_translation_reviewed": False,
                })
            except Exception as exc:
                failure = {
                    **row, "question_translation": "", "translation_model": MODEL_ID,
                    "translation_status": "BLOCKED_TRANSLATION_FAILURE", "review_status": "BLOCKED",
                    "translation_error": f"{type(exc).__name__}: {exc}",
                    "human_translation_reviewed": False,
                }
                translated_rows.append(failure)
                failures.append({"sft_id": row["sft_id"], "error": failure["translation_error"]})
        output_path = OUT / f"atomic_translation_drafts/{lang}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in translated_rows), encoding="utf-8")
        report["languages"][lang] = {
            "requested": len(rows), "drafted": sum(bool(r.get("question_translation")) for r in translated_rows),
            "blocked": len(failures), "failures": failures, "output_sha256": sha(output_path),
        }
        write_json(OUT / "atomic_translation_drafts/report.json", report)
        print(f"{lang}: {report['languages'][lang]['drafted']}/{len(rows)} drafted; {len(failures)} blocked", flush=True)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(OUT / "atomic_translation_drafts/report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", choices=tuple(LANG_CODES))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    print(json.dumps(run(args.batch_size, args.language), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
