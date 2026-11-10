#!/usr/bin/env python3
"""Draft independent fallbacks for blocked atomic question translations.

The source NLLB drafts remain unchanged. Every fallback is hash-bound to its
English request and stays pending independent Qwen review.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
V51 = REPO / "experiments/synthworlds_r1"
OUT = V51 / "training_release_draft"
sys.path.insert(0, str(V51 / "scripts"))
from repair_flagged_eval_translations_translategemma import protect, restore  # noqa: E402
from draft_eval_translations_nllb import language_script_ok  # noqa: E402

MODEL = "translategemma:4b"
MODEL_ID = "ollama:translategemma:4b"
API = "http://localhost:11434/api/chat"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")
MARKER = re.compile(r"ZXQ[A-Z]+QXZ")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def translate(text: str, lang: str, timeout: int = 180) -> str:
    language = LANGUAGE_NAMES[lang]
    prompt = (
        f"You are a professional English to {language} translator. Translate the question faithfully and "
        "naturally, preserving its meaning, relation direction, scope, and answer type. Every ZXQ marker "
        "must be copied exactly once, with identical capitalization and spelling. Do not translate, omit, "
        "or duplicate markers. Return only the translated question.\n\n"
        f"{text}"
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
        raise ValueError("TranslateGemma returned empty output")
    return output.strip()


def main() -> None:
    requests_path = OUT / "atomic_translation_requests.jsonl"
    request_rows = read_jsonl(requests_path)
    request_by_id = {(r["sft_id"], r["language"]): r for r in request_rows}
    if len(request_by_id) != 592:
        raise ValueError("Expected 592 unique source-bound atomic translation requests")

    report = {
        "status": "TRANSLATEGEMMA_FALLBACKS_PENDING_QWEN_REVIEW",
        "model_id": MODEL_ID,
        "source_requests_sha256": sha_file(requests_path),
        "human_translation_reviewed": False,
        "languages": {},
    }
    output_dir = OUT / "atomic_translation_repairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for lang in LANGUAGE_NAMES:
        draft_path = OUT / f"atomic_translation_drafts/{lang}.jsonl"
        drafts = read_jsonl(draft_path)
        blocked = [r for r in drafts if r.get("translation_status") == "BLOCKED_TRANSLATION_FAILURE"]
        repaired = []
        failures = []
        for row in blocked:
            source = request_by_id.get((row["sft_id"], lang))
            if not source or source["text_en"] != row["text_en"] or source["protected_entity_names"] != row["protected_entity_names"]:
                raise ValueError(f"Source request mismatch for {row['sft_id']}/{lang}")
            try:
                protected, values, expected = protect(row["text_en"], row["protected_entity_names"])
                translated_protected = translate(protected, lang)
                translated = restore(translated_protected, values, expected)
                source_names = [n for n in row["protected_entity_names"] if n and n in row["text_en"]]
                source_numbers = NUMBER.findall(row["text_en"])
                checks = {
                    "nonempty": bool(translated.strip()),
                    "target_script_present": language_script_ok(lang, translated),
                    "source_names_preserved_with_multiplicity": all(row["text_en"].count(n) == translated.count(n) for n in source_names),
                    "source_numbers_preserved_with_multiplicity": all(source_numbers.count(n) == NUMBER.findall(translated).count(n) for n in set(source_numbers)),
                    "no_marker_remains": not MARKER.search(translated),
                }
                if not all(checks.values()):
                    raise ValueError(f"Deterministic checks failed: {checks}")
                repaired.append({
                    "sft_id": row["sft_id"], "language": lang,
                    "text_en": row["text_en"],
                    "source_text_sha256": sha256(row["text_en"]),
                    "protected_entity_names": row["protected_entity_names"],
                    "question_translation": translated,
                    "translation_model": MODEL_ID,
                    "translation_route": "TRANSLATEGEMMA_FALLBACK_FOR_BLOCKED_NLLB",
                    "translation_status": "MACHINE_DRAFT_PENDING_QWEN_REVIEW",
                    "review_status": "PENDING_INDEPENDENT_AI_REVIEW",
                    "human_translation_reviewed": False,
                    "repaired_from_model": row.get("translation_model"),
                    "repaired_from_error": row.get("translation_error"),
                    "deterministic_checks": checks,
                })
            except Exception as exc:
                failures.append({"sft_id": row["sft_id"], "error": f"{type(exc).__name__}: {exc}"})
        out_path = output_dir / f"{lang}.jsonl"
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in repaired), encoding="utf-8")
        report["languages"][lang] = {
            "blocked_inputs": len(blocked), "repaired": len(repaired), "remaining_blocked": len(failures),
            "failures": failures, "output_sha256": sha_file(out_path),
        }
        write_json(output_dir / "report.json", report)
        print(f"{lang}: {len(repaired)}/{len(blocked)} repaired; {len(failures)} remain blocked", flush=True)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "report.json", report)


if __name__ == "__main__":
    main()
