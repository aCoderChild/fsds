#!/usr/bin/env python3
"""Create a separate TranslateGemma repair draft for flagged v5.1 prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from draft_eval_translations_nllb import alpha_code, LANG_CODES, RESPONSE_CONTRACT, language_script_ok, read_jsonl, sha256


MODEL = "translategemma:4b"
MODEL_ID = "ollama:translategemma:4b"
API = "http://localhost:11434/api/chat"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
DIRECT = {"direct_positive", "direct_negative_control"}
CONTRACT_RE = re.compile(r"\s*Reply with exactly YES or NO\.\s*$", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")
MARKER = re.compile(r"ZXQ[A-Z]+QXZ")
FLAG = "AI_ONLY_REVIEW_FLAGGED"
FAILED = "TRANSLATION_DRAFT_FAILED"


def protect(source: str, names: list[str]) -> tuple[str, dict[str, str], dict[str, int]]:
    values: dict[str, str] = {}
    expected: dict[str, int] = {}
    index = 0
    for name in sorted({x for x in names if isinstance(x, str) and x}, key=lambda x: (-len(x), x.casefold())):
        token = f"ZXQENT{alpha_code(index)}QXZ"
        source, count = re.subn(re.escape(name), token, source, flags=re.IGNORECASE)
        if count:
            values[token] = name
            expected[token] = count
            index += 1

    def hide_number(match: re.Match[str]) -> str:
        nonlocal index
        token = f"ZXQNUM{alpha_code(index)}QXZ"
        index += 1
        values[token] = match.group(0)
        expected[token] = 1
        return token

    return NUMBER.sub(hide_number, source), values, expected


def restore(output: str, values: dict[str, str], expected: dict[str, int]) -> str:
    found = {token: 0 for token in values}

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token not in values:
            raise ValueError(f"Unexpected protected marker: {token}")
        found[token] += 1
        return values[token]

    restored = MARKER.sub(replace, output).strip()
    errors = {token: {"expected": count, "found": found[token]}
              for token, count in expected.items() if found[token] != count}
    if errors:
        raise ValueError(f"Protected marker counts changed: {errors}")
    if MARKER.search(restored):
        raise ValueError("Unrestored protected marker remains")
    return restored


def translate(text: str, lang: str, timeout: int) -> str:
    language = LANGUAGE_NAMES[lang]
    prompt = (
        f"You are a professional English (en) to {language} ({lang}) translator. "
        f"Accurately convey the meaning and natural phrasing in {language}. "
        "Preserve every ZXQ marker exactly, including its capitalization and order; "
        "do not translate, omit, or duplicate markers. Produce only the translation, "
        f"without commentary. Translate this English text into {language}:\n\n\n{text}\n"
    )
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": 192},
        "keep_alive": "10m",
    }
    request = urllib.request.Request(
        API, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    text_out = result.get("message", {}).get("content", "")
    if not isinstance(text_out, str) or not text_out.strip():
        raise ValueError("TranslateGemma returned empty output")
    return text_out.strip()


def select_rows(drafts: list[dict], reviews: list[dict], max_per_language: int | None) -> list[dict]:
    review_by_id = {row["prompt_id"]: row for row in reviews}
    failed, flagged = [], []
    for draft in drafts:
        status = review_by_id[draft["prompt_id"]].get("review_status")
        if status == FAILED:
            failed.append(draft)
        elif status == FLAG:
            flagged.append(draft)
    if max_per_language is None:
        return failed + flagged
    return failed + flagged[:max(0, max_per_language - len(failed))]


def make_draft(request: dict, old: dict, lang: str, timeout: int) -> dict:
    text_en = request["text_en"]
    if request["probe"] in DIRECT:
        text_en, suffix_count = CONTRACT_RE.subn("", text_en)
        if suffix_count != 1:
            raise ValueError("Direct-probe YES/NO contract is missing or repeated")
    protected, values, expected = protect(text_en, request.get("protected_entity_names", []))
    output = translate(protected, lang, timeout)
    translated = restore(output, values, expected)
    if request["probe"] in DIRECT:
        translated = translated.rstrip() + " " + RESPONSE_CONTRACT[lang]
    checks = {
        "nonempty": bool(translated.strip()),
        "target_script_present": language_script_ok(lang, translated),
        "protected_names_and_numbers_restored": all(value in translated for value in values.values()),
        "yes_no_contract_preserved": request["probe"] not in DIRECT or ("YES" in translated and "NO" in translated),
        "no_marker_remains": not MARKER.search(translated),
    }
    if not all(checks.values()):
        raise ValueError(f"Deterministic repair check failed: {checks}")
    return {
        **request,
        "language": lang,
        "text_translation": translated,
        "translation_model": MODEL_ID,
        "translation_route": "TRANSLATEGEMMA_REPAIR_V1",
        "translation_status": "MACHINE_DRAFT",
        "review_status": "PENDING_INDEPENDENT_AI_REVIEW",
        "human_translation_reviewed": False,
        "repaired_from_model": old.get("translation_model"),
        "repaired_from_review_status": old.get("review_status"),
        "protected_text_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-language", type=int, help="Deterministic pilot cap; all failed rows are always included")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("translation_drafts/translategemma_repair_v1"))
    args = parser.parse_args()
    if args.max_per_language is not None and args.max_per_language <= 0:
        raise ValueError("--max-per-language must be a positive integer")
    root = Path(__file__).resolve().parents[1]
    request_path = root / "translation_requests/en.draft.jsonl"
    source_drafts = root / "translation_drafts/merged_v1"
    source_reviews = root / "translation_reviews/qwen_blinded_v1"
    output_root = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    requests = read_jsonl(request_path)
    request_by_id = {row["prompt_id"]: row for row in requests}
    if len(requests) != 585 or len(request_by_id) != 585:
        raise ValueError("Expected 585 unique, answer-free v5.1 translation requests")
    if any({"gold_label", "gold_answers", "final_answer_source_supported_en"} & set(row) for row in requests):
        raise ValueError("Translation requests must not contain gold labels")
    review_manifest = json.loads((source_reviews / "report.json").read_text(encoding="utf-8"))
    if review_manifest.get("source_requests_sha256") != sha256(request_path):
        raise ValueError("Review manifest is not bound to the current English request file")
    if review_manifest.get("unresolved_total") != 0:
        raise ValueError("Source review contains unresolved rows; repair selection is not reliable")
    report = {
        "status": "TRANSLATEGEMMA_REPAIR_DRAFT_PENDING_QWEN_REVIEW",
        "model": MODEL_ID,
        "source_requests_sha256": sha256(request_path),
        "input_translation_and_review_sha256": {},
        "max_per_language": args.max_per_language,
        "human_translation_reviewed": False,
        "repair_scope": "ALL_AI_FLAGGED_AND_TRANSLATION_FAILED_ROWS" if args.max_per_language is None else "PILOT_SUBSET_OF_FLAGGED_ROWS_PLUS_ALL_FAILED_ROWS",
        "languages": {},
    }
    for lang in LANG_CODES:
        draft_path, review_path = source_drafts / f"{lang}.jsonl", source_reviews / f"{lang}.jsonl"
        drafts, reviews = read_jsonl(draft_path), read_jsonl(review_path)
        draft_by_id = {row["prompt_id"]: row for row in drafts}
        review_by_id = {row["prompt_id"]: row for row in reviews}
        if (len(drafts) != 585 or len(draft_by_id) != 585 or len(reviews) != 585
                or len(review_by_id) != 585 or set(draft_by_id) != set(request_by_id)
                or set(review_by_id) != set(request_by_id)):
            raise ValueError(f"{lang}: source request, draft, and review IDs do not match")
        lang_manifest = review_manifest.get("languages", {}).get(lang, {})
        if lang_manifest.get("translation_sha256") != sha256(draft_path):
            raise ValueError(f"{lang}: source translation file hash differs from the review manifest")
        if lang_manifest.get("review_sha256") != sha256(review_path):
            raise ValueError(f"{lang}: source review file hash differs from the review manifest")
        for row in drafts:
            request = request_by_id[row["prompt_id"]]
            for field in ("source_qa_id", "group_id", "split", "probe", "text_en", "protected_entity_names"):
                if row.get(field) != request.get(field):
                    raise ValueError(f"{lang}/{row['prompt_id']}: source provenance mismatch for {field}")
        selected = select_rows(drafts, reviews, args.max_per_language)
        repaired, failures = {}, []
        for old in selected:
            prompt_id = old["prompt_id"]
            try:
                repaired[prompt_id] = make_draft(request_by_id[prompt_id], old, lang, args.timeout)
                repaired[prompt_id]["repaired_from_review_status"] = review_by_id[prompt_id]["review_status"]
            except Exception as exc:
                failed = {
                    **request_by_id[prompt_id], "language": lang, "text_translation": "",
                    "translation_model": MODEL_ID, "translation_route": "TRANSLATEGEMMA_REPAIR_V1",
                    "translation_status": "FAILED", "review_status": "BLOCKED",
                    "human_translation_reviewed": False,
                    "translation_error": f"{type(exc).__name__}: {exc}",
                }
                repaired[prompt_id] = failed
                failures.append({"prompt_id": prompt_id, "error": failed["translation_error"]})
            print(f"{lang}: {len(repaired)}/{len(selected)} repaired; latest={prompt_id}", flush=True)
        output_rows = [repaired.get(row["prompt_id"], row) for row in drafts]
        output_path = output_root / f"{lang}.jsonl"
        output_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
        report["input_translation_and_review_sha256"][lang] = {
            "translation": sha256(draft_path), "review": sha256(review_path),
        }
        report["languages"][lang] = {
            "flagged_or_failed_selected": len(selected),
            "translation_failures_after_repair": len(failures),
            "repaired_rows": len(selected) - len(failures),
            "repair_errors": failures,
            "output_sha256": sha256(output_path),
        }
    report["selected_prompt_ids_sha256_by_language"] = {
        lang: hashlib.sha256("\n".join(sorted(
            row["prompt_id"] for row in select_rows(
                read_jsonl(source_drafts / f"{lang}.jsonl"),
                read_jsonl(source_reviews / f"{lang}.jsonl"), args.max_per_language,
            )
        )).encode("utf-8")).hexdigest()
        for lang in LANG_CODES
    }
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_dir": str(output_root), "languages": report["languages"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
