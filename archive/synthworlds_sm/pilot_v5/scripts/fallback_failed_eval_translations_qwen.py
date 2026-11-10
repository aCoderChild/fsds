#!/usr/bin/env python3
"""Translate only NLLB marker-failure rows with local Qwen; keep failures blocked."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from draft_eval_translations_nllb import (
    LANG_CODES,
    RESPONSE_CONTRACT,
    DIRECT_CONTRACT,
    NUMBER,
    language_script_ok,
    protect,
    read_jsonl,
    restore,
    sha256,
)


MODEL = "qwen2.5:7b-instruct"
MODEL_ID = "ollama:qwen2.5:7b-instruct"
API = "http://localhost:11434/api/chat"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}

SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "issue": {"type": "string"},
    },
    "required": ["translation", "confidence", "issue"],
    "additionalProperties": False,
}


def source_for_translation(row: dict) -> str:
    source = row["text_en"]
    if row["probe"] in {"direct_positive", "direct_negative_control"}:
        source, count = DIRECT_CONTRACT.subn("", source)
        if count != 1:
            raise ValueError(f"Missing or duplicate English YES/NO contract: {row['prompt_id']}")
    return source


def saved_fallback_is_valid(saved: dict, request_row: dict, lang: str) -> bool:
    fields = ("prompt_id", "source_qa_id", "group_id", "split", "probe", "text_en", "protected_entity_names")
    if any(saved.get(key) != request_row.get(key) for key in fields):
        return False
    text = saved.get("text_translation", "")
    if not text or saved.get("translation_status") != "MACHINE_DRAFT":
        return False
    if not language_script_ok(lang, text):
        return False
    if request_row["probe"] in {"direct_positive", "direct_negative_control"} and not ("YES" in text and "NO" in text):
        return False
    if any(name in request_row["text_en"] and name not in text for name in request_row.get("protected_entity_names", [])):
        return False
    if any(number not in text for number in NUMBER.findall(request_row["text_en"])):
        return False
    return not re.search(r"ZXQ[A-Z]+QXZ", text)


def name_aware_marker_variant(source: str, mapping: dict, counts: dict):
    variant_source = source
    variant_mapping = {}
    variant_counts = {}
    used = set()
    for token, value in mapping.items():
        if token.startswith("ZXQENT"):
            name_part = re.sub(r"[^A-Za-z]", "", value.split()[0]).upper()[:12] or "ENTITY"
            replacement = f"ZXQ{name_part}QXZ"
            suffix = 0
            while replacement in used:
                suffix += 1
                replacement = f"ZXQ{name_part}{chr(64 + suffix)}QXZ"
            used.add(replacement)
        else:
            replacement = token
        variant_source = variant_source.replace(token, replacement)
        variant_mapping[replacement] = value
        variant_counts[replacement] = counts[token]
    return variant_source, variant_mapping, variant_counts


def restore_named_or_marked(text: str, mapping: dict, counts: dict) -> str:
    for token, value in mapping.items():
        token_count = text.count(token)
        if token_count == counts[token]:
            text = text.replace(token, value)
            continue
        if token_count == 0:
            pattern = rf"(?<!\w){re.escape(value)}(?!\w)"
            value_count = len(re.findall(pattern, text, flags=re.IGNORECASE))
            if value_count >= counts[token]:
                continue
        raise ValueError(
            f"Named marker or exact source string count mismatch for {token}: "
            f"expected={counts[token]}, markers={token_count}"
        )
    if re.search(r"ZXQ[A-Z]+QXZ", text):
        raise ValueError("Unrestored named placeholder remains")
    return text.strip()


def call_qwen(lang: str, row: dict, protected_source: str, timeout: int) -> dict:
    payload = {
        "model": MODEL,
        "stream": False,
        "format": SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Translate the English text into {LANGUAGE_NAMES[lang]}. Preserve its meaning, "
                    "question scope, requested answer type, and all placeholder strings exactly as written. "
                    "Do not answer the question. Do not add explanations or quotation marks around the full text. "
                    "Return the translation only in the required JSON field. Be conservative: if the source is "
                    "unclear or you cannot preserve a placeholder, report low confidence and explain briefly."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "english_source": protected_source,
                        "protected_entity_names": row.get("protected_entity_names", []),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "options": {"temperature": 0, "num_predict": 240},
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
    content = result.get("message", {}).get("content", "")
    parsed = json.loads(content)
    if not isinstance(parsed.get("translation"), str):
        raise ValueError("Qwen fallback response lacks translation text")
    return parsed


def run(root: Path, timeout: int = 180) -> dict:
    request_path = root / "translation_requests/en.draft.jsonl"
    requests = read_jsonl(request_path)
    request_by_id = {row["prompt_id"]: row for row in requests}
    input_root = root / "translation_drafts/nllb_v2"
    output_root = root / "translation_drafts/merged"
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "QWEN_FALLBACK_IN_PROGRESS",
        "fallback_model": MODEL_ID,
        "primary_translation_model": "facebook/nllb-200-distilled-600M",
        "translation_requests_sha256": sha256(request_path),
        "human_translation_reviewed": False,
        "languages": {},
    }
    total_failed = 0
    prior_report_path = output_root / "report.json"
    prior_report = json.loads(prior_report_path.read_text(encoding="utf-8")) if prior_report_path.is_file() else {}
    can_resume = prior_report.get("translation_requests_sha256") == sha256(request_path)
    for lang in LANG_CODES:
        nllb_rows = read_jsonl(input_root / f"{lang}.jsonl")
        if len(nllb_rows) != len(requests):
            raise ValueError(f"NLLB request coverage mismatch for {lang}")
        merged = []
        fallback_count = 0
        errors = []
        existing_path = output_root / f"{lang}.jsonl"
        existing_by_id = {row["prompt_id"]: row for row in read_jsonl(existing_path)} if can_resume and existing_path.is_file() else {}
        for nllb_row in nllb_rows:
            prompt_id = nllb_row["prompt_id"]
            request_row = request_by_id[prompt_id]
            if any(nllb_row.get(key) != request_row.get(key) for key in (
                "prompt_id", "source_qa_id", "group_id", "split", "probe", "text_en", "protected_entity_names",
            )):
                raise ValueError(f"NLLB draft provenance mismatch for {lang}/{prompt_id}")
            if nllb_row.get("translation_status") == "MACHINE_DRAFT":
                merged.append({**nllb_row, "translation_route": "NLLB_PRIMARY"})
                continue
            existing = existing_by_id.get(prompt_id)
            if existing and existing.get("translation_route") == "QWEN_FALLBACK_AFTER_NLLB_MARKER_FAILURE" and saved_fallback_is_valid(existing, request_row, lang):
                merged.append(existing)
                fallback_count += 1
                continue
            source = source_for_translation(request_row)
            protected, mapping, counts = protect(source, request_row.get("protected_entity_names", []))
            last_error = None
            recovered = None
            variants = [(protected, mapping, counts, "ZXQENT/NUM")]
            alternate_mapping = {}
            alternate_counts = {}
            alternate_source = protected
            for token, value in mapping.items():
                alternate_token = token.replace("ZXQENT", "ZXQALT").replace("ZXQNUM", "ZXQALT")
                alternate_source = alternate_source.replace(token, alternate_token)
                alternate_mapping[alternate_token] = value
                alternate_counts[alternate_token] = counts[token]
            if alternate_mapping:
                variants.append((alternate_source, alternate_mapping, alternate_counts, "ZXQALT"))
            named_source, named_mapping, named_counts = name_aware_marker_variant(protected, mapping, counts)
            if named_source != protected:
                variants.append((named_source, named_mapping, named_counts, "NAME_AWARE"))
            for variant_index, (protected_variant, mapping_variant, counts_variant, marker_scheme) in enumerate(variants):
                try:
                    response = call_qwen(lang, request_row, protected_variant, timeout)
                    if marker_scheme == "NAME_AWARE":
                        text = restore_named_or_marked(response["translation"], mapping_variant, counts_variant)
                    else:
                        text = restore(response["translation"], mapping_variant, counts_variant)
                    if request_row["probe"] in {"direct_positive", "direct_negative_control"}:
                        text = text.rstrip() + " " + RESPONSE_CONTRACT[lang]
                    checks = {
                        "nonempty": bool(text.strip()),
                        "target_script_present": language_script_ok(lang, text),
                        "yes_no_contract_preserved": request_row["probe"] not in {"direct_positive", "direct_negative_control"}
                        or ("YES" in text and "NO" in text),
                        "all_protected_names_present": all(
                            name not in source or name in text
                            for name in request_row.get("protected_entity_names", [])
                        ),
                    }
                    if not all(checks.values()):
                        raise ValueError(f"Fallback deterministic checks failed: {checks}")
                    recovered = {
                        **request_row,
                        "language": lang,
                        "text_translation": text,
                        "translation_model": MODEL_ID,
                        "translation_route": "QWEN_FALLBACK_AFTER_NLLB_MARKER_FAILURE",
                        "translation_status": "MACHINE_DRAFT",
                        "review_status": "PENDING_INDEPENDENT_AI_REVIEW",
                        "human_translation_reviewed": False,
                        "fallback_confidence": response["confidence"],
                        "fallback_model_issue": response["issue"],
                        "fallback_marker_scheme": marker_scheme,
                        "fallback_note": "Qwen self-report is retained for audit; deterministic marker checks and a separate review pass control use.",
                        "fallback_protected_checks": checks,
                    }
                    break
                except (
                    ValueError, json.JSONDecodeError, urllib.error.URLError, TimeoutError,
                ) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if variant_index + 1 < len(variants):
                        time.sleep(1)
            if recovered is None:
                errors.append({"prompt_id": prompt_id, "error": last_error})
                merged.append({
                    **request_row,
                    "language": lang,
                    "text_translation": "",
                    "translation_model": MODEL_ID,
                    "translation_route": "QWEN_FALLBACK_FAILED",
                    "translation_status": "FAILED",
                    "review_status": "BLOCKED",
                    "human_translation_reviewed": False,
                    "translation_error": last_error,
                })
            else:
                fallback_count += 1
                merged.append(recovered)
        output_path = output_root / f"{lang}.jsonl"
        output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged),
            encoding="utf-8",
        )
        fails = len(errors)
        total_failed += fails
        report["languages"][lang] = {
            "rows": len(merged),
            "nllb_primary_rows": len(merged) - fallback_count - fails,
            "qwen_fallback_rows": fallback_count,
            "unresolved_rows": fails,
            "failures": errors,
            "merged_sha256": sha256(output_path),
        }
        report_path = output_root / "report.json"
        report["status"] = "QWEN_FALLBACK_IN_PROGRESS"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{lang}: {len(merged)-fails} ready, {fallback_count} Qwen fallbacks, {fails} unresolved", flush=True)
    report["status"] = "MERGED_TRANSLATIONS_READY_FOR_AI_REVIEW" if total_failed == 0 else "MERGED_TRANSLATIONS_BLOCKED"
    report["human_reviewed_total"] = 0
    report["unresolved_total"] = total_failed
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    print(json.dumps(run(args.root, args.timeout), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
