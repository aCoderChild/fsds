#!/usr/bin/env python3
"""Create quarantined v5 prompt translations with NLLB-200.

Outputs are machine-translation drafts only. Protected entities, numbers, and
the literal YES/NO response contract are restored exactly; any placeholder
loss or generation failure is recorded and blocks downstream approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "facebook/nllb-200-distilled-600M"
LANG_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")
MARKER = re.compile(r"ZXQ[A-Z]+QXZ")
RESPONSE_CONTRACT = {
    "fr": "Répondez exactement par YES ou NO.",
    "de": "Antworten Sie exakt mit YES oder NO.",
    "hi": "केवल YES या NO में उत्तर दें।",
    "th": "โปรดตอบด้วย YES หรือ NO เท่านั้น",
}
DIRECT_CONTRACT = re.compile(r"\s*Reply with exactly YES or NO\.\s*$", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alpha_code(index: int) -> str:
    letters = string.ascii_uppercase
    result = ""
    while True:
        index, remainder = divmod(index, len(letters))
        result = letters[remainder] + result
        if index == 0:
            return result
        index -= 1


def protect(text: str, names: list[str]) -> tuple[str, dict[str, str], dict[str, int]]:
    mapping: dict[str, str] = {}
    counts: dict[str, int] = {}
    for index, name in enumerate(sorted({x for x in names if isinstance(x, str) and x}, key=lambda x: (-len(x), x.casefold()))):
        token = f"ZXQENT{alpha_code(index)}QXZ"
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
        text, count = re.subn(pattern, token, text, flags=re.IGNORECASE)
        if count:
            mapping[token] = name
            counts[token] = count
    number_index = len(mapping)

    def hide_number(match: re.Match[str]) -> str:
        nonlocal number_index
        token = f"ZXQNUM{alpha_code(number_index)}QXZ"
        number_index += 1
        mapping[token] = match.group(0)
        counts[token] = 1
        return token

    text = NUMBER.sub(hide_number, text)
    return text, mapping, counts


def restore(text: str, mapping: dict[str, str], counts: dict[str, int]) -> str:
    found = {token: 0 for token in mapping}

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token not in mapping:
            raise ValueError(f"Unexpected marker in translated prompt: {token}")
        found[token] += 1
        return mapping[token]

    restored = MARKER.sub(replace, text).strip()
    errors = {
        token: {"expected": expected, "found": found[token]}
        for token, expected in counts.items()
        if found[token] != expected
    }
    if errors:
        raise ValueError(f"Protected marker counts changed: {errors}")
    if MARKER.search(restored):
        raise ValueError("Unrestored marker remains in translated prompt")
    return restored


def translate_batch(model, tokenizer, torch, texts: list[str], target_code: str, device: str, batch_size: int) -> list[str]:
    tokenizer.src_lang = "eng_Latn"
    target_id = tokenizer.convert_tokens_to_ids(target_code)
    if target_id is None or target_id < 0:
        raise ValueError(f"NLLB target language code is unavailable: {target_code}")
    outputs: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size],
            return_tensors="pt", padding=True, truncation=True, max_length=256,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                forced_bos_token_id=target_id,
                max_length=None,
                max_new_tokens=128,
                num_beams=2,
                do_sample=False,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def language_script_ok(lang: str, text: str) -> bool:
    if lang == "hi":
        return bool(re.search(r"[\u0900-\u097F]", text))
    if lang == "th":
        return bool(re.search(r"[\u0E00-\u0E7F]", text))
    return bool(re.search(r"[A-Za-zÀ-ÿ]", text))


def run(
    root: Path,
    batch_size: int = 16,
    model_id: str = MODEL_ID,
    output_dir: Path | None = None,
    prompt_ids: set[str] | None = None,
    local_files_only: bool = True,
) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    source_path = root / "translation_requests/en.draft.jsonl"
    requests = read_jsonl(source_path)
    if len(requests) != 585 or len({row["prompt_id"] for row in requests}) != len(requests):
        raise ValueError("Expected 585 unique v5 English translation requests")
    if prompt_ids is not None:
        available_ids = {row["prompt_id"] for row in requests}
        unknown_ids = prompt_ids - available_ids
        if unknown_ids:
            raise ValueError(f"Unknown prompt IDs requested for translation: {sorted(unknown_ids)[:5]}")
        requests = [row for row in requests if row["prompt_id"] in prompt_ids]
        if len(requests) != len(prompt_ids) or not requests:
            raise ValueError("Selected prompt IDs must be nonempty and unique")
    if any(row.get("do_not_train") is not True for row in requests):
        raise ValueError("Translation input lost its training quarantine")
    forbidden = {"gold_label", "gold_answers", "final_answer_source_supported_en"}
    if any(forbidden & set(row) for row in requests):
        raise ValueError("Translation request contains answer labels")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn", local_files_only=local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, dtype=dtype, local_files_only=local_files_only,
    ).to(device).eval()
    output_root = output_dir or (root / "translation_drafts/nllb_v2")
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "status": "NLLB_MACHINE_TRANSLATION_DRAFTS_PENDING_INDEPENDENT_AI_REVIEW",
        "model_id": model_id,
        "model_license": "CC-BY-NC-4.0; noncommercial research use only",
        "device": device,
        "batch_size": batch_size,
        "source_sha256": sha256(source_path),
        "translation_scope": "SELECTED_PROMPT_IDS" if prompt_ids is not None else "FULL_CORPUS",
        "rows_per_language": len(requests),
        "prompt_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(row["prompt_id"] for row in requests)) + "\n").encode("utf-8")
        ).hexdigest(),
        "human_translation_reviewed": False,
        "languages": {},
    }
    for lang, target_code in LANG_CODES.items():
        print(f"Translating {lang}: {len(requests)} prompts", flush=True)
        prepared: list[tuple[dict, str, dict[str, str], dict[str, int]]] = []
        source_texts: list[str] = []
        for row in requests:
            source_text = row["text_en"]
            if row["probe"] in {"direct_positive", "direct_negative_control"}:
                source_text, suffix_count = DIRECT_CONTRACT.subn("", source_text)
                if suffix_count != 1:
                    raise ValueError(f"Direct-probe response contract is missing or repeated: {row['prompt_id']}")
            text, mapping, counts = protect(source_text, row.get("protected_entity_names", []))
            prepared.append((row, text, mapping, counts))
            source_texts.append(text)
        translations = translate_batch(model, tokenizer, torch, source_texts, target_code, device, batch_size)
        if len(translations) != len(requests):
            raise RuntimeError(f"NLLB returned {len(translations)} outputs for {lang}; expected {len(requests)}")
        records = []
        failures = []
        for (row, _protected_source, mapping, counts), translated in zip(prepared, translations):
            try:
                restored = restore(translated, mapping, counts)
                if row["probe"] in {"direct_positive", "direct_negative_control"}:
                    restored = restored.rstrip() + " " + RESPONSE_CONTRACT[lang]
                checks = {
                    "nonempty": bool(restored.strip()),
                    "target_script_present": language_script_ok(lang, restored),
                    "response_tokens_preserved": row["probe"] not in {"direct_positive", "direct_negative_control"}
                    or ("YES" in restored and "NO" in restored),
                }
                if not all(checks.values()):
                    raise ValueError(f"Deterministic translation check failed: {checks}")
                record = {
                    **row,
                    "language": lang,
                    "text_translation": restored,
                    "translation_model": model_id,
                    "translation_status": "MACHINE_DRAFT",
                    "review_status": "PENDING_INDEPENDENT_AI_REVIEW",
                    "human_translation_reviewed": False,
                    "protected_text_checks": checks,
                }
            except Exception as exc:
                failures.append({"prompt_id": row["prompt_id"], "error": f"{type(exc).__name__}: {exc}"})
                record = {
                    **row,
                    "language": lang,
                    "text_translation": "",
                    "translation_model": model_id,
                    "translation_status": "FAILED",
                    "review_status": "BLOCKED",
                    "human_translation_reviewed": False,
                    "translation_error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)
        path = output_root / f"{lang}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        report["languages"][lang] = {
            "rows": len(records),
            "failed_rows": len(failures),
            "failures": failures,
            "draft_sha256": sha256(path),
            "status": "PENDING_INDEPENDENT_AI_REVIEW" if not failures else "BLOCKED_TRANSLATION_FAILURES",
        }
        print(
            f"Completed {lang}: {len(records)} drafts, {len(failures)} failures",
            flush=True,
        )
    report["status"] = (
        "NLLB_DRAFTS_COMPLETE_PENDING_INDEPENDENT_AI_REVIEW"
        if all(info["failed_rows"] == 0 for info in report["languages"].values())
        else "NLLB_DRAFTS_INCOMPLETE"
    )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--download-model", action="store_true", help="Allow downloading model files not present in the local cache")
    parser.add_argument("--prompt-ids-jsonl", type=Path, help="Optional JSONL file with prompt_id fields to translate")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        import sentencepiece  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install v5/requirements-translation.txt in an isolated environment first") from exc
    selected_ids = None
    if args.prompt_ids_jsonl:
        selected_ids = {row["prompt_id"] for row in read_jsonl(args.prompt_ids_jsonl)}
    print(json.dumps(
        run(
            args.root,
            args.batch_size,
            model_id=args.model_id,
            output_dir=args.output_dir,
            prompt_ids=selected_ids,
            local_files_only=not args.download_model,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
