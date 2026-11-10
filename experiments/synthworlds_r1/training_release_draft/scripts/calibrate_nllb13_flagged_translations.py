#!/usr/bin/env python3
"""Diagnostic-only, paired 64-row calibration of NLLB 1.3B repair drafts.

The script samples rows flagged by both the first-stage screen and the frozen
Qwen14B second pass, stratified by language and question kind. It compares
NLLB drafts with the already-scored TranslateGemma 4B repair for those same
rows. It never edits source translations or approves a release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
V51 = SCRIPT.parents[2]
OUT = V51 / "training_release_draft"
LANGS = ("fr", "de", "hi", "th")
KINDS = ("atomic", "composition")
LANG_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
MODEL_ID = "facebook/nllb-200-distilled-1.3B"
MODEL_REVISION = "7be3e24664b38ce1cac29b8aeed6911aa0cf0576"
CALIBRATION_ID = "nllb13_strictflagged_cal64_v1"
PER_CELL = 8
REQUIRED_CHECKS = (
    "nonempty", "target_script_present", "protected_names_preserved",
    "protected_numbers_preserved", "no_unrestored_marker",
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

sys.path.insert(0, str(V51 / "scripts"))
sys.path.insert(0, str(OUT / "scripts"))
from draft_eval_translations_nllb import language_script_ok, protect, restore  # noqa: E402
from review_training_translations_qwen import (  # noqa: E402
    ask_qwen,
    collect_rows,
    deterministic_checks,
    masked_pair,
    read_jsonl,
    sha,
    write_json,
    write_jsonl,
)


def select_sample() -> tuple[list[dict], dict]:
    initial_root = OUT / "translation_reviews_roundtrip_qwen"
    final_root = OUT / "translation_reviews_roundtrip_qwen14b"
    repair_path = OUT / "translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03/repair_candidates.jsonl"
    final_report = json.loads((final_root / "report.json").read_text(encoding="utf-8"))
    initial_report = json.loads((initial_root / "report.json").read_text(encoding="utf-8"))
    if final_report.get("status") != "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE":
        raise ValueError("The frozen Qwen14B review is incomplete")
    if final_report.get("human_translation_reviewed") is not False or final_report.get("answers_sent_to_reviewer") is not False:
        raise ValueError("The frozen review metadata is not answer-blind AI-only")
    repairs = read_jsonl(repair_path)
    repair_by_key = {(r.get("language"), r.get("prompt_id"), r.get("kind")): r for r in repairs}
    if len(repair_by_key) != len(repairs):
        raise ValueError("TranslateGemma diagnostic has duplicate repair keys")

    selected = []
    source_hashes = {
        "initial_report_sha256": sha(initial_root / "report.json"),
        "final_report_sha256": sha(final_root / "report.json"),
        "translategemma_repair_report_sha256": sha(repair_path.parent / "report.json"),
        "translategemma_repair_rows_sha256": sha(repair_path),
        "script_sha256": sha(SCRIPT),
    }
    for lang in LANGS:
        current_rows, current_hashes = collect_rows(lang)
        source_hashes[f"current_sources_{lang}"] = current_hashes
        initial_rows = read_jsonl(initial_root / f"{lang}.jsonl")
        final_rows = read_jsonl(final_root / f"{lang}.jsonl")
        initial_by_key = {(r.get("prompt_id"), r.get("kind")): r for r in initial_rows}
        final_by_key = {(r.get("prompt_id"), r.get("kind")): r for r in final_rows}
        if len(initial_by_key) != 488 or len(final_by_key) != 488 or len(current_rows) != 488:
            raise ValueError(f"{lang}: expected exactly 488 source and review rows")
        if sha(initial_root / f"{lang}.jsonl") != initial_report["languages"][lang]["review_sha256"]:
            raise ValueError(f"{lang}: first-stage review hash mismatch")
        if sha(final_root / f"{lang}.jsonl") != final_report["languages"][lang]["adjudication_sha256"]:
            raise ValueError(f"{lang}: final review hash mismatch")
        source_hashes[f"initial_rows_{lang}_sha256"] = sha(initial_root / f"{lang}.jsonl")
        source_hashes[f"final_rows_{lang}_sha256"] = sha(final_root / f"{lang}.jsonl")
        current_by_key = {(r["prompt_id"], r["kind"]): r for r in current_rows}
        for kind in KINDS:
            eligible = []
            for key, source in current_by_key.items():
                if key[1] != kind:
                    continue
                first = initial_by_key.get(key)
                final = final_by_key.get(key)
                if not first or not final:
                    raise ValueError(f"{lang}/{key}: review row missing")
                if first.get("review_status") != "AI_ONLY_REVIEW_FLAGGED":
                    continue
                if final.get("final_review_status") != "AI_ONLY_REVIEW_FLAGGED":
                    continue
                if (source.get("text_en") != first.get("text_en")
                        or source.get("text_translation") != first.get("text_translation")
                        or source.get("text_en") != final.get("text_en")
                        or source.get("text_translation") != final.get("text_translation")):
                    raise ValueError(f"{lang}/{key}: current source differs from frozen review row")
                repair = repair_by_key.get((lang, key[0], kind))
                if (not repair or repair.get("status") not in {
                        "DIAGNOSTIC_QWEN_PASS", "DIAGNOSTIC_QWEN_FLAGGED", "DIAGNOSTIC_ERROR"
                    } or not repair.get("new_translation")):
                    raise ValueError(f"{lang}/{key}: paired TranslateGemma diagnostic candidate missing")
                if repair["status"] == "DIAGNOSTIC_ERROR":
                    if repair.get("qwen_pass") is not None:
                        raise ValueError(f"{lang}/{key}: failed TranslateGemma row has an unexpected reviewer decision")
                elif type(repair.get("qwen_pass")) is not bool:
                    raise ValueError(f"{lang}/{key}: paired TranslateGemma reviewer decision is invalid")
                if (repair.get("source_en_sha256") != sha_text(source["text_en"])
                        or repair.get("old_translation_sha256") != sha_text(source["text_translation"])):
                    raise ValueError(f"{lang}/{key}: paired TranslateGemma candidate source hash mismatch")
                if {"gold_answers", "gold_label", "answer", "split"}.intersection(source):
                    raise ValueError(f"{lang}/{key}: answer or split data reached translation calibration")
                eligible.append({
                    **source,
                    "language": lang,
                    "initial_status": first["review_status"],
                    "final_status": final["final_review_status"],
                    "translategemma_candidate": repair["new_translation"],
                    "translategemma_qwen_pass": repair.get("qwen_pass") is True,
                    "translategemma_status": repair["status"],
                    "selection_key": hashlib.sha256(
                        f"{CALIBRATION_ID}\0{lang}\0{kind}\0{key[0]}".encode("utf-8")
                    ).hexdigest(),
                })
            eligible.sort(key=lambda r: (r["selection_key"], r["prompt_id"]))
            if len(eligible) < PER_CELL:
                raise ValueError(f"{lang}/{kind}: only {len(eligible)} doubly flagged rows; need {PER_CELL}")
            selected.extend(eligible[:PER_CELL])
    if len(selected) != 64 or len({(r["language"], r["prompt_id"], r["kind"]) for r in selected}) != 64:
        raise ValueError("Calibration sample is not exactly 64 unique rows")
    return selected, source_hashes


def translate_sample(rows: list[dict], batch_size: int) -> tuple[list[str], dict]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    # Keep fp32 for consistent comparison with the 600M NLLB draft route.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, src_lang="eng_Latn")
    by_index: dict[int, dict] = {}
    batch_metrics = {"device": device, "dtype": "float32", "batch_size": batch_size, "num_beams": 2, "do_sample": False}
    for lang in LANGS:
        indices = [i for i, row in enumerate(rows) if row["language"] == lang]
        protected = []
        for i in indices:
            text, mapping, counts = protect(rows[i]["text_en"], rows[i].get("protected_entity_names", []))
            protected.append((i, text, mapping, counts))
        tokenizer.src_lang = "eng_Latn"
        target_id = tokenizer.convert_tokens_to_ids(LANG_CODES[lang])
        if target_id is None or target_id == tokenizer.unk_token_id:
            raise ValueError(f"NLLB target-language token unavailable: {LANG_CODES[lang]}")
        for start in range(0, len(protected), batch_size):
            chunk = protected[start:start + batch_size]
            encoded = tokenizer([item[1] for item in chunk], return_tensors="pt", padding=True, truncation=True, max_length=256)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(**encoded, forced_bos_token_id=target_id, max_length=None, max_new_tokens=128,
                                           num_beams=2, do_sample=False)
            outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
            if len(outputs) != len(chunk):
                raise ValueError(f"{lang}: NLLB output count differs from input count")
            for (index, _text, mapping, counts), output in zip(chunk, outputs):
                try:
                    by_index[index] = {"text": restore(output, mapping, counts), "error": None}
                except ValueError as exc:
                    by_index[index] = {"text": "", "error": f"{type(exc).__name__}: {exc}"}
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if len(by_index) != len(rows):
        raise ValueError(f"Only translated {len(by_index)}/{len(rows)} calibration rows")
    return [by_index[i] for i in range(len(rows))], batch_metrics


def review_with_fallback(lang: str, items: list[dict], batch_size: int = 4) -> list[dict]:
    """Use small batches; retry malformed batches item-by-item and fail closed."""
    decisions = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        try:
            batch_decisions = ask_qwen(lang, batch, timeout=240)
            if len(batch_decisions) != len(batch):
                raise ValueError("Qwen batch decision count mismatch")
            decisions.extend({**item, "review_error": None} for item in batch_decisions)
            continue
        except Exception as batch_error:
            batch_error_text = f"{type(batch_error).__name__}: {batch_error}"
        for item in batch:
            try:
                single = ask_qwen(lang, [item], timeout=240)
                if len(single) != 1:
                    raise ValueError("Qwen singleton decision count mismatch")
                decisions.append({**single[0], "review_error": None, "batch_retry_reason": batch_error_text})
            except Exception as single_error:
                decisions.append({
                    "decision": False,
                    "reason": "No decision: Qwen returned an invalid or incomplete response.",
                    "review_error": f"{type(single_error).__name__}: {single_error}",
                    "batch_retry_reason": batch_error_text,
                })
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate sample/provenance without loading weights")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=OUT / "translation_repair_diagnostics/nllb13_strictflagged_cal64_2026-11-03")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    rows, source_hashes = select_sample()
    sample_counts = {lang: {kind: sum(r["language"] == lang and r["kind"] == kind for r in rows) for kind in KINDS} for lang in LANGS}
    if args.dry_run:
        print(json.dumps({"status": "PREFLIGHT_PASS_NO_MODEL_LOADED", "sample_rows": len(rows),
                          "sample_counts": sample_counts, "model_id": MODEL_ID,
                          "model_revision": MODEL_REVISION, "source_hashes": source_hashes}, indent=2))
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output_dir}")

    translations, generation_config = translate_sample(rows, args.batch_size)
    output_rows = []
    for row, translation_result in zip(rows, translations):
        translated = translation_result["text"]
        translation_error = translation_result["error"]
        candidate = {**row, "text_translation": translated}
        checks = deterministic_checks(row["language"], candidate) if not translation_error else {}
        deterministic_pass = not translation_error and all(checks[key] for key in REQUIRED_CHECKS)
        output_rows.append({
            "language": row["language"], "prompt_id": row["prompt_id"], "kind": row["kind"],
            "text_en": row["text_en"], "text_translation": translated,
            "translation_error": translation_error,
            "initial_status": row["initial_status"], "final_status": row["final_status"],
            "source_en_sha256": sha_text(row["text_en"]),
            "base_translation_sha256": sha_text(row["text_translation"]),
            "nllb_translation_sha256": sha_text(translated) if translated else None,
            "translation_model": MODEL_ID, "model_revision": MODEL_REVISION,
            "deterministic_checks": checks, "deterministic_pass": deterministic_pass,
            "translategemma_qwen_pass": row["translategemma_qwen_pass"],
            "translategemma_status": row["translategemma_status"],
            "human_translation_reviewed": False, "answers_sent_to_translator": False,
            "answers_sent_to_reviewer": False,
            "status": ("PENDING_QWEN_REVIEW" if deterministic_pass else
                       ("DIAGNOSTIC_TRANSLATION_ERROR" if translation_error else "DIAGNOSTIC_DETERMINISTIC_FAIL")),
        })

    for lang in LANGS:
        lang_rows = [r for r in output_rows if r["language"] == lang and r["deterministic_pass"]]
        if not lang_rows:
            continue
        review_items = []
        for row in lang_rows:
            source = next(r for r in rows if r["language"] == lang and r["prompt_id"] == row["prompt_id"] and r["kind"] == row["kind"])
            review_items.append(masked_pair({**source, "text_translation": row["text_translation"]})[1])
        decisions = review_with_fallback(lang, review_items)
        if len(decisions) != len(lang_rows):
            raise ValueError(f"{lang}: Qwen decision count mismatch")
        for row, decision in zip(lang_rows, decisions):
            row["nllb_qwen_pass"] = decision["decision"] is True and decision["review_error"] is None
            row["nllb_qwen_reason"] = decision["reason"]
            row["nllb_qwen_review_error"] = decision["review_error"]
            row["status"] = ("DIAGNOSTIC_QWEN_PASS" if row["nllb_qwen_pass"] else
                              ("DIAGNOSTIC_REVIEW_ERROR" if decision["review_error"] else "DIAGNOSTIC_QWEN_FLAGGED"))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "candidates.jsonl"
    write_jsonl(rows_path, output_rows)
    per_cell = {}
    for lang in LANGS:
        per_cell[lang] = {}
        for kind in KINDS:
            cell = [r for r in output_rows if r["language"] == lang and r["kind"] == kind]
            per_cell[lang][kind] = {
                "selected": len(cell),
                "translation_success": sum(bool(r["text_translation"]) for r in cell),
                "deterministic_pass": sum(r["deterministic_pass"] for r in cell),
                "nllb_qwen_pass": sum(r.get("nllb_qwen_pass") is True for r in cell),
                "nllb_qwen_review_errors": sum(r.get("status") == "DIAGNOSTIC_REVIEW_ERROR" for r in cell),
                "paired_translategemma_qwen_pass": sum(r["translategemma_qwen_pass"] is True for r in cell),
                "paired_translategemma_qwen_fail": sum(r["translategemma_qwen_pass"] is False for r in cell),
                "paired_translategemma_diagnostic_error": sum(r["translategemma_status"] == "DIAGNOSTIC_ERROR" for r in cell),
            }
    status = "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
    report = {
        "status": status, "calibration_id": CALIBRATION_ID,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "reviewer_model": "ollama:qwen2.5:7b-instruct",
        "sample_rule": "8 rows per language x kind, flagged by both frozen review stages; stable SHA-256 order",
        "sample_size": len(output_rows), "sample_counts": sample_counts,
        "generation_config": generation_config,
        "review_config": {"model": "ollama:qwen2.5:7b-instruct", "batch_size": 4,
                          "malformed_batch_fallback": "retry each item individually; unresolved review errors count as failures"},
        "per_language_kind": per_cell,
        "answers_sent_to_translator": False, "answers_sent_to_reviewer": False,
        "human_translation_reviewed": False, "do_not_train": True,
        "source_hashes": source_hashes,
        "candidate_rows_sha256": sha(rows_path),
        "limits": [
            "This is a stratified calibration, not a representative estimate or release review.",
            "The Qwen reviewer is AI-only and does not certify human fluency.",
            "Passing results require full, hash-bound, blinded translation screening before release.",
            "No source translation, release manifest, or training artifact was modified.",
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
