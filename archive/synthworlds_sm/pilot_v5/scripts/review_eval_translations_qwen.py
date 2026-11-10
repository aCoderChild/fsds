#!/usr/bin/env python3
"""Blind entity names and probe labels to screen v5 translations with local Qwen.

Exact masked source/translation pairs share one decision across the full
language file and across resumed runs, avoiding batch-position inconsistencies.
"""
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

from draft_eval_translations_nllb import LANG_CODES, language_script_ok, read_jsonl, sha256


MODEL = "qwen2.5:7b-instruct"
MODEL_ID = "ollama:qwen2.5:7b-instruct"
API = "http://localhost:11434/api/chat"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")

SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {"type": "boolean"},
        },
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def blinded_review_item(row: dict) -> tuple[str, dict]:
    source = row["text_en"]
    target = row["text_translation"]
    names = [
        name for name in row.get("protected_entity_names", [])
        if name and name in source
    ]
    names = sorted(set(names), key=lambda name: source.index(name))
    replacements = {name: f"[ENTITY_{index + 1}]" for index, name in enumerate(names)}
    for name in sorted(names, key=len, reverse=True):
        source = source.replace(name, replacements[name])
        target = target.replace(name, replacements[name])
    probe_type = (
        "yes_no_question"
        if row["probe"] in {"direct_positive", "direct_negative_control"}
        else "multi_hop_question"
    )
    item = {
        "probe_type": probe_type,
        "source_english": source,
        "target_translation": target,
        "entity_placeholders_used": len(names),
    }
    key = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return key, item


def call_qwen(
    lang: str,
    batch: list[dict],
    timeout: int,
    decision_cache: dict[str, bool] | None = None,
) -> list[dict]:
    if decision_cache is None:
        decision_cache = {}
    unique_items: dict[str, dict] = {}
    key_by_prompt: dict[str, str] = {}
    for row in batch:
        key, item = blinded_review_item(row)
        key_by_prompt[row["prompt_id"]] = key
        if key not in decision_cache:
            unique_items.setdefault(key, item)
    review_items = list(unique_items.values())
    if review_items:
        payload = {
            "model": MODEL,
            "stream": False,
            "format": SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a translation-quality auditor. These are synthetic questions. Judge only "
                        "whether the target text faithfully and naturally translates the English source. "
                        "Entity names have been replaced in both texts by [ENTITY_n] placeholders so you "
                        "cannot answer the factual questions. Both direct probe variants are labeled only "
                        "as yes_no_question; do not infer whether a candidate answer is factually correct. "
                        "Approve only if meaning, relation direction, scope, answer type, grammar, and "
                        "naturalness are all preserved. For yes/no questions, retain the literal English "
                        "YES and NO response tokens. A placeholder mismatch or any semantic ambiguity is "
                        "a failure. Return one boolean decision per item in the same order; use false for "
                        "any concern. Return only the required JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "target_language": LANGUAGE_NAMES[lang],
                            "items": review_items,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "options": {"temperature": 0, "num_predict": max(64, len(review_items) * 4)},
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
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Qwen returned malformed JSON: {content[:300]!r}") from exc
        decisions = parsed.get("decisions")
        if (
            not isinstance(decisions, list)
            or len(decisions) != len(review_items)
            or any(not isinstance(value, bool) for value in decisions)
        ):
            raise ValueError(f"Qwen review response has invalid decision coverage: {parsed!r}")
        decision_cache.update(zip(unique_items, decisions))
    return [
        {
            "prompt_id": row["prompt_id"],
            "pass": decision_cache[key_by_prompt[row["prompt_id"]]],
            "review_pair_sha256": hashlib.sha256(key_by_prompt[row["prompt_id"]].encode("utf-8")).hexdigest(),
        }
        for row in batch
    ]


def deterministic_checks(lang: str, request: dict, draft: dict) -> dict:
    text = draft.get("text_translation", "")
    source_text = request["text_en"]
    names = request.get("protected_entity_names", [])
    missing_names = [name for name in names if name in source_text and name not in text]
    source_numbers = NUMBER.findall(source_text)
    missing_numbers = [number for number in source_numbers if number not in text]
    response_tokens_ok = (
        request["probe"] not in {"direct_positive", "direct_negative_control"}
        or ("YES" in text and "NO" in text)
    )
    return {
        "nonempty": bool(text.strip()),
        "target_script_present": language_script_ok(lang, text),
        "protected_names_preserved": not missing_names,
        "protected_numbers_preserved": not missing_numbers,
        "yes_no_contract_preserved": response_tokens_ok,
        "no_unrestored_marker": not re.search(r"ZXQ[A-Z]+QXZ", text),
        "missing_names": missing_names,
        "missing_numbers": missing_numbers,
    }


def independence_counts(drafts: list[dict], review_rows: list[dict]) -> dict:
    draft_by_id = {row["prompt_id"]: row for row in drafts}
    same_model = sum(
        draft_by_id[row["prompt_id"]].get("translation_model") == MODEL_ID
        for row in review_rows
    )
    return {
        "cross_model_review_rows": len(review_rows) - same_model,
        "same_model_self_check_rows": same_model,
    }


def run(
    root: Path,
    batch_size: int = 12,
    timeout: int = 180,
    max_batches: int | None = None,
    draft_dir: Path | None = None,
    review_dir: Path | None = None,
) -> dict:
    request_path = root / "translation_requests/en.draft.jsonl"
    requests = read_jsonl(request_path)
    request_by_id = {row["prompt_id"]: row for row in requests}
    # Review the merged set: the NLLB-only drafts contain marker failures
    # that were replaced by Qwen fallback translations.
    draft_root = draft_dir or (root / "translation_drafts/merged")
    review_root = review_dir or (root / "translation_reviews/qwen_blinded_v4")
    review_root.mkdir(parents=True, exist_ok=True)
    report_path = review_root / "report.json"
    previous_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report = {
        **previous_report,
        "status": "AI_ONLY_TRANSLATION_REVIEW_IN_PROGRESS",
        "reviewer_model": MODEL_ID,
        "review_protocol_id": "blind_entity_probe_neutral_global_deduplicated_v4",
        "review_mode": "AI_ONLY_BLINDED_TRANSLATION_SCREEN_ENTITY_NAMES_AND_DIRECT_LABELS_MASKED_GLOBAL_DEDUPLICATED",
        "deduplication_scope": "exact masked source/translation payload, across each complete language file and resumed runs",
        "human_translation_reviewed": False,
        "translation_model": "PER_ROW; see translation_model on each draft",
        "source_requests_sha256": sha256(request_path),
        "batch_size": batch_size,
        "languages": previous_report.get("languages", {}),
        "limitations": [
            "AI-only review is not certification by a fluent human reviewer.",
            "Automated model judgments can miss semantic or pragmatic translation errors.",
            "Entity names and direct-positive versus negative-control labels are masked from the reviewer to reduce factual-answer bias.",
            "Qwen fallback translations were screened by the same Qwen model; those rows are consistency checks, not independent-model verification.",
            "The evaluation prompts and translated text are synthetic and source-derived.",
        ],
    }
    batches_processed = 0
    for lang in LANG_CODES:
        draft_path = draft_root / f"{lang}.jsonl"
        if not draft_path.is_file():
            raise FileNotFoundError(f"Missing NLLB draft file: {draft_path}")
        drafts = read_jsonl(draft_path)
        if len(drafts) != len(requests) or len({row.get("prompt_id") for row in drafts}) != len(drafts):
            raise ValueError(f"Incomplete or duplicate NLLB draft coverage for {lang}")
        if any(row.get("translation_status") != "MACHINE_DRAFT" for row in drafts):
            raise ValueError(f"NLLB file contains failed or non-draft rows for {lang}")
        if any(row.get("language") != lang for row in drafts):
            raise ValueError(f"Language labels are inconsistent for {lang}")

        review_path = review_root / f"{lang}.jsonl"
        prior_language = report["languages"].get(lang, {})
        can_resume = (
            prior_language.get("translation_sha256") == sha256(draft_path)
            and review_path.is_file()
        )
        reviews_by_id = {
            row["prompt_id"]: row for row in read_jsonl(review_path)
        } if can_resume else {}
        draft_by_id = {row["prompt_id"]: row for row in drafts}
        expected_ids = {row["prompt_id"] for row in drafts}
        if not set(reviews_by_id).issubset(expected_ids):
            raise ValueError(f"Saved review contains unexpected prompt IDs for {lang}")
        decision_cache: dict[str, bool] = {}
        for prompt_id, review in reviews_by_id.items():
            key, _ = blinded_review_item({**request_by_id[prompt_id], **draft_by_id[prompt_id]})
            pair_sha256 = hashlib.sha256(key.encode("utf-8")).hexdigest()
            if review.get("review_pair_sha256") != pair_sha256:
                raise ValueError(f"Saved masked pair hash does not match current input: {prompt_id}")
            verdict = review.get("independent_model_checks", {}).get("blinded_translation_model_pass")
            if not isinstance(verdict, bool):
                raise ValueError(f"Saved review lacks a boolean model decision: {prompt_id}")
            if key in decision_cache and decision_cache[key] != verdict:
                raise ValueError(f"Prior review has inconsistent decisions for identical masked text: {prompt_id}")
            decision_cache[key] = verdict
        model_failures = list(prior_language.get("model_call_failures", [])) if can_resume else []
        for start in range(0, len(drafts), batch_size):
            batch = [
                row for row in drafts[start:start + batch_size]
                if row["prompt_id"] not in reviews_by_id
            ]
            if not batch:
                continue
            if max_batches is not None and batches_processed >= max_batches:
                break
            raw_review = None
            last_error = None
            for attempt in range(2):
                try:
                    raw_review = call_qwen(lang, batch, timeout, decision_cache)
                    break
                except (ValueError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt == 0:
                        time.sleep(1)
            if raw_review is None:
                for row in batch:
                    model_failures.append({"prompt_id": row["prompt_id"], "error": last_error})
                batches_processed += 1
                continue
            raw_by_id = {row["prompt_id"]: row for row in raw_review}
            for draft in batch:
                request = request_by_id[draft["prompt_id"]]
                model_row = raw_by_id[draft["prompt_id"]]
                checks = deterministic_checks(lang, request, draft)
                ai_checks = {
                    "blinded_translation_model_pass": model_row["pass"],
                }
                passed = all(
                    value for key, value in checks.items()
                    if key not in {"missing_names", "missing_numbers"}
                ) and all(ai_checks.values())
                reviews_by_id[draft["prompt_id"]] = {
                    "prompt_id": draft["prompt_id"],
                    "source_qa_id": draft["source_qa_id"],
                    "group_id": draft["group_id"],
                    "split": draft["split"],
                    "probe": draft["probe"],
                    "language": lang,
                    "translation_model": draft.get("translation_model"),
                    "translation_route": draft.get("translation_route"),
                    "review_independence": (
                        "SAME_MODEL_SELF_CHECK"
                        if draft.get("translation_model") == MODEL_ID
                        else "CROSS_MODEL_SCREEN"
                    ),
                    "review_status": "AI_ONLY_REVIEW_PASS" if passed else "AI_ONLY_REVIEW_FLAGGED",
                    "deterministic_checks": checks,
                    "independent_model_checks": ai_checks,
                    "model_issue": "none" if model_row["pass"] else "AI_REVIEW_FLAGGED",
                    "reviewer_model": MODEL_ID,
                    "review_protocol_id": "blind_entity_probe_neutral_global_deduplicated_v4",
                    "review_pair_sha256": model_row["review_pair_sha256"],
                    "human_translation_reviewed": False,
                }
            batches_processed += 1
            model_failures = [
                failure for failure in model_failures
                if failure.get("prompt_id") not in reviews_by_id
            ]
            # Save each successful batch so interruption or a slow model call is resumable.
            review_rows = [reviews_by_id[row["prompt_id"]] for row in drafts if row["prompt_id"] in reviews_by_id]
            review_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in review_rows),
                encoding="utf-8",
            )
            passes = sum(row["review_status"] == "AI_ONLY_REVIEW_PASS" for row in review_rows)
            flagged = sum(row["review_status"] == "AI_ONLY_REVIEW_FLAGGED" for row in review_rows)
            report["languages"][lang] = {
                "input_rows": len(drafts), "reviewed_rows": len(review_rows),
                "pass_rows": passes, "flagged_rows": flagged,
                "unresolved_rows": len(drafts) - len(review_rows),
                "model_call_failures": model_failures,
                **independence_counts(drafts, review_rows),
                "translation_sha256": sha256(draft_path), "review_sha256": sha256(review_path),
                "status": "AI_ONLY_REVIEW_IN_PROGRESS",
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"{lang}: checkpoint {len(review_rows)}/{len(drafts)} reviewed "
                f"({passes} pass, {flagged} flagged, {len(drafts) - len(review_rows)} unresolved)",
                flush=True,
            )

        review_rows = [reviews_by_id[row["prompt_id"]] for row in drafts if row["prompt_id"] in reviews_by_id]
        review_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in review_rows),
            encoding="utf-8",
        )
        passes = sum(row["review_status"] == "AI_ONLY_REVIEW_PASS" for row in review_rows)
        flagged = sum(row["review_status"] == "AI_ONLY_REVIEW_FLAGGED" for row in review_rows)
        unresolved = len(drafts) - len(review_rows)
        if len(review_rows) == len(drafts):
            lang_status = "AI_ONLY_REVIEW_COMPLETE"
        else:
            lang_status = "AI_ONLY_REVIEW_INCOMPLETE"
        report["languages"][lang] = {
            "input_rows": len(drafts),
            "reviewed_rows": len(review_rows),
            "pass_rows": passes,
            "flagged_rows": flagged,
            "unresolved_rows": unresolved,
            "model_call_failures": model_failures,
            **independence_counts(drafts, review_rows),
            "translation_sha256": sha256(draft_path),
            "review_sha256": sha256(review_path),
            "status": lang_status,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{lang}: {passes} pass, {flagged} flagged, {unresolved} unresolved", flush=True)

    complete_coverage = (
        len(report["languages"]) == len(LANG_CODES)
        and all(
            item.get("reviewed_rows") == item.get("input_rows")
            and item.get("unresolved_rows") == 0
            for item in report["languages"].values()
        )
    )
    duplicate_decisions: dict[tuple[str, str], bool] = {}
    duplicate_conflicts = []
    for lang, info in report["languages"].items():
        review_path = review_root / f"{lang}.jsonl"
        for row in read_jsonl(review_path):
            verdict = row.get("independent_model_checks", {}).get("blinded_translation_model_pass")
            key = (lang, row.get("review_pair_sha256"))
            if key in duplicate_decisions and duplicate_decisions[key] != verdict:
                duplicate_conflicts.append({"language": lang, "review_pair_sha256": key[1]})
            duplicate_decisions[key] = verdict
    if duplicate_conflicts:
        raise ValueError(f"Identical masked text has inconsistent saved verdicts: {duplicate_conflicts[:5]}")
    report["unique_masked_pair_payloads_reviewed"] = len(duplicate_decisions)
    report["translation_models"] = sorted({
        row.get("translation_model")
        for lang in LANG_CODES
        for row in read_jsonl(draft_root / f"{lang}.jsonl")
    })
    report["duplicate_pair_decision_conflicts"] = 0
    has_flags = any(item.get("flagged_rows", 0) for item in report["languages"].values())
    report["status"] = (
        "AI_ONLY_REVIEW_INCOMPLETE"
        if not complete_coverage
        else "AI_ONLY_REVIEW_REQUIRES_REPAIRS_OR_HUMAN_REVIEW"
        if has_flags
        else "AI_ONLY_REVIEW_ALL_PASS_HUMAN_REVIEW_NOT_PERFORMED"
    )
    report["human_reviewed_total"] = 0
    report["ai_review_pass_total"] = sum(item["pass_rows"] for item in report["languages"].values())
    report["ai_review_flagged_total"] = sum(item["flagged_rows"] for item in report["languages"].values())
    report["unresolved_total"] = sum(item["unresolved_rows"] for item in report["languages"].values())
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = review_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--draft-dir", type=Path, help="Draft directory; defaults to translation_drafts/merged")
    parser.add_argument("--review-dir", type=Path, help="Review output directory; defaults to translation_reviews/qwen_blinded_v4")
    args = parser.parse_args()
    if args.batch_size < 1 or args.timeout < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("batch size, timeout, and max batches must be positive")
    print(json.dumps(
        run(args.root, args.batch_size, args.timeout, args.max_batches, args.draft_dir, args.review_dir),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
