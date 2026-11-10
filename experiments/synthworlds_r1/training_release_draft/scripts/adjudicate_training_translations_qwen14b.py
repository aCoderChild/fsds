#!/usr/bin/env python3
"""Second-pass all training-question translations with a larger AI reviewer.

The first-stage screen is TranslateGemma back-translation plus Qwen2.5 7B
semantic/fluency review. This script sends every candidate row to Qwen2.5 14B,
so it can detect false negatives as well as adjudicate flags. It remains an
AI-only review, not human linguistic certification.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from review_training_translations_qwen import collect_rows, read_jsonl

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
V51 = REPO / "experiments/synthworlds_r1"
OUT = V51 / "training_release_draft"
INITIAL = OUT / "translation_reviews_roundtrip_qwen"
ADJUDICATION = OUT / "translation_reviews_roundtrip_qwen14b"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
MODEL = "qwen2.5:14b"
MODEL_ID = "ollama:qwen2.5:14b"
API = "http://localhost:11434/api/chat"
PROTOCOL = "translategemma_backtranslation_plus_qwen2.5_7b_primary_and_14b_all_candidate_second_pass_v1"
MARKER = re.compile(r"ZXQ[A-Z]+QXZ")
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values), encoding="utf-8")


def current_source_hashes(lang: str) -> tuple[list[dict], dict[str, str]]:
    rows, hashes = collect_rows(lang)
    hashes.pop("review_protocol", None)
    hashes = {**hashes, "roundtrip_protocol": PROTOCOL,
              "adjudication_script_sha256": sha(SCRIPT)}
    if len(rows) != 488:
        raise ValueError(f"{lang}: expected 488 current source questions, found {len(rows)}")
    return rows, hashes


def mask_triplet(row: dict) -> tuple[str, str, str]:
    """Mask source entities/numbers identically in source, target, and back-translation."""
    source = row["text_en"]
    target = row["text_translation"]
    back = row["backtranslation_en"]
    names = [n for n in row.get("protected_entity_names", []) if n and n in source]
    names = sorted(set(names), key=lambda n: source.index(n))
    replacements = {name: f"[ENTITY_{i + 1}]" for i, name in enumerate(names)}
    for name in sorted(names, key=len, reverse=True):
        source = source.replace(name, replacements[name])
        target = target.replace(name, replacements[name])
        back = back.replace(name, replacements[name])
    numbers = NUMBER.findall(source)
    for i, number in enumerate(numbers):
        placeholder = f"[NUMBER_{i + 1}]"
        source = source.replace(number, placeholder, 1)
        target = target.replace(number, placeholder, 1)
        back = back.replace(number, placeholder, 1)
    return source, target, back


def schema(count: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "semantic_pass": {"type": "boolean"},
            "fluency_pass": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["key", "semantic_pass", "fluency_pass", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"decisions": {"type": "array", "items": item, "minItems": count, "maxItems": count}},
        "required": ["decisions"],
        "additionalProperties": False,
    }


def judge_batch(lang: str, rows: list[dict], timeout: int) -> list[dict]:
    items = []
    by_key = {}
    for i, row in enumerate(rows):
        key = f"ROW_{i:03d}"
        source, target, back = mask_triplet(row)
        by_key[key] = row
        items.append({"key": key, "source_question_en": source,
                      "translated_question": target, "back_translation_en": back})
    system = (
        "You are a strict but linguistically informed translation adjudicator for synthetic QA prompts. "
        "For semantic_pass, decide whether the target-language question asks the same thing as the English "
        "source: preserve relation, direction, polarity, scope, tense/time, and answer type. Accept faithful "
        "paraphrases, inflection changes, and contextual synonyms. Do not reject merely because a word has "
        "another dictionary sense; judge the translated question in context. Use the English back-translation "
        "as independent diagnostic evidence, not as a requirement for word-for-word equality. For fluency_pass, "
        "judge only whether the translated question is grammatical and natural in the named target language. "
        "Names/numbers are opaque placeholders. Do not answer the questions or infer their facts. If a material "
        "meaning change or a substantial fluency problem is uncertain, return false. Give one concise reason. "
        "Return exactly one decision for each key, preserving the key strings and input order."
    )
    payload = {
        "model": MODEL,
        "stream": False,
        "format": schema(len(items)),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"target_language": LANGUAGE_NAMES[lang], "items": items}, ensure_ascii=False)},
        ],
        "options": {"temperature": 0, "num_predict": max(512, len(items) * 96)},
        "keep_alive": "10m",
    }
    request = urllib.request.Request(API, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    last_error = None
    for attempt in range(3):
        try:
            if attempt:
                payload["options"]["num_predict"] *= 2
                request = urllib.request.Request(API, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                                 headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            parsed = json.loads(result.get("message", {}).get("content", ""))
            decisions = parsed.get("decisions")
            if not isinstance(decisions, list) or len(decisions) != len(items):
                raise ValueError("Qwen14B returned incomplete decision coverage")
            keyed = {}
            for value in decisions:
                if (not isinstance(value, dict) or value.get("key") not in by_key
                        or type(value.get("semantic_pass")) is not bool
                        or type(value.get("fluency_pass")) is not bool
                        or not isinstance(value.get("reason"), str) or not value["reason"].strip()):
                    raise ValueError("Qwen14B returned a malformed decision row")
                if value["key"] in keyed:
                    raise ValueError("Qwen14B duplicated a decision key")
                keyed[value["key"]] = value
            if set(keyed) != set(by_key):
                raise ValueError("Qwen14B changed or omitted decision keys")
            return [keyed[f"ROW_{i:03d}"] for i in range(len(rows))]
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Qwen14B adjudication batch failed after retries: {type(last_error).__name__}: {last_error}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timeout < 1:
        parser.error("batch size and timeout must be positive")
    initial_report_path = INITIAL / "report.json"
    initial_report = json.loads(initial_report_path.read_text(encoding="utf-8"))
    if initial_report.get("status") != "AI_ONLY_ROUNDTRIP_TRANSLATION_SCREEN_COMPLETE":
        raise ValueError("The initial 7B translation screen is incomplete")
    if initial_report.get("human_translation_reviewed") is not False or initial_report.get("answers_sent_to_reviewer") is not False:
        raise ValueError("The initial review metadata must confirm AI-only, answer-blinded screening")

    report = {
        "status": "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_IN_PROGRESS",
        "protocol": PROTOCOL,
        "adjudication_script_sha256": sha(SCRIPT),
        "initial_screen_report_sha256": sha(initial_report_path),
        "adjudicator_model": MODEL_ID,
        "human_translation_reviewed": False,
        "answers_sent_to_reviewer": False,
        "languages": {},
        "limitations": [
            "AI-only review is not fluent-human certification.",
            "Qwen2.5 7B and Qwen2.5 14B share a model family, so their errors may be correlated.",
            "Machine back-translation and model judgments can still miss semantic or pragmatic errors.",
        ],
    }
    for lang in LANGUAGE_NAMES:
        sources, source_hashes = current_source_hashes(lang)
        _, base_source_hashes = collect_rows(lang)
        initial_path = INITIAL / f"{lang}.jsonl"
        initial_rows = read_jsonl(initial_path)
        initial_by_key = {(r.get("prompt_id"), r.get("kind")): r for r in initial_rows}
        source_by_key = {(r.get("prompt_id"), r.get("kind")): r for r in sources}
        if len(source_by_key) != 488 or len(initial_by_key) != 488 or set(source_by_key) != set(initial_by_key):
            raise ValueError(f"{lang}: source/initial-review coverage is not 488 unique questions")
        expected_hashes = initial_report.get("languages", {}).get(lang, {}).get("source_hashes")
        initial_script = SCRIPT.parent / "review_training_translations_roundtrip.py"
        base_source_hashes.pop("review_protocol", None)
        expected_initial_hashes = {**base_source_hashes,
                                   "roundtrip_protocol": initial_report["protocol"],
                                   "script_sha256": sha(initial_script)}
        if expected_hashes != expected_initial_hashes:
            raise ValueError(f"{lang}: initial screen source hashes differ from current translation source")
        detail = initial_report["languages"][lang]
        if detail.get("review_sha256") != sha(initial_path):
            raise ValueError(f"{lang}: initial review file hash mismatch")

        out_path = ADJUDICATION / f"{lang}.jsonl"
        old_by_key = {}
        if out_path.is_file():
            old_by_key = {(r.get("prompt_id"), r.get("kind")): r for r in read_jsonl(out_path)}
        final_rows = []
        pending = []
        for source in sources:
            key = (source["prompt_id"], source["kind"])
            initial = initial_by_key[key]
            if initial.get("text_en") != source.get("text_en") or initial.get("text_translation") != source.get("text_translation"):
                raise ValueError(f"{lang}/{key}: initial screen text does not match its source row")
            existing = old_by_key.get(key)
            if (existing and existing.get("initial_review_sha256") == detail.get("review_sha256")
                    and existing.get("adjudicator_model") == MODEL_ID
                    and existing.get("adjudication_protocol") == PROTOCOL
                    and existing.get("source_hashes") == source_hashes
                    and existing.get("answers_sent_to_reviewer") is False
                    and existing.get("qwen14b_semantic_pass") is not None
                    and existing.get("qwen14b_fluency_pass") is not None):
                final_rows.append(existing)
                continue
            row = {**initial, "source_hashes": source_hashes,
                   "initial_review_sha256": detail["review_sha256"],
                   "adjudicator_model": MODEL_ID,
                   "adjudication_protocol": PROTOCOL,
                   "answers_sent_to_reviewer": False,
                   "adjudicated_at_utc": None,
                   "qwen14b_semantic_pass": None,
                   "qwen14b_fluency_pass": None,
                   "qwen14b_reason": None}
            row["final_review_status"] = "PENDING_QWEN14B_INDEPENDENT_REVIEW"
            final_rows.append(row)
            pending.append(source)

        # Review every translation in the second stage. Restricting the larger
        # model to first-stage flags would leave primary-review false negatives.
        pending_keys = {(r["prompt_id"], r["kind"]) for r in pending}
        resolved = {k for k, r in old_by_key.items()
                    if k in pending_keys and r.get("initial_review_sha256") == detail.get("review_sha256")
                    and r.get("adjudicator_model") == MODEL_ID
                    and r.get("adjudication_protocol") == PROTOCOL
                    and r.get("source_hashes") == source_hashes
                    and r.get("answers_sent_to_reviewer") is False
                    and type(r.get("qwen14b_semantic_pass")) is bool
                    and type(r.get("qwen14b_fluency_pass")) is bool}
        if resolved:
            by_final_key = {(r["prompt_id"], r["kind"]): r for r in final_rows}
            for key in resolved:
                by_final_key[key] = old_by_key[key]
            final_rows = list(by_final_key.values())
            pending_keys -= resolved
        final_by_key = {(r["prompt_id"], r["kind"]): r for r in final_rows}
        pending_rows = [final_by_key[k] for k in sorted(pending_keys)]
        for start in range(0, len(pending_rows), args.batch_size):
            batch = pending_rows[start:start + args.batch_size]
            try:
                decisions = judge_batch(lang, batch, args.timeout)
            except Exception as exc:
                decisions = [{"semantic_pass": None, "fluency_pass": None,
                              "reason": f"REVIEW_ERROR: {type(exc).__name__}: {exc}"} for _ in batch]
            for source, decision in zip(batch, decisions):
                key = (source["prompt_id"], source["kind"])
                row = final_by_key[key]
                row["qwen14b_semantic_pass"] = decision["semantic_pass"]
                row["qwen14b_fluency_pass"] = decision["fluency_pass"]
                row["qwen14b_reason"] = decision["reason"]
                row["adjudicated_at_utc"] = datetime.now(timezone.utc).isoformat()
                if decision["semantic_pass"] is None or decision["fluency_pass"] is None:
                    row["final_review_status"] = "AI_ONLY_ADJUDICATION_BLOCKED"
                elif (decision["semantic_pass"] and decision["fluency_pass"]
                      and all(row.get("target_deterministic_checks", {}).values())
                      and bool(row.get("backtranslation_checks"))
                      and all(row.get("backtranslation_checks", {}).values())):
                    if row.get("review_status") == "AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS":
                        row["final_review_status"] = "AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS"
                    elif row.get("review_status") == "AI_ONLY_REVIEW_FLAGGED":
                        row["final_review_status"] = "AI_ONLY_14B_ADJUDICATED_PASS"
                    else:
                        row["final_review_status"] = "AI_ONLY_REVIEW_FLAGGED"
                else:
                    row["final_review_status"] = "AI_ONLY_REVIEW_FLAGGED"
            write_jsonl(out_path, list(final_by_key.values()))
            print(f"{lang}: second-pass reviewed {min(start + len(batch), len(pending_rows))}/{len(pending_rows)} questions", flush=True)

        final_rows = list(final_by_key.values())
        if len(final_rows) != 488 or any(r.get("final_review_status") in {"PENDING_QWEN14B_INDEPENDENT_REVIEW", "AI_ONLY_ADJUDICATION_BLOCKED"} for r in final_rows):
            write_jsonl(out_path, final_rows)
            counts = Counter(r.get("final_review_status") for r in final_rows)
            report["languages"][lang] = {"questions": len(final_rows), "review_status_counts": dict(counts),
                                         "source_hashes": source_hashes, "initial_review_sha256": detail["review_sha256"],
                                         "adjudication_sha256": sha(out_path)}
            write_json(ADJUDICATION / "report.json", report)
            raise ValueError(f"{lang}: adjudication incomplete or blocked: {dict(counts)}")
        write_jsonl(out_path, final_rows)
        report["languages"][lang] = {
            "questions": len(final_rows),
            "qwen14b_reviewed_count": sum(type(r.get("qwen14b_semantic_pass")) is bool and type(r.get("qwen14b_fluency_pass")) is bool for r in final_rows),
            "initial_status_counts": dict(Counter(r["review_status"] for r in final_rows)),
            "final_status_counts": dict(Counter(r["final_review_status"] for r in final_rows)),
            "semantic_model_disagreements": sum(type(r.get("qwen_semantic_pass")) is bool and r.get("qwen_semantic_pass") != r.get("qwen14b_semantic_pass") for r in final_rows),
            "fluency_model_disagreements": sum(type(r.get("qwen_fluency_pass")) is bool and r.get("qwen_fluency_pass") != r.get("qwen14b_fluency_pass") for r in final_rows),
            "source_hashes": source_hashes,
            "initial_review_sha256": detail["review_sha256"],
            "adjudication_sha256": sha(out_path),
        }
        write_json(ADJUDICATION / "report.json", report)
    report["status"] = "AI_ONLY_14B_ALL_CANDIDATE_SECOND_PASS_COMPLETE"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["reviewed_every_candidate"] = True
    report["release_ready_for_assembly"] = True
    report["do_not_train"] = True
    write_json(ADJUDICATION / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
