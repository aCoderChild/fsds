#!/usr/bin/env python3
"""Blind, cross-model Qwen screening for v5.1 training questions.

Only question text is sent to the reviewer. Entity names are masked on both
sides; answer labels, split labels, and fact labels are not sent to the model.
Every decision is bound to exact English/translation text and source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
V51 = REPO / "experiments/synthworlds_r1"
OUT = V51 / "training_release_draft"
sys.path.insert(0, str(V51 / "scripts"))
from draft_eval_translations_nllb import language_script_ok  # noqa: E402

MODEL = "qwen2.5:7b-instruct"
MODEL_ID = "ollama:qwen2.5:7b-instruct"
API = "http://localhost:11434/api/chat"
REVIEW_PROTOCOL = "reasoned_translation_equivalence_v2"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")
MARKER = re.compile(r"ZXQ[A-Z]+QXZ|<ENTITY_[A-Z]+>")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def alpha_label(index: int) -> str:
    """Return A..Z, AA.. labels so reviewer placeholders contain no digits."""
    value = index + 1
    label = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def masked_pair(row: dict) -> tuple[str, dict]:
    source, target = row["text_en"], row["text_translation"]
    names = [name for name in row.get("protected_entity_names", []) if name and name in source]
    names = sorted(set(names), key=lambda name: source.index(name))
    replacements = {name: f"[ENTITY_{alpha_label(index)}]" for index, name in enumerate(names)}
    for name in sorted(names, key=len, reverse=True):
        source = source.replace(name, replacements[name])
        target = target.replace(name, replacements[name])
    source_numbers = NUMBER.findall(source)
    for index, number in enumerate(source_numbers):
        token = f"[NUMBER_{alpha_label(index)}]"
        source = source.replace(number, token, 1)
        target = target.replace(number, token, 1)
    task = row.get("probe_type", "factual_question")
    item = {"task_type": task, "source_english": source, "target_translation": target}
    key = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return key, item


def deterministic_checks(lang: str, row: dict) -> dict:
    text, source = row["text_translation"], row["text_en"]
    names = [name for name in row.get("protected_entity_names", []) if name and name in source]
    missing_names = [name for name in names if source.count(name) != text.count(name)]
    numbers = NUMBER.findall(source)
    target_numbers = NUMBER.findall(text)
    missing_numbers = [number for number in sorted(set(numbers)) if numbers.count(number) != target_numbers.count(number)]
    return {
        "nonempty": bool(text.strip()),
        "target_script_present": language_script_ok(lang, text),
        "protected_names_preserved": not missing_names,
        "protected_numbers_preserved": not missing_numbers,
        "no_unrestored_marker": not MARKER.search(text),
        "missing_names": missing_names,
        "missing_numbers": missing_numbers,
    }


def schema(count: int) -> dict:
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": {"type": "boolean"}, "minItems": count, "maxItems": count},
        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": count, "maxItems": count}},
        "required": ["decisions", "reasons"], "additionalProperties": False}


def ask_qwen(lang: str, items: list[dict], timeout: int) -> list[dict]:
    prompt = {
        "model": MODEL, "stream": False, "format": schema(len(items)),
        "messages": [
            {"role": "system", "content": (
                "Review each target-language question as a translation of its English source. Decide whether it "
                "preserves the question's meaning, relation direction, scope, and answer type, and is understandable "
                "and grammatical. Accept idiomatic paraphrases and different word order; do not demand literal "
                "word-for-word matching. Bracketed entity and number placeholders are opaque labels; ignore their "
                "spelling and do not infer or judge answers. Mark false only for a clear omission, addition, meaning "
                "change, wrong relation direction, changed answer type, or seriously unnatural/ungrammatical wording. "
                "Give a concise reason for each decision. Return one decision and one reason per item in order."
            )},
            {"role": "user", "content": json.dumps({"target_language": LANGUAGE_NAMES[lang], "items": items}, ensure_ascii=False)},
        ],
        "options": {"temperature": 0, "num_predict": max(192, len(items) * 24)},
        "keep_alive": "10m",
    }
    request = urllib.request.Request(API, data=json.dumps(prompt, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    try:
        content = json.loads(result.get("message", {}).get("content", ""))
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen returned invalid JSON") from exc
    decisions, reasons = content.get("decisions"), content.get("reasons")
    if (not isinstance(decisions, list) or len(decisions) != len(items) or any(type(x) is not bool for x in decisions)
            or not isinstance(reasons, list) or len(reasons) != len(items)
            or any(not isinstance(x, str) or not x.strip() for x in reasons)):
        raise ValueError(f"Qwen returned invalid decision/reason coverage: {content!r}")
    return [{"decision": decision, "reason": reason} for decision, reason in zip(decisions, reasons)]


def collect_rows(lang: str) -> tuple[list[dict], dict[str, str]]:
    source_rows = []
    source_hashes = {"review_protocol": REVIEW_PROTOCOL}
    atomic_reuse_path = OUT / "atomic_reuse.jsonl"
    source_hashes["atomic_reuse"] = sha(atomic_reuse_path)
    for row in read_jsonl(atomic_reuse_path):
        if row.get("language") == lang and row.get("translation_status") != "SOURCE_ENGLISH":
            source_rows.append({
                "prompt_id": row["sft_id"], "kind": "atomic", "text_en": row["question_en"],
                "text_translation": row["question"], "protected_entity_names": row["protected_entity_names"],
                "translation_model": row.get("translation_model", "unknown"),
            })
    draft_path = OUT / f"atomic_translation_drafts/{lang}.jsonl"
    if not draft_path.is_file():
        raise FileNotFoundError(f"Missing {lang} atomic draft file")
    source_hashes[f"atomic_drafts_{lang}"] = sha(draft_path)
    drafts = read_jsonl(draft_path)
    if len(drafts) != 148 or any(r.get("language") != lang for r in drafts):
        raise ValueError(f"{lang} atomic draft coverage is invalid")
    repair_path = OUT / f"atomic_translation_repairs/{lang}.jsonl"
    if repair_path.is_file():
        source_hashes[f"atomic_repairs_{lang}"] = sha(repair_path)
        repairs = read_jsonl(repair_path)
        repair_by_id = {row["sft_id"]: row for row in repairs}
        if len(repairs) != len(repair_by_id):
            raise ValueError(f"{lang}: duplicate atomic repair IDs")
        for row in drafts:
            repair = repair_by_id.get(row["sft_id"])
            if repair:
                if row.get("translation_status") != "BLOCKED_TRANSLATION_FAILURE":
                    raise ValueError(f"Repair supplied for a non-blocked row: {row['sft_id']}/{lang}")
                if repair.get("source_text_sha256") != hashlib.sha256(row["text_en"].encode("utf-8")).hexdigest():
                    raise ValueError(f"Atomic repair source text hash mismatch: {row['sft_id']}/{lang}")
                if repair.get("translation_status") != "MACHINE_DRAFT_PENDING_QWEN_REVIEW" or not repair.get("question_translation"):
                    raise ValueError(f"Atomic translation repair is still blocked: {row['sft_id']}/{lang}")
                row = {**row, **repair}
            source_row = row
            if source_row.get("translation_status") != "MACHINE_DRAFT_PENDING_QWEN_REVIEW" or not source_row.get("question_translation"):
                raise ValueError(f"Blocked or missing atomic draft: {lang}/{source_row.get('sft_id')}")
            source_rows.append({
                "prompt_id": source_row["sft_id"], "kind": "atomic", "text_en": source_row["text_en"],
                "text_translation": source_row["question_translation"], "protected_entity_names": source_row["protected_entity_names"],
                "translation_model": source_row["translation_model"],
            })
        extra_repairs = set(repair_by_id) - {r["sft_id"] for r in drafts}
        if extra_repairs:
            raise ValueError(f"Atomic repairs contain unknown IDs: {sorted(extra_repairs)}")
    else:
        source_hashes[f"atomic_repairs_{lang}"] = "ABSENT"
        for row in drafts:
            if row.get("translation_status") != "MACHINE_DRAFT_PENDING_QWEN_REVIEW" or not row.get("question_translation"):
                raise ValueError(f"Blocked or missing atomic draft: {lang}/{row.get('sft_id')}")
            source_rows.append({
                "prompt_id": row["sft_id"], "kind": "atomic", "text_en": row["text_en"],
                "text_translation": row["question_translation"], "protected_entity_names": row["protected_entity_names"],
                "translation_model": row["translation_model"],
            })
    comp_path = OUT / "composition_translation_review_queue.jsonl"
    source_hashes["composition_review_queue"] = sha(comp_path)
    for row in read_jsonl(comp_path):
        if row.get("language") == lang:
            source_rows.append({**row, "kind": "composition"})
    by_id = {}
    for row in source_rows:
        key = (row["prompt_id"], row["kind"])
        if key in by_id:
            raise ValueError(f"Duplicate translation review key {key}")
        checks = deterministic_checks(lang, row)
        row["deterministic_checks"] = checks
        row["review_pair_sha256"] = hashlib.sha256(masked_pair(row)[0].encode("utf-8")).hexdigest()
        row["review_status"] = "PENDING_INDEPENDENT_AI_REVIEW" if all(checks[k] for k in ("nonempty", "target_script_present", "protected_names_preserved", "protected_numbers_preserved", "no_unrestored_marker")) else "AI_ONLY_REVIEW_FLAGGED"
        row["reviewer_model"] = MODEL_ID
        by_id[f"{key[0]}::{key[1]}"] = row
    repairs_path = OUT / f"translation_flagged_repairs/{lang}.jsonl"
    if repairs_path.is_file():
        source_hashes[f"flagged_repairs_{lang}"] = sha(repairs_path)
        repair_rows = read_jsonl(repairs_path)
        repair_by_key = {(r["prompt_id"], r["kind"]): r for r in repair_rows}
        if len(repair_by_key) != len(repair_rows):
            raise ValueError(f"{lang}: duplicate flagged-translation repair key")
        for (prompt_id, kind), repair in repair_by_key.items():
            key = f"{prompt_id}::{kind}"
            row = by_id.get(key)
            if row is None:
                raise ValueError(f"{lang}: repair references unknown review row {key}")
            if repair.get("language") != lang or repair.get("source_text_sha256") != hashlib.sha256(row["text_en"].encode("utf-8")).hexdigest():
                raise ValueError(f"{lang}: repair source provenance mismatch for {key}")
            if repair.get("old_text_translation_sha256") != hashlib.sha256(row["text_translation"].encode("utf-8")).hexdigest():
                raise ValueError(f"{lang}: repair does not match the current base translation for {key}")
            if repair.get("translation_status") != "MACHINE_DRAFT_PENDING_QWEN_RERUN" or not repair.get("text_translation"):
                raise ValueError(f"{lang}: repaired question is empty or improperly labeled for {key}")
            row["text_translation"] = repair["text_translation"]
            row["translation_model"] = repair["translation_model"]
            row["translation_route"] = repair["translation_route"]
    else:
        source_hashes[f"flagged_repairs_{lang}"] = "ABSENT"
    for row in by_id.values():
        row["review_pair_sha256"] = hashlib.sha256(masked_pair(row)[0].encode("utf-8")).hexdigest()
    expected = 164 + 148 + 176
    if len(by_id) != expected:
        raise ValueError(f"{lang}: expected {expected} questions for cross-model review, found {len(by_id)}")
    return list(by_id.values()), source_hashes


def run(batch_size: int = 12, timeout: int = 180) -> dict:
    review_dir = OUT / "translation_reviews_qwen"
    review_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "AI_ONLY_CROSS_MODEL_TRANSLATION_SCREEN_IN_PROGRESS",
        "reviewer_model": MODEL_ID, "review_mode": "AI_ONLY_BLINDED_CROSS_MODEL",
        "human_translation_reviewed": False, "answers_sent_to_reviewer": False,
        "protocol": REVIEW_PROTOCOL,
        "batch_size": batch_size, "source_hashes": {}, "languages": {},
    }
    for lang in LANGUAGE_NAMES:
        rows, source_hashes = collect_rows(lang)
        report["source_hashes"][lang] = source_hashes
        out_path = review_dir / f"{lang}.jsonl"
        old = {}
        if out_path.is_file():
            for row in read_jsonl(out_path):
                if row.get("reviewer_model") == MODEL_ID and row.get("review_pair_sha256"):
                    old[(row.get("prompt_id"), row.get("kind"))] = row
        pending = []
        for row in rows:
            previous = old.get((row["prompt_id"], row["kind"]))
            if previous and previous.get("review_pair_sha256") == row["review_pair_sha256"] and previous.get("source_hashes") == source_hashes:
                row["qwen_pass"] = previous.get("qwen_pass")
                row["qwen_reason"] = previous.get("qwen_reason")
                row["review_status"] = previous.get("review_status", row["review_status"])
                row["reviewed_at_utc"] = previous.get("reviewed_at_utc")
                row["source_hashes"] = source_hashes
            elif row["review_status"] == "PENDING_INDEPENDENT_AI_REVIEW":
                pending.append(row)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            unique: dict[str, dict] = {}
            keys = []
            for row in batch:
                key, item = masked_pair(row)
                keys.append(key)
                unique.setdefault(key, item)
            unique_keys = list(unique)
            decisions = ask_qwen(lang, [unique[k] for k in unique_keys], timeout)
            decision_map = dict(zip(unique_keys, decisions))
            for row, key in zip(batch, keys):
                decision = decision_map[key]
                passed = decision["decision"]
                row["qwen_pass"] = passed
                row["qwen_reason"] = decision["reason"]
                if not passed:
                    row["review_status"] = "AI_ONLY_REVIEW_FLAGGED"
                else:
                    row["review_status"] = "AI_ONLY_CROSS_MODEL_PASS"
                row["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
                row["source_hashes"] = source_hashes
            reviewed_by_key = {(r["prompt_id"], r["kind"]): r for r in rows}
            for key, prev in old.items():
                if key in reviewed_by_key and reviewed_by_key[key].get("review_status") == "PENDING_INDEPENDENT_AI_REVIEW":
                    reviewed_by_key[key] = prev
            rows = list(reviewed_by_key.values())
            write_jsonl(out_path, rows)
            print(f"{lang}: reviewed {min(start + len(batch), len(pending))}/{len(pending)} new questions", flush=True)
        # A deterministic failure is a fail, even if the Qwen call was skipped.
        for row in rows:
            if row["review_status"] == "AI_ONLY_CROSS_MODEL_PASS" and row.get("deterministic_checks", {}).get("protected_names_preserved") is not True:
                raise ValueError(f"Inconsistent pass with a missing entity: {row['prompt_id']}/{lang}")
            if row["review_status"] == "AI_ONLY_CROSS_MODEL_PASS" and row.get("qwen_pass") is not True:
                raise ValueError(f"Pass status lacks a true model decision: {row['prompt_id']}/{lang}")
            if row["review_status"] == "AI_ONLY_CROSS_MODEL_PASS" and not row.get("qwen_reason"):
                raise ValueError(f"Pass status lacks its model rationale: {row['prompt_id']}/{lang}")
        write_jsonl(out_path, rows)
        counts = Counter(r["review_status"] for r in rows)
        report["languages"][lang] = {
            "questions": len(rows), "review_status_counts": dict(counts),
            "source_hashes": source_hashes, "review_sha256": sha(out_path),
        }
        write_json(review_dir / "report.json", report)
    report["status"] = "AI_ONLY_CROSS_MODEL_REVIEW_COMPLETE"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(review_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timeout < 1:
        parser.error("batch size and timeout must be positive")
    print(json.dumps(run(args.batch_size, args.timeout), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
