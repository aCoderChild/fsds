#!/usr/bin/env python3
"""Cross-model round-trip and fluency screening for training questions.

TranslateGemma back-translates each non-English training question into English.
Qwen separately checks English semantic equivalence and target-language fluency.
Any deterministic, semantic, or fluency concern is flagged; this is AI-only
screening and does not claim fluent-human validation.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
V51 = REPO / "experiments/synthworlds_r1"
OUT = V51 / "training_release_draft"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V51 / "scripts"))
from review_training_translations_qwen import collect_rows, masked_pair, alpha_label  # noqa: E402
from repair_flagged_eval_translations_translategemma import protect, restore  # noqa: E402
from draft_eval_translations_nllb import language_script_ok  # noqa: E402

TRANSLATE_MODEL = "translategemma:4b"
TRANSLATE_MODEL_ID = "ollama:translategemma:4b"
JUDGE_MODEL = "qwen2.5:7b-instruct"
JUDGE_MODEL_ID = "ollama:qwen2.5:7b-instruct"
API = "http://localhost:11434/api/chat"
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
NUMBER = re.compile(r"\d+(?:[-/.]\d+)*")
MARKER = re.compile(r"ZXQ[A-Z]+QXZ")
PROTOCOL = "translategemma_backtranslation_plus_qwen_english_semantics_and_target_fluency_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def object_schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def translate_schema(count: int) -> dict:
    item = object_schema({"key": {"type": "string"}, "english": {"type": "string"}}, ["key", "english"])
    return object_schema({"items": {"type": "array", "items": item, "minItems": count, "maxItems": count}}, ["items"])


def decision_schema(count: int) -> dict:
    return object_schema({
        "decisions": {"type": "array", "items": {"type": "boolean"}, "minItems": count, "maxItems": count},
        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": count, "maxItems": count},
    }, ["decisions", "reasons"])


def api_chat(model: str, messages: list[dict], response_schema: dict, num_predict: int, timeout: int = 180) -> dict:
    """Use constrained JSON with bounded retries for length-truncated responses."""
    last_error = None
    for attempt in range(3):
        payload = {
            "model": model, "stream": False, "format": response_schema, "messages": messages,
            "options": {"temperature": 0, "num_predict": num_predict * (2 ** attempt)}, "keep_alive": "10m",
        }
        request = urllib.request.Request(API, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        content = result.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            last_error = ValueError(
                f"{model} returned invalid JSON on attempt {attempt + 1} "
                f"(done_reason={result.get('done_reason')!r}): {content!r}"
            )
            continue
        if not isinstance(parsed, dict):
            last_error = ValueError(f"{model} returned a non-object response")
            continue
        return parsed
    raise last_error or ValueError(f"{model} failed to return valid JSON")


def target_deterministic_checks(lang: str, row: dict) -> dict:
    target, source = row["text_translation"], row["text_en"]
    names = [name for name in row.get("protected_entity_names", []) if name and name in source]
    source_numbers, target_numbers = NUMBER.findall(source), NUMBER.findall(target)
    return {
        "nonempty": bool(target.strip()),
        "target_script_present": language_script_ok(lang, target),
        "protected_names_preserved_with_multiplicity": all(source.count(n) == target.count(n) for n in names),
        "protected_numbers_preserved_with_multiplicity": all(source_numbers.count(n) == target_numbers.count(n) for n in set(source_numbers)),
        "no_unrestored_marker": not MARKER.search(target),
    }


def translate_batch(lang: str, prepared: list[dict], timeout: int) -> dict[str, str]:
    key_to_row = {f"ROW_{alpha_label(i)}": row for i, row in enumerate(prepared)}
    payload_items = [{"key": key, "text": row["protected_text"]} for key, row in key_to_row.items()]
    prompt = (
        f"Translate each {LANGUAGE_NAMES[lang]} factual question into natural English. Preserve its complete "
        "meaning, relation direction, scope, and answer type. Each item is independent. Copy every ZXQ marker "
        "exactly once in the corresponding translation. Return the input keys unchanged and only the required JSON.\n\n"
        + json.dumps({"items": payload_items}, ensure_ascii=False)
    )
    try:
        result = api_chat(TRANSLATE_MODEL, [{"role": "user", "content": prompt}], translate_schema(len(prepared)), max(256, len(prepared) * 72), timeout)
        items = result.get("items")
        if not isinstance(items, list) or len(items) != len(prepared):
            raise ValueError("TranslateGemma returned incomplete batch")
        found = {}
        for item in items:
            if not isinstance(item, dict) or item.get("key") not in key_to_row or not isinstance(item.get("english"), str) or not item["english"].strip():
                raise ValueError(f"TranslateGemma returned malformed row: {item!r}")
            key = item["key"]
            if key in found:
                raise ValueError(f"TranslateGemma duplicated row key: {key}")
            found[key] = item["english"].strip()
        if set(found) != set(key_to_row):
            raise ValueError("TranslateGemma changed or omitted item keys")
        return found
    except Exception:
        # A bad batch must not scramble row alignment. Retry each item with a
        # one-row schema; rows that still fail remain explicitly blocked.
        found = {}
        for key, row in key_to_row.items():
            single = {"key": key, "text": row["protected_text"]}
            prompt = (
                f"Translate this {LANGUAGE_NAMES[lang]} factual question into natural English. Preserve its full "
                "meaning, relation direction, scope, and answer type. Copy every ZXQ marker exactly once. "
                "Return the same key and only the required JSON.\n\n" + json.dumps({"items": [single]}, ensure_ascii=False)
            )
            response = api_chat(TRANSLATE_MODEL, [{"role": "user", "content": prompt}], translate_schema(1), 192, timeout)
            items = response.get("items")
            if not isinstance(items, list) or len(items) != 1 or items[0].get("key") != key or not isinstance(items[0].get("english"), str):
                raise ValueError(f"TranslateGemma single-row retry failed for {key}: {response!r}")
            found[key] = items[0]["english"].strip()
        return found


def backtranslate_language(lang: str, rows: list[dict], source_hashes: dict[str, str], batch_size: int, timeout: int) -> list[dict]:
    cache_path = OUT / "translation_backtranslation_drafts" / f"{lang}.jsonl"
    old = {}
    if cache_path.is_file():
        for row in read_jsonl(cache_path):
            old[(row.get("prompt_id"), row.get("kind"))] = row
    output = {}
    pending = []
    for row in rows:
        key = (row["prompt_id"], row["kind"])
        target_hash = sha_text(row["text_translation"])
        previous = old.get(key)
        if (previous and previous.get("target_text_sha256") == target_hash
                and previous.get("source_hashes") == source_hashes
                and previous.get("backtranslation_status") in {"BACKTRANSLATION_DRAFT_READY_FOR_QWEN", "BLOCKED_TARGET_DETERMINISTIC_CHECK"}):
            output[key] = previous
        else:
            target_checks = target_deterministic_checks(lang, row)
            prepared = {**row, "source_hashes": source_hashes, "target_text_sha256": target_hash,
                        "target_deterministic_checks": target_checks}
            if not all(target_checks.values()):
                output[key] = {**prepared, "backtranslation_en": "", "backtranslation_status": "BLOCKED_TARGET_DETERMINISTIC_CHECK"}
                continue
            protected, mapping, counts = protect(row["text_translation"], row.get("protected_entity_names", []))
            prepared.update({"protected_text": protected, "restore_mapping": mapping, "restore_counts": counts})
            pending.append(prepared)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        try:
            translations = translate_batch(lang, batch, timeout)
        except Exception as exc:
            translations = {}
            batch_error = f"{type(exc).__name__}: {exc}"
        else:
            batch_error = ""
        for offset, row in enumerate(batch):
            key_label = f"ROW_{alpha_label(offset)}"
            key = (row["prompt_id"], row["kind"])
            try:
                if key_label not in translations:
                    raise ValueError(batch_error or "Missing backtranslation item")
                back = restore(translations[key_label], row["restore_mapping"], row["restore_counts"])
                checks = {
                    "nonempty": bool(back.strip()),
                    "english_latin_script_present": any("A" <= char <= "Z" or "a" <= char <= "z" for char in back),
                    "source_entity_names_preserved_with_multiplicity": all(row["text_en"].count(n) == back.count(n) for n in row["protected_entity_names"] if n in row["text_en"]),
                    "source_numbers_preserved_with_multiplicity": all(NUMBER.findall(row["text_en"]).count(n) == NUMBER.findall(back).count(n) for n in set(NUMBER.findall(row["text_en"]))),
                    "no_marker_remains": not MARKER.search(back),
                }
                if not all(checks.values()):
                    raise ValueError(f"Backtranslation deterministic checks failed: {checks}")
                output[key] = {**{k: v for k, v in row.items() if k not in {"protected_text", "restore_mapping", "restore_counts"}},
                               "backtranslation_en": back, "backtranslation_checks": checks,
                               "source_text_sha256": sha_text(row["text_en"]),
                               "backtranslation_text_sha256": sha_text(back),
                               "backtranslation_status": "BACKTRANSLATION_DRAFT_READY_FOR_QWEN"}
            except Exception as exc:
                output[key] = {**{k: v for k, v in row.items() if k not in {"protected_text", "restore_mapping", "restore_counts"}},
                               "backtranslation_en": "", "backtranslation_checks": {},
                               "backtranslation_status": "BLOCKED_BACKTRANSLATION_FAILURE",
                               "backtranslation_error": f"{type(exc).__name__}: {exc}"}
        write_jsonl(cache_path, [output[(r["prompt_id"], r["kind"])] for r in rows if (r["prompt_id"], r["kind"]) in output])
        print(f"{lang}: back-translated {min(start + len(batch), len(pending))}/{len(pending)} pending", flush=True)
    final = [output[(r["prompt_id"], r["kind"])] for r in rows]
    write_jsonl(cache_path, final)
    return final


def ask_qwen(kind: str, lang: str, items: list[dict], timeout: int) -> list[dict]:
    if kind == "semantic":
        system = (
            "Compare each English source question with an English back-translation produced from a French, German, "
            "Hindi, or Thai translation. Decide whether the same question is asked with the same relation direction, "
            "scope, and answer type. Accept faithful paraphrases and different word order; reject clear changes, "
            "omissions, additions, or lost relation direction. Bracketed entity and number placeholders are opaque. "
            "Do not solve the questions. Give a concise reason for every decision."
        )
        target = {"language": "English back-translation", "items": items}
    elif kind == "fluency":
        system = (
            "Judge only whether each isolated question is grammatically understandable and natural in the stated "
            "target language. Do not compare it with English or infer whether its meaning is correct. Bracketed "
            "entity and number placeholders are opaque names. Accept normal borrowing and non-literal wording; "
            "reject only clearly broken grammar, unreadable wording, or substantial untranslated English outside "
            "placeholders. Give a concise reason for every decision."
        )
        target = {"language": LANGUAGE_NAMES[lang], "items": items}
    else:
        raise ValueError(f"Unsupported Qwen task {kind}")
    messages = [
        {"role": "system", "content": system + " Keep each reason to at most 12 words."},
        {"role": "user", "content": json.dumps(target, ensure_ascii=False)},
    ]

    def checked_call(call_items: list[dict]) -> list[dict]:
        result = api_chat(JUDGE_MODEL, messages if call_items is items else [
            {"role": "system", "content": system + " Keep the reason to at most 12 words."},
            {"role": "user", "content": json.dumps({"language": target["language"], "items": call_items}, ensure_ascii=False)},
        ], decision_schema(len(call_items)), max(512, len(call_items) * 80), timeout)
        decisions, reasons = result.get("decisions"), result.get("reasons")
        if (not isinstance(decisions, list) or len(decisions) != len(call_items) or any(type(x) is not bool for x in decisions)
                or not isinstance(reasons, list) or len(reasons) != len(call_items)
                or any(not isinstance(x, str) or not x.strip() for x in reasons)):
            raise ValueError(f"Qwen {kind} response has invalid coverage: {result!r}")
        return [{"decision": decision, "reason": reason} for decision, reason in zip(decisions, reasons)]

    try:
        return checked_call(items)
    except Exception as batch_error:
        # A schema/truncation/API problem must not abort the entire language.
        # Retry each row independently; unresolved rows are blocked, never passed.
        results = []
        for item in items:
            try:
                results.extend(checked_call([item]))
            except Exception as row_error:
                results.append({"decision": None,
                                "reason": f"REVIEW_ERROR: batch={type(batch_error).__name__}; row={type(row_error).__name__}: {row_error}"})
        return results


def screen_language(lang: str, sources: list[dict], backtranslations: list[dict], source_hashes: dict[str, str], batch_size: int, timeout: int) -> dict:
    review_dir = OUT / "translation_reviews_roundtrip_qwen"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / f"{lang}.jsonl"
    old = {}
    if review_path.is_file():
        for row in read_jsonl(review_path):
            old[(row.get("prompt_id"), row.get("kind"))] = row
    source_by_key = {(r["prompt_id"], r["kind"]): r for r in sources}
    if len(source_by_key) != len(sources):
        raise ValueError(f"{lang}: duplicate source translation review rows")
    bt_by_key = {(r["prompt_id"], r["kind"]): r for r in backtranslations}
    if set(bt_by_key) != set(source_by_key):
        raise ValueError(f"{lang}: backtranslation coverage differs from source questions")
    rows = []
    for key, source in source_by_key.items():
        bt = bt_by_key[key]
        check_source_hash = {**source_hashes, "backtranslation_cache": sha(OUT / "translation_backtranslation_drafts" / f"{lang}.jsonl")}
        target_checks = target_deterministic_checks(lang, source)
        semantic_row = {"text_en": source["text_en"], "text_translation": bt.get("backtranslation_en", ""),
                        "protected_entity_names": source["protected_entity_names"]}
        semantic_checks = {
            "backtranslation_available": bt.get("backtranslation_status") == "BACKTRANSLATION_DRAFT_READY_FOR_QWEN",
            "backtranslation_deterministic_checks_pass": bool(bt.get("backtranslation_checks")) and all(bt.get("backtranslation_checks", {}).values()),
        }
        if semantic_checks["backtranslation_available"]:
            semantic_pair = masked_pair(semantic_row)[1]
            semantic_item = {"source_english": semantic_pair["source_english"], "backtranslation_english": semantic_pair["target_translation"]}
            fluency_item = {"target_question": masked_pair(source)[1]["target_translation"]}
            pair_hash = sha_text(json.dumps({"semantic": semantic_item, "fluency": fluency_item}, ensure_ascii=False, sort_keys=True))
        else:
            semantic_item, fluency_item, pair_hash = None, None, ""
        previous = old.get(key)
        if (previous and previous.get("review_pair_sha256") == pair_hash
                and previous.get("source_hashes") == check_source_hash
                and previous.get("qwen_semantic_pass") is not None and previous.get("qwen_fluency_pass") is not None
                and not str(previous.get("qwen_semantic_reason", "")).startswith("REVIEW_ERROR:")
                and not str(previous.get("qwen_fluency_reason", "")).startswith("REVIEW_ERROR:")):
            rows.append(previous)
            continue
        rows.append({
            "prompt_id": source["prompt_id"], "kind": source["kind"], "language": lang,
            "text_en": source["text_en"], "text_translation": source["text_translation"],
            "backtranslation_en": bt.get("backtranslation_en", ""),
            "translation_model": source.get("translation_model"), "backtranslation_model": TRANSLATE_MODEL_ID,
            "reviewer_model": JUDGE_MODEL_ID, "review_protocol": PROTOCOL,
            "target_deterministic_checks": target_checks, "backtranslation_checks": bt.get("backtranslation_checks", {}),
            "semantic_checks": semantic_checks, "semantic_item": semantic_item, "fluency_item": fluency_item,
            "review_pair_sha256": pair_hash, "source_hashes": check_source_hash,
            "qwen_semantic_pass": None, "qwen_semantic_reason": "",
            "qwen_fluency_pass": None, "qwen_fluency_reason": "",
            "review_status": "PENDING_AI_ONLY_ROUNDTRIP_SCREEN" if all(target_checks.values()) and all(semantic_checks.values()) else "AI_ONLY_REVIEW_FLAGGED",
            "human_translation_reviewed": False,
        })

    pending_indices = [i for i, row in enumerate(rows) if row["review_status"] == "PENDING_AI_ONLY_ROUNDTRIP_SCREEN"]
    for start in range(0, len(pending_indices), batch_size):
        indices = pending_indices[start:start + batch_size]
        batch = [rows[i] for i in indices]
        semantic = ask_qwen("semantic", lang, [r["semantic_item"] for r in batch], timeout)
        fluency = ask_qwen("fluency", lang, [r["fluency_item"] for r in batch], timeout)
        for index, sem, flu in zip(indices, semantic, fluency):
            row = rows[index]
            row["qwen_semantic_pass"], row["qwen_semantic_reason"] = sem["decision"], sem["reason"]
            row["qwen_fluency_pass"], row["qwen_fluency_reason"] = flu["decision"], flu["reason"]
            row["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
            if sem["decision"] is None or flu["decision"] is None:
                row["review_status"] = "AI_ONLY_REVIEW_BLOCKED"
            else:
                row["review_status"] = "AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS" if sem["decision"] and flu["decision"] else "AI_ONLY_REVIEW_FLAGGED"
        write_jsonl(review_path, [{k: v for k, v in row.items() if k not in {"semantic_item", "fluency_item"}} for row in rows])
        print(f"{lang}: screened {min(start + len(indices), len(pending_indices))}/{len(pending_indices)} translations", flush=True)

    for row in rows:
        if row["review_status"] == "PENDING_AI_ONLY_ROUNDTRIP_SCREEN":
            raise ValueError(f"Unscreened translation remained: {row['prompt_id']}/{lang}")
    serialized = [{k: v for k, v in row.items() if k not in {"semantic_item", "fluency_item"}} for row in rows]
    write_jsonl(review_path, serialized)
    counts = Counter(r["review_status"] for r in rows)
    return {"questions": len(rows), "review_status_counts": dict(counts), "source_hashes": source_hashes,
            "review_sha256": sha(review_path), "backtranslation_sha256": sha(OUT / "translation_backtranslation_drafts" / f"{lang}.jsonl")}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8, help="TranslateGemma back-translation batch size")
    parser.add_argument("--review-batch-size", type=int, default=8, help="Qwen review batch size")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if min(args.batch_size, args.review_batch_size, args.timeout) < 1:
        parser.error("batch sizes and timeout must be positive")
    report = {
        "status": "AI_ONLY_ROUNDTRIP_TRANSLATION_SCREEN_IN_PROGRESS",
        "protocol": PROTOCOL, "translation_model": TRANSLATE_MODEL_ID,
        "semantic_and_fluency_reviewer": JUDGE_MODEL_ID, "human_translation_reviewed": False,
        "answers_sent_to_reviewer": False, "languages": {},
    }
    for lang in LANGUAGE_NAMES:
        sources, source_hashes = collect_rows(lang)
        if len(sources) != 488:
            raise ValueError(f"{lang}: expected 488 training questions, found {len(sources)}")
        source_hashes.pop("review_protocol", None)
        source_hashes = {**source_hashes, "roundtrip_protocol": PROTOCOL,
                         "script_sha256": sha(Path(__file__).resolve())}
        backtranslations = backtranslate_language(lang, sources, source_hashes, args.batch_size, args.timeout)
        report["languages"][lang] = screen_language(lang, sources, backtranslations, source_hashes,
                                                       args.review_batch_size, args.timeout)
        write_json(OUT / "translation_reviews_roundtrip_qwen" / "report.json", report)
    report["status"] = "AI_ONLY_ROUNDTRIP_TRANSLATION_SCREEN_COMPLETE"
    report["release_ready"] = all(
        details.get("review_status_counts") == {"AI_ONLY_ROUNDTRIP_AND_FLUENCY_PASS": details.get("questions")}
        for details in report["languages"].values()
    )
    report["do_not_train"] = not report["release_ready"]
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(OUT / "translation_reviews_roundtrip_qwen" / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
