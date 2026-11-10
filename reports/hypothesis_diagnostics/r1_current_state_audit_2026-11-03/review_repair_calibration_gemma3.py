#!/usr/bin/env python3
"""Blind, read-only cross-family review of a stratified repair sample.

The 64-row output is a model-agreement diagnostic, not gold data and not an
approval to train. It sends no answers or evaluation labels to the reviewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "experiments/synthworlds_r1/training_release_draft"
OUT = Path(__file__).resolve().parent / "evidence/gemma3_repair_calibration"
LANGS = ("fr", "de", "hi", "th")
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
MODEL = "gemma3:4b"
API = "http://localhost:11434/api/chat"
REPAIR_DIR = TRAINING / "translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03"
SYSTEM_PROMPT = (
    "You are a multilingual translation reviewer for synthetic question-answering prompts. "
    "For each item, compare the translated question directly with its English source. "
    "Check that it asks the same question, preserving relation direction, scope, polarity, "
    "time, and expected answer type. Accept faithful paraphrases and contextual synonyms; "
    "do not require word-for-word correspondence. Judge grammar and naturalness only in the "
    "named target language. Bracketed ENTITY and NUMBER placeholders are opaque and must "
    "not be interpreted. Do not answer the questions, infer facts, or use outside knowledge. "
    "If semantic equivalence or fluency is uncertain, set the relevant field false. "
    "Return only the requested JSON object with one decision for every key."
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def mask_entities_and_numbers(source: str, target: str, names: list[str]) -> tuple[str, str]:
    # The repair candidates are selected only after deterministic checks. Mask
    # names identically so the reviewer judges the relation language, not facts.
    ordered = sorted({name for name in names if name and name in source and name in target},
                     key=lambda value: source.index(value))
    for index, name in enumerate(ordered, 1):
        source = source.replace(name, f"[ENTITY_{index}]")
        target = target.replace(name, f"[ENTITY_{index}]")
    import re
    source_numbers = re.findall(r"\d+(?:[-/.]\d+)*", source)
    target_numbers = re.findall(r"\d+(?:[-/.]\d+)*", target)
    if source_numbers != target_numbers:
        raise ValueError("Protected number sequence differs after deterministic candidate checks")
    for index, number in enumerate(source_numbers, 1):
        marker = f"[NUMBER_{index}]"
        source = source.replace(number, marker, 1)
        target = target.replace(number, marker, 1)
    return source, target


def select_sample(limit_per_cell: int) -> tuple[list[dict], dict]:
    source_path = REPAIR_DIR / "repair_candidates.jsonl"
    report_path = REPAIR_DIR / "report.json"
    repair_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (repair_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or sha256_file(source_path) != repair_report.get("decisions_sha256")):
        raise RuntimeError("Repair source status/hash does not match its report")

    originals = {}
    final_report_path = TRAINING / "translation_reviews_roundtrip_qwen14b/report.json"
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    for language in LANGS:
        path = TRAINING / f"translation_reviews_roundtrip_qwen14b/{language}.jsonl"
        language_rows = read_jsonl(path)
        if sha256_file(path) != final_report["languages"][language].get("adjudication_sha256"):
            raise RuntimeError(f"Final review source hash mismatch for {language}")
        for row in language_rows:
            originals[(language, row["prompt_id"], row["kind"])] = row

    candidates = read_jsonl(source_path)
    selected = []
    for language in LANGS:
        for kind in ("atomic", "composition"):
            cell = []
            for candidate in candidates:
                if (candidate.get("language") != language or candidate.get("kind") != kind
                        or candidate.get("status") != "DIAGNOSTIC_QWEN_PASS"
                        or candidate.get("deterministic_pass") is not True
                        or candidate.get("answers_sent_to_reviewer") is not False):
                    continue
                key = (language, candidate["prompt_id"], kind)
                original = originals.get(key)
                if not original:
                    raise RuntimeError(f"Repair candidate has no source review row: {key}")
                if "protected_entity_names" not in original:
                    raise RuntimeError("Historical review rows omit protected entity names; use a source-bound review instead")
                if (candidate.get("source_en_sha256") != sha256_text(original["text_en"])
                        or candidate.get("old_translation_sha256") != sha256_text(original["text_translation"])):
                    raise RuntimeError(f"Repair candidate provenance mismatch: {key}")
                source, target = mask_entities_and_numbers(
                    original["text_en"], candidate["new_translation"],
                    original.get("protected_entity_names", []),
                )
                cell.append({
                    "language": language,
                    "kind": kind,
                    "prompt_id": candidate["prompt_id"],
                    "source_qa_id": original.get("source_qa_id"),
                    "source_question_masked": source,
                    "candidate_translation_masked": target,
                    "source_en_sha256": candidate["source_en_sha256"],
                    "candidate_translation_sha256": candidate["new_translation_sha256"],
                    "qwen7b_preliminary_pass": candidate.get("status") == "DIAGNOSTIC_QWEN_PASS",
                    "qwen7b_preliminary_status": candidate.get("status"),
                })
            cell.sort(key=lambda row: sha256_text(f"{language}|{kind}|{row['prompt_id']}"))
            if len(cell) < limit_per_cell:
                raise RuntimeError(f"Not enough Qwen7B-passed candidates in {language}/{kind}: {len(cell)}")
            selected.extend(cell[:limit_per_cell])

    selection_meta = {
        "sample_rule": f"lowest SHA-256(language|kind|prompt_id), {limit_per_cell} Qwen7B-passed repair candidates per language/kind cell",
        "sample_size": len(selected),
        "selected_by_cell": {
            language: {
                kind: sum(row["language"] == language and row["kind"] == kind for row in selected)
                for kind in ("atomic", "composition")
            }
            for language in LANGS
        },
        "candidate_rows_sha256": sha256_file(source_path),
        "candidate_report_sha256": sha256_file(report_path),
        "final_review_report_sha256": sha256_file(final_report_path),
    }
    return selected, selection_meta


def select_all_candidates() -> tuple[list[dict], dict]:
    source_path = REPAIR_DIR / "repair_candidates.jsonl"
    report_path = REPAIR_DIR / "report.json"
    repair_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (repair_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or sha256_file(source_path) != repair_report.get("decisions_sha256")):
        raise RuntimeError("Repair source status/hash does not match its report")

    final_report_path = TRAINING / "translation_reviews_roundtrip_qwen14b/report.json"
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    originals = {}
    for language in LANGS:
        path = TRAINING / f"translation_reviews_roundtrip_qwen14b/{language}.jsonl"
        if sha256_file(path) != final_report["languages"][language].get("adjudication_sha256"):
            raise RuntimeError(f"Final review source hash mismatch for {language}")
        for row in read_jsonl(path):
            originals[(language, row["prompt_id"], row["kind"])] = row

    selected = []
    skipped = Counter()
    for candidate in read_jsonl(source_path):
        language, kind, prompt_id = candidate.get("language"), candidate.get("kind"), candidate.get("prompt_id")
        key = (language, prompt_id, kind)
        original = originals.get(key)
        if not original:
            raise RuntimeError(f"Repair candidate has no source review row: {key}")
        if "protected_entity_names" not in original:
            raise RuntimeError("Historical review rows omit protected entity names; use a source-bound review instead")
        if (candidate.get("source_en_sha256") != sha256_text(original["text_en"])
                or candidate.get("old_translation_sha256") != sha256_text(original["text_translation"])):
            raise RuntimeError(f"Repair candidate provenance mismatch: {key}")
        translation = candidate.get("new_translation")
        if not isinstance(translation, str) or not translation.strip():
            skipped["empty_candidate_translation"] += 1
            continue
        if candidate.get("deterministic_pass") is not True:
            skipped["deterministic_failure"] += 1
            continue
        source, target = mask_entities_and_numbers(
            original["text_en"], translation, original.get("protected_entity_names", [])
        )
        selected.append({
            "language": language,
            "kind": kind,
            "prompt_id": prompt_id,
            "source_qa_id": original.get("source_qa_id"),
            "source_question_masked": source,
            "candidate_translation_masked": target,
            "source_en_sha256": candidate["source_en_sha256"],
            "candidate_translation_sha256": candidate.get("new_translation_sha256"),
            "qwen7b_preliminary_pass": candidate.get("status") == "DIAGNOSTIC_QWEN_PASS",
            "qwen7b_preliminary_status": candidate.get("status"),
        })
    selected.sort(key=lambda row: (LANGS.index(row["language"]), row["kind"], row["prompt_id"]))
    return selected, {
        "sample_rule": "all nonempty, deterministic-pass repair candidates across every target-language/kind cell, regardless of Qwen7B result",
        "sample_size": len(selected),
        "selected_by_cell": {
            language: {
                kind: sum(row["language"] == language and row["kind"] == kind for row in selected)
                for kind in ("atomic", "composition")
            }
            for language in LANGS
        },
        "skipped_rows": dict(skipped),
        "candidate_rows_sha256": sha256_file(source_path),
        "candidate_report_sha256": sha256_file(report_path),
        "final_review_report_sha256": sha256_file(final_report_path),
    }


def response_schema(count: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "semantic_equivalent": {"type": "boolean"},
            "relation_direction_preserved": {"type": "boolean"},
            "answer_type_preserved": {"type": "boolean"},
            "fluent": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["key", "semantic_equivalent", "relation_direction_preserved", "answer_type_preserved", "fluent", "reason"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {
        "decisions": {"type": "array", "items": item, "minItems": count, "maxItems": count}
    }, "required": ["decisions"], "additionalProperties": False}


def judge_batch(rows: list[dict], timeout: int) -> list[dict]:
    def invoke(batch: list[dict], token_limit: int) -> list[dict]:
        by_key = {f"ROW_{index:03d}": row for index, row in enumerate(batch)}
        items = [{
            "key": key,
            "language": LANGUAGE_NAMES[row["language"]],
            "question_type": "atomic factual question" if row["kind"] == "atomic" else "multi-hop or chain-practice question",
            "source_question_en": row["source_question_masked"],
            "candidate_translation": row["candidate_translation_masked"],
        } for key, row in by_key.items()]
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ],
            "format": response_schema(len(batch)),
            "stream": False,
            "options": {"temperature": 0, "num_predict": token_limit},
            "keep_alive": "10m",
        }
        request = urllib.request.Request(API, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if result.get("done_reason") == "length":
            raise ValueError("Gemma 3 response reached the token limit")
        parsed = json.loads(result.get("message", {}).get("content", ""))
        decisions = parsed.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != len(batch):
            raise ValueError("Gemma 3 returned incomplete decision coverage")
        keyed = {}
        for decision in decisions:
            key = decision.get("key") if isinstance(decision, dict) else None
            if key not in by_key or key in keyed:
                raise ValueError("Gemma 3 returned a missing, unknown, or duplicate key")
            if any(type(decision.get(field)) is not bool for field in (
                "semantic_equivalent", "relation_direction_preserved", "answer_type_preserved", "fluent"
            )) or not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                raise ValueError(f"Gemma 3 returned a malformed decision for {key}")
            keyed[key] = decision
        if set(keyed) != set(by_key):
            raise ValueError("Gemma 3 omitted one or more keys")
        return [keyed[f"ROW_{index:03d}"] for index in range(len(batch))]

    errors = []
    for token_limit in (1024, 2048):
        try:
            result = invoke(rows, token_limit)
            for decision in result:
                decision["review_call_mode"] = f"batch_{len(rows)}_tokens_{token_limit}"
            return result
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if len(rows) > 1:
        result = []
        for row in rows:
            single = judge_batch([row], timeout)
            single[0]["review_call_mode"] = "single_row_retry_after_batch_error"
            result.extend(single)
        return result
    raise ValueError("Gemma 3 single-row review failed: " + "; ".join(errors))


def ollama_version() -> str:
    result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def model_digest() -> str:
    request = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.load(response)
    matches = [row for row in result.get("models", []) if row.get("name") == MODEL]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise RuntimeError(f"Could not uniquely resolve the local digest for {MODEL}")
    return matches[0]["digest"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-cell", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--all-candidates", action="store_true",
                        help="review every deterministic-pass repair candidate; writes a separate full diagnostic")
    args = parser.parse_args()
    if args.limit_per_cell < 1 or args.batch_size < 1:
        raise ValueError("Sample and batch sizes must be positive")
    sample, selection = select_all_candidates() if args.all_candidates else select_sample(args.limit_per_cell)
    run_out = OUT.parent / "gemma3_all_repair_candidates" if args.all_candidates else OUT
    model_hash = model_digest()

    manifest_path = run_out / "manifest.json"
    manifest = {
        "status": "IN_PROGRESS_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW",
        "model": MODEL,
        "model_digest": model_hash,
        "ollama_version": ollama_version(),
        "prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "sample_mode": "all_candidates" if args.all_candidates else "stratified_calibration",
        "answers_sent_to_reviewer": False,
        "human_reviewed": False,
        **selection,
    }
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old_manifest and any(old_manifest.get(key) != manifest.get(key) for key in (
        "model", "model_digest", "sample_size", "candidate_rows_sha256", "selected_by_cell", "prompt_sha256", "sample_mode"
    )):
        raise RuntimeError("Existing calibration manifest differs; refusing unsafe resume")
    write_json(manifest_path, manifest)

    checkpoint_path = run_out / "checkpoint.jsonl"
    completed = read_jsonl(checkpoint_path) if checkpoint_path.exists() else []
    for row in completed:
        # Earlier successful batches were durably written only after the full
        # structured response validated; tag them explicitly when resuming the
        # run after retry handling was added.
        row.setdefault("review_status", "AI_REVIEW_COMPLETE")
        row.setdefault("review_call_mode", "validated_checkpoint_before_retry_hardening")
    key_of = lambda row: (row["language"], row["kind"], row["prompt_id"])
    selected_by_id = {key_of(row): row for row in sample}
    if any(key_of(row) not in selected_by_id for row in completed):
        raise RuntimeError("Checkpoint contains a prompt outside the fixed calibration sample")
    completed_by_id = {key_of(row): row for row in completed}
    if len(completed_by_id) != len(completed):
        raise RuntimeError("Checkpoint contains duplicate prompt IDs")

    pending = [
        row for row in sample
        if key_of(row) not in completed_by_id
        or completed_by_id[key_of(row)].get("review_status") == "AI_REVIEW_ERROR"
    ]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        try:
            decisions = judge_batch(batch, args.timeout)
        except Exception as exc:
            decisions = [{
                "semantic_equivalent": False,
                "relation_direction_preserved": False,
                "answer_type_preserved": False,
                "fluent": False,
                "reason": f"Review error: {type(exc).__name__}",
                "review_status": "AI_REVIEW_ERROR",
                "review_call_mode": "error",
            } for _ in batch]
        for source, decision in zip(batch, decisions):
            completed_by_id[key_of(source)] = {
                **source,
                **decision,
                "gemma_pass": all(decision[field] is True for field in (
                    "semantic_equivalent", "relation_direction_preserved", "answer_type_preserved", "fluent"
                )),
                "reviewer_model": MODEL,
                "answers_sent_to_reviewer": False,
                "review_status": decision.get("review_status", "AI_REVIEW_COMPLETE"),
            }
        ordered = [completed_by_id[key_of(row)] for row in sample if key_of(row) in completed_by_id]
        write_jsonl(checkpoint_path, ordered)
        print(f"Reviewed {len(ordered)}/{len(sample)}", flush=True)

    decisions_path = run_out / "decisions.jsonl"
    decisions = [completed_by_id[key_of(row)] for row in sample]
    write_jsonl(decisions_path, decisions)
    by_cell = {}
    for language in LANGS:
        by_cell[language] = {}
        for kind in ("atomic", "composition"):
            rows = [row for row in decisions if row["language"] == language and row["kind"] == kind]
            by_cell[language][kind] = {
                "reviewed": len(rows),
                "gemma3_passes": sum(row["gemma_pass"] for row in rows),
                "review_errors": sum(row.get("review_status") == "AI_REVIEW_ERROR" for row in rows),
                "qwen7b_preliminary_passes": sum(row["qwen7b_preliminary_pass"] for row in rows),
                "agreement": sum(row["gemma_pass"] and row["qwen7b_preliminary_pass"] for row in rows),
            }
    report = {
        **manifest,
        "status": (
            "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            if all(row.get("review_status") == "AI_REVIEW_COMPLETE" for row in decisions)
            else "COMPLETE_WITH_UNREVIEWED_ROWS_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
        ),
        "completed_rows": len(decisions),
        "review_error_rows": sum(row.get("review_status") == "AI_REVIEW_ERROR" for row in decisions),
        "decision_sha256": sha256_file(decisions_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "results_by_language_kind": by_cell,
        "total_gemma3_passes": sum(row["gemma_pass"] for row in decisions),
        "total_qwen7b_agreements": sum(row["gemma_pass"] and row["qwen7b_preliminary_pass"] for row in decisions),
        "limitations": [
            ("This covers all selected repair candidates, not a gold-standard translation audit."
             if args.all_candidates else
             f"This is a deterministic {len(sample)}-row sample of Qwen7B-passed repair candidates, not a gold-standard translation audit."),
            "Model agreement cannot establish correctness or fluent-human certification.",
            "Gemma3 and the TranslateGemma repair translator share the Gemma model family.",
            "No repair candidate is promoted into the training release by this script.",
        ],
    }
    write_json(run_out / "report.json", report)
    manifest["status"] = report["status"]
    manifest["decision_sha256"] = report["decision_sha256"]
    write_json(manifest_path, manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
