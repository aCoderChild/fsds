#!/usr/bin/env python3
"""Build a masked, source-bound final review for TG candidates not covered by ChatGPT passes."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
PROJECT = HERE.parents[2]
INPUTS = HERE / "independent_review_inputs_2026-11-04"
TG_DIR = RELEASE / "translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03"
MERGED_DIR = HERE / "remote_output/merged_full799/version_1"
TEMPLATE = HERE / "candidate_review_kaggle/review.py"
OUT = HERE / "tg_complement_review_kaggle"
MODE = "tg_qwen7_complement_v1"
KERNEL_ID = "bunnoob2005/synthworlds-r1-tg-complement-review-v1"
LANGS = ("fr", "de", "hi", "th")

sys.path.insert(0, str(RELEASE / "scripts"))
from review_training_translations_qwen import collect_rows, deterministic_checks, masked_pair  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    full_input_path = INPUTS / "full799.jsonl"
    private_map_path = INPUTS / "full799.private_map.jsonl"
    full_rows = read_jsonl(full_input_path)
    map_rows = read_jsonl(private_map_path)
    full_by_key = {r["review_key"]: r for r in full_rows}
    map_by_review_key = {r["review_key"]: r for r in map_rows}
    if (len(full_by_key) != 799 or len(map_by_review_key) != 799
            or len(full_rows) != len(map_rows)
            or sha(full_input_path.read_bytes()) != read_json(INPUTS / "full799.manifest.json").get("input_sha256")
            or sha(private_map_path.read_bytes()) != read_json(INPUTS / "full799.manifest.json").get("private_map_sha256")):
        raise ValueError("Frozen masked input or local map is incomplete or changed")

    merged_path = MERGED_DIR / "merged_full799_diagnostic_reviews.jsonl"
    merged_receipt_path = MERGED_DIR / "merge_verification.json"
    merged = read_jsonl(merged_path)
    receipt = read_json(merged_receipt_path)
    if (len(merged) != 799 or receipt.get("status") != "VERIFIED_799_ROW_AI_ONLY_DIAGNOSTIC_MERGE_NOT_RELEASE_APPROVAL"
            or receipt.get("merged_rows_sha256") != sha(merged_path.read_bytes())
            or receipt.get("release_approved") is not False):
        raise ValueError("Merged ChatGPT candidate review is not hash-bound or diagnostic-only")
    chat_by_key = {r.get("review_key"): r for r in merged}
    if len(chat_by_key) != 799 or set(chat_by_key) != set(full_by_key):
        raise ValueError("ChatGPT review does not cover the exact frozen 799 rows")
    for review_key, review in chat_by_key.items():
        mapping = map_by_review_key[review_key]
        source = full_by_key[review_key]
        if (review.get("candidate_translation_sha256") != mapping.get("candidate_translation_sha256")
                or review.get("source_sha256") != sha(source["source_question_en"].encode("utf-8"))
                or review.get("language") != source.get("language")
                or review.get("kind") != source.get("kind")):
            raise ValueError(f"{review_key}: ChatGPT review does not bind to its exact candidate/source row")
    chat_pass = {k for k, r in chat_by_key.items() if r.get("review_status") == "AI_ONLY_REPAIR_REVIEW_PASS"}

    tg_path = TG_DIR / "repair_candidates.jsonl"
    tg_report_path = TG_DIR / "report.json"
    tg_rows = read_jsonl(tg_path)
    tg_report = read_json(tg_report_path)
    if (tg_report.get("status") != "DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW"
            or tg_report.get("decisions_sha256") != sha(tg_path.read_bytes())
            or tg_report.get("human_translation_reviewed") is not False):
        raise ValueError("TranslateGemma candidate receipt is incomplete or mislabeled")
    tg_by_key = {}
    for row in tg_rows:
        key = (row.get("language"), row.get("prompt_id"), row.get("kind"))
        if key in tg_by_key:
            raise ValueError(f"Duplicate TranslateGemma candidate: {key}")
        tg_by_key[key] = row

    selected = []
    selected_map = []
    deterministic_exclusions = []
    source_rows_by_lang = {}
    for lang in LANGS:
        rows, _ = collect_rows(lang)
        source_rows_by_lang[lang] = {(r["prompt_id"], r["kind"]): r for r in rows}
        if len(source_rows_by_lang[lang]) != 488:
            raise ValueError(f"{lang}: current source input is incomplete")

    for review_key, source in full_by_key.items():
        mapping = map_by_review_key[review_key]
        key = (source["language"], mapping["prompt_id"], source["kind"])
        candidate = tg_by_key.get(key)
        if candidate is None or review_key in chat_pass:
            continue
        if not (candidate.get("deterministic_pass") is True and candidate.get("qwen_pass") is True):
            continue
        if (candidate.get("human_translation_reviewed") is not False
                or candidate.get("answers_sent_to_reviewer") is not False
                or candidate.get("source_en_sha256") != mapping.get("source_text_sha256")
                or candidate.get("old_translation_sha256") != mapping.get("old_translation_sha256")
                or sha(candidate.get("new_translation", "").encode("utf-8")) != candidate.get("new_translation_sha256")):
            raise ValueError(f"{review_key}: TG candidate is not source-bound and answer-blind")

        source_row = source_rows_by_lang[source["language"]].get((mapping["prompt_id"], source["kind"]))
        if source_row is None:
            raise ValueError(f"{review_key}: current source row missing")
        if (sha(source_row["text_en"].encode("utf-8")) != mapping["source_text_sha256"]
                or sha(source_row["text_translation"].encode("utf-8")) != mapping["old_translation_sha256"]
                or candidate.get("new_translation") is None):
            raise ValueError(f"{review_key}: source or candidate text hash mismatch")
        replaced = {**source_row, "text_translation": candidate["new_translation"]}
        checks = deterministic_checks(source["language"], replaced)
        failed_checks = [k for k in (
            "nonempty", "target_script_present", "protected_names_preserved",
            "protected_numbers_preserved", "no_unrestored_marker") if checks.get(k) is not True]
        if failed_checks:
            deterministic_exclusions.append({
                "source_review_key": review_key, "language": source["language"],
                "prompt_id": mapping["prompt_id"], "kind": source["kind"],
                "failed_checks": failed_checks,
            })
            continue
        _, pair = masked_pair(replaced)
        new_review_key = f"T{len(selected) + 1:04d}"
        selected.append({
            "review_key": new_review_key,
            "language": source["language"], "kind": source["kind"],
            "source_question_en": pair["source_english"],
            "translated_question": pair["target_translation"],
        })
        selected_map.append({
            "review_key": new_review_key, "source_review_key": review_key,
            "language": source["language"], "prompt_id": mapping["prompt_id"], "kind": source["kind"],
            "source_text_sha256": mapping["source_text_sha256"],
            "old_translation_sha256": mapping["old_translation_sha256"],
            "candidate_translation_sha256": sha(candidate["new_translation"].encode("utf-8")),
            "candidate_model": candidate.get("translator_model"),
            "candidate_screen": "TranslateGemma4B_plus_Qwen2.5_7B_preliminary",
            "human_translation_reviewed": False,
        })

    if (len(selected) != 167 - len(deterministic_exclusions) or len({r["review_key"] for r in selected}) != len(selected)
            or {r["review_key"] for r in selected} != {r["review_key"] for r in selected_map}):
        raise ValueError(f"TG complement/input map coverage mismatch: {len(selected)} rows, {len(deterministic_exclusions)} deterministic exclusions")

    compact = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    input_sha = sha(compact.encode("utf-8"))
    template = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        'MODE = "full799"': f"MODE = {MODE!r}",
        'INPUT_SHA256 = "14fd830ce6a8733b0e808b46a7f783aac9d1da14be2d30d58e5d26f4606d8a29"': f"INPUT_SHA256 = {input_sha!r}",
    }
    for old, new in replacements.items():
        if template.count(old) != 1:
            raise ValueError("Reviewer template no longer matches the expected input declaration")
        template = template.replace(old, new, 1)
    raw_line = next((line for line in template.splitlines() if line.startswith("RAW_INPUT = ")), None)
    if raw_line is None:
        raise ValueError("Reviewer template lacks an embedded input declaration")
    template = template.replace(raw_line, f"RAW_INPUT = {compact!r}", 1)
    compile(template, "review.py", "exec")

    OUT.mkdir(parents=True, exist_ok=True)
    code_path = OUT / "review.py"
    input_path = HERE / "tg_complement_input_v1.jsonl"
    map_path = HERE / "tg_complement_private_map_v1.jsonl"
    write_jsonl(input_path, selected)
    write_jsonl(map_path, selected_map)
    write_json(HERE / "tg_complement_deterministic_exclusions_v1.json", {
        "status": "PRELIMINARY_CANDIDATES_EXCLUDED_BY_RECOMPUTED_DETERMINISTIC_CHECKS",
        "expected_preliminary_complement_rows": 167,
        "excluded_rows": len(deterministic_exclusions),
        "excluded_rows_by_language": dict(Counter(r["language"] for r in deterministic_exclusions)),
        "exclusions": deterministic_exclusions,
        "training_eligible": False,
    })
    code_path.write_text(template, encoding="utf-8")
    meta = read_json(TEMPLATE.parent / "kernel-metadata.json")
    meta.update({"id": KERNEL_ID, "title": "SynthWorlds R1 TG Complement Review v1", "code_file": "review.py"})
    write_json(OUT / "kernel-metadata.json", meta)
    manifest = {
        "status": "MASKED_TG_COMPLEMENT_REVIEW_READY_NOT_APPROVED",
        "mode": MODE, "kernel_id": KERNEL_ID, "rows": len(selected),
        "preliminary_complement_rows": 167, "deterministic_exclusions": len(deterministic_exclusions),
        "input_jsonl_sha256": sha(input_path.read_bytes()), "input_compact_array_sha256": input_sha,
        "private_map_sha256": sha(map_path.read_bytes()), "review_script_sha256": sha(template.encode("utf-8")),
        "chatgpt_merge_sha256": sha(merged_path.read_bytes()), "tg_candidate_file_sha256": sha(tg_path.read_bytes()),
        "selected_rule": "TG deterministic+Qwen7 preliminary pass AND ChatGPT/NLLB/Qwen14 diagnostic is not pass",
        "answers_or_splits_in_remote_input": False, "private_map_uploaded": False,
        "human_translation_reviewed": False, "release_approved": False,
        "counts_by_language": dict(Counter(r["language"] for r in selected)),
        "limitation": "This is a standardized AI-only candidate screen; no row is approved for training by this queue.",
    }
    write_json(HERE / "tg_complement_build_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
