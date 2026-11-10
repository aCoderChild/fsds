#!/usr/bin/env python3
"""Reconcile existing repair candidates against the 799 final-flagged rows.

This audit checks identity and candidate availability only. It does not judge
translation quality, approve any candidate, or change release inputs.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
TRAINING = REPO / "experiments/synthworlds_r1/training_release_draft"
AUDIT = SCRIPT.parent
REPAIR_ROOT = TRAINING / "translation_repair_diagnostics"
CHATGPT = REPAIR_ROOT / "chatgpt_fullflagged_queue_2026-11-04"
TG = REPAIR_ROOT / "translategemma4b_qwen7b_all_flagged_2026-11-03"
LANGS = ("fr", "de", "hi", "th")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def marker_counts(text: str) -> Counter:
    import re
    return Counter(re.findall(r"ZXQ[A-Z]+QXZ", text))


def run() -> dict:
    flagged: dict[tuple[str, str, str], dict] = {}
    flag_file_hashes = {}
    for lang in LANGS:
        path = TRAINING / "translation_reviews_roundtrip_qwen14b" / f"{lang}.jsonl"
        rows = read_jsonl(path)
        flag_file_hashes[lang] = sha_bytes(path.read_bytes())
        if len(rows) != 488:
            raise ValueError(f"{lang}: expected 488 final-review rows, got {len(rows)}")
        for row in rows:
            if row.get("final_review_status") == "AI_ONLY_REVIEW_FLAGGED":
                key = (lang, row["prompt_id"], row["kind"])
                if key in flagged:
                    raise ValueError(f"Duplicate final-review flag: {key}")
                flagged[key] = row
    if len(flagged) != 799:
        raise ValueError(f"Expected 799 final-review flags, found {len(flagged)}")

    tg_path = TG / "repair_candidates.jsonl"
    tg_rows = read_jsonl(tg_path)
    tg_report_path = TG / "report.json"
    tg_report = read_json(tg_report_path)
    if tg_report.get("processed_rows") != len(tg_rows) or tg_report.get("decisions_sha256") != sha_bytes(tg_path.read_bytes()):
        raise ValueError("TranslateGemma candidate file is not bound to its completion report")
    tg_by_key = {}
    for row in tg_rows:
        key = (row.get("language"), row.get("prompt_id"), row.get("kind"))
        if key in tg_by_key:
            raise ValueError(f"Duplicate TranslateGemma candidate key: {key}")
        tg_by_key[key] = row

    private_rows = read_jsonl(CHATGPT / "private_row_map.jsonl")
    private_by_id = {}
    for row in private_rows:
        key = (row["language"], row["row_id"])
        if key in private_by_id:
            raise ValueError(f"Duplicate private row-map key: {key}")
        private_by_id[key] = row
    chat_rows = {}
    response_hashes = {}
    chat_manifest_path = CHATGPT / "responses/candidate_draft_manifest.json"
    manifest_entries = {
        (row["language"], row["batch"]): row
        for row in read_json(chat_manifest_path).get("candidates", [])
    }
    for lang in LANGS:
        for path in sorted((CHATGPT / "responses" / lang).glob("*.jsonl")):
            response_hashes[f"{lang}/{path.name}"] = sha_bytes(path.read_bytes())
            batch = path.stem
            entry = manifest_entries.get((lang, batch))
            if not entry or entry.get("candidate_sha256") != response_hashes[f"{lang}/{path.name}"]:
                raise ValueError(f"ChatGPT response file is not hash-bound by its manifest: {lang}/{batch}")
            for row in read_jsonl(path):
                map_row = private_by_id.get((lang, row.get("row_id")))
                if not map_row or set(row) != {"row_id", "translation"} or not isinstance(row.get("translation"), str) or not row["translation"].strip():
                    raise ValueError(f"Malformed or unmapped ChatGPT response row in {lang}/{path.name}")
                key = (lang, map_row["prompt_id"], map_row["kind"])
                base = flagged.get(key)
                if (base is None or map_row.get("final_review_status") != "AI_ONLY_REVIEW_FLAGGED"
                        or map_row.get("source_question_en_sha256") != sha_text(base.get("text_en", ""))
                        or map_row.get("rejected_translation_sha256") != sha_text(base.get("text_translation", ""))):
                    raise ValueError(f"ChatGPT candidate is not bound to the current flagged source: {lang}/{row['row_id']}")
                expected = Counter(map_row.get("expected_markers", {}))
                if marker_counts(row["translation"]) != expected:
                    raise ValueError(f"Masked marker mismatch for ChatGPT candidate {lang}/{row['row_id']}")
                if key in chat_rows:
                    raise ValueError(f"Duplicate ChatGPT candidate key: {key}")
                chat_rows[key] = row

    queue_report_path = CHATGPT / "queue_validation.json"
    queue_report = read_json(queue_report_path)
    chat_manifest = read_json(chat_manifest_path)
    if (queue_report.get("status") != "PASS_SOURCE_BOUND_MASKED_TRANSLATION_QUEUE"
            or queue_report.get("request_rows") != 799 or queue_report.get("unique_row_ids") != 799
            or chat_manifest.get("coverage", {}).get("rows_received") != len(chat_rows)
            or chat_manifest.get("coverage", {}).get("rows_total") != 799
            or chat_manifest.get("release_approved") is not False
            or chat_manifest.get("training_eligible") is not False):
        raise ValueError("ChatGPT queue/response manifest does not match the audited scope")

    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    tg_valid = set()
    tg_qwen_pass = set()
    for key, base in flagged.items():
        lang, prompt_id, kind = key
        cell = cells[(lang, kind)]
        cell["flagged"] += 1
        tg = tg_by_key.get(key)
        if tg is not None:
            if (tg.get("source_en_sha256") != sha_text(base.get("text_en", ""))
                    or tg.get("old_translation_sha256") != sha_text(base.get("text_translation", ""))
                    or tg.get("new_translation_sha256") != sha_text(tg.get("new_translation", ""))):
                raise ValueError(f"TranslateGemma candidate hash/source mismatch for {key}")
            tg_valid.add(key)
            cell["translategemma_hash_bound_candidate"] += 1
            if tg.get("qwen_pass") is True and tg.get("status") == "DIAGNOSTIC_QWEN_PASS":
                tg_qwen_pass.add(key)
                cell["translategemma_qwen7_pass_candidate"] += 1
        if key in chat_rows:
            cell["chatgpt_structural_candidate"] += 1
        if key in tg_qwen_pass or key in chat_rows:
            cell["has_candidate_for_further_review"] += 1
        else:
            cell["no_candidate_or_qwen7_pass"] += 1

    result = {
        "status": "VERIFIED_CANDIDATE_AVAILABILITY_ONLY_NOT_RELEASE",
        "final_flagged_rows": len(flagged),
        "translategemma_hash_bound_flagged_candidates": len(tg_valid & set(flagged)),
        "translategemma_qwen7_pass_candidates": len(tg_qwen_pass),
        "chatgpt_structurally_valid_candidates": len(chat_rows),
        "union_of_qwen7_pass_tg_or_chatgpt_candidates": len(tg_qwen_pass | set(chat_rows)),
        "flagged_rows_without_these_candidates": len(set(flagged) - tg_qwen_pass - set(chat_rows)),
        "chatgpt_release_approved": False,
        "chatgpt_training_eligible": False,
        "translation_quality_approved": False,
        "by_language_kind": {f"{lang}/{kind}": dict(cells[(lang, kind)]) for lang in LANGS for kind in ("atomic", "composition")},
        "input_hashes": {
            "final_flag_files_by_language": flag_file_hashes,
            "translategemma_candidates": sha_bytes(tg_path.read_bytes()),
            "translategemma_report": sha_bytes(tg_report_path.read_bytes()),
            "chatgpt_private_map": sha_bytes((CHATGPT / "private_row_map.jsonl").read_bytes()),
            "chatgpt_queue_validation": sha_bytes(queue_report_path.read_bytes()),
            "chatgpt_candidate_manifest": sha_bytes(chat_manifest_path.read_bytes()),
            "chatgpt_response_files": response_hashes,
        },
        "limitations": [
            "Candidate availability and source binding are not translation-quality judgments.",
            "TranslateGemma Qwen7B passes are preliminary and share a reviewer family with the primary screen.",
            "ChatGPT responses are only structurally checked; backend model/version is not exposed.",
            "No candidate in this audit is approved or training-eligible.",
        ],
    }
    output = AUDIT / "repair_candidate_availability.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["output"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
