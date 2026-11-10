#!/usr/bin/env python3
"""Build a masked, reproducible queue for a third translation-candidate pass."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKBENCH = HERE.parent / "chatgpt_fullflagged_queue_2026-11-04"
INPUT_DIR = WORKBENCH / "independent_review_inputs_2026-11-04"
MERGED = WORKBENCH / "remote_output/merged_full799/version_1"
TG = WORKBENCH.parent / "translategemma4b_qwen7b_all_flagged_2026-11-03"
OUT = HERE / "requests.jsonl"
MANIFEST = HERE / "manifest.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    full_path = INPUT_DIR / "full799.jsonl"
    map_path = INPUT_DIR / "full799.private_map.jsonl"
    merged_path = MERGED / "merged_full799_diagnostic_reviews.jsonl"
    receipt_path = MERGED / "merge_verification.json"
    tg_path = TG / "repair_candidates.jsonl"

    full_rows = read_jsonl(full_path)
    map_rows = read_jsonl(map_path)
    merged_rows = read_jsonl(merged_path)
    tg_rows = read_jsonl(tg_path)
    receipt = read_json(receipt_path)

    by_map = {row["review_key"]: row for row in map_rows}
    by_review = {row["review_key"]: row for row in merged_rows}
    by_tg = {(row["language"], row["prompt_id"], row["kind"]): row for row in tg_rows}
    if (len(full_rows) != 799 or len(by_map) != 799 or len(by_review) != 799
            or receipt.get("status") != "VERIFIED_799_ROW_AI_ONLY_DIAGNOSTIC_MERGE_NOT_RELEASE_APPROVAL"
            or receipt.get("merged_rows_sha256") != sha(merged_path.read_bytes())
            or receipt.get("release_approved") is not False
            or receipt.get("training_eligible") is not False):
        raise ValueError("Frozen 799-row candidate review evidence is incomplete or changed")

    requests = []
    for source in full_rows:
        key = source["review_key"]
        mapping = by_map[key]
        review = by_review[key]
        tg = by_tg.get((source["language"], mapping["prompt_id"], source["kind"]))
        chat_pass = review.get("review_status") == "AI_ONLY_REPAIR_REVIEW_PASS"
        tg_pass = bool(tg and tg.get("deterministic_pass") is True and tg.get("qwen_pass") is True)
        if chat_pass or tg_pass:
            continue
        source_text = source["source_question_en"]
        requests.append({
            "source_review_key": key,
            "prompt_id": mapping["prompt_id"],
            "language": source["language"],
            "kind": source["kind"],
            "source_question_en": source_text,
            "source_question_sha256": sha(source_text.encode("utf-8")),
            "prior_chatgpt_review_status": review["review_status"],
            "prior_translategemma_status": tg.get("status") if tg else "NO_CANDIDATE",
        })

    if len(requests) != 187 or len({r["source_review_key"] for r in requests}) != 187:
        raise ValueError(f"Expected 187 unresolved source rows, got {len(requests)}")
    if any("answer" in key.lower() or "split" in key.lower() for row in requests for key in row):
        raise ValueError("Answer/split field leaked into the masked candidate queue")

    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in requests)
    OUT.write_text(payload, encoding="utf-8")
    manifest = {
        "status": "MASKED_THIRD_CANDIDATE_QUEUE_READY_PENDING_TRANSLATION_AND_REVIEW",
        "rows": len(requests),
        "counts_by_language": dict(sorted(collections.Counter(r["language"] for r in requests).items())),
        "counts_by_language_kind": {
            f"{lang}/{kind}": count
            for (lang, kind), count in sorted(collections.Counter((r["language"], r["kind"]) for r in requests).items())
        },
        "request_sha256": sha(OUT.read_bytes()),
        "input_sha256": sha(full_path.read_bytes()),
        "private_map_sha256": sha(map_path.read_bytes()),
        "merged_review_sha256": sha(merged_path.read_bytes()),
        "merged_review_receipt_sha256": sha(receipt_path.read_bytes()),
        "translategemma_candidate_sha256": sha(tg_path.read_bytes()),
        "candidate_generation_model": "OpenAI ChatGPT conversation; exact served model/version not exposed",
        "human_translation_reviewed": False,
        "answers_or_split_labels_in_queue": False,
        "release_approved": False,
        "training_eligible": False,
        "limitations": [
            "Translations are candidates only; they require source-bound deterministic checks and independent AI review.",
            "AI-only screening is not fluent-human certification.",
            "The frozen 195-case split and all source files remain unchanged.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
