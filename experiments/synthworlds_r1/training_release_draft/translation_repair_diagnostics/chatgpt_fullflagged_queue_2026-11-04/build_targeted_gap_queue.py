#!/usr/bin/env python3
"""Build a masked, answer-blind translation queue for the remaining R1 gaps.

The queue is diagnostic only. It identifies flagged translations whose repair
could complete a required atomic case or composition chain; it does not approve
translations or change the frozen source/review files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
OUTPUT = HERE / "targeted_gap_queue_2026-11-05"
LANGS = ("fr", "de", "hi", "th")
LANGUAGE_NAMES = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
BASE_PASS = {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RELEASE / "scripts"))
from analyze_candidate_union_ceiling import main as analyze_union  # noqa: E402
from review_training_translations_qwen import collect_rows, masked_pair, read_jsonl  # noqa: E402


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return sha_bytes(payload.encode("utf-8"))


def main() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing queue: {OUTPUT}")

    union = analyze_union()  # Revalidates and hash-binds all source/candidate/review receipts.
    if union.get("status") != "OPTIMISTIC_CANDIDATE_UNION_CEILING_NOT_RELEASE_APPROVAL":
        raise ValueError("The current candidate-union audit did not verify")

    pass_by_language = {lang: set() for lang in LANGS}
    source_by_language = {}
    source_hashes_by_language = {}
    for lang in LANGS:
        sources, source_hashes = collect_rows(lang)
        source_by_language[lang] = {(r["prompt_id"], r["kind"]): r for r in sources}
        if len(source_by_language[lang]) != 488:
            raise ValueError(f"{lang}: expected 488 unique source questions")
        source_hashes_by_language[lang] = source_hashes
        reviews = read_jsonl(RELEASE / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl")
        for row in reviews:
            if row.get("final_review_status") in BASE_PASS:
                pass_by_language[lang].add((row["prompt_id"], row["kind"]))

    merged_path = HERE / "remote_output/merged_full799/version_1/merged_full799_diagnostic_reviews.jsonl"
    for row in read_jsonl(merged_path):
        if row.get("review_status") == "AI_ONLY_REPAIR_REVIEW_PASS":
            pass_by_language[row["language"]].add((row["prompt_id"], row["kind"]))

    tg_path = RELEASE / "translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03/repair_candidates.jsonl"
    for row in read_jsonl(tg_path):
        if row.get("deterministic_pass") is True and row.get("qwen_pass") is True:
            pass_by_language[row["language"]].add((row["prompt_id"], row["kind"]))

    # Collect only rows missing from at least one unfinished required case/chain.
    needed: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    atomic_cases = defaultdict(list)
    for row in read_jsonl(RELEASE / "atomic_reuse.jsonl"):
        if row.get("split") in {"forget", "retain"}:
            atomic_cases[row["source_qa_id"]].append(row)
    for case_id, rows in atomic_cases.items():
        missing = [(lang, row["sft_id"], "atomic") for row in rows for lang in LANGS
                   if (row["sft_id"], "atomic") not in pass_by_language[lang]]
        if missing:
            for lang, prompt_id, kind in missing:
                needed[(lang, prompt_id, kind)].add(f"atomic:{case_id}")

    composition_chains = defaultdict(list)
    for row in read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"):
        composition_chains[row["source_qa_id"]].append(row)
    for chain_id, rows in composition_chains.items():
        missing = [row for row in rows
                   if (row["prompt_id"], "composition") not in pass_by_language[row["language"]]]
        if missing:
            for row in missing:
                needed[(row["language"], row["prompt_id"], "composition")].add(f"composition:{chain_id}")

    if len(needed) != 187:
        raise ValueError(f"Expected 187 unique missing translation rows from the verified ceiling; found {len(needed)}")

    chat_rows = {(r["language"], r["prompt_id"], r["kind"]): r for r in read_jsonl(
        HERE / "remote_output/merged_full799/version_1/merged_full799_diagnostic_reviews.jsonl")}
    tg_rows = {(r["language"], r["prompt_id"], r["kind"]): r for r in read_jsonl(tg_path)}
    request_rows_by_language = {lang: [] for lang in LANGS}
    private_rows = []
    local_report_rows = []

    # Stable pseudorandom ordering prevents the request order from revealing split/case structure.
    sorted_keys = sorted(needed, key=lambda key: hashlib.sha256("|".join(key).encode()).hexdigest())
    for index, key in enumerate(sorted_keys, start=1):
        lang, prompt_id, kind = key
        source = source_by_language[lang].get((prompt_id, kind))
        if source is None:
            raise ValueError(f"No source row for required gap {key}")
        masked = masked_pair({**source, "text_translation": ""})[1]["source_english"]
        row_id = f"G{index:04d}"
        request_rows_by_language[lang].append({
            "row_id": row_id,
            "target_language": LANGUAGE_NAMES[lang],
            "source_english": masked,
        })
        private_rows.append({
            "row_id": row_id,
            "language": lang,
            "prompt_id": prompt_id,
            "kind": kind,
            "source_text_sha256": hashlib.sha256(source["text_en"].encode("utf-8")).hexdigest(),
            "source_qa_ids_for_coverage_only": sorted(needed[key]),
            "chatgpt_review_status": chat_rows.get(key, {}).get("review_status", "NO_CANDIDATE_REVIEW"),
            "translategemma_candidate_status": (
                "PASS" if tg_rows.get(key, {}).get("deterministic_pass") is True and tg_rows.get(key, {}).get("qwen_pass") is True
                else "FLAGGED_OR_UNAVAILABLE"
            ),
        })
        local_report_rows.append({"language": lang, "kind": kind, "prompt_id": prompt_id,
                                  "missing_from_candidate_union_for": sorted(needed[key])})

    language_counts = {lang: len(request_rows_by_language[lang]) for lang in LANGS}
    if language_counts != {"fr": 21, "de": 19, "hi": 62, "th": 85}:
        raise ValueError(f"Targeted queue language counts changed: {language_counts}")

    OUTPUT.mkdir(parents=True)
    request_hashes = {}
    for lang in LANGS:
        rows = request_rows_by_language[lang]
        for batch_index, start in enumerate(range(0, len(rows), 16), start=1):
            rel = Path("requests") / lang / f"batch_{batch_index:02d}.jsonl"
            request_hashes[str(rel)] = write_jsonl(OUTPUT / rel, rows[start:start + 16])
    private_hash = write_jsonl(OUTPUT / "private_row_map.jsonl", private_rows)
    report = {
        "status": "VERIFIED_TARGETED_TRANSLATION_GAP_QUEUE_ONLY_NOT_APPROVAL",
        "rows": len(private_rows),
        "rows_by_language": language_counts,
        "rows_by_kind": dict(Counter(row["kind"] for row in local_report_rows)),
        "release_counts_covered_by_candidate_union": union["optimistic_candidate_union_coverage"],
        "queue_row_map_sha256": private_hash,
        "request_sha256_by_file": request_hashes,
        "source_hashes_by_language": source_hashes_by_language,
        "source_review_file_sha256_by_language": {
            lang: sha_file(RELEASE / f"translation_reviews_roundtrip_qwen14b/{lang}.jsonl") for lang in LANGS
        },
        "chatgpt_merged_review_sha256": sha_file(merged_path),
        "translategemma_candidate_sha256": sha_file(tg_path),
        "answers_or_labels_in_request": False,
        "split_or_case_ids_in_request": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "limitations": [
            "This 187-row queue targets only gaps that block currently incomplete required cases/chains.",
            "Requests are masked questions only. Never upload private_row_map.jsonl.",
            "Returned text remains an AI-generated candidate and must pass the frozen production review and source hash checks.",
            "The candidate-union ceiling combines different diagnostic criteria; it is not a valid merged review or approval.",
        ],
    }
    (OUTPUT / "gap_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "local_gap_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in local_report_rows), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
