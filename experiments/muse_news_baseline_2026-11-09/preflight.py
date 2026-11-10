#!/usr/bin/env python3
"""Read-only source, tokenizer, and fixed-split audit for the MUSE baseline."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
DATASET = "muse-bench/MUSE-News"
DATASET_REV = "506bd5b150b92814d45e4404a82f120ab2d748bf"
MODEL = "muse-bench/MUSE-news_target"
MODEL_REV = "a2f39769e9a0b98ec1cdd12f65e9962502208935"
TOKENIZER = "meta-llama/Llama-2-7b-hf"
TOKENIZER_REV = "01c7f73d771dfac7d292323805ebc428287df4f9"
TOKENIZER_MIRROR = "UIC-AI-lab/llama2-tokenizer"
TOKENIZER_MIRROR_REV = "f7be1db2ca91f1c00c1f7b0dafb98374823f85b4"
EVALUATOR_COMMIT = "6d4fdcbdebe4ad46dccaf70f8526cd23ecff609e"
TOKENIZER_FILES = (
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def get_json(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": "fsds-muse-baseline-preflight/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fsds-muse-baseline-preflight/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def git_blob_oid(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def dataset_rows(config: str, split: str, length: int) -> list[dict]:
    query = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": config,
        "split": split,
        "offset": 0,
        "length": length,
        "revision": DATASET_REV,
    })
    result = get_json(f"https://datasets-server.huggingface.co/rows?{query}")
    rows = [row["row"] for row in result["rows"]]
    if result["num_rows_total"] != length or len(rows) != length:
        raise ValueError(f"Unexpected size for {config}/{split}: {result['num_rows_total']} total, {len(rows)} fetched")
    return rows


def row_keys(rows: list[dict]) -> set[tuple[str, str]]:
    return {(str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()) for row in rows}


def question_keys(rows: list[dict]) -> set[str]:
    return {" ".join(str(row.get("question", "")).casefold().split()) for row in rows}


def main() -> None:
    model_meta = get_json(f"https://huggingface.co/api/models/{MODEL}?blobs=true")
    data_meta = get_json(f"https://huggingface.co/api/datasets/{DATASET}?blobs=true")
    tokenizer_meta = get_json(f"https://huggingface.co/api/models/{TOKENIZER}?blobs=true")
    mirror_meta = get_json(f"https://huggingface.co/api/models/{TOKENIZER_MIRROR}?blobs=true")
    model_tree = get_json(f"https://huggingface.co/api/models/{TOKENIZER}/tree/{TOKENIZER_REV}?recursive=true&expand=true")
    mirror_tree = get_json(f"https://huggingface.co/api/models/{TOKENIZER_MIRROR}/tree/{TOKENIZER_MIRROR_REV}?recursive=true&expand=true")
    official_entries = {item["path"]: item for item in model_tree if item.get("path") in TOKENIZER_FILES}
    mirror_entries = {item["path"]: item for item in mirror_tree if item.get("path") in TOKENIZER_FILES}
    mirror_hashes = {}
    for name in TOKENIZER_FILES:
        blob = get_bytes(f"https://huggingface.co/{TOKENIZER_MIRROR}/resolve/{TOKENIZER_MIRROR_REV}/{name}")
        actual_oid = git_blob_oid(blob)
        official_entry = official_entries.get(name, {})
        mirror_entry = mirror_entries.get(name, {})
        same_git_object = official_entry.get("oid") == mirror_entry.get("oid")
        mirror_lfs = mirror_entry.get("lfs") or {}
        same_lfs_payload = (
            not mirror_lfs
            or (hashlib.sha256(blob).hexdigest() == mirror_lfs.get("oid")
                and len(blob) == mirror_lfs.get("size"))
        )
        same_plain_blob = bool(mirror_lfs) or actual_oid == mirror_entry.get("oid")
        if not (same_git_object and same_lfs_payload and same_plain_blob):
            raise ValueError(f"Tokenizer mirror does not match the official file object: {name}")
        mirror_hashes[name] = {
            "bytes": len(blob),
            "git_blob_oid_of_downloaded_bytes": actual_oid,
            "official_and_mirror_git_tree_oid": official_entry.get("oid"),
            "mirror_lfs_sha256": mirror_lfs.get("oid"),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "official_and_mirror_point_to_same_bytes": True,
        }

    forget = dataset_rows("knowmem", "forget_qa", 100)
    retain = dataset_rows("knowmem", "retain_qa", 100)
    forget_icl = dataset_rows("knowmem", "forget_qa_icl", 10)
    retain_icl = dataset_rows("knowmem", "retain_qa_icl", 10)
    verbmem = dataset_rows("verbmem", "forget", 100)
    eval_overlap = sorted(row_keys(forget) & row_keys(retain))
    question_overlaps = {
        "forget_eval_vs_retain_eval": len(question_keys(forget) & question_keys(retain)),
        "forget_eval_vs_forget_icl": len(question_keys(forget) & question_keys(forget_icl)),
        "retain_eval_vs_retain_icl": len(question_keys(retain) & question_keys(retain_icl)),
        "forget_eval_vs_retain_icl": len(question_keys(forget) & question_keys(retain_icl)),
        "retain_eval_vs_forget_icl": len(question_keys(retain) & question_keys(forget_icl)),
    }
    icleval_overlap = {
        "forget_eval_vs_forget_icl": len(row_keys(forget) & row_keys(forget_icl)),
        "retain_eval_vs_retain_icl": len(row_keys(retain) & row_keys(retain_icl)),
        "forget_eval_vs_retain_eval": len(eval_overlap),
    }

    evaluator_url = f"https://api.github.com/repos/jaechan-repo/muse_bench/commits/{EVALUATOR_COMMIT}"
    evaluator = get_json(evaluator_url)
    if evaluator.get("sha") != EVALUATOR_COMMIT:
        raise ValueError("MUSE evaluator commit lookup did not return the pinned commit")

    checks = {
        "dataset_revision_match": data_meta.get("sha") == DATASET_REV,
        "target_model_revision_match": model_meta.get("sha") == MODEL_REV,
        "official_tokenizer_revision_match": tokenizer_meta.get("sha") == TOKENIZER_REV,
        "tokenizer_mirror_revision_match": mirror_meta.get("sha") == TOKENIZER_MIRROR_REV,
        "evaluator_commit_match": evaluator.get("sha") == EVALUATOR_COMMIT,
        "tokenizer_files_byte_identical_to_official": all(
            value["official_and_mirror_point_to_same_bytes"] for value in mirror_hashes.values()
        ),
        "expected_qa_sizes": [len(forget), len(retain), len(forget_icl), len(retain_icl)] == [100, 100, 10, 10],
        "expected_verbmem_size": len(verbmem) == 100,
        "unique_questions_within_each_qa_split": all(
            len(question_keys(rows)) == len(rows)
            for rows in (forget, retain, forget_icl, retain_icl)
        ),
        "no_normalized_question_overlap": not any(question_overlaps.values()),
        "forget_retain_exact_qa_overlap": len(eval_overlap),
        "model_license_tag_missing": model_meta.get("cardData", {}).get("license") is None,
    }
    preflight = {
        "audit_id": "muse_news_baseline_preflight_v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "revisions": {
            "dataset": {"repo": DATASET, "sha": data_meta.get("sha")},
            "model": {"repo": MODEL, "sha": model_meta.get("sha"), "gated": model_meta.get("gated"), "license": model_meta.get("cardData", {}).get("license")},
            "official_tokenizer": {"repo": TOKENIZER, "sha": tokenizer_meta.get("sha"), "gated": tokenizer_meta.get("gated"), "license": tokenizer_meta.get("cardData", {}).get("license")},
            "tokenizer_public_mirror_audit_only": {"repo": TOKENIZER_MIRROR, "sha": mirror_meta.get("sha"), "license": mirror_meta.get("cardData", {}).get("license"), "used_for_download": False},
            "official_evaluator": {"repo": "jaechan-repo/muse_bench", "sha": evaluator.get("sha"), "commit_message": evaluator["commit"]["message"].splitlines()[0]},
        },
        "tokenizer_file_identity": mirror_hashes,
        "dataset_rows": {
            "verbmem_forget": len(verbmem),
            "knowmem_forget_qa": len(forget),
            "knowmem_retain_qa": len(retain),
            "knowmem_forget_icl": len(forget_icl),
            "knowmem_retain_icl": len(retain_icl),
            "unique_questions": {
                "knowmem_forget_qa": len(question_keys(forget)),
                "knowmem_retain_qa": len(question_keys(retain)),
                "knowmem_forget_qa_icl": len(question_keys(forget_icl)),
                "knowmem_retain_qa_icl": len(question_keys(retain_icl)),
            },
            "normalized_question_overlaps": question_overlaps,
            "question_answer_overlaps": icleval_overlap,
        },
        "limitations": [
            "The target model card has no license tag. This package permits inference-only evaluation and forbids redistribution; confirm terms before training or releasing derivatives.",
            "Official tokenizer access remains gated. The public mirror was used only for byte-identity audit and will not be used to bypass the gate.",
            "The benchmark has no source/article IDs. Report row-level bootstrap intervals only; do not claim source-cluster robustness.",
        ],
    }
    (OUT / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(preflight, indent=2))


if __name__ == "__main__":
    main()
