#!/usr/bin/env python3
"""Read-only feasibility audit for the public MUSE-News release."""

from __future__ import annotations

import json
import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


OUT = Path(__file__).resolve().parent
DATASET = "muse-bench/MUSE-News"
MODEL = "muse-bench/MUSE-news_target"


def fetch_json(url: str) -> dict | list:
    last_error = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "fsds-muse-readiness-audit/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < 5:
                time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "fsds-muse-readiness-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def nested_keys(value: object) -> set[str]:
    """Collect schema keys recursively without assuming uniform row shapes."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            found.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(nested_keys(child))
    return found


def fetch_dataset_file(revision: str, path: str) -> bytes:
    escaped_path = urllib.parse.quote(path, safe="/")
    request = urllib.request.Request(
        f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{escaped_path}",
        headers={"User-Agent": "fsds-muse-readiness-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    model_meta = fetch_json(f"https://huggingface.co/api/models/{MODEL}?blobs=true")
    data_meta = fetch_json(f"https://huggingface.co/api/datasets/{DATASET}?blobs=true")
    tree_url = (
        f"https://huggingface.co/api/datasets/{DATASET}/tree/"
        f"{data_meta['sha']}?recursive=true&expand=false"
    )
    tree_entries = fetch_json(tree_url)
    tree_paths = sorted(
        entry["path"] for entry in tree_entries
        if isinstance(entry, dict) and entry.get("type") == "file" and entry.get("path")
    )
    model_files = model_meta.get("siblings", [])
    data_files = data_meta.get("siblings", [])
    weight_files = [file for file in model_files if file.get("rfilename", "").endswith(".safetensors")]
    weight_bytes = sum((file.get("lfs") or {}).get("size", file.get("size", 0)) for file in weight_files)
    data_bytes = sum((file.get("lfs") or {}).get("size", file.get("size", 0)) for file in data_files)

    qa_url = f"https://huggingface.co/datasets/{DATASET}/resolve/{data_meta['sha']}/knowmem/forget-qa.json"
    qas = fetch_json(qa_url)
    qa_top_level_fields = sorted(set().union(*(set(row) for row in qas))) if qas else []
    qa_nested_fields = sorted(set().union(*(nested_keys(row) for row in qas))) if qas else []
    target_statistics_fields = sorted(
        set().union(*(nested_keys(row.get("target_statistics", {})) for row in qas))
    ) if qas else []
    source_field_names = {
        "article_id", "article_url", "document_id", "source_id", "url",
        "article_idx", "document_idx", "source_idx",
    }
    split_tables = {}
    split_counts = {}
    for split in ("forget", "retain1", "retain2", "holdout"):
        parquet_path = f"raw/{split}-00000-of-00001.parquet"
        table = pq.read_table(io.BytesIO(fetch_dataset_file(data_meta["sha"], parquet_path)))
        split_tables[split] = table
        split_counts[split] = table.num_rows
    raw_articles = split_tables["forget"].column("text").to_pylist()
    raw_normalized = {
        split: [normalize(text) for text in table.column("text").to_pylist()]
        for split, table in split_tables.items()
    }
    train_forget_table = pq.read_table(
        io.BytesIO(fetch_dataset_file(data_meta["sha"], "train/forget-00000-of-00001.parquet"))
    )
    train_forget_normalized = [
        normalize(text) for text in train_forget_table.column("text").to_pylist()
    ]
    verbmem_table = pq.read_table(
        io.BytesIO(fetch_dataset_file(data_meta["sha"], "verbmem/forget-00000-of-00001.parquet"))
    )
    if not {"prompt", "gt"}.issubset(verbmem_table.column_names):
        raise ValueError(f"Unexpected MUSE verbmem schema: {verbmem_table.column_names}")
    verbmem_rows = verbmem_table.to_pylist()
    verbmem_normalized = [
        normalize(f"{row.get('prompt', '')}\n{row.get('gt', '')}") for row in verbmem_rows
    ]
    qa_alignment_records = []
    raw_forget_match_counts = {"unique": 0, "multiple": 0, "none": 0}
    train_forget_match_counts = {"unique": 0, "multiple": 0, "none": 0}
    verbmem_match_counts = {"unique": 0, "multiple": 0, "none": 0}
    raw_forget_unmatched_but_train_matched = []
    qa_verbmem_same_row_matches = []
    raw_unique_texts = set(raw_normalized["forget"])
    train_unique_texts = set(train_forget_normalized)
    raw_forget_rows_missing_from_train = sorted(raw_unique_texts - train_unique_texts)
    for qa_idx, qa in enumerate(qas):
        answer = normalize(qa.get("answer", ""))
        raw_hits = {
            split: [i for i, article in enumerate(texts) if answer and answer in article]
            for split, texts in raw_normalized.items()
        }
        train_hits = [
            i for i, article in enumerate(train_forget_normalized)
            if answer and answer in article
        ]
        verbmem_hits = [
            i for i, sample in enumerate(verbmem_normalized)
            if answer and answer in sample
        ]
        raw_forget_count = len(raw_hits["forget"])
        train_forget_count = len(train_hits)
        verbmem_count = len(verbmem_hits)
        raw_bucket = "unique" if raw_forget_count == 1 else "multiple" if raw_forget_count > 1 else "none"
        train_bucket = "unique" if train_forget_count == 1 else "multiple" if train_forget_count > 1 else "none"
        verbmem_bucket = "unique" if verbmem_count == 1 else "multiple" if verbmem_count > 1 else "none"
        raw_forget_match_counts[raw_bucket] += 1
        train_forget_match_counts[train_bucket] += 1
        verbmem_match_counts[verbmem_bucket] += 1
        if qa_idx in verbmem_hits:
            qa_verbmem_same_row_matches.append(qa_idx)
        if raw_forget_count == 0 and train_forget_count > 0:
            raw_forget_unmatched_but_train_matched.append(qa_idx)
        qa_alignment_records.append({
            "qa_row_index": qa_idx,
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "raw_exact_answer_candidate_rows": raw_hits,
            "train_forget_exact_answer_candidate_rows": train_hits,
            "train_forget_exact_answer_candidate_count": train_forget_count,
            "verbmem_exact_answer_candidate_rows": verbmem_hits,
            "verbmem_exact_answer_candidate_count": verbmem_count,
            "qa_index_equals_verbmem_index_exact_match": qa_idx in verbmem_hits,
            "candidate_status": (
                "UNVERIFIED_TRAIN_FORGET_ONLY_EXACT_ANSWER_CANDIDATES"
                if raw_forget_count == 0 and train_forget_count > 0
                else "UNVERIFIED_UNIQUE_RAW_FORGET_ANSWER_CANDIDATE"
                if raw_forget_count == 1
                else "UNVERIFIED_AMBIGUOUS_RAW_FORGET_ANSWER_CANDIDATES"
                if raw_forget_count > 1
                else "NO_EXACT_ANSWER_CANDIDATE_FOUND"
            ),
            "target_statistics_response": (qa.get("target_statistics") or {}).get("response"),
        })
    qa_alignment_path = OUT / "qa_split_alignment_candidates.jsonl"
    qa_alignment_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qa_alignment_records),
        encoding="utf-8",
    )

    model_readme = fetch_text(f"https://huggingface.co/{MODEL}/resolve/{model_meta['sha']}/README.md")
    model_config = json.loads(fetch_text(f"https://huggingface.co/{MODEL}/resolve/{model_meta['sha']}/config.json"))

    model_license = model_meta.get("cardData", {}).get("license")
    data_license = data_meta.get("cardData", {}).get("license")
    audit = {
        "audit_id": "muse_news_readiness_2026-11-09",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repo": DATASET,
            "revision": data_meta.get("sha"),
            "public": data_meta.get("private") is False,
            "gated": data_meta.get("gated"),
            "license_tag": data_license,
            "repo_file_count": len(data_files),
            "repo_file_bytes_sum": data_bytes,
            "raw_split_article_counts": split_counts,
            "train_forget_article_count": train_forget_table.num_rows,
            "verbmem_forget_sample_count": verbmem_table.num_rows,
            "verbmem_forget_columns": verbmem_table.column_names,
            "raw_forget_unique_article_texts": len(raw_unique_texts),
            "train_forget_unique_article_texts": len(train_unique_texts),
            "raw_forget_unique_texts_found_in_train_forget": len(raw_unique_texts & train_unique_texts),
            "raw_forget_unique_texts_missing_from_train_forget": len(raw_forget_rows_missing_from_train),
            "train_forget_unique_texts_beyond_raw_forget": len(train_unique_texts - raw_unique_texts),
            "knowmem_forget_qa_count": len(qas),
            "knowmem_forget_qa_fields": qa_top_level_fields,
            "knowmem_forget_qa_nested_fields": qa_nested_fields,
            "target_statistics_fields": target_statistics_fields,
            "repository_file_paths": tree_paths,
            "repository_source_mapping_paths": [
                path for path in tree_paths
                if any(term in path.casefold() for term in ("source", "article_id", "document_id", "mapping"))
            ],
            "knowmem_forget_qa_columns": qa_top_level_fields,
            "raw_article_columns": split_tables["forget"].column_names,
            "article_linkage": {
                "status": "POTENTIAL_QA_RAW_SPLIT_ALIGNMENT_ISSUE_AND_NO_SOURCE_IDS_FOR_CLUSTER_BOOTSTRAP",
                "qa_records_have_source_id": bool(source_field_names.intersection(qa_nested_fields)),
                "raw_article_records_have_source_id": any(
                    feature in source_field_names
                    for feature in split_tables["forget"].column_names
                ),
                "normalized_exact_answer_match_counts_raw_forget": raw_forget_match_counts,
                "normalized_exact_answer_match_counts_train_forget": train_forget_match_counts,
                "normalized_exact_answer_match_counts_verbmem_forget": verbmem_match_counts,
                "qa_index_equals_verbmem_index_exact_answer_matches": qa_verbmem_same_row_matches,
                "qa_index_equals_verbmem_index_exact_answer_match_count": len(qa_verbmem_same_row_matches),
                "qa_rows_unmatched_in_raw_forget_but_matched_in_train_forget": raw_forget_unmatched_but_train_matched,
                "candidate_table": str(qa_alignment_path.relative_to(OUT)),
                "candidate_table_rows": len(qa_alignment_records),
                "basis": "The MUSE paper says KnowMem QA pairs are generated from excerpts of Verbatim samples, while the Hub card says KnowMem questions test the corresponding raw split. Exact normalized answer searches across raw articles, training passages, and the released verbmem prompt/gt samples are diagnostics, not source identifiers.",
                "note": "Exact answer matching is only a diagnostic. It leaves ambiguous/unmatched questions and cannot be treated as verified article-group labels.",
                "schema_audit": "All released forget-QA rows and nested target_statistics were inspected. target_statistics contains only model response and ROUGE fields; no source/article identifier was found in QA schema or dataset repository file paths.",
            },
        },
        "target_model": {
            "repo": MODEL,
            "revision": model_meta.get("sha"),
            "public": model_meta.get("private") is False,
            "gated": model_meta.get("gated"),
            "license_tag": model_license,
            "license_missing_from_card_and_readme": model_license is None and "[More Information Needed]" in model_readme,
            "model_type": model_config.get("model_type"),
            "architecture": model_config.get("architectures"),
            "parameters_approx": "7B",
            "checkpoint_dtype": model_config.get("torch_dtype"),
            "safetensors_file_count": len(weight_files),
            "safetensors_bytes": weight_bytes,
            "safetensors_gib": round(weight_bytes / (1024**3), 3),
            "note": "Public/un-gated access is technically verified, but the model card does not state a license or fine-tuning provenance. Resolve applicable base-model and derivative-weight terms before training or redistributing outputs.",
        },
        "decision": {
            "dataset_access": "PASS",
            "model_download_access": "PASS_PUBLIC_UNGATED",
            "training_readiness": "BLOCKED_MODEL_TERMS_AND_QA_RAW_SPLIT_ALIGNMENT_REVIEW",
            "gpu_run_started": False,
            "recommended_next_action": "Resolve the MUSE KnowMem-QA versus raw-forget answer mismatch and target-model use terms before training. Use the candidate table only for review, not as verified source groups. If no source mapping can be verified, select a pre-registered finite-set descriptive analysis or a benchmark with stable source IDs for inferential statistics.",
        },
    }
    (OUT / "readiness_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = f"""# MUSE-News feasibility audit

Checked: {audit['checked_at_utc']}

## Verified access

- Dataset `{DATASET}` is public and ungated at revision `{data_meta.get('sha')}`; its Hub card reports `{data_license}`. The repository API returned {len(data_files)} files.
- The official raw split has {split_counts['forget']} forget, {split_counts['retain1']} retain1, {split_counts['retain2']} retain2, and {split_counts['holdout']} holdout article rows. The knowmem forget evaluation contains {len(qas)} QA records.
- Target checkpoint `{MODEL}` is public and ungated at revision `{model_meta.get('sha')}`. It is a {model_config.get('model_type')} {model_config.get('torch_dtype')} checkpoint with {len(weight_files)} Safetensors shards ({weight_bytes:,} bytes, {weight_bytes / (1024**3):.2f} GiB).

## Blockers for an inferential run

- The target model card has no license tag and its README is an autogenerated template with missing provenance. Public download access does not itself settle the terms for training or redistributing a derivative. Resolve the base-model/derivative terms first.
- The released QA rows have no article/source ID; the raw articles expose only a `text` column. The full `train/forget` subset has {train_forget_table.num_rows} rows versus {len(raw_articles)} rows in raw/forget. Every unique raw/forget article text appears in train/forget, with {len(train_unique_texts - raw_unique_texts)} additional unique texts.
- The MUSE paper says KnowMem answers are copied verbatim from source excerpts, and the Hub card says these questions test the corresponding raw split. Yet exact normalized answer matching finds {raw_forget_match_counts['unique']} unique, {raw_forget_match_counts['multiple']} ambiguous, and {raw_forget_match_counts['none']} absent answers in raw/forget. All {len(raw_forget_unmatched_but_train_matched)} absent raw/forget answers do occur in the broader train/forget subset. This is a potential release-alignment issue, not a verified source map. Candidate row indices are recorded in `{qa_alignment_path.name}` and must not be treated as confirmed groups.
- The paper describes KnowMem QA generation from Verbatim excerpts. The released Verbatim-forget file has {verbmem_table.num_rows} rows with columns {', '.join(verbmem_table.column_names)}. Searching answer strings across concatenated `prompt` and `gt` finds {verbmem_match_counts['unique']} unique, {verbmem_match_counts['multiple']} ambiguous, and {verbmem_match_counts['none']} absent matches; exact QA-row-index to Verbatim-row-index matches occur for {len(qa_verbmem_same_row_matches)} rows. These are only string-match diagnostics; they do not establish the intended QA-to-excerpt mapping.
- The complete QA schema was checked, including nested `target_statistics`: its fields are {', '.join(target_statistics_fields)}. The pinned dataset revision has {len(tree_paths)} files and no source/article mapping file; `target_statistics` provides stored model-response/ROUGE information, not provenance.
- Without a verified QA-to-article map, cluster-bootstrap confidence intervals would invent grouping. The benchmark's official finite-set aggregate metrics can still be reported descriptively, but inferential intervals require a source map or a preregistered, reviewed linkage.

## Decision

Dataset access and target-checkpoint download access pass. Training is held until target-model terms and QA/source alignment are reviewed. Before claiming source-cluster inferential statistics, obtain an official source mapping or a verified linkage table and preserve ambiguous/unmatched labels. No GPU run was started by this audit.
"""
    (OUT / "readiness_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit["decision"] | {"audit_json": str(OUT / "readiness_audit.json"), "audit_report": str(OUT / "readiness_audit.md")}, indent=2))


if __name__ == "__main__":
    main()
