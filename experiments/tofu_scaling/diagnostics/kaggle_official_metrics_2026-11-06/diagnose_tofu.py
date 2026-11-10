#!/usr/bin/env python3
"""Audit pinned TOFU data and pinned Open-Unlearning checkpoints without loading weights."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATASET_ID = "locuslab/TOFU"
DATASET_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"
OPEN_UNLEARNING_REVISION = "2d98027677f3468d52b6baa3fc4f2b9f2dc08820"
DATA_FILES = (
    "full.json",
    "forget10.json",
    "retain90.json",
    "holdout10.json",
    "forget10_perturbed.json",
    "retain_perturbed.json",
    "real_authors.json",
    "real_authors_perturbed.json",
    "world_facts.json",
    "world_facts_perturbed.json",
)
EXPECTED_ROWS = {
    "full.json": 4000,
    "forget10.json": 400,
    "retain90.json": 3600,
    "holdout10.json": 400,
    "forget10_perturbed.json": 400,
    "retain_perturbed.json": 400,
    "real_authors.json": 100,
    "real_authors_perturbed.json": 100,
    "world_facts.json": 117,
    "world_facts_perturbed.json": 117,
}
PROFILE_QA_COUNT = 20

CHECKPOINTS = {
    "llama_3_2_3b_full": (
        "open-unlearning/tofu_Llama-3.2-3B-Instruct_full",
        "24f31ca19f6966dcb6f6b29abc511cce71222d4a",
    ),
    "llama_3_2_3b_retain90": (
        "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90",
        "2c35819e9a731d096df9abd6efbcc361870c5333",
    ),
    "llama_3_1_8b_full": (
        "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
        "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e",
    ),
    "llama_3_1_8b_retain90": (
        "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90",
        "f02956a9f25a18f38ad7808fbefb2f957347273e",
    ),
}


def get_bytes(url: str, timeout: int = 45, attempts: int = 3) -> bytes:
    request = Request(url, headers={"User-Agent": "fsds-tofu-audit/1.0"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                break
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last_error}")


def load_jsonl(raw: bytes, name: str) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def duplicate_summary(rows: list[dict]) -> dict:
    qa_counts = collections.Counter((row.get("question"), row.get("answer")) for row in rows)
    question_answers: dict[str, set[str]] = collections.defaultdict(set)
    question_counts = collections.Counter()
    for row in rows:
        question = row.get("question")
        answer = row.get("answer")
        if isinstance(question, str):
            question_counts[question] += 1
            question_answers[question].add(answer)
    repeated_questions = [
        {"question": question, "count": count, "answers": sorted(question_answers[question])}
        for question, count in question_counts.items()
        if count > 1
    ]
    return {
        "exact_duplicate_qa_rows": sum(count - 1 for count in qa_counts.values()),
        "repeated_question_rows": sum(count - 1 for count in question_counts.values()),
        "repeated_question_strings": len(repeated_questions),
        "conflicting_answer_questions": sum(len(item["answers"]) > 1 for item in repeated_questions),
        "repeated_questions": repeated_questions,
    }


def audit_data() -> dict:
    rows_by_file: dict[str, list[dict]] = {}
    file_info: dict[str, dict] = {}
    for name in DATA_FILES:
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{name}"
        raw = get_bytes(url)
        rows = load_jsonl(raw, name)
        rows_by_file[name] = rows
        file_info[name] = {
            "url": url,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": len(rows),
            "expected_rows": EXPECTED_ROWS[name],
            "required_fields": ["question", "answer"],
            "missing_or_empty_required_fields": sum(
                1
                for row in rows
                if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("question", "answer"))
            ),
            "duplicates": duplicate_summary(rows),
        }

    base_splits = ("forget10.json", "retain90.json", "holdout10.json")
    qa_sets = {
        name: {(row["question"], row["answer"]) for row in rows_by_file[name]}
        for name in base_splits
    }
    question_sets = {
        name: {row["question"] for row in rows_by_file[name]}
        for name in base_splits
    }
    intersections = {}
    for i, left in enumerate(base_splits):
        for right in base_splits[i + 1 :]:
            key = f"{left.removesuffix('.json')}__{right.removesuffix('.json')}"
            intersections[key] = {
                "exact_qa_pairs": len(qa_sets[left] & qa_sets[right]),
                "question_strings": len(question_sets[left] & question_sets[right]),
            }

    full_counter = collections.Counter(
        (row["question"], row["answer"]) for row in rows_by_file["full.json"]
    )
    forget_retain_counter = collections.Counter(
        (row["question"], row["answer"])
        for name in ("forget10.json", "retain90.json")
        for row in rows_by_file[name]
    )

    aligned_pairs = {
        "forget10_perturbed_matches_forget10_in_order": (
            "forget10.json",
            "forget10_perturbed.json",
            5,
            True,
            True,
        ),
        "real_authors_perturbed_matches_source_in_order": (
            "real_authors.json",
            "real_authors_perturbed.json",
            3,
            False,
            False,
        ),
        "world_facts_perturbed_matches_source_in_order": (
            "world_facts.json",
            "world_facts_perturbed.json",
            3,
            False,
            False,
        ),
    }
    alignment_checks = {}
    for check_name, (source_name, perturbed_name, wrong_count, has_paraphrased_question, has_paraphrased_answer) in aligned_pairs.items():
        source_rows = rows_by_file[source_name]
        perturbed_rows = rows_by_file[perturbed_name]
        aligned = len(source_rows) == len(perturbed_rows) and all(
            base.get("question") == pert.get("question") and base.get("answer") == pert.get("answer")
            for base, pert in zip(source_rows, perturbed_rows)
        )
        fields_valid = all(
            (
                not has_paraphrased_answer
                or (isinstance(row.get("paraphrased_answer"), str) and bool(row["paraphrased_answer"].strip()))
            )
            and isinstance(row.get("perturbed_answer"), list)
            and len(row["perturbed_answer"]) == wrong_count
            and all(isinstance(answer, str) and answer.strip() for answer in row["perturbed_answer"])
            and (
                not has_paraphrased_question
                or (isinstance(row.get("paraphrased_question"), str) and bool(row["paraphrased_question"].strip()))
            )
            for row in perturbed_rows
        )
        alignment_checks[check_name] = aligned
        alignment_checks[f"{perturbed_name}_answer_variants_valid"] = fields_valid

    retain_rows = rows_by_file["retain90.json"]
    retain_pairs = collections.defaultdict(list)
    for row_index, row in enumerate(retain_rows):
        retain_pairs[(row["question"], row["answer"])].append(row_index)
    retain_perturbed = rows_by_file["retain_perturbed.json"]
    retain_perturbed_aligned = all(
        len(retain_pairs.get((row["question"], row["answer"]), [])) == 1
        for row in retain_perturbed
    )
    retain_perturbation_fields_valid = all(
        isinstance(row.get("paraphrased_answer"), str)
        and bool(row["paraphrased_answer"].strip())
        and isinstance(row.get("perturbed_answer"), list)
        and len(row["perturbed_answer"]) == 5
        and all(isinstance(answer, str) and answer.strip() for answer in row["perturbed_answer"])
        for row in retain_perturbed
    )
    retain_source_indices = [
        retain_pairs[(row["question"], row["answer"])][0]
        for row in retain_perturbed
        if len(retain_pairs.get((row["question"], row["answer"]), [])) == 1
    ]
    retain_profile_blocks = collections.Counter(index // PROFILE_QA_COUNT for index in retain_source_indices)
    retain_eval_profile_blocks_valid = (
        len(retain_source_indices) == len(set(retain_source_indices))
        and sorted(retain_source_indices) == list(range(20 * PROFILE_QA_COUNT))
        and set(retain_profile_blocks) == set(range(20))
        and all(count == PROFILE_QA_COUNT for count in retain_profile_blocks.values())
    )
    alignment_checks["retain_perturbed_rows_map_uniquely_to_retain90"] = retain_perturbed_aligned
    alignment_checks["retain_perturbed_answer_variants_valid"] = retain_perturbation_fields_valid
    alignment_checks["retain_perturbed_is_20_complete_profile_blocks"] = retain_eval_profile_blocks_valid

    perturbation_checks_pass = all(alignment_checks.values())

    count_checks = all(file_info[name]["rows"] == file_info[name]["expected_rows"] for name in DATA_FILES)
    schema_checks = all(file_info[name]["missing_or_empty_required_fields"] == 0 for name in DATA_FILES)
    disjoint_checks = all(
        item["exact_qa_pairs"] == 0 and item["question_strings"] == 0
        for item in intersections.values()
    )
    partition_check = full_counter == forget_retain_counter
    structural_checks_pass = count_checks and schema_checks and disjoint_checks and partition_check and perturbation_checks_pass

    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "files": file_info,
        "split_intersections": intersections,
        "checks": {
            "expected_record_counts": count_checks,
            "required_schema_and_nonempty_text": schema_checks,
            "forget_retain_holdout_pairwise_disjoint_by_question_and_answer": disjoint_checks,
            "full_equals_forget10_plus_retain90_as_multiset": partition_check,
            **alignment_checks,
            "all_official_eval_and_perturbation_alignment_checks_pass": perturbation_checks_pass,
            "structural_checks_pass": structural_checks_pass,
        },
        "canonical_retain_eval": {
            "file": "retain_perturbed.json",
            "rows": len(retain_perturbed),
            "source_profile_blocks_in_retain90": sorted(retain_profile_blocks),
            "source_row_indices_in_retain90": retain_source_indices,
        },
        "profile_grouping": {
            "published_question_answer_pairs_per_profile": PROFILE_QA_COUNT,
            "inferred_profile_counts_from_split_sizes": {
                name.removesuffix(".json"): len(rows_by_file[name]) // PROFILE_QA_COUNT
                for name in ("forget10.json", "retain90.json", "holdout10.json")
            },
            "author_id_column_present": False,
            "limitation": "The release has question/answer text but no author_id field; author isolation is the benchmark's documented construction, not independently proven from a structured ID column here.",
        },
    }


def audit_checkpoints() -> dict:
    results = {}
    for label, (repo, revision) in CHECKPOINTS.items():
        tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
        files = json.loads(get_bytes(tree_url))
        config_raw = get_bytes(f"https://huggingface.co/{repo}/resolve/{revision}/config.json")
        config = json.loads(config_raw)
        weight_files = [item for item in files if item.get("path", "").endswith(".safetensors")]
        available_paths = {item.get("path") for item in files}
        required_tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
        tokenizer_files_present = required_tokenizer_files <= available_paths
        results[label] = {
            "repo_id": repo,
            "revision": revision,
            "config_http_access": "PASS",
            "config_sha256": hashlib.sha256(config_raw).hexdigest(),
            "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "weight_file_count": len(weight_files),
            "weight_bytes_from_hub_metadata": sum(int(item["size"]) for item in weight_files),
            "required_tokenizer_files": sorted(required_tokenizer_files),
            "tokenizer_files_available": tokenizer_files_present,
            "weights_downloaded": False,
        }
        results[label]["checkpoint_access_checks_pass"] = (
            results[label]["config_http_access"] == "PASS"
            and tokenizer_files_present
            and bool(weight_files)
            and all(int(item.get("size", 0)) > 0 for item in weight_files)
        )
    return results


def runtime_context() -> dict:
    try:
        import torch

        return {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
    except Exception as error:  # Runtime metadata is informative, not a data gate.
        return {"platform": platform.platform(), "torch_probe_error": f"{type(error).__name__}: {error}"}


def build_markdown(report: dict) -> str:
    lines = [
        "# TOFU 3B/8B readiness diagnostic",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Data integrity",
        "",
        f"Status: **{'PASS' if report['data_audit']['checks']['structural_checks_pass'] else 'FAIL'}; one known data ambiguity is reported below**",
        "",
        "| File | Rows | SHA-256 | Empty required fields | Exact duplicate QA rows | Repeated questions |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, info in report["data_audit"]["files"].items():
        dup = info["duplicates"]
        lines.append(
            f"| `{name}` | {info['rows']:,} | `{info['sha256']}` | {info['missing_or_empty_required_fields']} | {dup['exact_duplicate_qa_rows']} | {dup['repeated_question_strings']} |"
        )
    lines += [
        "",
        "Forget10 and retain90 form the full 4,000-row training set exactly. Forget10, retain90, and holdout10 have no exact question overlap with one another. The official evaluation perturbation files are checked against their source rows, field schemas, and expected wrong-answer counts. The 400-row `retain_perturbed` evaluation set maps exactly to the first 20 complete retain90 profile blocks; the evaluator must use this canonical set rather than substitute a custom sample.",
        "",
        "### Data warning",
        "",
    ]
    retain_duplicates = report["data_audit"]["files"]["retain90.json"]["duplicates"]["repeated_questions"]
    if retain_duplicates:
        for item in retain_duplicates:
            answer_text = " / ".join(item["answers"])
            lines.append(
                f"- `retain90` contains the same question {item['count']} times with {len(item['answers'])} different answers: {item['question']} ({answer_text}). This is one ambiguous QA collision; preserve the official split for comparability and report a sensitivity analysis excluding both rows."
            )
    else:
        lines.append("- No repeated question strings found.")
    pg = report["data_audit"]["profile_grouping"]
    counts = pg["inferred_profile_counts_from_split_sizes"]
    lines += [
        "",
        "The published design gives 20 QA pairs per fictitious author. Split sizes therefore correspond to 20 forget profiles, 180 retain profiles, and 20 holdout profiles. The released JSON has no explicit author ID, so this audit does not claim an independent structured-ID leakage proof.",
        "",
        "## Checkpoint availability and resource facts",
        "",
        "| Checkpoint | Immutable revision | Weight shards | Weight size (decimal GB) | Config | Tokenizer files | Weights downloaded? |",
        "|---|---|---:|---:|---|---|",
    ]
    for label, item in report["checkpoints"].items():
        model = "Llama 3.2 3B" if "3b" in label else "Llama 3.1 8B"
        role = "full" if label.endswith("_full") else "retain90 oracle"
        size_gb = item["weight_bytes_from_hub_metadata"] / 1e9
        lines.append(
            f"| {model} {role} | `{item['revision']}` | {item['weight_file_count']} | {size_gb:.2f} | {item['config_http_access']} | {'PASS' if item['tokenizer_files_available'] else 'FAIL'} | no |"
        )
    rt = report["local_runtime"]
    lines += [
        "",
        f"Local runtime: `{rt.get('platform')}`; CUDA available: `{rt.get('cuda_available', 'unknown')}`; CUDA devices: `{rt.get('cuda_device_count', 'unknown')}`; MPS available: `{rt.get('mps_available', 'unknown')}`.",
        "",
        "An earlier Kaggle T4×2 smoke passed one synthetic 8B LoRA step and one synthetic NPO update. That verifies only the synthetic runtime path; it is not a TOFU score or an unlearning result. This local diagnostic did not download model weights or start training.",
        "",
        "## Interpretation and next gate",
        "",
        "TOFU is a workable public benchmark for a controlled pilot: the file-level splits are internally consistent and disjoint, and published 3B/8B full and retain90 checkpoints are reachable by immutable revision. The repeated generic retain question is a small but real label ambiguity. The author-group structure has no explicit IDs in the downloaded schema, which limits local leakage verification.",
        "",
        "TOFU uses synthetic fictional author profiles. Results on it do not establish deletion of real personal data or real-world privacy protection.",
        "",
        "An earlier Kaggle run measured answer likelihood on 1,200 examples per checkpoint, but it used a custom retain sample and did not include generation, Truth Ratio, model utility, or the retain-only oracle comparison; treat it as exploratory. The new pinned official-metric evaluator is separate and has not yet run. Run it on Kaggle before NPO, archive all question-level outputs and per-profile uncertainty, then review the results before choosing a training recipe. Treat 3B vs 8B as a comparison of two released model checkpoints, not a pure parameter-count effect: they are Llama 3.2 and Llama 3.1 respectively.",
        "",
        "## Provenance",
        "",
        f"- TOFU dataset revision: `{report['data_audit']['dataset_revision']}`",
        f"- Open-Unlearning source revision for configs/metric semantics: `{report['open_unlearning_revision']}`",
        "- Dataset audit does not include or download model weights.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnostics",
        help="Directory for the JSON and Markdown audit reports.",
    )
    args = parser.parse_args()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "open_unlearning_revision": OPEN_UNLEARNING_REVISION,
        "data_audit": audit_data(),
        "checkpoints": audit_checkpoints(),
        "local_runtime": runtime_context(),
    }
    report["checkpoint_access_checks_pass"] = all(
        item["checkpoint_access_checks_pass"] for item in report["checkpoints"].values()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tofu_readiness_audit.json"
    md_path = args.output_dir / "tofu_readiness_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Data structural checks: {'PASS' if report['data_audit']['checks']['structural_checks_pass'] else 'FAIL'}")
    print(f"Checkpoint configs available: {sum(item['config_http_access'] == 'PASS' for item in report['checkpoints'].values())}/{len(report['checkpoints'])}")
    print(f"Checkpoint files available: {'PASS' if report['checkpoint_access_checks_pass'] else 'FAIL'}")
    print("Model weights downloaded: 0; training started: no")


if __name__ == "__main__":
    main()
