#!/usr/bin/env python3
"""Reconstruct prepared SFT inputs and bind a release to its verified sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
TRAINING = PROJECT / "training_release_draft"
sys.path.insert(0, str(TRAINING / "scripts"))
import prepare_release as prepare
import assemble_training_release as legacy

PREPARED = ("atomic_reuse.jsonl", "atomic_translation_requests.jsonl",
            "composition_reuse.jsonl", "composition_translation_review_queue.jsonl")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_prepared_inputs() -> tuple[dict[str, dict], dict]:
    """Recompute all prepared rows using the hash-bound candidate and source data."""
    draft = read_json(PROJECT / "manifest.draft.json")
    validation = read_json(PROJECT / "LOCAL_VALIDATION.json")
    prep = read_json(TRAINING / "PREPARATION_REPORT.json")
    if (draft.get("status") != "DRAFT_NOT_APPROVED" or draft.get("do_not_train") is not True
            or validation.get("status") != "PASS_DRAFT_SOURCE_AND_SPLIT_VALIDATION"
            or prep.get("do_not_train") is not True or prep.get("approval_status") != "NOT_APPROVED_FOR_TRAINING"):
        raise ValueError("Draft or preparation quarantine metadata is invalid")
    for name, digest in draft.get("artifact_sha256", {}).items():
        path = PROJECT / name
        if validation.get("artifact_hashes", {}).get(name) != digest or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Source draft artifact checksum mismatch: {name}")
    if not draft.get("artifact_sha256") or set(validation.get("artifact_hashes", {})) != set(draft["artifact_sha256"]):
        raise ValueError("Source draft validation does not cover the full artifact set")
    source_files = {"qa": prepare.SOURCE / "qa-sm.jsonl", "facts": prepare.SOURCE / "qa-sm-facts.jsonl",
                    "documents": prepare.SOURCE / "qa-sm-docs.jsonl"}
    if any(sha(path) != draft.get("source_hashes", {}).get(name) for name, path in source_files.items()):
        raise ValueError("Original SynthWorlds source hashes differ from the draft manifest")
    expected_hashes = {
        "candidate_cases": sha(PROJECT / "candidate_cases.draft.jsonl"),
        "labels": sha(PROJECT / "labels/gold_labels.draft.jsonl"),
        "requests": sha(PROJECT / "translation_requests/en.draft.jsonl"),
        "source_qa": sha(source_files["qa"]), "source_facts": sha(source_files["facts"]),
        "v4_atomic_corpus": sha(prepare.V4 / "kaggle/r1_experiment_dataset_disjoint/sft_training/corpus.jsonl"),
    }
    if prep.get("source_hashes") != expected_hashes:
        raise ValueError("Preparation report source hashes do not match the current inputs")
    cases = read_jsonl(PROJECT / "candidate_cases.draft.jsonl")
    qa_rows = read_jsonl(source_files["qa"])
    source_qa = {row["instance_id"]: row for row in qa_rows}
    if len(source_qa) != len(qa_rows):
        raise ValueError("Duplicate source QA IDs")
    atomic, requests, atomic_plan = prepare.prepare_atomic(cases, source_qa)
    composition, queue, composition_plan = prepare.prepare_composition(cases, source_qa)
    expected = dict(zip(PREPARED, (atomic, requests, composition, queue)))
    for name, rows in expected.items():
        if read_jsonl(TRAINING / name) != rows:
            raise ValueError(f"Prepared SFT input differs from source reconstruction: {name}")
    if prep.get("atomic_plan") != atomic_plan or prep.get("composition_plan") != composition_plan:
        raise ValueError("Prepared SFT plan differs from the source reconstruction")
    canonical = {}
    for row in atomic + composition:
        if row.get("language") != "en":
            continue
        if row["sft_id"] in canonical:
            raise ValueError(f"Duplicate canonical SFT ID: {row['sft_id']}")
        canonical[row["sft_id"]] = row
    if len(canonical) != 488:
        raise ValueError(f"Expected 488 canonical SFT IDs, found {len(canonical)}")
    return canonical, {"source_inputs_sha256": expected_hashes,
                       "source_manifest_sha256": sha(PROJECT / "manifest.draft.json"),
                       "source_validation_sha256": sha(PROJECT / "LOCAL_VALIDATION.json"),
                       "preparation_report_sha256": sha(TRAINING / "PREPARATION_REPORT.json"),
                       "prepared_files_sha256": {name: sha(TRAINING / name) for name in PREPARED},
                       "prepare_script_sha256": sha(Path(prepare.__file__)),
                       "legacy_assembler_sha256": sha(Path(legacy.__file__)),
                       "canonical_sft_ids": len(canonical)}


def validate_corpus_source_rows(rows: list[dict], canonical: dict[str, dict]) -> None:
    expected_keys = {(lang, sft_id) for lang in ("en", "fr", "de", "hi", "th") for sft_id in canonical}
    keys = [(row.get("language"), row.get("sft_id")) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Corpus does not contain exactly one row per canonical ID and language")
    for row in rows:
        expected = canonical[row["sft_id"]]
        for field in ("source_qa_id", "split", "hop", "question_en", "answer_en"):
            if row.get(field) != expected.get(field):
                raise ValueError(f"Corpus source {field} differs from verified source: {row['sft_id']}/{row['language']}")
        if (row.get("answer") != expected["answer_en"]
                or row.get("fact_id") != expected.get("fact_id")
                or row.get("fact_ids", []) != expected.get("fact_ids", [])):
            raise ValueError(f"Corpus answer or fact lineage differs from verified source: {row['sft_id']}/{row['language']}")
        if row["language"] == "en" and row.get("question") != expected["question_en"]:
            raise ValueError(f"English corpus question differs from source: {row['sft_id']}")


def validate_release(corpus_path: Path, manifest_path: Path) -> dict:
    canonical, sources = validate_prepared_inputs()
    corpus, manifest = read_jsonl(corpus_path), read_json(manifest_path)
    if manifest.get("corpus_sha256") != sha(corpus_path):
        raise ValueError("Release manifest does not bind the provided corpus")
    validate_corpus_source_rows(corpus, canonical)
    reviews = legacy.checked_translation_reviews()
    rebuilt, _ = legacy.build_corpus(reviews)
    if corpus != rebuilt:
        raise ValueError("Corpus differs from the complete reviewed release reconstruction")
    legacy.validate_split_and_evaluation(rebuilt)
    release_validation = corpus_path.parent / "release_validation.json"
    release = read_json(release_validation)
    if (release.get("status") != "PILOT_TRAINING_RELEASE_READY" or release.get("do_not_train") is not False
            or release.get("corpus_sha256") != sha(corpus_path)
            or release.get("manifest_sha256") != sha(manifest_path)):
        raise ValueError("Legacy release validation is missing or does not bind the release")
    return {"status": "PASS_SOURCE_BOUND_SFT_VALIDATION", "corpus_sha256": sha(corpus_path),
            "manifest_sha256": sha(manifest_path), "release_validation_sha256": sha(release_validation),
            "verifier_sha256": sha(Path(__file__)), "corpus_rows": len(corpus),
            "canonical_sft_ids": len(canonical), "languages": ["en", "fr", "de", "hi", "th"],
            "source_checks": sources, "human_translation_reviewed": False,
            "note": "Source alignment and AI-review provenance only; not a human language audit."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-only", action="store_true")
    parser.add_argument("--corpus", type=Path, default=PROJECT / "sft_training/corpus.jsonl")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "sft_training/manifest.json")
    args = parser.parse_args()
    if args.prepared_only:
        _, sources = validate_prepared_inputs()
        print(json.dumps({"status": "PASS_PREPARED_SOURCE_ALIGNMENT_ONLY", **sources}, indent=2))
        return
    proof = validate_release(args.corpus, args.manifest)
    output = args.corpus.parent / "source_integrity_validation.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": proof["status"], "proof": str(output)}, indent=2))


if __name__ == "__main__":
    main()
