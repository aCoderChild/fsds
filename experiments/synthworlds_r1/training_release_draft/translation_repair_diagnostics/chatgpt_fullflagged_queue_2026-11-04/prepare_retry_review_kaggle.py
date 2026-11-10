#!/usr/bin/env python3
"""Build a small Kaggle retry for only blocked/pending rows in the verified full799 run."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "independent_review_inputs_2026-11-04"
PARTIAL_DIR = HERE / "remote_output/full799/version_5"
SOURCE_SCRIPT = HERE / "candidate_review_kaggle/review.py"
OUTPUT_DIR = HERE / "candidate_review_retry_kaggle"
MODE = "retry_full799_unresolved_v1"
REQUIRED_KEYS = {"review_key", "language", "kind", "source_question_en", "translated_question"}
PLACEHOLDER = re.compile(r"\[(ENTITY|NUMBER)_([A-Z]+)\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    source_path = INPUTS / "full799.jsonl"
    launch_path = INPUTS / "full799.kaggle_launch_manifest.json"
    input_rows = jsonl(source_path)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    result_path = PARTIAL_DIR / "repair_review.json"
    verification_path = PARTIAL_DIR / "verification.json"
    verified_path = PARTIAL_DIR / "partial_verified_repair_reviews.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verified = jsonl(verified_path)
    records = result.get("records", [])
    by_key = {row.get("review_key"): row for row in records}
    input_by_key = {row.get("review_key"): row for row in input_rows}
    if (launch.get("mode") != "full799" or len(input_rows) != 799
            or launch.get("input_jsonl_sha256") != sha(source_path.read_bytes())
            or result.get("input_sha256") != launch.get("input_compact_array_sha256")
            or verification.get("status") != "VERIFIED_KILLED_PARTIAL_AI_ONLY_REPAIR_REVIEW"
            or verification.get("output_sha256") != sha(result_path.read_bytes())
            or verification.get("verified_rows_sha256") != sha(verified_path.read_bytes())
            or verification.get("submitted_script_matches_launch_manifest") is not True
            or len(by_key) != 799 or set(by_key) != set(input_by_key)):
        raise ValueError("The source-bound partial review checkpoint failed integrity checks")
    if len(verified) != 733 or len({row.get("review_key") for row in verified}) != len(verified):
        raise ValueError("Expected exactly 733 unique, locally verified checkpoint decisions")

    retries = []
    reasons = Counter()
    for source in input_rows:
        record = by_key[source["review_key"]]
        decision = record.get("decision")
        needs_retry = not isinstance(decision, dict) or "error" in decision or record.get("backtranslation_error") is not None
        if not needs_retry:
            continue
        if (record.get("source_sha256") != sha(source["source_question_en"].encode("utf-8"))
                or record.get("translation_sha256") != sha(source["translated_question"].encode("utf-8"))):
            raise ValueError(f"{source['review_key']}: partial checkpoint is bound to different source text")
        reuse = record.get("backtranslation_en") if record.get("backtranslation_error") is None else None
        reuse_sha = record.get("backtranslation_sha256") if reuse else None
        if reuse is not None:
            if (not isinstance(reuse, str) or not reuse.strip() or reuse_sha != sha(reuse.encode("utf-8"))
                    or Counter(PLACEHOLDER.findall(reuse)) != Counter(PLACEHOLDER.findall(source["source_question_en"]))
                    or Counter(PLACEHOLDER.findall(reuse)) != Counter(PLACEHOLDER.findall(source["translated_question"]))):
                raise ValueError(f"{source['review_key']}: reusable back-translation failed hash/marker checks")
            reasons["reuse_existing_backtranslation"] += 1
        else:
            reasons["regenerate_backtranslation"] += 1
        retries.append({**source, "reuse_backtranslation_en": reuse,
                        "reuse_backtranslation_sha256": reuse_sha})
    if len(retries) != 97 or reasons != Counter({"reuse_existing_backtranslation": 69,
                                                  "regenerate_backtranslation": 28}):
        raise ValueError(f"Unexpected retry composition: rows={len(retries)}, reasons={dict(reasons)}")

    raw_input = json.dumps(retries, ensure_ascii=False, separators=(",", ":"))
    input_hash = sha(raw_input.encode("utf-8"))
    source = SOURCE_SCRIPT.read_text(encoding="utf-8")
    lines = source.splitlines()
    replaced = {"MODE": 0, "INPUT_SHA256": 0, "RAW_INPUT": 0}
    new_lines = []
    for line in lines:
        if line.startswith("MODE = "):
            line = f"MODE = {MODE!r}"; replaced["MODE"] += 1
        elif line.startswith("INPUT_SHA256 = "):
            line = f"INPUT_SHA256 = {input_hash!r}"; replaced["INPUT_SHA256"] += 1
        elif line.startswith("RAW_INPUT = "):
            line = f"RAW_INPUT = {raw_input!r}"; replaced["RAW_INPUT"] += 1
        new_lines.append(line)
    if replaced != {"MODE": 1, "INPUT_SHA256": 1, "RAW_INPUT": 1}:
        raise ValueError(f"Could not replace exact reviewer input declarations: {replaced}")
    generated = "\n".join(new_lines) + "\n"

    old_schema = '''if (set(row) != {"review_key", "language", "kind", "source_question_en", "translated_question"}
                or row["language"] not in LANGUAGE_NAMES or row["kind"] not in {"atomic", "composition"}
                or not row["source_question_en"].strip() or not row["translated_question"].strip()):
            raise ValueError(f"invalid blinded review input: {row.get('review_key')}")'''
    new_schema = '''required = {"review_key", "language", "kind", "source_question_en", "translated_question",
                    "reuse_backtranslation_en", "reuse_backtranslation_sha256"}
        if (set(row) != required or row["language"] not in LANGUAGE_NAMES
                or row["kind"] not in {"atomic", "composition"}
                or not row["source_question_en"].strip() or not row["translated_question"].strip()
                or ((row["reuse_backtranslation_en"] is None) != (row["reuse_backtranslation_sha256"] is None))):
            raise ValueError(f"invalid blinded review input: {row.get('review_key')}")
        if row["reuse_backtranslation_en"] is not None:
            reused = row["reuse_backtranslation_en"]
            if not isinstance(reused, str) or not reused.strip():
                raise ValueError(f"invalid reusable backtranslation for {row.get('review_key')}")
            markers = Counter(PLACEHOLDER.findall(reused))
            if (sha(reused) != row["reuse_backtranslation_sha256"]
                    or markers != Counter(PLACEHOLDER.findall(row["source_question_en"]))
                    or markers != Counter(PLACEHOLDER.findall(row["translated_question"]))):
                raise ValueError(f"invalid reusable backtranslation for {row.get('review_key')}")'''
    if old_schema not in generated:
        raise ValueError("Could not locate the reviewer input schema for a fail-closed retry update")
    generated = generated.replace(old_schema, new_schema, 1)

    old_bt = '''        try:
            protected, originals, expected = mask_placeholders(row["translated_question"])'''
    new_bt = '''        reused = row["reuse_backtranslation_en"]
        if reused is not None:
            record["backtranslation_en"] = reused
            record["backtranslation_sha256"] = sha(reused)
            records.append(record)
            checkpoint("IN_PROGRESS_BACKTRANSLATION", rows, records, runtime)
            print(f"Reused verified backtranslation {index}/{len(rows)}", flush=True)
            continue
        try:
            protected, originals, expected = mask_placeholders(row["translated_question"])'''
    if old_bt not in generated:
        raise ValueError("Could not locate the backtranslation loop for safe checkpoint reuse")
    generated = generated.replace(old_bt, new_bt, 1)
    compile(generated, "retry_review.py", "exec")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    code_path = OUTPUT_DIR / "review_retry.py"
    code_path.write_text(generated, encoding="utf-8")
    retry_input_path = HERE / "retry_input_full799_unresolved_v1.json"
    retry_input_path.write_text(raw_input + "\n", encoding="utf-8")
    kernel_meta = json.loads((SOURCE_SCRIPT.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    kernel_meta.update({"id": "bunnoob2005/synthworlds-translation-repair-retry-v1",
                        "title": "SynthWorlds Translation Repair Retry v1",
                        "code_file": code_path.name})
    (OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8")
    build = {
        "status": "MASKED_RETRY_INPUT_READY_NOT_APPROVED",
        "mode": MODE,
        "rows": len(retries),
        "retries_by_reason": dict(reasons),
        "review_input_sha256": input_hash,
        "review_input_json_sha256": sha(retry_input_path.read_bytes()),
        "review_script_sha256": sha(generated.encode("utf-8")),
        "source_input_sha256": launch["input_jsonl_sha256"],
        "partial_output_sha256": sha(result_path.read_bytes()),
        "partial_verification_sha256": sha(verification_path.read_bytes()),
        "reused_backtranslations_are_source_bound": True,
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
    }
    (HERE / "retry_review_build_manifest.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(build, indent=2))


if __name__ == "__main__":
    main()
