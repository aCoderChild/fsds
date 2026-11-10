#!/usr/bin/env python3
"""Verify a completed 97-row retry against its source-bound build and Kaggle artifacts.

This produces diagnostic evidence only. It never changes canonical translations or
approves a training corpus. Incomplete/worker-killed retries fail closed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "independent_review_inputs_2026-11-04"
BUILD = HERE / "retry_review_build_manifest.json"
RETRY_INPUT = HERE / "retry_input_full799_unresolved_v1.json"
ORIGINAL_PARTIAL = HERE / "remote_output/full799/version_5"
MODE = "retry_full799_unresolved_v1"
KERNEL_ID = "bunnoob2005/synthworlds-translation-repair-retry-v1"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha(value: str) -> str:
    return sha(value.encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def log_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    chunks = []
    for line in raw.splitlines():
        try:
            entry = json.loads(line.lstrip(","))
        except json.JSONDecodeError:
            continue
        if entry.get("stream_name") in {"stdout", "stderr"}:
            chunks.append(entry.get("data", ""))
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True,
                        help="Directory downloaded from the completed Kaggle retry kernel")
    args = parser.parse_args()
    remote = args.remote_dir

    full_path = INPUTS / "full799.jsonl"
    full_launch_path = INPUTS / "full799.kaggle_launch_manifest.json"
    full_rows = read_jsonl(full_path)
    full_launch = read_json(full_launch_path)
    full_map = read_jsonl(INPUTS / "full799.private_map.jsonl")
    full_by_key = {row["review_key"]: row for row in full_rows}
    map_by_key = {row["review_key"]: row for row in full_map}
    if (len(full_rows) != 799 or len(full_by_key) != 799 or len(map_by_key) != 799
            or full_launch.get("input_jsonl_sha256") != sha(full_path.read_bytes())
            or full_launch.get("private_map_sha256") != sha((INPUTS / "full799.private_map.jsonl").read_bytes())):
        raise ValueError("Canonical full-review input or private map failed its source launch-manifest checks")

    partial_result_path = ORIGINAL_PARTIAL / "repair_review.json"
    partial_receipt_path = ORIGINAL_PARTIAL / "verification.json"
    partial_rows_path = ORIGINAL_PARTIAL / "partial_verified_repair_reviews.jsonl"
    partial_result = read_json(partial_result_path)
    partial_receipt = read_json(partial_receipt_path)
    partial_verified = read_jsonl(partial_rows_path)
    partial_records = partial_result.get("records", [])
    partial_by_key = {row.get("review_key"): row for row in partial_records}
    if (partial_receipt.get("status") != "VERIFIED_KILLED_PARTIAL_AI_ONLY_REPAIR_REVIEW"
            or partial_receipt.get("output_sha256") != sha(partial_result_path.read_bytes())
            or partial_receipt.get("verified_rows_sha256") != sha(partial_rows_path.read_bytes())
            or partial_result.get("input_sha256") != full_launch.get("input_compact_array_sha256")
            or len(partial_by_key) != 799 or set(partial_by_key) != set(full_by_key)):
        raise ValueError("Original killed-partial receipt is invalid or does not match the 799-row source")

    expected_retry = []
    for row in full_rows:
        checkpoint = partial_by_key[row["review_key"]]
        decision = checkpoint.get("decision")
        unresolved = (not isinstance(decision, dict) or "error" in decision
                      or checkpoint.get("backtranslation_error") is not None)
        if unresolved:
            reused = checkpoint.get("backtranslation_en") if checkpoint.get("backtranslation_error") is None else None
            expected_retry.append({
                **row,
                "reuse_backtranslation_en": reused,
                "reuse_backtranslation_sha256": checkpoint.get("backtranslation_sha256") if reused else None,
            })
    if len(expected_retry) != 97:
        raise ValueError(f"Expected 97 unresolved rows from the partial, found {len(expected_retry)}")

    build = read_json(BUILD)
    input_rows = read_json(RETRY_INPUT)
    compact_input = json.dumps(input_rows, ensure_ascii=False, separators=(",", ":"))
    if (len(input_rows) != 97 or input_rows != expected_retry
            or build.get("status") != "MASKED_RETRY_INPUT_READY_NOT_APPROVED"
            or build.get("mode") != MODE or build.get("rows") != 97
            or build.get("review_input_sha256") != sha(compact_input.encode("utf-8"))
            or build.get("review_input_json_sha256") != sha(RETRY_INPUT.read_bytes())
            or build.get("source_input_sha256") != full_launch.get("input_jsonl_sha256")
            or build.get("partial_output_sha256") != sha(partial_result_path.read_bytes())
            or build.get("partial_verification_sha256") != sha(partial_receipt_path.read_bytes())):
        raise ValueError("Retry input/build manifest does not exactly reconstruct the unresolved source rows")

    result_path = remote / "repair_review.json"
    log_path_candidates = sorted(remote.glob("*.log"))
    if not result_path.is_file() or len(log_path_candidates) != 1:
        raise ValueError("Expected repair_review.json and exactly one Kaggle run log")
    result = read_json(result_path)
    log_path = log_path_candidates[0]
    source_candidates = [p for p in remote.rglob("*.py") if p.is_file()]
    source_matches = [p for p in source_candidates if sha(p.read_bytes()) == build.get("review_script_sha256")]
    if len(source_matches) != 1:
        raise ValueError("Kaggle-pulled retry source does not uniquely match the submitted script hash")
    source_path = source_matches[0]

    if (result.get("mode") != MODE
            or result.get("input_sha256") != build.get("review_input_sha256")
            or result.get("nllb_model_id") != NLLB_ID
            or result.get("nllb_model_revision") != NLLB_REVISION
            or result.get("qwen_model_id") != QWEN_ID
            or result.get("qwen_model_revision") != QWEN_REVISION
            or result.get("quantization") != "4-bit NF4"
            or result.get("bitsandbytes") != "0.50.2"
            or result.get("human_translation_reviewed") is not False
            or result.get("answers_or_splits_sent") is not False):
        raise ValueError("Retry output model, input, quantization, or blinding provenance mismatch")

    records = result.get("records")
    if not isinstance(records, list) or len(records) != len(input_rows):
        raise ValueError("Retry output does not contain exactly 97 records")
    retry_by_key = {row.get("review_key"): row for row in records}
    retry_input_by_key = {row["review_key"]: row for row in input_rows}
    if (len(retry_by_key) != 97 or set(retry_by_key) != set(retry_input_by_key)
            or [r.get("review_key") for r in records] != [r["review_key"] for r in input_rows]):
        raise ValueError("Retry output has duplicate, missing, extra, or reordered row keys")

    statuses = collections.Counter()
    cells = collections.defaultdict(collections.Counter)
    verified_rows = []
    for key, row in retry_input_by_key.items():
        record = retry_by_key[key]
        canonical = full_by_key[key]
        local_map = map_by_key[key]
        if (record.get("language") != row["language"] or record.get("kind") != row["kind"]
                or record.get("source_sha256") != text_sha(canonical["source_question_en"])
                or record.get("translation_sha256") != text_sha(canonical["translated_question"])
                or local_map.get("language") != row["language"] or local_map.get("kind") != row["kind"]):
            raise ValueError(f"{key}: returned row identity is not bound to its frozen source and candidate")

        source_markers = collections.Counter(PLACEHOLDER.findall(canonical["source_question_en"]))
        target_markers = collections.Counter(PLACEHOLDER.findall(canonical["translated_question"]))
        bt = record.get("backtranslation_en")
        bt_error = record.get("backtranslation_error")
        if row["reuse_backtranslation_en"] is not None:
            if (bt != row["reuse_backtranslation_en"]
                    or record.get("backtranslation_sha256") != row["reuse_backtranslation_sha256"]):
                raise ValueError(f"{key}: reused backtranslation differs from its verified checkpoint")
        if bt_error is None:
            if (not isinstance(bt, str) or not bt.strip()
                    or record.get("backtranslation_sha256") != text_sha(bt)
                    or collections.Counter(PLACEHOLDER.findall(bt)) != source_markers
                    or source_markers != target_markers):
                raise ValueError(f"{key}: backtranslation hash or protected-marker check failed")
            markers_pass = True
        else:
            if bt not in {"", None}:
                raise ValueError(f"{key}: errored backtranslation unexpectedly contains text")
            markers_pass = False

        decision = record.get("decision")
        if not isinstance(decision, dict):
            raise ValueError(f"{key}: completed retry has no explicit decision/block")
        if "error" in decision:
            if set(decision) != {"error"} or not isinstance(decision["error"], str) or not decision["error"].strip():
                raise ValueError(f"{key}: malformed blocked decision")
            if record.get("raw_output") is not None:
                raise ValueError(f"{key}: blocked decision should not include an accepted raw output")
            status = "AI_ONLY_REPAIR_REVIEW_BLOCKED"
        else:
            parsed = json.loads(record.get("raw_output", ""), object_pairs_hook=unique_object)
            if (set(parsed) != {"decisions"} or not isinstance(parsed["decisions"], list)
                    or len(parsed["decisions"]) != 1 or parsed["decisions"][0] != decision
                    or set(decision) != {"key", "semantic_pass", "fluency_pass", "reason"}
                    or decision["key"] != key or type(decision["semantic_pass"]) is not bool
                    or type(decision["fluency_pass"]) is not bool
                    or not isinstance(decision["reason"], str) or not decision["reason"].strip()
                    or bt_error is not None):
                raise ValueError(f"{key}: raw adjudicator output is invalid or differs from parsed decision")
            status = ("AI_ONLY_REPAIR_REVIEW_PASS" if markers_pass and decision["semantic_pass"]
                      and decision["fluency_pass"] else "AI_ONLY_REPAIR_REVIEW_FLAGGED")
        statuses[status] += 1
        cells[f"{row['language']}/{row['kind']}"][status] += 1
        verified_rows.append({
            "review_key": key,
            "prompt_id": local_map["prompt_id"],
            "language": row["language"],
            "kind": row["kind"],
            "source_sha256": text_sha(canonical["source_question_en"]),
            "translation_sha256": text_sha(canonical["translated_question"]),
            "candidate_translation_sha256": local_map["candidate_translation_sha256"],
            "backtranslation_en": bt or "",
            "backtranslation_sha256": record.get("backtranslation_sha256"),
            "backtranslation_markers_pass": markers_pass,
            "decision": decision,
            "review_status": status,
            "human_translation_reviewed": False,
        })

    remote_status = result.get("status")
    if remote_status not in {"AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY",
                             "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS"}:
        raise ValueError(f"Retry did not finish cleanly; refusing partial or unexpected status {remote_status!r}")
    log = log_text(log_path)
    progress_values = [int(value) for value in re.findall(r"Reviewed (\d+)/97", log)]
    if not progress_values or max(progress_values) != 97 or "completed_at_utc" not in result:
        raise ValueError("Run log/output do not prove all 97 semantic reviews reached finalization")
    observed_blocked = sum(status == "AI_ONLY_REPAIR_REVIEW_BLOCKED" for status in statuses.elements())
    if (remote_status == "AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY" and observed_blocked != 0
            or remote_status == "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS" and observed_blocked == 0):
        raise ValueError("Remote terminal status disagrees with independently counted blocked rows")

    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in verified_rows)
    report = {
        "status": "VERIFIED_COMPLETE_RETRY_AI_ONLY_DIAGNOSTIC",
        "remote_status": remote_status,
        "kernel_id": KERNEL_ID,
        "mode": MODE,
        "rows": 97,
        "review_script_sha256": sha(source_path.read_bytes()),
        "submitted_script_matches_build_manifest": True,
        "retry_input_sha256": sha(compact_input.encode("utf-8")),
        "retry_input_json_sha256": sha(RETRY_INPUT.read_bytes()),
        "source_full799_sha256": sha(full_path.read_bytes()),
        "candidate_map_sha256": sha((INPUTS / "full799.private_map.jsonl").read_bytes()),
        "original_partial_output_sha256": sha(partial_result_path.read_bytes()),
        "original_partial_verification_sha256": sha(partial_receipt_path.read_bytes()),
        "output_sha256": sha(result_path.read_bytes()),
        "run_log_sha256": sha(log_path.read_bytes()),
        "verified_retry_rows_sha256": sha(payload.encode("utf-8")),
        "models": {"backtranslator": NLLB_ID, "backtranslator_revision": NLLB_REVISION,
                   "adjudicator": QWEN_ID, "adjudicator_revision": QWEN_REVISION},
        "gpu": result.get("gpu"),
        "counts": dict(statuses),
        "counts_by_language_kind": {key: dict(value) for key, value in sorted(cells.items())},
        "release_approved": False,
        "training_eligible": False,
        "human_translation_reviewed": False,
        "answers_or_splits_sent": False,
        "limitations": [
            "This is an AI-only diagnostic; it is not fluent-human translation certification.",
            "A pass is only a candidate for the frozen production re-screen, not corpus approval.",
            "NLLB back-translation and Qwen judgments are not ground truth; reviewer errors may be correlated.",
        ],
    }
    (remote / "retry_verified_repair_reviews.jsonl").write_text(payload, encoding="utf-8")
    (remote / "retry_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
