#!/usr/bin/env python3
"""Verify private Kaggle review outputs against the local masked input and map."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "independent_review_inputs_2026-11-04"
PACKAGE = HERE / "candidate_review_kaggle"
KERNEL_ID = "bunnoob2005/synthworlds-ai-only-translation-repair-review"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot16", "full799"), default="pilot16")
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--allow-killed-partial", action="store_true",
                        help="Verify a hash-bound in-progress checkpoint after a logged Kaggle worker kill; never treat it as complete.")
    args = parser.parse_args()
    input_path = INPUT_DIR / f"{args.mode}.jsonl"
    map_path = INPUT_DIR / f"{args.mode}.private_map.jsonl"
    launch_path = INPUT_DIR / f"{args.mode}.kaggle_launch_manifest.json"
    result_path = args.remote_dir / "repair_review.json"
    source_path = args.remote_dir / "source/synthworlds-ai-only-translation-repair-review.py"
    log_path = args.remote_dir / "synthworlds-ai-only-translation-repair-review.log"

    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    rows = read_jsonl(input_path)
    mapping = read_jsonl(map_path)
    if (launch.get("kernel_id") != KERNEL_ID or launch.get("mode") != args.mode
            or launch.get("input_jsonl_sha256") != sha(input_path.read_bytes())
            or launch.get("private_map_sha256") != sha(map_path.read_bytes())
            or len(rows) != len(mapping) or launch.get("rows") != len(rows)):
        raise ValueError("Local launch manifest, input, or row map failed integrity checks")
    if not source_path.is_file() or sha(source_path.read_bytes()) != launch.get("review_script_sha256"):
        raise ValueError("Kaggle-pulled reviewer does not match the exact script hash in the launch manifest")

    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("mode") != args.mode or result.get("input_sha256") != sha(raw.encode("utf-8"))
            or result.get("nllb_model_id") != NLLB_ID or result.get("nllb_model_revision") != NLLB_REVISION
            or result.get("qwen_model_id") != QWEN_ID or result.get("qwen_model_revision") != QWEN_REVISION
            or result.get("quantization") != "4-bit NF4" or result.get("bitsandbytes") != "0.50.2"):
        raise ValueError("Remote model, input, or quantization provenance mismatch")
    output = result.get("records")
    by_key = {row.get("review_key"): row for row in output or []}
    input_by_key = {row["review_key"]: row for row in rows}
    map_by_key = {row["review_key"]: row for row in mapping}
    if (len(by_key) != len(rows) or len(map_by_key) != len(mapping)
            or set(by_key) != set(input_by_key) or set(input_by_key) != set(map_by_key)):
        raise ValueError("Remote output coverage differs from the complete local input/map")

    remote_status = result.get("status")
    if remote_status not in {
        "AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY",
        "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS",
        "IN_PROGRESS_QWEN14B_REVIEW",
    }:
        raise ValueError(f"Unexpected remote reviewer status: {remote_status}")
    pending_keys = {key for key, record in by_key.items() if not isinstance(record.get("decision"), dict)}
    if pending_keys and not args.allow_killed_partial:
        raise ValueError("At least one input row has no final Qwen decision or explicit block")

    counts = collections.Counter()
    cells = collections.defaultdict(collections.Counter)
    reviewed_rows = []
    pending_rows = []
    for key, input_row in input_by_key.items():
        record = by_key[key]
        local_map = map_by_key[key]
        if (record.get("language") != input_row["language"] or record.get("kind") != input_row["kind"]
                or record.get("source_sha256") != sha(input_row["source_question_en"].encode("utf-8"))
                or record.get("translation_sha256") != sha(input_row["translated_question"].encode("utf-8"))
                or local_map.get("language") != input_row["language"] or local_map.get("kind") != input_row["kind"]):
            raise ValueError(f"{key}: output source, target, or map identity mismatch")

        source_markers = collections.Counter(PLACEHOLDER.findall(input_row["source_question_en"]))
        target_markers = collections.Counter(PLACEHOLDER.findall(input_row["translated_question"]))
        bt = record.get("backtranslation_en", "")
        bt_markers = collections.Counter(PLACEHOLDER.findall(bt))
        markers_pass = bool(bt.strip()) and source_markers == target_markers == bt_markers
        if record.get("backtranslation_error") is None and not markers_pass:
            raise ValueError(f"{key}: backtranslation did not preserve all source/target placeholders")
        if bt and record.get("backtranslation_sha256") != sha(bt.encode("utf-8")):
            raise ValueError(f"{key}: backtranslation hash mismatch")

        decision = record.get("decision")
        if not isinstance(decision, dict):
            pending_rows.append(key)
            continue
        if isinstance(decision, dict) and "error" not in decision:
            parsed = json.loads(record.get("raw_output", ""), object_pairs_hook=unique_object)
            if (set(parsed) != {"decisions"} or not isinstance(parsed["decisions"], list)
                    or len(parsed["decisions"]) != 1 or parsed["decisions"][0] != decision
                    or set(decision) != {"key", "semantic_pass", "fluency_pass", "reason"}
                    or decision["key"] != key or type(decision["semantic_pass"]) is not bool
                    or type(decision["fluency_pass"]) is not bool
                    or not isinstance(decision["reason"], str) or not decision["reason"].strip()):
                raise ValueError(f"{key}: raw Qwen output is malformed or differs from parsed decision")
            status = "AI_ONLY_REPAIR_REVIEW_PASS" if markers_pass and decision["semantic_pass"] and decision["fluency_pass"] else "AI_ONLY_REPAIR_REVIEW_FLAGGED"
            counts[status] += 1
            cells[f"{input_row['language']}/{input_row['kind']}"][status] += 1
        else:
            status = "AI_ONLY_REPAIR_REVIEW_BLOCKED"
            counts[status] += 1
            cells[f"{input_row['language']}/{input_row['kind']}"][status] += 1
        reviewed_rows.append({
            "review_key": key, "prompt_id": local_map["prompt_id"], "language": input_row["language"],
            "kind": input_row["kind"], "candidate_translation_sha256": local_map["candidate_translation_sha256"],
            "backtranslation_en": bt, "backtranslation_sha256": record.get("backtranslation_sha256"),
            "backtranslation_markers_pass": markers_pass, "decision": decision,
            "review_status": status, "human_translation_reviewed": False,
        })

    log_raw = log_path.read_text(encoding="utf-8", errors="replace")
    log_parts = []
    for line in log_raw.splitlines():
        try:
            entry = json.loads(line.lstrip(","))
        except json.JSONDecodeError:
            continue
        if entry.get("stream_name") in {"stdout", "stderr"}:
            log_parts.append(entry.get("data", ""))
    log = "".join(log_parts)
    progress_values = [int(value) for value in re.findall(r"Reviewed (\d+)/", log)]
    progress = max(progress_values, default=0)
    finalization_error = None
    partial_run = False
    if remote_status == "IN_PROGRESS_QWEN14B_REVIEW":
        expected_aggregation_error = (
            "TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'" in log
            and 'blocked = sum("error" in (r.get("decision") or {}) or r.get("backtranslation_error") for r in records)' in log
        )
        worker_killed = "Killed" in log or "killed" in log
        if expected_aggregation_error and progress == len(rows):
            finalization_error = "Remote script failed in the final blocked-row tally after all per-row decisions were checkpointed."
        elif args.allow_killed_partial and worker_killed and 0 < progress < len(rows):
            partial_run = True
            finalization_error = f"Kaggle worker was killed after {progress}/{len(rows)} semantic review rows; remaining rows are pending."
            if progress != len(rows) - len(pending_keys):
                raise ValueError("Checkpoint decisions disagree with the last logged semantic-review progress")
        else:
            raise ValueError("Remote checkpoint is incomplete or stopped for an unrecognized reason")
    elif progress != len(rows):
        raise ValueError(f"Kaggle run log shows {progress}/{len(rows)} semantic review rows")
    elif pending_keys:
        raise ValueError("A completed remote status cannot contain pending Qwen decisions")
    derived_blocked = sum(
        "error" in (record.get("decision") or {}) or record.get("backtranslation_error") is not None
        for record in by_key.values()
    )
    derived_reviewed = sum(
        isinstance(record.get("decision"), dict) and "error" not in record["decision"]
        and record.get("backtranslation_error") is None
        for record in by_key.values()
    )
    verified_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in reviewed_rows
    )
    report = {
        "status": ("VERIFIED_KILLED_PARTIAL_AI_ONLY_REPAIR_REVIEW" if partial_run
                   else "VERIFIED_AI_ONLY_REPAIR_REVIEW_DIAGNOSTIC" if finalization_error is None
                   else "VERIFIED_ROWS_REMOTE_FINALIZATION_FAILED_AFTER_ALL_DECISIONS"),
        "remote_status": remote_status,
        "remote_finalization_error": finalization_error,
        "release_approved": False,
        "training_eligible": False,
        "mode": args.mode,
        "rows": len(rows),
        "semantic_review_progress": progress,
        "kernel_id": KERNEL_ID,
        "model_id": QWEN_ID,
        "model_revision": QWEN_REVISION,
        "backtranslator_model_id": NLLB_ID,
        "backtranslator_model_revision": NLLB_REVISION,
        "input_sha256": sha(raw.encode("utf-8")),
        "review_script_sha256": sha(source_path.read_bytes()),
        "submitted_script_matches_launch_manifest": True,
        "output_sha256": sha(result_path.read_bytes()),
        "run_log_sha256": sha(log_path.read_bytes()),
        "gpu": result.get("gpu"), "torch": result.get("torch"),
        "transformers": result.get("transformers"), "accelerate": result.get("accelerate"),
        "bitsandbytes": result.get("bitsandbytes"), "quantization": result.get("quantization"),
        "reviewed_rows": derived_reviewed, "blocked_rows": derived_blocked,
        "pending_rows": len(pending_rows), "pending_review_keys": pending_rows,
        "verified_rows_sha256": sha(verified_payload.encode("utf-8")),
        "counts": dict(counts),
        "counts_by_language_kind": {key: dict(value) for key, value in sorted(cells.items())},
        "human_translation_reviewed": False,
        "answers_or_splits_sent": False,
        "limitations": ["AI-only review is not fluent-human certification.",
                        "The NLLB backtranslation is lower-capacity and is diagnostic evidence, not ground truth.",
                        "Qwen2.5 7B and Qwen2.5 14B share a model family; reviewer errors can be correlated.",
                        "This diagnostic does not modify source translation files or approve an SFT release."],
    }
    (args.remote_dir / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verified_path = (args.remote_dir / "partial_verified_repair_reviews.jsonl" if partial_run
                     else INPUT_DIR / f"{args.mode}.verified_repair_reviews.jsonl")
    verified_path.write_text(verified_payload, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
