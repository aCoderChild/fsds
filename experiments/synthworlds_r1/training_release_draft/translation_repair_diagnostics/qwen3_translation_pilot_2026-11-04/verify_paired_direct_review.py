#!/usr/bin/env python3
"""Verify the direct paired screen's provenance, runtime, calibration, and row coverage."""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "paired_direct_review_input.json"
PRIVATE_MAP = HERE / "paired_direct_review_private_map.json"
BUILD = HERE / "paired_direct_review_build_manifest.json"
SCRIPT = HERE / "paired_direct_review.py"
KERNEL_DIR = HERE / "paired_direct_review_kernel"
REMOTE_SCRIPT = HERE / "paired_direct_review_remote_source" / "swsm-qwen14-direct-paired-calibration-2026-11-04.py"
OUTPUT = HERE / "paired_direct_review_remote_output" / "paired_direct_review.json"
REMOTE_LOG = HERE / "paired_direct_review_remote_output" / "swsm-qwen14-direct-paired-calibration-2026-11-04.log"
RUN_RECEIPT = HERE / "paired_direct_review_kaggle_run.json"
REPORT = HERE / "paired_direct_review_verification.json"
CONTROL_REVIEWER = HERE.parents[4] / "reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls/review.py"
CONTROL_INPUT = HERE.parents[4] / "reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls/input_array.json"
PAIR_INPUT = HERE / "paired_review_input.json"
PAIR_PRIVATE_MAP = HERE / "paired_review_private_map.json"
PAIR_OUTPUT = HERE / "paired_review_remote_output" / "repair_review.json"
PAIR_VERIFICATION = HERE / "paired_review_verification.json"
QWEN_INPUT = HERE / "generation_input.json"
PACKAGES = {"transformers": "4.51.3", "accelerate": "1.15.0"}
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode("utf-8"))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assignment_string(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError(f"String assignment missing: {name}")


def main() -> None:
    rows = read(INPUT)
    private = read(PRIVATE_MAP)
    build = read(BUILD)
    result = read(OUTPUT)
    run_receipt = read(RUN_RECEIPT)
    pair_rows = read(PAIR_INPUT)
    pair_private = read(PAIR_PRIVATE_MAP)
    pair_output = read(PAIR_OUTPUT)
    pair_verification = read(PAIR_VERIFICATION)
    qwen_rows = read(QWEN_INPUT)
    controls = read(CONTROL_INPUT)
    script_hash = sha_bytes(SCRIPT.read_bytes())
    if not REMOTE_SCRIPT.is_file():
        raise ValueError("Pulled Kaggle direct-review source is missing")
    remote_script_hash = sha_bytes(REMOTE_SCRIPT.read_bytes())
    if remote_script_hash != script_hash:
        raise ValueError("Pulled Kaggle reviewer source differs from the reviewed local script")
    if (run_receipt.get("status") != "KernelWorkerStatus.COMPLETE"
            or run_receipt.get("status_api") != "Authenticated Kaggle API kernels_status: COMPLETE"
            or run_receipt.get("kernel_id") != "bunnoob2005/swsm-qwen14-direct-paired-calibration-2026-11-04"
            or run_receipt.get("review_script_sha256") != script_hash
            or run_receipt.get("output_sha256") != sha_bytes(OUTPUT.read_bytes())
            or run_receipt.get("log_sha256") != sha_bytes(REMOTE_LOG.read_bytes())):
        raise ValueError("Kaggle completion receipt does not bind to this exact output/run")
    log_text = REMOTE_LOG.read_text(encoding="utf-8")
    log_times = [float(value) for value in re.findall(r'"time":([0-9.]+)', log_text)]
    if (not log_times
            or abs(max(log_times) - run_receipt.get("runtime_seconds_log_last_event", 0)) > 0.005):
        raise ValueError("Kaggle completion receipt runtime does not match the downloaded log")
    if sha_bytes(PRIVATE_MAP.read_bytes()) != build.get("direct_private_map_sha256"):
        raise ValueError("Private direct-screen row map hash mismatch")
    if sha_bytes(PAIR_PRIVATE_MAP.read_bytes()) != build.get("candidate_private_map_sha256"):
        raise ValueError("Original candidate map hash changed after review setup")
    if sha_bytes((HERE / "paired_review_build_manifest.json").read_bytes()) != build.get("paired_roundtrip_build_sha256"):
        raise ValueError("Paired round-trip build manifest changed")
    if (pair_verification.get("status") != "VERIFIED_BLINDED_PAIRED_AI_SCREEN_DIAGNOSTIC_ONLY"
            or pair_verification.get("output_sha256") != sha_bytes(PAIR_OUTPUT.read_bytes())):
        raise ValueError("Paired round-trip result receipt is missing or changed")
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    if (sha_text(raw) != build.get("direct_input_sha256")
            or result.get("input_sha256") != sha_text(raw)):
        raise ValueError("Direct input hash mismatch")
    if result.get("prompt_sha256") != build.get("calibration_instruction_sha256"):
        raise ValueError("Direct-screen prompt hash mismatch")
    model = ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
    if (build.get("model_id"), build.get("model_revision")) != model:
        raise ValueError("Build manifest model revision mismatch")
    if (result.get("model_id"), result.get("model_revision")) != model:
        raise ValueError("Remote result model revision mismatch")
    if {name: result.get(name) for name in PACKAGES} != PACKAGES:
        raise ValueError("Runtime packages differ from the pinned review setup")
    if (result.get("cuda_device_count", 0) < 2 or len(result.get("gpu_devices", [])) < 2
            or any("T4" not in device for device in result["gpu_devices"])):
        raise ValueError("Review did not run on the configured two-T4 runtime")
    device_map = result.get("device_map") or {}
    mapped = {str(value).replace("cuda:", "") for value in device_map.values()}
    if not {"0", "1"}.issubset(mapped) or "disk" in {value.lower() for value in mapped}:
        raise ValueError("FP16 model did not use both GPUs or unexpectedly offloaded to disk")
    if (result.get("status") not in {
            "AI_ONLY_DIRECT_SCREEN_COMPLETE_DIAGNOSTIC_ONLY",
            "AI_ONLY_DIRECT_SCREEN_WITH_BLOCKED_ROWS"}
            or result.get("answers_or_splits_sent") is not False
            or result.get("human_translation_reviewed") is not False):
        raise ValueError("Unexpected result status or misrepresented review metadata")
    if not result.get("started_at_utc") or run_receipt.get("runtime_seconds_log_last_event", 0) <= 0:
        raise ValueError("Remote result lacks a valid start time or completed-run duration")
    kernel_files = {path.name for path in KERNEL_DIR.iterdir() if path.is_file()}
    if kernel_files != {"README.md", "kernel-metadata.json", "review.py"}:
        raise ValueError(f"Unexpected files uploaded to private Kaggle kernel: {sorted(kernel_files)}")

    pair_private_by_key = {row["review_key"]: row for row in pair_private}
    candidate_by_key = {row["review_key"]: row for row in pair_rows}
    controls_by_id = {row["control_id"]: row for row in controls}
    expected_candidate_keys = set(candidate_by_key)
    if len(candidate_by_key) != 32 or set(pair_private_by_key) != expected_candidate_keys:
        raise ValueError("Original paired candidate input or private map is incomplete")
    private_by_key = {row["review_key"]: row for row in private}
    row_by_key = {row["review_key"]: row for row in rows}
    record_keys = [record.get("review_key") for record in result.get("records", [])]
    if (len(rows) != 48 or len(row_by_key) != 48 or set(private_by_key) != set(row_by_key)
            or len(record_keys) != 48 or len(set(record_keys)) != 48 or set(record_keys) != set(row_by_key)):
        raise ValueError("Expected 48 unique blinded records and exact private map coverage")
    if [record.get("review_key") for record in result["records"]] != [row["review_key"] for row in rows]:
        raise ValueError("Remote review rows are missing, duplicated, extra, or reordered")
    if set(private_by_key) != set(row_by_key):
        raise ValueError("Remote rows and local private map differ")

    pair_input_by_key = {row["review_key"]: row for row in pair_rows}
    roundtrip_by_source_method = {}
    for record in pair_output.get("records", []):
        hidden_pair = pair_private_by_key.get(record.get("review_key"))
        source_pair = pair_input_by_key.get(record.get("review_key"))
        if hidden_pair is None or source_pair is None:
            raise ValueError("Round-trip output contains a row outside its private candidate map")
        if (record.get("translation_sha256") != hidden_pair.get("candidate_sha256")
                or record.get("source_sha256") != sha_text(source_pair["source_question_en"])):
            raise ValueError("Round-trip result hashes do not match paired candidate input")
        decision = record.get("decision") or {}
        if record.get("backtranslation_error") or "error" in decision:
            roundtrip_label = "blocked"
        else:
            roundtrip_label = "pass" if decision.get("semantic_pass") is True and decision.get("fluency_pass") is True else "flag"
        roundtrip_by_source_method[(hidden_pair["source_review_key"], hidden_pair["candidate_method"])] = roundtrip_label
    if len(roundtrip_by_source_method) != 32:
        raise ValueError("Round-trip results do not cover all paired candidates")

    labels_by_source: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    candidate_counts = collections.Counter()
    candidate_by_method_language = collections.defaultdict(collections.Counter)
    roundtrip_direct_transitions = collections.defaultdict(collections.Counter)
    sentinel_results = []
    case_results = []
    blocked = 0
    for record in result["records"]:
        key = record["review_key"]
        source = row_by_key[key]
        hidden = private_by_key[key]
        if (record.get("language"), record.get("kind")) != (source["language"], source["kind"]):
            raise ValueError(f"Language/kind mismatch at {key}")
        if record.get("source_sha256") != sha_text(source["source_question_en"]):
            raise ValueError(f"Source text hash mismatch at {key}")
        translation_hash = sha_text(source["translated_question"])
        if (record.get("translation_sha256") != translation_hash
                or hidden.get("candidate_sha256") != translation_hash):
            raise ValueError(f"Candidate hash mismatch at {key}")
        decision = record.get("decision") or {}
        if "error" in decision:
            label = "blocked"
            blocked += 1
        else:
            if (set(decision) != {"semantic_pass", "fluency_pass", "reason"}
                    or type(decision["semantic_pass"]) is not bool
                    or type(decision["fluency_pass"]) is not bool
                    or not isinstance(decision["reason"], str) or not decision["reason"].strip()):
                raise ValueError(f"Invalid decision schema at {key}")
            label = "pass" if decision["semantic_pass"] and decision["fluency_pass"] else "flag"

        if hidden.get("row_role") == "candidate":
            method = hidden["candidate_method"]
            if method not in {"qwen3_14b", "assistant_ai_candidate"}:
                raise ValueError(f"Unexpected candidate label in local map at {key}")
            if key not in expected_candidate_keys or hidden["source_review_key"] not in {
                    row["review_key"] for row in read(QWEN_INPUT)}:
                raise ValueError(f"Candidate row does not map to the original source set at {key}")
            if (pair_private_by_key[key].get("candidate_method") != method
                    or pair_private_by_key[key].get("source_review_key") != hidden["source_review_key"]
                    or pair_private_by_key[key].get("candidate_sha256") != hidden["candidate_sha256"]):
                raise ValueError(f"Candidate method map changed at {key}")
            candidate_counts[(method, label)] += 1
            candidate_by_method_language[(method, source["language"])][label] += 1
            old_label = roundtrip_by_source_method.get((hidden["source_review_key"], method))
            if old_label is None:
                raise ValueError(f"Candidate missing its round-trip decision at {key}")
            roundtrip_direct_transitions[method][f"{old_label}_to_{label}"] += 1
            labels_by_source[hidden["source_review_key"]][method] = {
                "label": label,
                "roundtrip_label": old_label,
                "semantic_pass": decision.get("semantic_pass"),
                "fluency_pass": decision.get("fluency_pass"),
                "reason": decision.get("reason"),
                "review_key": key,
                "translation_sha256": translation_hash,
                "translated_question": source["translated_question"],
            }
        elif hidden.get("row_role") == "known_mismatch_sentinel":
            control = controls_by_id.get(hidden.get("control_id"))
            if (control is None
                    or source["language"] != control["language"]
                    or source["kind"] != control["kind"]
                    or source["source_question_en"] != control["source"]
                    or source["translated_question"] != control["translation"]
                    or hidden["candidate_sha256"] != sha_text(control["translation"])):
                raise ValueError(f"Known-mismatch sentinel source/hash binding failed at {key}")
            sentinel_results.append({
                "control_id": hidden["control_id"],
                "language": source["language"],
                "kind": source["kind"],
                "semantic_pass": decision.get("semantic_pass"),
                "fluency_pass": decision.get("fluency_pass"),
                "label": label,
                "reason": decision.get("reason"),
            })
        else:
            raise ValueError(f"Unknown row role in local private map at {key}")

    expected_methods = {"qwen3_14b", "assistant_ai_candidate"}
    if (set(labels_by_source) != {row["review_key"] for row in qwen_rows}
            or any(set(by_method) != expected_methods for by_method in labels_by_source.values())
            or len(sentinel_results) != 16
            or {row["control_id"] for row in sentinel_results} != set(controls_by_id)):
        raise ValueError("Paired candidate or calibration sentinel coverage is incomplete")
    if result.get("reviewed_rows") != 48 - blocked or result.get("blocked_rows") != blocked:
        raise ValueError("Remote counters do not agree with row decisions")

    control_source = CONTROL_REVIEWER.read_text(encoding="utf-8")
    control_instruction = assignment_string(control_source, "instruction")
    direct_instruction = assignment_string(SCRIPT.read_text(encoding="utf-8"), "PROMPT_INSTRUCTION")
    prompt_matches_calibration = direct_instruction == control_instruction
    if not prompt_matches_calibration:
        raise ValueError("Direct-review instruction differs from the calibrated control prompt")
    semantic_rejects = sum(item["semantic_pass"] is False for item in sentinel_results)
    semantic_accepts = sum(item["semantic_pass"] is True for item in sentinel_results)
    sentinel_blocked = sum(item["label"] == "blocked" for item in sentinel_results)
    calibration_passed = semantic_rejects >= 15 and sentinel_blocked == 0
    transitions = collections.Counter(
        f"qwen3_{labels['qwen3_14b']['label']}__assistant_{labels['assistant_ai_candidate']['label']}"
        for labels in labels_by_source.values())
    for source_key, methods in labels_by_source.items():
        source = next(row for row in qwen_rows if row["review_key"] == source_key)
        case_results.append({
            "source_review_key": source_key,
            "language": source["language"],
            "kind": source["kind"],
            "source_question_en": source["source_question_en"],
            "qwen3_14b": methods["qwen3_14b"],
            "assistant_ai_candidate": methods["assistant_ai_candidate"],
        })

    report = {
        "status": "VERIFIED_BLINDED_DIRECT_PAIRED_AI_SCREEN_DIAGNOSTIC_ONLY"
        if calibration_passed and blocked == 0 else "VERIFIED_DIRECT_SCREEN_BUT_REVIEWER_CALIBRATION_OR_COMPLETENESS_FAILED",
        "candidate_screen_interpretable": calibration_passed and blocked == 0,
        "rows_reviewed": len(result["records"]),
        "candidate_rows": 32,
        "paired_source_questions": len(labels_by_source),
        "candidate_counts": {
            method: {label: candidate_counts[(method, label)] for label in ("pass", "flag", "blocked")}
            for method in sorted(expected_methods)
        },
        "counts_by_method_and_language": {
            method: {language: {label: candidate_by_method_language[(method, language)][label]
                               for label in ("pass", "flag", "blocked")}
                     for language in ("fr", "de", "hi", "th")}
            for method in sorted(expected_methods)
        },
        "paired_transitions": dict(sorted(transitions.items())),
        "roundtrip_to_direct_transitions": {
            method: dict(sorted(counts.items()))
            for method, counts in sorted(roundtrip_direct_transitions.items())
        },
        "calibration": {
            "sentinel_rows": len(sentinel_results),
            "semantic_rejects": semantic_rejects,
            "semantic_accepts_on_known_mismatches": semantic_accepts,
            "blocked": sentinel_blocked,
            "predeclared_threshold": "at least 15/16 known mismatches semantically rejected; zero blocked",
            "passed": calibration_passed,
            "sentinel_decisions": sentinel_results,
        },
        "models": {"reviewer": {"id": result["model_id"], "revision": result["model_revision"]}},
        "runtime": {key: result.get(key) for key in (
            "torch", "transformers", "accelerate", "gpu_devices", "cuda_device_count", "device_map")},
        "cpu_offload_detected": "cpu" in {str(value).replace("cuda:", "").lower() for value in device_map.values()},
        "kaggle_run": {key: run_receipt.get(key) for key in (
            "kernel_id", "run_id", "status", "kernel_last_run_time_utc",
            "runtime_seconds_log_last_event", "observed_at_utc")},
        "output_completion_timestamp_missing": result.get("completed_at_utc") is None,
        "prompt_sha256": result["prompt_sha256"],
        "prompt_matches_calibrated_template": prompt_matches_calibration,
        "review_script_sha256": script_hash,
        "pulled_kaggle_review_script_sha256": remote_script_hash,
        "pulled_kaggle_review_script_matches": True,
        "input_sha256": sha_text(raw),
        "output_sha256": sha_bytes(OUTPUT.read_bytes()),
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "note": "A blinded paired AI-only diagnostic with a same-run known-mismatch sentinel check. It compares direct-review and NLLB round-trip labels on this selected 16-question sample, but protocol differences mean it cannot identify NLLB as the sole source of disagreement. It is neither human-gold translation validation nor a dataset release decision.",
        "comparison_limitations": [
            "The direct reviewer is FP16 with unconstrained JSON generation, while the round-trip reviewer used NF4 and constrained JSON decoding; label changes cannot be attributed to NLLB alone.",
            "The direct reviewer used device_map=auto and offloaded some FP16 layers to CPU; the same-run sentinels check the reviewer under this actual placement, which may differ from earlier calibration placement.",
            "The direct screen and calibration sentinels are AI judgments, not fluent-human gold labels.",
            "The 16 source questions are a selected diagnostic pilot, not a representative corpus sample.",
        ],
        "case_results": case_results,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "candidate_screen_interpretable", "candidate_counts", "paired_transitions",
        "roundtrip_to_direct_transitions", "calibration")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
