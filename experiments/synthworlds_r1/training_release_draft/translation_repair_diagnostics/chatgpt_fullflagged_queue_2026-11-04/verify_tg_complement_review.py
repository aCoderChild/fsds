#!/usr/bin/env python3
"""Verify the masked Kaggle review and report pilot-only source coverage."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
PROJECT = HERE.parents[2]
INPUT_PATH = HERE / "tg_complement_input_v1.jsonl"
MAP_PATH = HERE / "tg_complement_private_map_v1.jsonl"
BUILD_PATH = HERE / "tg_complement_build_manifest.json"
TG_DIR = RELEASE / "translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03"
QUEUE = HERE / "independent_review_inputs_2026-11-04"
LANGS = ("fr", "de", "hi", "th")
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
MODE = "tg_qwen7_complement_v1"
KERNEL_ID = "bunnoob2005/synthworlds-r1-tg-complement-review-v1"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
INPUT_FIELDS = ("review_key", "language", "kind", "source_question_en", "translated_question")

sys.path.insert(0, str(RELEASE / "scripts"))
from review_training_translations_qwen import collect_rows, deterministic_checks, masked_pair, read_jsonl  # noqa: E402

sys.path.insert(0, str(HERE))
from analyze_chatgpt_repair_coverage import compute_coverage  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha(value: str) -> str:
    return sha(value.encode("utf-8"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def remote_log(path: Path) -> str:
    chunks = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line.lstrip(","))
        except json.JSONDecodeError:
            continue
        if record.get("stream_name") in {"stdout", "stderr"}:
            chunks.append(record.get("data", ""))
    return "".join(chunks)


def compact_input_payload(rows: list[dict]) -> str:
    """Rebuild the builder's compact array hash independent of JSONL key order."""
    canonical_rows = [{field: row[field] for field in INPUT_FIELDS} for row in rows]
    return json.dumps(canonical_rows, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--source-script", type=Path, required=True,
                        help="Exact .py source retrieved with `kaggle kernels pull`")
    args = parser.parse_args()
    build = read_json(BUILD_PATH)
    inputs, maps = read_jsonl(INPUT_PATH), read_jsonl(MAP_PATH)
    if (build.get("status") != "MASKED_TG_COMPLEMENT_REVIEW_READY_NOT_APPROVED"
            or build.get("mode") != MODE or build.get("kernel_id") != KERNEL_ID
            or build.get("rows") != len(inputs) or not inputs
            or build.get("input_jsonl_sha256") != sha(INPUT_PATH.read_bytes())
            or build.get("private_map_sha256") != sha(MAP_PATH.read_bytes())
            or len(maps) != len(inputs)):
        raise ValueError("Local build manifest/input/map failed integrity checks")
    if (any(set(row) != {"review_key", "language", "kind", "source_question_en", "translated_question"} for row in inputs)
            or len({r["review_key"] for r in inputs}) != len(inputs)
            or {r["review_key"] for r in inputs} != {r["review_key"] for r in maps}):
        raise ValueError("Masked review input schema or row-map coverage is invalid")

    compact = compact_input_payload(inputs)
    if build.get("input_compact_array_sha256") != sha(compact.encode("utf-8")):
        raise ValueError("Embedded compact input hash differs from the local row set")
    input_by_key = {r["review_key"]: r for r in inputs}
    map_by_key = {r["review_key"]: r for r in maps}
    tg_path = TG_DIR / "repair_candidates.jsonl"
    tg_report_path = TG_DIR / "report.json"
    tg_rows = read_jsonl(tg_path)
    tg_by_key = {(r["language"], r["prompt_id"], r["kind"]): r for r in tg_rows}
    if (build.get("tg_candidate_file_sha256") != sha(tg_path.read_bytes())
            or build.get("chatgpt_merge_sha256") != sha((HERE / "remote_output/merged_full799/version_1/merged_full799_diagnostic_reviews.jsonl").read_bytes())
            or read_json(tg_report_path).get("decisions_sha256") != sha(tg_path.read_bytes())):
        raise ValueError("Candidate source artifacts changed after queue preparation")

    source_by_lang = {}
    for lang in LANGS:
        rows, _ = collect_rows(lang)
        source_by_lang[lang] = {(r["prompt_id"], r["kind"]): r for r in rows}
    for key, row in input_by_key.items():
        mapping = map_by_key[key]
        if mapping.get("human_translation_reviewed") is not False:
            raise ValueError(f"{key}: local row map misstates human review")
        source = source_by_lang[row["language"]].get((mapping["prompt_id"], row["kind"]))
        candidate = tg_by_key.get((row["language"], mapping["prompt_id"], row["kind"]))
        if source is None or candidate is None:
            raise ValueError(f"{key}: source or candidate row is missing")
        checks = deterministic_checks(row["language"], {**source, "text_translation": candidate["new_translation"]})
        if (text_sha(source["text_en"]) != mapping.get("source_text_sha256")
                or text_sha(source["text_translation"]) != mapping.get("old_translation_sha256")
                or text_sha(candidate["new_translation"]) != mapping.get("candidate_translation_sha256")
                or not all(checks.get(k) is True for k in (
                    "nonempty", "target_script_present", "protected_names_preserved",
                    "protected_numbers_preserved", "no_unrestored_marker"))):
            raise ValueError(f"{key}: current source/candidate/deterministic check mismatch")
        _, pair = masked_pair({**source, "text_translation": candidate["new_translation"]})
        if (row["source_question_en"] != pair["source_english"]
                or row["translated_question"] != pair["target_translation"]):
            raise ValueError(f"{key}: remote masked text does not bind to the selected candidate")

    remote = args.remote_dir
    result_path = remote / "repair_review.json"
    log_candidates = list(remote.glob("*.log"))
    source_path = args.source_script
    if (not result_path.is_file() or len(log_candidates) != 1
            or not source_path.is_file()
            or sha(source_path.read_bytes()) != build.get("review_script_sha256")):
        raise ValueError("Expected one output JSON/log and the exact Kaggle-pulled reviewer source")
    result = read_json(result_path)
    if (result.get("mode") != MODE or result.get("input_sha256") != build["input_compact_array_sha256"]
            or result.get("nllb_model_id") != NLLB_ID or result.get("nllb_model_revision") != NLLB_REVISION
            or result.get("qwen_model_id") != QWEN_ID or result.get("qwen_model_revision") != QWEN_REVISION
            or result.get("quantization") != "4-bit NF4" or result.get("bitsandbytes") != "0.50.2"
            or result.get("human_translation_reviewed") is not False
            or result.get("answers_or_splits_sent") is not False):
        raise ValueError("Remote model, input, quantization, or blinding provenance mismatch")
    records = result.get("records")
    if (not isinstance(records, list) or len(records) != len(inputs)
            or [r.get("review_key") for r in records] != [r["review_key"] for r in inputs]):
        raise ValueError("Remote output has missing, extra, duplicate, or reordered rows")

    statuses = collections.Counter()
    verified = []
    for row, record in zip(inputs, records):
        key = row["review_key"]
        mapping = map_by_key[key]
        markers = collections.Counter(PLACEHOLDER.findall(row["source_question_en"]))
        if (record.get("language") != row["language"] or record.get("kind") != row["kind"]
                or record.get("source_sha256") != text_sha(row["source_question_en"])
                or record.get("translation_sha256") != text_sha(row["translated_question"])):
            raise ValueError(f"{key}: remote row identity/hash mismatch")
        backtranslation = record.get("backtranslation_en", "")
        bt_error = record.get("backtranslation_error")
        decision = record.get("decision")
        if not isinstance(decision, dict):
            raise ValueError(f"{key}: no explicit model decision or block")
        if bt_error is not None or "error" in decision:
            if "error" not in decision:
                raise ValueError(f"{key}: a backtranslation failure lacks an explicit block")
            status = "AI_ONLY_REPAIR_REVIEW_BLOCKED"
        else:
            if (not backtranslation.strip() or record.get("backtranslation_sha256") != text_sha(backtranslation)
                    or collections.Counter(PLACEHOLDER.findall(backtranslation)) != markers
                    or collections.Counter(PLACEHOLDER.findall(row["translated_question"])) != markers):
                raise ValueError(f"{key}: backtranslation hash or marker check failed")
            raw = json.loads(record.get("raw_output", ""), object_pairs_hook=unique_object)
            if (set(raw) != {"decisions"} or len(raw["decisions"]) != 1 or raw["decisions"][0] != decision
                    or decision.get("key") != key or type(decision.get("semantic_pass")) is not bool
                    or type(decision.get("fluency_pass")) is not bool or not decision.get("reason", "").strip()):
                raise ValueError(f"{key}: raw model output differs from its typed decision")
            status = ("AI_ONLY_REPAIR_REVIEW_PASS" if decision["semantic_pass"] and decision["fluency_pass"]
                      else "AI_ONLY_REPAIR_REVIEW_FLAGGED")
        statuses[status] += 1
        verified.append({
            "review_key": key, "source_review_key": mapping["source_review_key"],
            "language": row["language"], "prompt_id": mapping["prompt_id"], "kind": row["kind"],
            "candidate_translation_sha256": mapping["candidate_translation_sha256"],
            "backtranslation_sha256": record.get("backtranslation_sha256"),
            "decision": decision, "review_status": status, "human_translation_reviewed": False,
        })

    if result.get("status") not in {"AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY", "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS"}:
        raise ValueError("Remote review did not reach an accepted terminal status")
    log = remote_log(log_candidates[0])
    progress = [int(n) for n in re.findall(rf"Reviewed (\d+)/{len(inputs)}", log)]
    if not progress or max(progress) != len(inputs) or "completed_at_utc" not in result:
        raise ValueError("Remote log/result does not prove all rows completed")

    verified_path = HERE / "tg_complement_verified_reviews_v1.jsonl"
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in verified)
    verified_path.write_text(payload, encoding="utf-8")
    submitted_source_path = HERE / "tg_complement_submitted_source_v1.py"
    submitted_source_path.write_bytes(source_path.read_bytes())
    receipt = {
        "status": "VERIFIED_TG_COMPLEMENT_AI_ONLY_FINAL_SCREEN_DIAGNOSTIC",
        "kernel_id": KERNEL_ID, "mode": MODE, "rows": len(inputs),
        "review_script_sha256": sha(submitted_source_path.read_bytes()),
        "submitted_source_copy": submitted_source_path.name,
        "input_compact_array_sha256": build["input_compact_array_sha256"],
        "output_sha256": sha(result_path.read_bytes()), "run_log_sha256": sha(log_candidates[0].read_bytes()),
        "verified_rows_sha256": sha(verified_path.read_bytes()),
        "counts": dict(statuses), "human_translation_reviewed": False,
        "answers_or_splits_sent": False, "release_approved": False, "training_eligible": False,
        "limitations": ["AI-only candidate screening; not fluent-human certification.",
                         "Candidate availability/review does not approve a production corpus.",
                         "The frozen full-corpus release protocol remains unchanged."],
    }
    receipt_path = HERE / "tg_complement_verification_v1.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Planning ceiling after applying this same final screen to the 75 selected TG replacements.
    pass_by_language = {lang: set() for lang in LANGS}
    source_review_root = RELEASE / "translation_reviews_roundtrip_qwen14b"
    source_pass = {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}
    for lang in LANGS:
        for row in read_jsonl(source_review_root / f"{lang}.jsonl"):
            if row.get("final_review_status") in source_pass:
                pass_by_language[lang].add((row["prompt_id"], row["kind"]))
    merged_rows = read_jsonl(HERE / "remote_output/merged_full799/version_1/merged_full799_diagnostic_reviews.jsonl")
    full_map = read_jsonl(QUEUE / "full799.private_map.jsonl")
    full_map_by_key = {r["review_key"]: r for r in full_map}
    for row in merged_rows:
        if row.get("review_status") == "AI_ONLY_REPAIR_REVIEW_PASS":
            mapping = full_map_by_key[row["review_key"]]
            pass_by_language[row["language"]].add((mapping["prompt_id"], row["kind"]))
    for row in verified:
        if row["review_status"] == "AI_ONLY_REPAIR_REVIEW_PASS":
            pass_by_language[row["language"]].add((row["prompt_id"], row["kind"]))
    coverage = compute_coverage(
        pass_by_language,
        read_jsonl(RELEASE / "atomic_reuse.jsonl"),
        read_jsonl(RELEASE / "composition_translation_review_queue.jsonl"),
    )
    case_manifest = {r["source_qa_id"]: r for r in read_jsonl(PROJECT / "candidate_cases.draft.jsonl")}
    atomic_by_case = collections.defaultdict(list)
    for row in read_jsonl(RELEASE / "atomic_reuse.jsonl"):
        atomic_by_case[row["source_qa_id"]].append(row)
    eligible_by_split = {"forget": [], "retain": []}
    for case_id, rows in atomic_by_case.items():
        split = rows[0]["split"]
        if split not in eligible_by_split:
            continue
        ids = {r["sft_id"] for r in rows}
        if len(ids) == 2 and all((sid, "atomic") in pass_by_language[lang] for sid in ids for lang in LANGS):
            eligible_by_split[split].append(case_id)
    group_counts = {
        split: len({case_manifest[cid]["group_id"] for cid in case_ids})
        for split, case_ids in eligible_by_split.items()
    }
    planning = {
        "status": "PILOT_COVERAGE_AFTER_STANDARDIZED_AI_SCREEN_DIAGNOSTIC_NOT_RELEASE",
        "review_rows": len(inputs), "review_counts": dict(statuses),
        "ai_candidate_complete_cases": {
            split: {"cases": len(case_ids), "entity_groups": group_counts[split]}
            for split, case_ids in eligible_by_split.items()
        },
        "coverage": coverage,
        "baseline_competence_gate": "NOT_RUN; only a model baseline can establish this",
        "full_r1_release_approved": False, "training_eligible": False,
        "human_translation_reviewed": False,
    }
    plan_path = HERE / "tg_complement_candidate_coverage_v1.json"
    plan_path.write_text(json.dumps(planning, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verification": receipt, "coverage": planning}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
