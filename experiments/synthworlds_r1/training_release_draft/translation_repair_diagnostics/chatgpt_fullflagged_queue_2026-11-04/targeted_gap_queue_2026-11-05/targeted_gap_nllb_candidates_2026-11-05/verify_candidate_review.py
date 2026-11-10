#!/usr/bin/env python3
"""Verify NLLB candidate rows and Qwen review without granting release approval."""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETED = HERE.parent
REVIEW = TARGETED / "candidate_review_kaggle"
RETRY = TARGETED / "targeted_gap_retry_2026-11-05"
NLLB_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
QWEN_ID = "Qwen/Qwen2.5-14B-Instruct"
QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha(text.encode("utf-8"))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def embedded_source_rows() -> tuple[str, list[dict]]:
    tree = ast.parse((REVIEW / "review.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RAW_INPUT" for target in node.targets
        ):
            raw = ast.literal_eval(node.value)
            return raw, json.loads(raw, object_pairs_hook=unique_object)
    raise ValueError("missing original 187-row blinded input")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--pulled-script", type=Path, required=True)
    args = parser.parse_args()

    input_path = HERE / "candidate_input.json"
    manifest_path = HERE / "candidate_input_manifest.json"
    script_path = HERE / "candidate_review.py"
    output_path = args.remote_dir / "candidate_review.json"
    input_bytes = input_path.read_bytes()
    input_payload = json.loads(input_bytes, object_pairs_hook=unique_object)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    local_script = script_path.read_bytes()
    pulled_script = args.pulled_script.read_bytes()
    output_bytes = output_path.read_bytes()
    output = json.loads(output_bytes, object_pairs_hook=unique_object)

    source_raw, source_rows = embedded_source_rows()
    source_by_key = {row["review_key"]: row for row in source_rows}
    previous = json.loads((REVIEW / "remote_output/targeted_gap_review.json").read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    retry_path = RETRY / "remote_output/retry_output.json"
    retry_bytes = retry_path.read_bytes()
    retry = json.loads(retry_bytes, object_pairs_hook=unique_object)
    passed = set()
    for row in previous["records"]:
        decision = row.get("decision") or {}
        if row.get("backtranslation_error") is None and decision.get("semantic_pass") is True and decision.get("fluency_pass") is True:
            passed.add(row["review_key"])
    for row in retry["records"]:
        decision = row.get("decision") or {}
        if row.get("backtranslation_error") is None and "error" not in decision and decision.get("semantic_pass") is True and decision.get("fluency_pass") is True:
            passed.add(row["review_key"])
    expected_rows = [{
        "review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
        "source_question_en": row["source_question_en"], "source_sha256": sha_text(row["source_question_en"])
    } for row in source_rows if row["review_key"] not in passed]
    expected_rows.sort(key=lambda row: row["review_key"])

    if (input_payload.get("schema") != "synthworlds-nllb-alternative-translation-v1"
            or input_payload.get("mode") != "AI_ONLY_DIAGNOSTIC_CANDIDATE_NO_RELEASE_APPROVAL"
            or input_payload.get("rows") != expected_rows
            or len(expected_rows) != 148
            or sha(input_bytes) != manifest.get("input_sha256")
            or sha(source_raw.encode()) != input_payload.get("source_review_input_sha256")
            or sha((REVIEW / "remote_output/targeted_gap_review.json").read_bytes()) != input_payload.get("prior_review_output_sha256")
            or sha(retry_bytes) != input_payload.get("prior_retry_output_sha256")
            or sha((TARGETED / "private_row_map.jsonl").read_bytes()) != input_payload.get("row_map_sha256")
            or manifest.get("candidate_script_sha256") != sha(local_script)
            or pulled_script != local_script
            or manifest.get("kernel_id") != "bunnoob2005/sw-gap187-nllb-148-candidate-review-2026-11-05"):
        raise ValueError("candidate input selection, prior evidence, or submitted source hash mismatch")
    if (output.get("status") not in {"AI_ONLY_NLLB_CANDIDATE_REVIEW_COMPLETE", "AI_ONLY_NLLB_CANDIDATE_REVIEW_HAS_BLOCKS"}
            or output.get("input_sha256") != sha(input_bytes)
            or output.get("input_payload") != input_payload
            or output.get("nllb_model_id") != NLLB_ID
            or output.get("nllb_model_revision") != NLLB_REVISION
            or output.get("qwen_model_id") != QWEN_ID
            or output.get("qwen_model_revision") != QWEN_REVISION
            or output.get("human_translation_reviewed") is not False
            or output.get("answers_or_split_labels_included") is not False
            or output.get("release_approved") is not False
            or output.get("training_eligible") is not False):
        raise ValueError("remote candidate review provenance or scope is invalid")
    gpu = output.get("gpu")
    if not isinstance(gpu, list) or len(gpu) != 2 or not all("T4" in name for name in gpu):
        raise ValueError(f"expected exactly two T4 GPUs; got {gpu!r}")

    expected_by_key = {row["review_key"]: row for row in expected_rows}
    records = output.get("records")
    by_key = {row.get("review_key"): row for row in records or []}
    if len(records or []) != 148 or len(by_key) != 148 or set(by_key) != set(expected_by_key):
        raise ValueError("remote records do not exactly cover the 148 selected unresolved rows")
    counts = collections.Counter()
    by_language = {language: collections.Counter() for language in ("fr", "de", "hi", "th")}
    for key, row in expected_by_key.items():
        record = by_key[key]
        lang = row["language"]
        if (record.get("language") != lang or record.get("kind") != row["kind"]
                or record.get("source_sha256") != row["source_sha256"]):
            raise ValueError(f"{key}: record metadata/source hash differs from input")
        if record.get("candidate_error") is None:
            candidate = record.get("candidate_translation")
            if (not isinstance(candidate, str) or not candidate.strip()
                    or record.get("candidate_sha256") != sha_text(candidate)
                    or collections.Counter(MARKER.findall(candidate)) != collections.Counter(MARKER.findall(row["source_question_en"]))):
                raise ValueError(f"{key}: candidate translation/hash/placeholder multiset invalid")
            counts["candidate_rows"] += 1
            by_language[lang]["candidate_rows"] += 1
        elif record.get("candidate_translation") or record.get("candidate_sha256"):
            raise ValueError(f"{key}: failed candidate has nonempty candidate text/hash")

        if record.get("candidate_error") is not None or record.get("backtranslation_error") is not None:
            if not isinstance(record.get("decision"), dict) or "error" not in record["decision"]:
                raise ValueError(f"{key}: technical failure is not reflected in a blocked decision")
            counts["blocked"] += 1
            by_language[lang]["blocked"] += 1
            continue
        backtranslation = record.get("backtranslation_en")
        if (not isinstance(backtranslation, str) or not backtranslation.strip()
                or record.get("backtranslation_sha256") != sha_text(backtranslation)
                or collections.Counter(MARKER.findall(backtranslation)) != collections.Counter(MARKER.findall(row["source_question_en"]))):
            raise ValueError(f"{key}: invalid back-translation/hash/placeholder multiset")
        decision = record.get("decision")
        if not isinstance(decision, dict) or "error" in decision:
            counts["blocked"] += 1
            by_language[lang]["blocked"] += 1
            continue
        if (decision.get("key") != key or type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip() or len(decision["reason"]) > 120
                or not isinstance(record.get("raw_output"), str) or not record["raw_output"].strip()):
            raise ValueError(f"{key}: invalid Qwen decision schema/key")
        counts["reviewed"] += 1
        by_language[lang]["reviewed"] += 1
        counts["semantic_pass"] += decision["semantic_pass"]
        by_language[lang]["semantic_pass"] += decision["semantic_pass"]
        counts["fluency_pass"] += decision["fluency_pass"]
        by_language[lang]["fluency_pass"] += decision["fluency_pass"]
        joint = decision["semantic_pass"] and decision["fluency_pass"]
        counts["joint_pass"] += joint
        by_language[lang]["joint_pass"] += joint

    if (output.get("candidate_rows") != counts["candidate_rows"]
            or output.get("reviewed_rows") != counts["reviewed"]
            or output.get("blocked_rows") != counts["blocked"]
            or output.get("status") != ("AI_ONLY_NLLB_CANDIDATE_REVIEW_COMPLETE" if counts["blocked"] == 0
                                         else "AI_ONLY_NLLB_CANDIDATE_REVIEW_HAS_BLOCKS")):
        raise ValueError("remote summary counts/status differ from row-level review")
    receipt = {
        "status": "VERIFIED_AI_ONLY_NLLB_CANDIDATES_QUARANTINED",
        "input_sha256": sha(input_bytes),
        "output_sha256": sha(output_bytes),
        "submitted_script_sha256": sha(local_script),
        "pulled_script_sha256": sha(pulled_script),
        "rows": len(expected_rows),
        "counts": dict(counts),
        "counts_by_language": {lang: dict(values) for lang, values in by_language.items()},
        "gpu": gpu,
        "nllb_model_revision": NLLB_REVISION,
        "qwen_model_revision": QWEN_REVISION,
        "human_translation_reviewed": False,
        "answers_or_split_labels_included": False,
        "release_approved": False,
        "training_eligible": False,
        "interpretation": "AI-only alternate candidate screen; no translation is certified or admitted to the training release.",
    }
    (args.remote_dir / "candidate_verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
