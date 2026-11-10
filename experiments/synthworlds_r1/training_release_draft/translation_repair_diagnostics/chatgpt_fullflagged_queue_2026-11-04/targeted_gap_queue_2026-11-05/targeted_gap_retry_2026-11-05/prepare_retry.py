#!/usr/bin/env python3
"""Prepare a source-bound, answer-blind retry set for the 17 quarantined rows."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW_DIR = HERE.parent / "candidate_review_kaggle"
INPUT_SCRIPT = REVIEW_DIR / "review.py"
INPUT_OUTPUT = REVIEW_DIR / "remote_output" / "targeted_gap_review.json"
INPUT_RECEIPT = REVIEW_DIR / "run_receipt.json"
INPUT_VERIFICATION = REVIEW_DIR / "review_verification.json"
EXPECTED_BLOCKED = 17


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def constants_from_script(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"RAW_INPUT", "INPUT_SHA256", "NLLB_ID", "NLLB_REVISION", "QWEN_ID", "QWEN_REVISION"}
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                result[name] = ast.literal_eval(node.value)
    if set(result) != wanted:
        raise ValueError(f"missing source constants: {sorted(wanted - set(result))}")
    return result


def main() -> None:
    constants = constants_from_script(INPUT_SCRIPT)
    raw_input = constants["RAW_INPUT"].encode("utf-8")
    if sha(raw_input) != constants["INPUT_SHA256"]:
        raise ValueError("original review input does not match its embedded SHA-256")
    source_rows = json.loads(raw_input, object_pairs_hook=unique_object)
    old_output_bytes = INPUT_OUTPUT.read_bytes()
    receipt = json.loads(INPUT_RECEIPT.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    verification = json.loads(INPUT_VERIFICATION.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    old_output = json.loads(old_output_bytes, object_pairs_hook=unique_object)
    if sha(old_output_bytes) != receipt["output_sha256"]:
        raise ValueError("previous review output hash differs from its Kaggle receipt")
    if verification.get("input_sha256") != constants["INPUT_SHA256"]:
        raise ValueError("previous review receipt is for a different input")
    if (verification.get("artifact_sha256") != sha(old_output_bytes)
            or verification.get("human_translation_reviewed") is not False
            or verification.get("release_approved") is not False
            or verification.get("training_eligible") is not False):
        raise ValueError("previous review receipt has unexpected approval flags")

    prior_by_key = {record["review_key"]: record for record in old_output["records"]}
    if len(prior_by_key) != len(old_output["records"]):
        raise ValueError("previous output has duplicate review keys")
    rows = []
    for row in source_rows:
        prior = prior_by_key.get(row["review_key"])
        if prior is None:
            raise ValueError(f"missing prior review record: {row['review_key']}")
        failed_bt = prior.get("backtranslation_error") is not None
        failed_judge = "error" in (prior.get("decision") or {})
        if not (failed_bt or failed_judge):
            continue
        expected = {
            "source_sha256": sha(row["source_question_en"].encode("utf-8")),
            "translation_sha256": sha(row["translated_question"].encode("utf-8")),
        }
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError(f"prior record is not bound to source/candidate: {row['review_key']}")
        item = {
            "review_key": row["review_key"], "language": row["language"], "kind": row["kind"],
            "source_question_en": row["source_question_en"],
            "translated_question": row["translated_question"],
            **expected,
            "retry_backtranslation": "ALIAS_NLLB" if failed_bt else "REUSE_VERIFIED_PRIOR",
        }
        if failed_judge and not failed_bt:
            backtranslation = prior.get("backtranslation_en")
            if not isinstance(backtranslation, str) or not backtranslation.strip():
                raise ValueError(f"missing prior backtranslation: {row['review_key']}")
            item["prior_backtranslation_en"] = backtranslation
            item["prior_backtranslation_sha256"] = sha(backtranslation.encode("utf-8"))
        rows.append(item)

    if len(rows) != EXPECTED_BLOCKED or len({row["review_key"] for row in rows}) != EXPECTED_BLOCKED:
        raise ValueError(f"expected {EXPECTED_BLOCKED} unique blocked records; got {len(rows)}")
    rows.sort(key=lambda row: row["review_key"])
    payload = {
        "schema": "synthworlds-targeted-gap-retry-v1",
        "mode": "AI_ONLY_DIAGNOSTIC_RETRY_NO_RELEASE_APPROVAL",
        "source_input_sha256": constants["INPUT_SHA256"],
        "prior_output_sha256": sha(old_output_bytes),
        "prior_nllb_model": {"id": constants["NLLB_ID"], "revision": constants["NLLB_REVISION"]},
        "prior_qwen_model": {"id": constants["QWEN_ID"], "revision": constants["QWEN_REVISION"]},
        "rows": rows,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "answers_or_split_labels_included": False,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (HERE / "retry_input.json").write_bytes(encoded)
    manifest = {
        "status": "VERIFIED_RETRY_INPUT_PREPARED_DIAGNOSTIC_ONLY",
        "input_sha256": sha(encoded),
        "source_input_sha256": constants["INPUT_SHA256"],
        "prior_output_sha256": sha(old_output_bytes),
        "rows": len(rows),
        "alias_backtranslation_rows": sum(r["retry_backtranslation"] == "ALIAS_NLLB" for r in rows),
        "judge_retry_rows": sum(r["retry_backtranslation"] == "REUSE_VERIFIED_PRIOR" for r in rows),
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "answers_or_split_labels_included": False,
    }
    (HERE / "retry_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    template = (HERE / "retry_template.py").read_text(encoding="utf-8")
    if template.count("__RETRY_INPUT_B64__") != 1 or template.count("__RETRY_INPUT_SHA256__") != 1:
        raise ValueError("retry template placeholders are missing or duplicated")
    source = template.replace("__RETRY_INPUT_B64__", base64.b64encode(encoded).decode())
    source = source.replace("__RETRY_INPUT_SHA256__", sha(encoded))
    (HERE / "retry.py").write_text(source, encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/sw-gap187-alias-retry-2026-11-05",
        "title": "SW Gap187 Alias Retry 2026-11-05",
        "code_file": "retry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest["retry_script_sha256"] = sha(source.encode("utf-8"))
    manifest["kernel_id"] = metadata["id"]
    (HERE / "retry_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
