#!/usr/bin/env python3
"""Prepare the existing Qwen2.5+NLLB blind screen for completed pilot output."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GEN_INPUT = HERE / "generation_input.json"
GEN_OUTPUT = HERE / "remote_output" / "qwen3_translation_pilot.json"
GEN_VERIFY = HERE / "verification.json"
TEMPLATE = HERE.parent / "chatgpt_third_candidate_2026-11-04" / "kaggle_review" / "review.py"
SCREEN_ID = "bunnoob2005/qwen3-pilot-qwen14-review-2026-11-04"
KERNEL_DIR = HERE / "qwen14_review_kernel"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: object) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    if not GEN_VERIFY.exists():
        raise ValueError("First verify the completed Qwen3 translation output")
    generation_report = json.loads(GEN_VERIFY.read_text(encoding="utf-8"))
    if generation_report.get("status") != "VERIFIED_QWEN3_TRANSLATION_PILOT_DIAGNOSTIC_ONLY":
        raise ValueError("Qwen3 generation output is not verified")
    rows = json.loads(GEN_INPUT.read_text(encoding="utf-8"))
    result = json.loads(GEN_OUTPUT.read_text(encoding="utf-8"))
    if result.get("status") != "AI_ONLY_TRANSLATION_PILOT_COMPLETE":
        raise ValueError("Qwen3 translation generation did not complete cleanly")
    by_key = {record["review_key"]: record for record in result["records"]}
    if len(by_key) != 16 or set(by_key) != {row["review_key"] for row in rows}:
        raise ValueError("Qwen3 output does not cover the exact 16 source rows")
    review_rows = []
    for source in rows:
        candidate = by_key[source["review_key"]]
        if candidate.get("error") is not None or not candidate.get("translation"):
            raise ValueError(f"Missing candidate at {source['review_key']}")
        review_rows.append({
            "review_key": source["review_key"],
            "language": source["language"],
            "kind": source["kind"],
            "source_question_en": source["source_question_en"],
            "translated_question": candidate["translation"],
        })
    raw = json.dumps(review_rows, ensure_ascii=False, separators=(",", ":"))
    input_hash = sha_json(review_rows)

    text = TEMPLATE.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    replacements = {
        "MODE": 'MODE = "qwen3_translation_pilot_qwen14_roundtrip_screen_16"',
        "INPUT_SHA256": f'INPUT_SHA256 = "{input_hash}"',
        "RAW_INPUT": "RAW_INPUT = " + repr(raw),
    }
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    if node.lineno != node.end_lineno or target.id in found:
                        raise ValueError(f"Unexpected {target.id} template assignment")
                    lines[node.lineno - 1] = replacements[target.id]
                    found.add(target.id)
    if found != set(replacements):
        raise ValueError("Expected review-template constants are missing")
    screen_script = "\n".join(lines) + "\n"
    ast.parse(screen_script)

    KERNEL_DIR.mkdir(exist_ok=True)
    (HERE / "review_input.json").write_text(raw + "\n", encoding="utf-8")
    (HERE / "review.py").write_text(screen_script, encoding="utf-8")
    metadata = {
        "id": SCREEN_ID,
        "title": "Qwen3 Pilot Qwen14 Review 2026-11-04",
        "code_file": "review.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
    }
    (KERNEL_DIR / "review.py").write_text(screen_script, encoding="utf-8")
    (KERNEL_DIR / "review_input.json").write_text(raw + "\n", encoding="utf-8")
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (KERNEL_DIR / "README.md").write_text(
        "# Blinded Qwen2.5-14B + NLLB review\n\n"
        "This private kernel screens the 16 Qwen3 translation-pilot rows. It receives only masked English questions and candidate translations; it has no answers, facts, or split labels. All rows remain diagnostic-only.\n",
        encoding="utf-8",
    )
    report = {
        "status": "PREPARED_BLINDED_QWEN14_NLLB_SCREEN_DIAGNOSTIC_ONLY",
        "rows": len(review_rows),
        "review_input_sha256": input_hash,
        "review_script_sha256": sha_bytes((HERE / "review.py").read_bytes()),
        "qwen3_translation_output_sha256": sha_bytes(GEN_OUTPUT.read_bytes()),
        "qwen3_translation_verification_sha256": sha_bytes(GEN_VERIFY.read_bytes()),
        "reviewer_template_sha256": sha_bytes(TEMPLATE.read_bytes()),
        "qwen_model_id": "Qwen/Qwen2.5-14B-Instruct",
        "qwen_model_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "backtranslation_model_id": "facebook/nllb-200-distilled-600M",
        "backtranslation_model_revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    (HERE / "review-build-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
