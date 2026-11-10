#!/usr/bin/env python3
"""Build the same relation controls with the back-translation used in production review."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STANDALONE = HERE.parent / "qwen14b_hard_negative_controls"
sys.path.insert(0, str(STANDALONE))
sys.path.insert(0, str(HERE.parents[4] / "experiments/synthworlds_r1/training_release_draft/scripts"))
from build_controls import candidates_for_cell  # noqa: E402
from review_training_translations_qwen import NUMBER, alpha_label, collect_rows, masked_pair  # noqa: E402

TRAINING = HERE.parents[4] / "experiments/synthworlds_r1/training_release_draft"
LANGS = ("fr", "de", "hi", "th")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def masked_backtranslation(source: dict, text: str) -> str:
    original = source["text_en"]
    names = [name for name in source.get("protected_entity_names", []) if name and name in original]
    names = sorted(set(names), key=lambda name: original.index(name))
    replacements = {name: f"[ENTITY_{alpha_label(i)}]" for i, name in enumerate(names)}
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(name, replacements[name])
    for i, number in enumerate(NUMBER.findall(original)):
        text = text.replace(number, f"[NUMBER_{alpha_label(i)}]", 1)
    return text


def main() -> None:
    inputs = []
    controls = []
    for lang in LANGS:
        source_rows, _ = collect_rows(lang)
        source_by_key = {(r["prompt_id"], r["kind"]): r for r in source_rows}
        review_path = TRAINING / "translation_reviews_roundtrip_qwen14b" / f"{lang}.jsonl"
        review_rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        review_by_key = {(r["prompt_id"], r["kind"]): r for r in review_rows}
        for kind in ("atomic", "composition"):
            for base in candidates_for_cell(lang, kind):
                key = (base["prompt_id"], kind)
                source, review = source_by_key[key], review_by_key[key]
                if (source["text_en"] != review["text_en"]
                        or source["text_translation"] != review["text_translation"]
                        or review.get("final_review_status") not in {
                            "AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}):
                    raise ValueError(f"Base source/review mismatch for {lang}/{key}")
                if not review.get("backtranslation_en") or not all(review.get("backtranslation_checks", {}).values()):
                    raise ValueError(f"Base review lacks a clean back-translation for {lang}/{key}")
                _, pair = masked_pair(source)
                bt = masked_backtranslation(source, review["backtranslation_en"])
                control_id = f"PH{len(inputs) + 1:03d}"
                item = {
                    "control_id": control_id,
                    "language": lang,
                    "kind": kind,
                    "source": base["source_counterfactual"],
                    "translation": pair["target_translation"],
                    "back_translation_en": bt,
                }
                inputs.append(item)
                controls.append({
                    **base,
                    "control_id": control_id,
                    "translation": pair["target_translation"],
                    "back_translation_en": bt,
                    "backtranslation_sha256": sha(review["backtranslation_en"].encode("utf-8")),
                    "source_review_file_sha256": sha(review_path.read_bytes()),
                    "backtranslation_model": review.get("backtranslation_model"),
                })
    if len(inputs) != 16 or len({r["control_id"] for r in inputs}) != 16:
        raise ValueError(f"Expected 16 unique pipeline controls, got {len(inputs)}")
    raw = json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
    jsonl = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in inputs)
    manifest = {
        "status": "BLINDED_FULL_REVIEW_PIPELINE_CONTROLS_DIAGNOSTIC_ONLY",
        "rows": len(inputs),
        "controls_per_language_kind": 2,
        "construction": "Reuse the same 16 relation/event counterfactuals as the standalone control; provide the unchanged AI-screened target translation and its masked English back-translation to mirror production adjudication inputs.",
        "rule": "The reviewer must semantically reject all 16 mismatches with zero malformed decisions. This is a reviewer screen only, not translation approval or release authorization.",
        "input_sha256_compact_array": sha(raw.encode("utf-8")),
        "input_jsonl_sha256": sha(jsonl.encode("utf-8")),
        "inputs_include_answer_or_split": False,
        "human_translation_reviewed": False,
        "controls": controls,
        "limitations": [
            "The translations and back-translations are AI-screened, not human-gold.",
            "The back-translation corresponds to the original positive English source while the reviewed source changes one relation/event phrase.",
            "This tests rejection of relation-level mismatches with the production back-translation evidence; it does not establish real-corpus precision, recall, or fluency accuracy.",
        ],
    }
    (HERE / "input.jsonl").write_text(jsonl, encoding="utf-8")
    (HERE / "input_array.json").write_text(raw + "\n", encoding="utf-8")
    (HERE / "control_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(inputs), "input_sha256": manifest["input_sha256_compact_array"],
                      "cells": {f"{lang}/{kind}": sum(r["language"] == lang and r["kind"] == kind for r in inputs)
                                for lang in LANGS for kind in ("atomic", "composition")}}, indent=2))


if __name__ == "__main__":
    main()
