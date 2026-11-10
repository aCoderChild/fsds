#!/usr/bin/env python3
"""Build balanced AI-only positive consistency controls from the source-bound PH pairs."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEG = HERE.parent / "qwen14b_pipeline_hard_controls"
TRAINING = HERE.parents[4] / "experiments/synthworlds_r1/training_release_draft"
sys.path.insert(0, str(TRAINING / "scripts"))
from review_training_translations_qwen import NUMBER, alpha_label, collect_rows, masked_pair  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mask_backtranslation(source: dict, text: str) -> str:
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
    negative_manifest_path = NEG / "control_manifest.json"
    negative_manifest = json.loads(negative_manifest_path.read_text(encoding="utf-8"))
    source_manifest_sha = sha(negative_manifest_path.read_bytes())
    inputs, controls = [], []
    for ph in negative_manifest["controls"]:
        lang = ph["language"]
        source_rows, _ = collect_rows(lang)
        source = next(r for r in source_rows if (r["prompt_id"], r["kind"]) == (ph["prompt_id"], ph["kind"]))
        review_path = TRAINING / "translation_reviews_roundtrip_qwen14b" / f"{lang}.jsonl"
        review_rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        review = next(r for r in review_rows if (r["prompt_id"], r["kind"]) == (ph["prompt_id"], ph["kind"]))
        if review.get("final_review_status") not in {
            "AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"
        }:
            raise ValueError(f"Base pair is not an AI-screened pass: {lang}/{ph['prompt_id']}/{ph['kind']}")
        if not review.get("backtranslation_en") or not all(review.get("backtranslation_checks", {}).values()):
            raise ValueError(f"Base pair lacks complete back-translation checks: {lang}/{ph['prompt_id']}/{ph['kind']}")
        _, pair = masked_pair(source)
        back = mask_backtranslation(source, review["backtranslation_en"])
        if (pair["source_english"] != ph["source_original"] or pair["target_translation"] != ph["translation"]
                or back != ph["back_translation_en"]):
            raise ValueError(f"Source-bound positive pair mismatch: {lang}/{ph['prompt_id']}/{ph['kind']}")
        control_id = f"PP{len(inputs) + 1:03d}"
        item = {"control_id": control_id, "language": lang, "kind": ph["kind"],
                "source": ph["source_original"], "translation": pair["target_translation"],
                "back_translation_en": back}
        inputs.append(item)
        controls.append({
            "control_id": control_id, "source_control_id": ph["control_id"],
            "prompt_id": ph["prompt_id"], "language": lang, "kind": ph["kind"],
            "base_review_status": review["final_review_status"],
            "base_review_file_sha256": sha(review_path.read_bytes()),
            "translation_sha256": sha(pair["target_translation"].encode()),
            "backtranslation_sha256": sha(back.encode()),
        })
    if len(inputs) != 16 or len({r["control_id"] for r in inputs}) != 16:
        raise ValueError(f"Expected 16 balanced controls, got {len(inputs)}")
    cells = {f"{lang}/{kind}": sum(r["language"] == lang and r["kind"] == kind for r in inputs)
             for lang in ("fr", "de", "hi", "th") for kind in ("atomic", "composition")}
    if set(cells.values()) != {2}:
        raise ValueError(f"Unexpected cell balance: {cells}")
    raw = json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inputs)
    manifest = {
        "status": "AI_ONLY_POSITIVE_REVIEW_CONSISTENCY_DIAGNOSTIC_ONLY",
        "rows": len(inputs), "controls_per_language_kind": 2,
        "source_negative_control_manifest_sha256": source_manifest_sha,
        "input_sha256_compact_array": sha(raw.encode()), "input_jsonl_sha256": sha(jsonl.encode()),
        "inputs_include_answer_or_split": False, "human_translation_reviewed": False,
        "interpretation": "Tests acceptance consistency on original source/translation pairs that previously passed AI-only review; these are not human-gold positive controls.",
        "controls": controls,
        "limitations": [
            "All base translations and back-translations were reviewed by AI only, not fluent human reviewers.",
            "Pairs were selected for passing prior AI review; the set is a small consistency diagnostic, not an unbiased estimate of false-rejection rate.",
            "This does not establish real-corpus precision, recall, or fluency accuracy.",
        ],
    }
    (HERE / "input.jsonl").write_text(jsonl, encoding="utf-8")
    (HERE / "input_array.json").write_text(raw + "\n", encoding="utf-8")
    (HERE / "control_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(inputs), "input_sha256": manifest["input_sha256_compact_array"], "cells": cells}, indent=2))


if __name__ == "__main__":
    main()
