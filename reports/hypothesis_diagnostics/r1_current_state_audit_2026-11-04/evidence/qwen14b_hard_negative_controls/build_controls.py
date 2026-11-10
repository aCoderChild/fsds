#!/usr/bin/env python3
"""Build blinded near-mismatch controls from AI-screened positive translations.

Each control reuses a hash-bound translation that passed both existing Qwen
screens and the round-trip checks, then changes exactly one English relation
word. The control is diagnostic, not a human gold translation.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TRAINING = REPO / "experiments/synthworlds_r1/training_release_draft"
TARGET_LANGS = ("fr", "de", "hi", "th")
PASS_STATUSES = {"AI_ONLY_TWO_MODEL_ROUNDTRIP_PASS", "AI_ONLY_14B_ADJUDICATED_PASS"}

sys.path.insert(0, str(TRAINING / "scripts"))
from review_training_translations_qwen import collect_rows, masked_pair  # noqa: E402


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def swap_word(text: str, left: str, right: str) -> str | None:
    pattern = re.compile(rf"\b{re.escape(left)}\b", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    old = match.group(0)
    replacement = right.capitalize() if old[0].isupper() else right
    return text[:match.start()] + replacement + text[match.end():]


def candidates_for_cell(lang: str, kind: str) -> list[dict]:
    source_rows, _ = collect_rows(lang)
    source_by_key = {(r["prompt_id"], r["kind"]): r for r in source_rows}
    review_path = TRAINING / "translation_reviews_roundtrip_qwen14b" / f"{lang}.jsonl"
    reviews = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    review_by_key = {(r["prompt_id"], r["kind"]): r for r in reviews}

    if kind == "atomic":
        families = {
            "kinship": (("father", "mother"), ("mother", "father")),
            "event": (("officially open", "officially close"), ("officially close", "officially open")),
        }
    elif kind == "composition" and lang in {"fr", "th"}:
        families = {"kinship": (("father", "mother"), ("mother", "father")),
                    "family_relation": (("spouse", "sibling"), ("sibling", "spouse"))}
    elif kind == "composition" and lang in {"de", "hi"}:
        families = {"kinship": (("father", "mother"), ("mother", "father")),
                    "profession": (("editor", "director"), ("director", "editor"))}
    else:
        raise ValueError(f"Unsupported cell: {lang}/{kind}")

    by_family = {family: [] for family in families}
    for (prompt_id, row_kind), source in source_by_key.items():
        if row_kind != kind:
            continue
        review = review_by_key.get((prompt_id, kind))
        if not review or review.get("final_review_status") not in PASS_STATUSES:
            continue
        if source.get("text_en") != review.get("text_en") or source.get("text_translation") != review.get("text_translation"):
            raise ValueError(f"Accepted source mismatch: {lang}/{prompt_id}/{kind}")
        _, pair = masked_pair(source)
        original = pair["source_english"]
        target = pair["target_translation"]
        for family, swaps in families.items():
            for left, right in swaps:
                mutated = swap_word(original, left, right)
                if mutated and mutated != original:
                    by_family[family].append({
                        "language": lang,
                        "kind": kind,
                        "prompt_id": prompt_id,
                        "family": family,
                        "source_original": original,
                        "source_counterfactual": mutated,
                        "translation": target,
                        "translation_sha256": sha_text(target),
                        "base_review_status": review["final_review_status"],
                        "base_review_pair_sha256": review.get("review_pair_sha256"),
                        "base_review_file_sha256": sha_bytes(review_path.read_bytes()),
                    })
                    break
            if by_family[family] and by_family[family][-1]["prompt_id"] == prompt_id:
                continue

    chosen = []
    for family, rows in by_family.items():
        # Keep one control per unique source template; select deterministically.
        unique = {}
        for row in rows:
            unique.setdefault(row["source_original"], row)
        if not unique:
            raise ValueError(f"No accepted positive source for {lang}/{kind}/{family}")
        ordered = sorted(unique.values(), key=lambda row: (
            hashlib.sha256(f"r1-qwen14b-hard-control-v1\0{lang}\0{kind}\0{family}\0{row['source_original']}".encode()).hexdigest(),
            row["prompt_id"],
        ))
        chosen.append(ordered[0])
    return chosen


def main() -> None:
    rows = []
    manifest = []
    for lang in TARGET_LANGS:
        for kind in ("atomic", "composition"):
            for control in candidates_for_cell(lang, kind):
                control_id = f"HN{len(rows) + 1:03d}"
                input_row = {
                    "control_id": control_id,
                    "language": lang,
                    "kind": kind,
                    "source": control["source_counterfactual"],
                    "translation": control["translation"],
                }
                row_with_id = {**control, "control_id": control_id}
                manifest.append(row_with_id)
                rows.append(input_row)
    expected = len(TARGET_LANGS) * 2 * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} hard controls, built {len(rows)}")
    keyset = {(r["language"], r["kind"], r["control_id"]) for r in rows}
    if len(keyset) != len(rows):
        raise ValueError("Duplicate hard-control IDs")
    input_jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    input_array = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    (HERE / "input.jsonl").write_text(input_jsonl, encoding="utf-8")
    control_hash = sha_bytes(input_array.encode("utf-8"))
    (HERE / "control_manifest.json").write_text(json.dumps({
        "status": "BLINDED_HARD_NEGATIVE_CONTROLS_DIAGNOSTIC_ONLY",
        "rows": len(rows),
        "controls_per_language_kind": 2,
        "construction": "Use a translation with a passing two-model+roundtrip AI status, mask entity names, and change exactly one English relation word while keeping the target translation fixed.",
        "predeclared_diagnostic_rule": "All 16 known mismatches must be semantically rejected, with zero malformed outputs, to proceed to a larger calibration. Any semantic pass or schema error fails this screen. Passing this screen does not approve translations or release the corpus.",
        "input_sha256_compact_array": control_hash,
        "input_jsonl_sha256": sha_bytes(input_jsonl.encode("utf-8")),
        "builder_sha256": sha_bytes(Path(__file__).read_bytes()),
        "answers_included": False,
        "human_translation_reviewed": False,
        "controls": manifest,
        "limitations": [
            "The source/candidate pairing is intentionally wrong by one relation-word substitution.",
            "The reused translations are AI-screened, not human-gold translations.",
            "This tests rejection of relation-level mismatches; it does not certify target-language fluency or the full corpus.",
        ],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "input_array.json").write_text(input_array + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sha256": control_hash, "cells": {
        f"{lang}/{kind}": sum(r["language"] == lang and r["kind"] == kind for r in rows)
        for lang in TARGET_LANGS for kind in ("atomic", "composition")
    }}, indent=2))


if __name__ == "__main__":
    main()
