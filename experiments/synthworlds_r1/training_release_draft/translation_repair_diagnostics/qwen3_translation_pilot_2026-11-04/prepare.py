#!/usr/bin/env python3
"""Build a balanced 16-row blinded Qwen3 translation pilot."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "chatgpt_third_candidate_2026-11-04" / "kaggle_review" / "review_input.json"
TEMPLATE = HERE / "generation_template.py"
MODEL_ID = "Qwen/Qwen3-14B"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
LANGUAGES = ("fr", "de", "hi", "th")
KINDS = ("atomic", "composition")


def canonical_sha(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    selected = []
    for lang in LANGUAGES:
        for kind in KINDS:
            rows = [row for row in source_rows if row["language"] == lang and row["kind"] == kind]
            if len(rows) < 2:
                raise ValueError(f"Need at least two {lang}/{kind} rows for balanced pilot")
            for row in rows[:2]:
                selected.append({
                    "review_key": row["review_key"],
                    "language": lang,
                    "kind": kind,
                    "source_question_en": row["source_question_en"],
                })
    if len(selected) != 16 or len({row["review_key"] for row in selected}) != 16:
        raise ValueError("Pilot selection is not 16 unique rows")
    raw = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    input_hash = canonical_sha(selected)

    text = TEMPLATE.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    values = {
        "INPUT_SHA256": f'INPUT_SHA256 = "{input_hash}"',
        "RAW_INPUT": "RAW_INPUT = " + repr(raw),
    }
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in values:
                    if node.lineno != node.end_lineno or target.id in found:
                        raise ValueError(f"Unexpected {target.id} template assignment")
                    lines[node.lineno - 1] = values[target.id]
                    found.add(target.id)
    if found != set(values):
        raise ValueError("Expected input constants are missing from template")
    script = "\n".join(lines) + "\n"
    ast.parse(script)

    (HERE / "generation_input.json").write_text(raw + "\n", encoding="utf-8")
    (HERE / "generation.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/synthworlds-qwen3-translation-pilot-2026-11-04",
        "title": "SynthWorlds Qwen3 Translation Pilot 2026-11-04",
        "code_file": "generation.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "PREPARED_BLINDED_QWEN3_TRANSLATION_PILOT_DIAGNOSTIC_ONLY",
        "rows": len(selected),
        "counts_by_language_kind": {
            f"{lang}/{kind}": sum(row["language"] == lang and row["kind"] == kind for row in selected)
            for lang in LANGUAGES for kind in KINDS
        },
        "input_sha256": input_hash,
        "generation_script_sha256": file_sha(HERE / "generation.py"),
        "source_review_input_sha256": file_sha(SOURCE),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    (HERE / "build_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
