#!/usr/bin/env python3
"""Package a pilot or full masked repair-candidate review for private Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "independent_review_inputs_2026-11-04"
TEMPLATE = HERE / "review_repair_candidates_kaggle_template.py"
PACKAGE = HERE / "candidate_review_kaggle"
KERNEL_ID = "bunnoob2005/synthworlds-ai-only-translation-repair-review"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot16", "full799"), default="pilot16")
    args = parser.parse_args()
    input_path = INPUT_DIR / f"{args.mode}.jsonl"
    map_path = INPUT_DIR / f"{args.mode}.private_map.jsonl"
    manifest_path = INPUT_DIR / f"{args.mode}.manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_rows = read_jsonl(input_path)
    mapping = read_jsonl(map_path)
    if (input_manifest.get("status") != "MASKED_AI_REVIEW_INPUT_ONLY_NOT_APPROVED"
            or input_manifest.get("input_sha256") != sha(input_path.read_bytes())
            or input_manifest.get("private_map_sha256") != sha(map_path.read_bytes())
            or len(input_rows) != input_manifest.get("rows")
            or len(mapping) != len(input_rows)):
        raise ValueError("Review input or private map failed its local integrity checks")

    raw = json.dumps(input_rows, ensure_ascii=False, separators=(",", ":"))
    raw_hash = sha(raw.encode("utf-8"))
    source = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        'MODE = "__MODE__"': f'MODE = "{args.mode}"',
        'INPUT_SHA256 = "__INPUT_SHA256__"': f'INPUT_SHA256 = "{raw_hash}"',
        "RAW_INPUT = __INPUT_JSON__": "RAW_INPUT = " + repr(raw),
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError(f"Expected one template slot: {before}")
        source = source.replace(before, after)
    compile(source, "review.py", "exec")

    PACKAGE.mkdir(parents=True, exist_ok=True)
    review_path = PACKAGE / "review.py"
    review_path.write_text(source, encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": "SynthWorlds AI-only Translation Repair Review",
        "code_file": "review.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (PACKAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (PACKAGE / "README.md").write_text(
        "# SynthWorlds translation-repair review\n\n"
        f"Mode: `{args.mode}`; input rows: {len(input_rows)}. This private Kaggle run generates NLLB-200 back-translations "
        "and Qwen2.5-14B semantic/fluency judgments for masked repair candidates. It sends no answers, split labels, "
        "or hidden facts. The output is AI-only diagnostic evidence; it does not approve translations or start training. "
        "NLLB-200 is under CC-BY-NC 4.0; use is limited to this noncommercial research task.\n",
        encoding="utf-8",
    )
    launch = {
        "status": "PACKAGED_PRIVATE_KAGGLE_AI_REVIEW_NOT_APPROVAL",
        "mode": args.mode,
        "kernel_id": KERNEL_ID,
        "rows": len(input_rows),
        "input_manifest_sha256": sha(manifest_path.read_bytes()),
        "input_jsonl_sha256": sha(input_path.read_bytes()),
        "input_compact_array_sha256": raw_hash,
        "private_map_sha256": sha(map_path.read_bytes()),
        "template_sha256": sha(TEMPLATE.read_bytes()),
        "review_script_sha256": sha(review_path.read_bytes()),
        "review_script_uploaded_matches_generated_source": True,
        "answers_or_splits_in_input": False,
        "human_translation_reviewed": False,
        "training_approved": False,
    }
    write_path = INPUT_DIR / f"{args.mode}.kaggle_launch_manifest.json"
    write_path.write_text(json.dumps(launch, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"package_dir": str(PACKAGE), "launch_manifest": str(write_path),
                      "kernel_id": KERNEL_ID, "mode": args.mode, "rows": len(input_rows),
                      "input_sha256": raw_hash, "script_sha256": launch["review_script_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
