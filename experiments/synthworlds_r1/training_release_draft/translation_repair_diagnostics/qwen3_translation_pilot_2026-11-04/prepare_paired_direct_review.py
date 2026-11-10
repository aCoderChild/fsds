#!/usr/bin/env python3
"""Build the blinded direct review plus same-run reviewer sentinels."""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
INPUT = HERE / "paired_review_input.json"
PRIVATE_CANDIDATE_MAP = HERE / "paired_review_private_map.json"
PAIRED_BUILD = HERE / "paired_review_build_manifest.json"
CONTROL_INPUT = ROOT / "reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls/input_array.json"
CONTROL_REVIEWER = ROOT / "reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls/review.py"
TEMPLATE = HERE / "paired_direct_review_template.py"
KERNEL_DIR = HERE / "paired_direct_review_kernel"
KERNEL_ID = "bunnoob2005/swsm-qwen14-direct-paired-calibration-2026-11-04"
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode("utf-8"))


def sha_json(value: object) -> str:
    return sha_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def calibrated_instruction(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "instruction" for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("Could not extract the calibration instruction")


def set_constant(source: str, name: str, replacement: str) -> str:
    lines = source.splitlines()
    tree = ast.parse(source)
    found = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            if node.lineno != node.end_lineno:
                raise ValueError(f"Expected one-line template constant: {name}")
            lines[node.lineno - 1] = replacement
            found += 1
    if found != 1:
        raise ValueError(f"Expected exactly one template constant: {name}")
    result = "\n".join(lines) + "\n"
    ast.parse(result)
    return result


def main() -> None:
    candidates = read_json(INPUT)
    private = read_json(PRIVATE_CANDIDATE_MAP)
    paired_build = read_json(PAIRED_BUILD)
    controls = read_json(CONTROL_INPUT)
    control_script = CONTROL_REVIEWER.read_text(encoding="utf-8")
    instruction = calibrated_instruction(control_script)
    if sha_bytes(PRIVATE_CANDIDATE_MAP.read_bytes()) != paired_build.get("private_map_sha256"):
        raise ValueError("The candidate method map changed after paired review preparation")
    if (len(candidates) != 32 or len(private) != 32 or len(controls) != 16
            or len({row["review_key"] for row in candidates}) != 32):
        raise ValueError("Expected 32 blinded candidates and 16 known-mismatch sentinels")
    candidate_by_key = {row["review_key"]: row for row in candidates}
    map_by_key = {row["review_key"]: row for row in private}
    if set(candidate_by_key) != set(map_by_key):
        raise ValueError("Candidate rows and local method map do not match")

    rows = []
    private_map = []
    for row in candidates:
        rows.append(row)
        hidden = map_by_key[row["review_key"]]
        if hidden.get("candidate_sha256") != sha_text(row["translated_question"]):
            raise ValueError(f"Candidate mapping hash mismatch at {row['review_key']}")
        private_map.append({
            "review_key": row["review_key"],
            "row_role": "candidate",
            "source_review_key": hidden["source_review_key"],
            "candidate_method": hidden["candidate_method"],
            "candidate_sha256": hidden["candidate_sha256"],
        })

    for control in controls:
        key = "R" + hashlib.sha256(f"direct-sentinel-v1:{control['control_id']}".encode()).hexdigest()[:11].upper()
        rows.append({
            "review_key": key,
            "language": control["language"],
            "kind": control["kind"],
            "source_question_en": control["source"],
            "translated_question": control["translation"],
        })
        private_map.append({
            "review_key": key,
            "row_role": "known_mismatch_sentinel",
            "control_id": control["control_id"],
            "candidate_sha256": sha_text(control["translation"]),
        })

    if len({row["review_key"] for row in rows}) != 48:
        raise ValueError("Candidate and calibration keys collide")
    for row in rows:
        if (set(row) != {"review_key", "language", "kind", "source_question_en", "translated_question"}
                or row["language"] not in {"fr", "de", "hi", "th"}
                or row["kind"] not in {"atomic", "composition"}
                or Counter(PLACEHOLDER.findall(row["source_question_en"]))
                != Counter(PLACEHOLDER.findall(row["translated_question"]))):
            raise ValueError(f"Invalid or marker-mismatched blinded row: {row['review_key']}")
    random.Random(41021104).shuffle(rows)
    random.Random(41021104).shuffle(private_map)
    if {row["review_key"] for row in rows} != {row["review_key"] for row in private_map}:
        raise ValueError("Randomized private mapping lost row coverage")

    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    input_hash = sha_text(raw)
    prompt_hash = sha_text(instruction)
    script = TEMPLATE.read_text(encoding="utf-8")
    for name, value in (
        ("INPUT_SHA256", input_hash),
        ("PROMPT_SHA256", prompt_hash),
    ):
        script = set_constant(script, name, f'{name} = "{value}"')
    script = set_constant(script, "RAW_INPUT", "RAW_INPUT = " + repr(raw))
    script = set_constant(script, "PROMPT_INSTRUCTION", "PROMPT_INSTRUCTION = " + repr(instruction))
    script = set_constant(script, "MODE", 'MODE = "blinded_direct_translation_screen_with_sentinels"')
    ast.parse(script)
    generated_prompt = ast.literal_eval(next(
        node.value for node in ast.walk(ast.parse(script))
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PROMPT_INSTRUCTION"
            for target in node.targets)))
    if generated_prompt != instruction:
        raise ValueError("Generated reviewer instruction differs from the calibrated direct-review instruction")

    KERNEL_DIR.mkdir(exist_ok=True)
    (HERE / "paired_direct_review_input.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    private_path = HERE / "paired_direct_review_private_map.json"
    private_path.write_text(json.dumps(private_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    script_path = HERE / "paired_direct_review.py"
    script_path.write_text(script, encoding="utf-8")
    (KERNEL_DIR / "review.py").write_text(script, encoding="utf-8")
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID,
        "title": "SWSM Qwen14 Direct Paired Calibration 2026-11-04",
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
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (KERNEL_DIR / "README.md").write_text(
        "# Blinded direct translation screen\n\n"
        "A bounded AI-only diagnostic of 32 paired translation candidates, mixed with 16 known-mismatch "
        "reviewer sentinels. Each row is reviewed independently by the same pinned Qwen2.5-14B checkpoint "
        "and instruction used by the direct hard-negative calibration. The kernel receives English sources, "
        "target-language questions, opaque row keys, and language/kind only. It receives no answers, facts, "
        "candidate method labels, split assignments, or private mapping. The sentinels are not labeled in the "
        "kernel. AI judgments are diagnostic only and do not constitute fluent-human review or release approval.\n",
        encoding="utf-8",
    )
    build = {
        "status": "PREPARED_BLINDED_DIRECT_PAIRED_AI_SCREEN_WITH_SENTINELS",
        "kaggle_kernel_id": KERNEL_ID,
        "rows_total": 48,
        "candidate_rows": 32,
        "known_mismatch_sentinels": 16,
        "candidate_input_sha256": sha_json(candidates),
        "candidate_private_map_sha256": sha_bytes(PRIVATE_CANDIDATE_MAP.read_bytes()),
        "direct_input_sha256": input_hash,
        "direct_private_map_sha256": sha_bytes(private_path.read_bytes()),
        "review_script_sha256": sha_bytes(script.encode("utf-8")),
        "calibration_instruction_sha256": prompt_hash,
        "calibration_reviewer_sha256": sha_bytes(CONTROL_REVIEWER.read_bytes()),
        "paired_roundtrip_build_sha256": sha_bytes(PAIRED_BUILD.read_bytes()),
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "model_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "runtime": "FP16, device_map=auto, two NVIDIA T4 GPUs",
        "screen_route": "direct English source to target question; no NLLB/back-translation",
        "method_labels_sent": False,
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    build_path = HERE / "paired_direct_review_build_manifest.json"
    build_path.write_text(json.dumps(build, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(build, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
