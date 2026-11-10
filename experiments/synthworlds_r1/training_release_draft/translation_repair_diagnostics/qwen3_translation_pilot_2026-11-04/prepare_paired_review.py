#!/usr/bin/env python3
"""Prepare a blinded, paired AI screen for two 16-row translation candidates."""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_INPUT = HERE / "generation_input.json"
QWEN_OUTPUT = HERE / "remote_output" / "qwen3_translation_pilot.json"
QWEN_VERIFICATION = HERE / "verification.json"
QWEN_BUILD = HERE / "build_manifest.json"
ALT_CANDIDATES = HERE / "alternative_translation_candidates_v1.json"
ALT_MANIFEST = HERE / "alternative_candidate_manifest.json"
TEMPLATE = HERE.parent / "chatgpt_third_candidate_2026-11-04" / "kaggle_review" / "review.py"
KERNEL_DIR = HERE / "paired_review_kernel"
KERNEL_ID = "bunnoob2005/swsm-qwen3-pilot-paired-review-2026-11-04"
REMOTE_SOURCE = HERE / "paired_review_remote_source" / "swsm-qwen3-pilot-paired-review-2026-11-04.py"
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: object) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source_rows = read_json(SOURCE_INPUT)
    qwen_output = read_json(QWEN_OUTPUT)
    qwen_verification = read_json(QWEN_VERIFICATION)
    qwen_build = read_json(QWEN_BUILD)
    alt_rows = read_json(ALT_CANDIDATES)
    alt_manifest = read_json(ALT_MANIFEST)

    if len(source_rows) != 16 or len({r["review_key"] for r in source_rows}) != 16:
        raise ValueError("Expected the exact 16-row source pilot")
    if qwen_verification.get("status") != "VERIFIED_QWEN3_TRANSLATION_PILOT_DIAGNOSTIC_ONLY":
        raise ValueError("The Qwen3 baseline generation receipt is not verified")
    if (qwen_build.get("generation_script_sha256") != qwen_verification.get("generation_script_sha256")
            or qwen_build.get("input_sha256") != qwen_verification.get("input_sha256")):
        raise ValueError("Qwen3 baseline generation build and result receipts disagree")
    if (sha_json(source_rows) != qwen_build.get("input_sha256")
            or sha_bytes(QWEN_OUTPUT.read_bytes()) != qwen_verification.get("output_sha256")):
        raise ValueError("Qwen3 baseline input/output does not match its verified receipt")
    if alt_manifest.get("status") != "AI_GENERATED_BLINDED_CANDIDATE_ONLY" or alt_manifest.get("release_approved") is not False:
        raise ValueError("Alternative candidates are not explicitly quarantined AI-only artifacts")
    if sha_bytes(ALT_CANDIDATES.read_bytes()) != alt_manifest.get("candidate_file_sha256"):
        raise ValueError("Alternative-candidate file hash does not match its manifest")
    if sha_bytes(SOURCE_INPUT.read_bytes()) != alt_manifest.get("input_source_sha256"):
        raise ValueError("Alternative candidate manifest is bound to a different source input")
    if len(alt_rows) != 16 or {r["review_key"] for r in alt_rows} != {r["review_key"] for r in source_rows}:
        raise ValueError("Alternative candidate set does not cover the exact pilot source rows")

    source_by_key = {r["review_key"]: r for r in source_rows}
    alt_by_key = {r["review_key"]: r for r in alt_rows}
    qwen_by_key = {r["review_key"]: r for r in qwen_output.get("records", [])}
    if set(qwen_by_key) != set(source_by_key) or len(qwen_by_key) != 16:
        raise ValueError("Qwen3 candidates do not cover the exact pilot rows")

    review_rows = []
    private_map = []
    for source in source_rows:
        key = source["review_key"]
        alt = alt_by_key[key]
        if (alt["language"], alt["kind"], alt["source_question_en"]) != (
                source["language"], source["kind"], source["source_question_en"]):
            raise ValueError(f"Alternative source row mismatch: {key}")
        if Counter(PLACEHOLDER.findall(source["source_question_en"])) != Counter(PLACEHOLDER.findall(alt["translated_question"])):
            raise ValueError(f"Alternative placeholder mismatch: {key}")
        if source["language"] == "hi" and not any("\u0900" <= char <= "\u097f" for char in alt["translated_question"]):
            raise ValueError(f"Alternative Hindi candidate lacks Devanagari: {key}")
        if source["language"] == "th":
            if not any("\u0e00" <= char <= "\u0e7f" for char in alt["translated_question"]):
                raise ValueError(f"Alternative Thai candidate lacks Thai script: {key}")
            if any("\u0900" <= char <= "\u097f" for char in alt["translated_question"]):
                raise ValueError(f"Alternative Thai candidate contains unexpected Devanagari: {key}")
        qwen_candidate = qwen_by_key[key]
        if (qwen_candidate.get("source_sha256") != sha_bytes(source["source_question_en"].encode("utf-8"))
                or qwen_candidate.get("translation_sha256") != sha_bytes(qwen_candidate.get("translation", "").encode("utf-8"))):
            raise ValueError(f"Qwen3 candidate row/hash mismatch: {key}")
        for method, translation in (
                ("qwen3_14b", qwen_by_key[key].get("translation")),
                ("assistant_ai_candidate", alt.get("translated_question"))):
            if not isinstance(translation, str) or not translation.strip():
                raise ValueError(f"Missing {method} candidate at {key}")
            review_key = "P" + hashlib.sha256(f"paired-review-v1:{key}:{method}".encode()).hexdigest()[:10].upper()
            review_rows.append({
                "review_key": review_key,
                "language": source["language"],
                "kind": source["kind"],
                "source_question_en": source["source_question_en"],
                "translated_question": translation,
            })
            private_map.append({
                "review_key": review_key,
                "source_review_key": key,
                "candidate_method": method,
                "candidate_sha256": sha_bytes(translation.encode("utf-8")),
            })

    random.Random(4102026).shuffle(review_rows)
    random.Random(4102026).shuffle(private_map)
    review_keys = [r["review_key"] for r in review_rows]
    if len(set(review_keys)) != 32:
        raise ValueError("Paired review keys are not unique")
    raw = json.dumps(review_rows, ensure_ascii=False, separators=(",", ":"))
    input_hash = sha_json(review_rows)
    expected_fields = {"review_key", "language", "kind", "source_question_en", "translated_question"}
    if any(set(row) != expected_fields for row in review_rows):
        raise ValueError("Blinded screen input contains fields outside the approved question/candidate schema")

    template_text = TEMPLATE.read_text(encoding="utf-8")
    lines = template_text.splitlines()
    tree = ast.parse(template_text)
    replacements = {
        "MODE": 'MODE = "qwen3_translation_pilot_blinded_paired_screen_32"',
        "INPUT_SHA256": f'INPUT_SHA256 = "{input_hash}"',
        "RAW_INPUT": "RAW_INPUT = " + repr(raw),
    }
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    if node.lineno != node.end_lineno or target.id in found:
                        raise ValueError(f"Unexpected template assignment for {target.id}")
                    lines[node.lineno - 1] = replacements[target.id]
                    found.add(target.id)
    if found != set(replacements):
        raise ValueError("Expected mode/input constants missing from reviewer template")

    # NLLB drops the synthetic ZXQ...QXZ strings. Replace them only during the
    # back-translation call with ordinary names/numerals, then restore exact
    # original markers before Qwen sees the round-trip pair.
    tree = ast.parse("\n".join(lines))
    mask_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mask_placeholders"), None)
    if mask_node is None:
        raise ValueError("Template mask_placeholders() not found")
    mask_function = '''def mask_placeholders(text):
    counts = Counter()
    originals = {}
    entity_aliases = {
        "A": "Alice", "B": "Bob", "C": "Carol", "D": "David", "E": "Emma",
        "F": "Frank", "G": "Grace", "H": "Henry", "I": "Iris", "J": "James",
        "K": "Kate", "L": "Liam", "M": "Maya", "N": "Noah", "O": "Olivia",
        "P": "Peter", "Q": "Quinn", "R": "Rose", "S": "Sam", "T": "Tara",
        "U": "Uma", "V": "Victor", "W": "Willow", "X": "Xavier", "Y": "Yara", "Z": "Zane",
    }
    number_aliases = {
        "A": "17", "B": "23", "C": "31", "D": "43", "E": "59", "F": "61",
        "G": "73", "H": "89", "I": "97", "J": "101", "K": "113", "L": "127",
        "M": "139", "N": "149", "O": "157", "P": "167", "Q": "179", "R": "191",
        "S": "199", "T": "211", "U": "223", "V": "227", "W": "229", "X": "233",
        "Y": "239", "Z": "241",
    }
    def replace(match):
        category, letter = match.group(1), match.group(2)
        if len(letter) != 1:
            raise ValueError(f"Unsupported marker suffix for safe back-translation: {match.group(0)}")
        token = (entity_aliases if category == "ENTITY" else number_aliases)[letter]
        counts[token] += 1
        originals[token] = match.group(0)
        return token
    return PLACEHOLDER.sub(replace, text), originals, counts
'''
    lines[mask_node.lineno - 1:mask_node.end_lineno] = mask_function.rstrip("\n").splitlines()
    script_text = "\n".join(lines) + "\n"
    script_tree = ast.parse(script_text)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "mask_placeholders" for n in script_tree.body):
        raise ValueError("Generated alias-masking reviewer did not parse")
    if 'opaque placeholders: expected=' in script_text:
        script_text = script_text.replace('opaque placeholders: expected=', 'temporary alias markers: expected=')
    ast.parse(script_text)
    if any(secret in script_text for secret in ("qwen3_14b", "assistant_ai_candidate", "paired_review_private_map")):
        raise ValueError("Reviewer source leaks private candidate labels or mapping")

    KERNEL_DIR.mkdir(exist_ok=True)
    (HERE / "paired_review_input.json").write_text(json.dumps(review_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "paired_review_private_map.json").write_text(json.dumps(private_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "paired_review.py").write_text(script_text, encoding="utf-8")
    (KERNEL_DIR / "review.py").write_text(script_text, encoding="utf-8")
    (KERNEL_DIR / "review_input.json").write_text(json.dumps(review_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": "SWSM Qwen3 Pilot Paired Review 2026-11-04",
        "code_file": "review.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (KERNEL_DIR / "README.md").write_text(
        "# Blinded paired translation screen\n\n"
        "This private kernel screens 32 masked question/candidate pairs: two candidate translations for each of 16 source questions. It receives no method labels, answer keys, facts, split assignments, or private pair map.\n\n"
        "For NLLB back-translation only, `[ENTITY_X]` markers are replaced with common names and `[NUMBER_X]` markers with numerals, then exact source markers are restored before semantic adjudication. Any missing/changed alias blocks that row; it is never silently repaired. All decisions remain diagnostic-only.\n",
        encoding="utf-8",
    )

    build = {
        "status": "PREPARED_BLINDED_PAIRED_AI_SCREEN_DIAGNOSTIC_ONLY",
        "rows": 32,
        "kaggle_kernel_id": KERNEL_ID,
        "unique_source_questions": 16,
        "candidates_per_source_question": 2,
        "review_input_sha256": input_hash,
        "review_script_sha256": sha_bytes(script_text.encode("utf-8")),
        "private_map_sha256": sha_bytes((HERE / "paired_review_private_map.json").read_bytes()),
        "qwen3_candidate_file_sha256": sha_bytes(QWEN_OUTPUT.read_bytes()),
        "qwen3_candidate_verification_sha256": sha_bytes(QWEN_VERIFICATION.read_bytes()),
        "alternative_candidate_file_sha256": sha_bytes(ALT_CANDIDATES.read_bytes()),
        "alternative_candidate_manifest_sha256": sha_bytes(ALT_MANIFEST.read_bytes()),
        "reviewer_template_sha256": sha_bytes(TEMPLATE.read_bytes()),
        "qwen_model_id": "Qwen/Qwen2.5-14B-Instruct",
        "qwen_model_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "backtranslation_model_id": "facebook/nllb-200-distilled-600M",
        "backtranslation_model_revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "placeholder_masking": "common names and numerals for NLLB only; exact markers restored before adjudication",
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }
    (HERE / "paired_review_build_manifest.json").write_text(json.dumps(build, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(build, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
