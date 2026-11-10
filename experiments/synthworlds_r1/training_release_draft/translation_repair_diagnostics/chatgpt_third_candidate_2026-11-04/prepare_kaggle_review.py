#!/usr/bin/env python3
"""Prepare an answer-blind, source-bound Kaggle AI review of 187 repair candidates."""
from __future__ import annotations
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_REVIEW = HERE / "requests.jsonl"
CANDIDATES = HERE / "proposed_translations.jsonl"
TEMPLATE = HERE.parent / "chatgpt_fullflagged_queue_2026-11-04" / "candidate_review_kaggle" / "review.py"
OUT = HERE / "kaggle_review"
TARGET = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
PLACEHOLDER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    request_rows = load_jsonl(SOURCE_REVIEW)
    candidate_rows = load_jsonl(CANDIDATES)
    if len(request_rows) != 187 or len(candidate_rows) != 187:
        raise ValueError("Expected exactly 187 masked source rows and 187 candidate rows")
    if len({r.get("source_review_key") for r in request_rows}) != 187:
        raise ValueError("Duplicate source-review key")
    if len({r.get("source_review_key") for r in candidate_rows}) != 187:
        raise ValueError("Duplicate candidate key")
    by_candidate = {r["source_review_key"]: r for r in candidate_rows}
    review_rows = []
    counts = collections.Counter()
    for source in request_rows:
        key = source["source_review_key"]
        candidate = by_candidate.get(key)
        if candidate is None or set(candidate) != {"source_review_key", "translated_question"}:
            raise ValueError(f"Missing or malformed candidate {key}")
        if set(source) != {
            "source_review_key", "prompt_id", "language", "kind", "source_question_en",
            "source_question_sha256", "prior_chatgpt_review_status", "prior_translategemma_status",
        }:
            raise ValueError(f"Unexpected source fields for {key}")
        language = source["language"]
        src = source["source_question_en"]
        trg = candidate["translated_question"]
        if language not in TARGET or source["kind"] not in {"atomic", "composition"}:
            raise ValueError(f"Invalid language or kind for {key}")
        if sha(src.encode("utf-8")) != source["source_question_sha256"]:
            raise ValueError(f"Source hash mismatch for {key}")
        if not src.strip() or not trg.strip():
            raise ValueError(f"Empty source or candidate for {key}")
        if collections.Counter(PLACEHOLDER.findall(src)) != collections.Counter(PLACEHOLDER.findall(trg)):
            raise ValueError(f"Placeholder multiset mismatch for {key}")
        if language == "hi" and not any("\u0900" <= c <= "\u097f" for c in trg):
            raise ValueError(f"Hindi candidate lacks Devanagari for {key}")
        if language == "th" and not any("\u0e00" <= c <= "\u0e7f" for c in trg):
            raise ValueError(f"Thai candidate lacks Thai script for {key}")
        review_rows.append({
            "review_key": key,
            "language": language,
            "kind": source["kind"],
            "source_question_en": src,
            "translated_question": trg,
        })
        counts[(language, source["kind"])] += 1
    if set(by_candidate) != {r["source_review_key"] for r in request_rows}:
        raise ValueError("Candidate/source coverage mismatch")

    raw_input = json.dumps(review_rows, ensure_ascii=False, separators=(",", ":"))
    input_hash = sha(raw_input.encode("utf-8"))
    source_bytes = TEMPLATE.read_bytes()
    source_text = source_bytes.decode("utf-8")
    tree = ast.parse(source_text)
    required = {"MODE", "INPUT_SHA256", "RAW_INPUT"}
    replacements = {
        "MODE": 'MODE = "third_candidate_187"',
        "INPUT_SHA256": f'INPUT_SHA256 = "{input_hash}"',
        "RAW_INPUT": "RAW_INPUT = " + repr(raw_input),
    }
    found = set()
    lines = source_text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in required:
                    if target.id in found or node.lineno != node.end_lineno:
                        raise ValueError(f"Unexpected template assignment shape: {target.id}")
                    lines[node.lineno - 1] = replacements[target.id]
                    found.add(target.id)
    if found != required:
        raise ValueError(f"Template constants missing: {required - found}")
    rendered = "\n".join(lines) + "\n"
    ast.parse(rendered)
    (OUT / "review.py").write_text(rendered, encoding="utf-8")
    (OUT / "review_input.json").write_text(raw_input + "\n", encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/synthworlds-third-candidate-review-2026-11-04",
        "title": "SynthWorlds Third Candidate AI-only Review 2026-11-04",
        "code_file": "review.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "PREPARED_FOR_MASKED_INDEPENDENT_AI_ONLY_REVIEW",
        "rows": len(review_rows),
        "input_sha256": input_hash,
        "source_queue_sha256": sha(SOURCE_REVIEW.read_bytes()),
        "candidate_sha256": sha(CANDIDATES.read_bytes()),
        "template_sha256": sha(source_bytes),
        "prepared_review_script_sha256": sha((OUT / "review.py").read_bytes()),
        "counts_by_language_kind": {f"{a}/{b}": n for (a,b), n in sorted(counts.items())},
        "model_ids_and_revisions": {
            "backtranslation": "facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d",
            "judge": "Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        },
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "limitations": [
            "This is an AI-only semantic, fluency, and back-translation screen; it cannot certify human fluency.",
            "All outputs remain quarantined diagnostic candidates; no output automatically changes the frozen corpus.",
            "The exact serving/version of the ChatGPT used to draft candidate translations was not exposed.",
        ],
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readback = ast.parse(rendered)
    constants = {t.id: ast.literal_eval(n.value) for n in readback.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name) and t.id in required}
    if constants["MODE"] != "third_candidate_187" or constants["INPUT_SHA256"] != input_hash or constants["RAW_INPUT"] != raw_input:
        raise ValueError("Rendered review script constants do not match the frozen input")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
