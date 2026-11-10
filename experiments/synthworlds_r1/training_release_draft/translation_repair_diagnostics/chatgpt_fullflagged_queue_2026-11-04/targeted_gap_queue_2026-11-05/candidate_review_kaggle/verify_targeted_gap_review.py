#!/usr/bin/env python3
"""Verify a downloaded 187-row candidate-review artifact; never approve it."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedded_input() -> tuple[str, list[dict]]:
    source = (ROOT / "review.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    raw = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RAW_INPUT" for target in node.targets
        ):
            raw = ast.literal_eval(node.value)
            break
    if not isinstance(raw, str):
        raise ValueError("RAW_INPUT is missing or not a literal string")
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError("embedded input must be a list")
    return raw, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, help="Downloaded targeted_gap_review.json")
    parser.add_argument("--output", type=Path, default=ROOT / "review_verification.json")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "review_input_manifest.json").read_text(encoding="utf-8"))
    raw, inputs = load_embedded_input()
    expected_input_hash = manifest["review_input_sha256"]
    errors: list[str] = []
    if sha(raw) != expected_input_hash:
        errors.append("embedded input hash differs from the handoff manifest")
    if len(inputs) != 187 or len({r.get("review_key") for r in inputs}) != 187:
        errors.append("embedded input does not contain 187 unique rows")

    output = json.loads(args.artifact.read_text(encoding="utf-8"))
    if output.get("mode") != "targeted_gap_187":
        errors.append("review mode mismatch")
    if output.get("input_sha256") != expected_input_hash:
        errors.append("review output input hash mismatch")
    if output.get("nllb_model_revision") != manifest["reviewer_models"]["backtranslation"]["revision"]:
        errors.append("NLLB revision mismatch")
    if output.get("qwen_model_revision") != manifest["reviewer_models"]["judge"]["revision"]:
        errors.append("Qwen revision mismatch")
    records = output.get("records")
    if not isinstance(records, list):
        errors.append("review records are missing")
        records = []
    expected_by_key = {r["review_key"]: r for r in inputs}
    actual_keys = [r.get("review_key") for r in records]
    if len(actual_keys) != len(set(actual_keys)):
        errors.append("duplicate review keys in output")
    if set(actual_keys) != set(expected_by_key):
        errors.append("review output key coverage differs from input")

    counts = {"semantic_pass": 0, "fluency_pass": 0, "joint_pass": 0, "blocked": 0}
    by_language: dict[str, Counter] = {lang: Counter() for lang in ("fr", "de", "hi", "th")}
    by_kind: dict[str, Counter] = {kind: Counter() for kind in ("atomic", "composition")}
    for record in records:
        key = record.get("review_key")
        source = expected_by_key.get(key)
        if source is None:
            continue
        if record.get("source_sha256") != sha(source["source_question_en"]):
            errors.append(f"{key}: source hash mismatch")
        if record.get("translation_sha256") != sha(source["translated_question"]):
            errors.append(f"{key}: candidate hash mismatch")
        if record.get("language") != source["language"] or record.get("kind") != source["kind"]:
            errors.append(f"{key}: language/kind mismatch")
        decision = record.get("decision")
        if record.get("backtranslation_error") or not isinstance(decision, dict) or "error" in decision:
            counts["blocked"] += 1
            by_language[source["language"]]["blocked"] += 1
            by_kind[source["kind"]]["blocked"] += 1
            continue
        if set(decision) != {"key", "semantic_pass", "fluency_pass", "reason"}:
            errors.append(f"{key}: decision schema mismatch")
            continue
        if decision.get("key") != key or type(decision.get("semantic_pass")) is not bool or type(decision.get("fluency_pass")) is not bool:
            errors.append(f"{key}: invalid decision values")
            continue
        if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            errors.append(f"{key}: empty decision reason")
        if decision["semantic_pass"]:
            counts["semantic_pass"] += 1
        if decision["fluency_pass"]:
            counts["fluency_pass"] += 1
        if decision["semantic_pass"] and decision["fluency_pass"]:
            counts["joint_pass"] += 1
        for counter in (by_language[source["language"]], by_kind[source["kind"]]):
            counter["reviewed"] += 1
            counter["semantic_pass"] += int(decision["semantic_pass"])
            counter["fluency_pass"] += int(decision["fluency_pass"])
            counter["joint_pass"] += int(decision["semantic_pass"] and decision["fluency_pass"])

    if output.get("human_translation_reviewed") is not False:
        errors.append("output does not explicitly record AI-only review")
    if output.get("answers_or_splits_sent") is not False:
        errors.append("output does not attest that answers/splits were excluded")
    if output.get("blocked_rows") != counts["blocked"]:
        errors.append("blocked-row total disagrees with records")
    if output.get("reviewed_rows") != len(records) - counts["blocked"]:
        errors.append("reviewed-row total disagrees with records")

    verification = {
        "status": "VERIFIED_AI_ONLY_DIAGNOSTIC_QUARANTINED" if not errors else "VERIFICATION_FAILED",
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "input_sha256": expected_input_hash,
        "reviewed_input_rows": len(inputs),
        "record_rows": len(records),
        "counts": counts,
        "by_language": {lang: dict(c) for lang, c in by_language.items()},
        "by_kind": {kind: dict(c) for kind, c in by_kind.items()},
        "model_revisions": {
            "nllb": output.get("nllb_model_revision"),
            "qwen": output.get("qwen_model_revision"),
        },
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "errors": errors,
    }
    args.output.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
