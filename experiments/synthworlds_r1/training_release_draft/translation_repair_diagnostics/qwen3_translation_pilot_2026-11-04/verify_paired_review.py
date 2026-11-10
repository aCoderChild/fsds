#!/usr/bin/env python3
"""Verify provenance and paired results for the blinded 32-row translation screen."""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "paired_review_input.json"
PRIVATE_MAP = HERE / "paired_review_private_map.json"
BUILD = HERE / "paired_review_build_manifest.json"
SCRIPT = HERE / "paired_review.py"
REMOTE_SCRIPT = HERE / "paired_review_remote_source" / "swsm-qwen3-pilot-paired-review-2026-11-04.py"
OUTPUT = HERE / "paired_review_remote_output" / "repair_review.json"
REPORT = HERE / "paired_review_verification.json"
QWEN_INPUT = HERE / "generation_input.json"
QWEN_OUTPUT = HERE / "remote_output" / "qwen3_translation_pilot.json"
ALT_CANDIDATES = HERE / "alternative_translation_candidates_v1.json"
NLLB = ("facebook/nllb-200-distilled-600M", "f8d333a098d19b4fd9a8b18f94170487ad3f821d")
QWEN = ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
PACKAGES = {"transformers": "4.51.3", "accelerate": "1.15.0", "bitsandbytes": "0.50.2"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: object) -> str:
    return sha(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = read(INPUT)
    private = read(PRIVATE_MAP)
    build = read(BUILD)
    result = read(OUTPUT)
    script_hash = sha(SCRIPT.read_bytes())
    if not REMOTE_SCRIPT.is_file():
        raise ValueError("Pulled Kaggle reviewer source is missing")
    remote_script_hash = sha(REMOTE_SCRIPT.read_bytes())
    input_hash = canonical_sha(rows)
    keys = [r["review_key"] for r in rows]
    if len(rows) != 32 or len(set(keys)) != 32 or len(private) != 32:
        raise ValueError("Expected 32 unique blinded paired-review records")
    if remote_script_hash != script_hash:
        raise ValueError("Pulled Kaggle reviewer source differs from the reviewed script")
    if (build.get("review_input_sha256") != input_hash
            or build.get("review_script_sha256") != script_hash
            or build.get("private_map_sha256") != sha(PRIVATE_MAP.read_bytes())
            or result.get("input_sha256") != input_hash):
        raise ValueError("Review input, private map, script, build, or remote output hash mismatch")
    if (build.get("qwen3_candidate_file_sha256") != sha(QWEN_OUTPUT.read_bytes())
            or build.get("alternative_candidate_file_sha256") != sha(ALT_CANDIDATES.read_bytes())):
        raise ValueError("A candidate source changed after paired review preparation")
    if (result.get("qwen_model_id"), result.get("qwen_model_revision")) != QWEN:
        raise ValueError("Qwen reviewer model pin mismatch")
    if (result.get("nllb_model_id"), result.get("nllb_model_revision")) != NLLB:
        raise ValueError("NLLB back-translation model pin mismatch")
    if {name: result.get(name) for name in PACKAGES} != PACKAGES:
        raise ValueError("Runtime package pins mismatch")
    if result.get("gpu") != "Tesla T4" or result.get("quantization") != "4-bit NF4":
        raise ValueError("Unexpected reviewer runtime profile")
    if (result.get("answers_or_splits_sent") is not False
            or result.get("human_translation_reviewed") is not False):
        raise ValueError("Blinding/human-review metadata is misrepresented")
    if result.get("status") not in {"AI_ONLY_REPAIR_REVIEW_COMPLETE_DIAGNOSTIC_ONLY", "AI_ONLY_REPAIR_REVIEW_HAS_BLOCKED_ROWS"}:
        raise ValueError("Unexpected reviewer run status")
    if not result.get("started_at_utc") or not result.get("completed_at_utc"):
        raise ValueError("Remote output lacks runtime timestamps")

    private_by_key = {r["review_key"]: r for r in private}
    if set(private_by_key) != set(keys):
        raise ValueError("Private mapping does not match the blinded input keys")
    records = result.get("records")
    if not isinstance(records, list) or [r.get("review_key") for r in records] != keys:
        raise ValueError("Remote output has missing, duplicate, reordered, or extra rows")
    rows_by_key = {r["review_key"]: r for r in rows}
    labels_by_source: dict[str, dict[str, str]] = collections.defaultdict(dict)
    counts = collections.Counter()
    by_lang = collections.defaultdict(collections.Counter)
    seen_private_pairs = set()
    for record in records:
        key = record["review_key"]
        source = rows_by_key[key]
        hidden = private_by_key[key]
        if (record.get("language"), record.get("kind")) != (source["language"], source["kind"]):
            raise ValueError(f"Language/kind mismatch at {key}")
        if record.get("source_sha256") != sha(source["source_question_en"].encode("utf-8")):
            raise ValueError(f"Source hash mismatch at {key}")
        if record.get("translation_sha256") != sha(source["translated_question"].encode("utf-8")):
            raise ValueError(f"Candidate hash mismatch at {key}")
        if hidden.get("candidate_sha256") != record.get("translation_sha256"):
            raise ValueError(f"Private-map candidate hash mismatch at {key}")
        seen_private_pairs.add((hidden["source_review_key"], hidden["candidate_method"]))
        decision = record.get("decision") or {}
        if record.get("backtranslation_error") or "error" in decision:
            label = "blocked"
        else:
            if (decision.get("key") != key
                    or type(decision.get("semantic_pass")) is not bool
                    or type(decision.get("fluency_pass")) is not bool
                    or not isinstance(decision.get("reason"), str)
                    or not decision["reason"].strip()):
                raise ValueError(f"Invalid adjudication schema at {key}")
            label = "pass" if decision["semantic_pass"] and decision["fluency_pass"] else "flag"
        if label != "blocked":
            markers = collections.Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", source["translated_question"]))
            back_markers = collections.Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", record.get("backtranslation_en", "")))
            if markers != back_markers:
                raise ValueError(f"Back-translation marker restoration mismatch at {key}")
        labels_by_source[hidden["source_review_key"]][hidden["candidate_method"]] = label
        counts[(hidden["candidate_method"], label)] += 1
        by_lang[(hidden["candidate_method"], source["language"])][label] += 1

    expected_sources = {r["review_key"] for r in read(QWEN_INPUT)}
    expected_private_pairs = {(source, method) for source in expected_sources
                              for method in ("qwen3_14b", "assistant_ai_candidate")}
    if (set(labels_by_source) != expected_sources
            or seen_private_pairs != expected_private_pairs
            or any(set(pair) != {"qwen3_14b", "assistant_ai_candidate"} for pair in labels_by_source.values())):
        raise ValueError("Every source question must have exactly one candidate from each blinded method")
    blocked = sum(counts[(method, "blocked")] for method in ("qwen3_14b", "assistant_ai_candidate"))
    if result.get("blocked_rows") != blocked or result.get("reviewed_rows") != 32 - blocked:
        raise ValueError("Reviewer row counters disagree with row-level decisions")
    transitions = collections.Counter(
        f"qwen3_{labels['qwen3_14b']}__assistant_{labels['assistant_ai_candidate']}"
        for labels in labels_by_source.values()
    )
    summary = {
        "status": "VERIFIED_BLINDED_PAIRED_AI_SCREEN_DIAGNOSTIC_ONLY",
        "rows_reviewed": len(records),
        "paired_source_questions": len(labels_by_source),
        "candidate_counts": {
            method: {label: counts[(method, label)] for label in ("pass", "flag", "blocked")}
            for method in ("qwen3_14b", "assistant_ai_candidate")
        },
        "counts_by_method_and_language": {
            method: {lang: {label: by_lang[(method, lang)][label] for label in ("pass", "flag", "blocked")}
                     for lang in ("fr", "de", "hi", "th")}
            for method in ("qwen3_14b", "assistant_ai_candidate")
        },
        "paired_transitions": dict(sorted(transitions.items())),
        "input_sha256": input_hash,
        "output_sha256": sha(OUTPUT.read_bytes()),
        "review_script_sha256": script_hash,
        "pulled_kaggle_review_script_sha256": remote_script_hash,
        "pulled_kaggle_review_script_matches": True,
        "models": {"qwen": {"id": QWEN[0], "revision": QWEN[1]}, "nllb": {"id": NLLB[0], "revision": NLLB[1]}},
        "placeholder_masking": "common names/numerals during NLLB only; exact markers restored before Qwen adjudication",
        "answers_or_splits_sent": False,
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
        "note": "Paired AI-only screen on one selected 16-question diagnostic sample; not human-gold quality evidence and not a release decision.",
    }
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
