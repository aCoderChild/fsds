#!/usr/bin/env python3
"""Verify identity, coverage, schema, hashes, and decisions for the 32-row review."""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = Path(__file__).resolve().parents[3] / "r1_current_state_audit_2026-11-03" / "evidence" / "chatgpt_translation_calibration32_kaggle" / "input.jsonl"
OUTPUT = ROOT / "review.json"
LOG = ROOT / "synthworlds-chatgpt-translation-32-review.log"
EXPECTED_INPUT_SHA = "7fe75e5cb4ab9b3ca9bb96c4cbd713d0a88a39feffc4b7af330d3e18259ccc14"
EXPECTED_MODEL = "Qwen/Qwen2.5-14B-Instruct"
EXPECTED_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MARKER = re.compile(r"ZXQ[A-Z]+XZ")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    input_bytes = INPUT.read_bytes()
    expected = read_jsonl(INPUT)
    output = json.loads(OUTPUT.read_text(encoding="utf-8"))
    records = output.get("records")
    assert hashlib.sha256(input_bytes).hexdigest() == EXPECTED_INPUT_SHA == output.get("input_sha256")
    assert output.get("status") == "DIAGNOSTIC_AI_REVIEW_COMPLETE"
    assert output.get("model_id") == EXPECTED_MODEL
    assert output.get("model_revision") == EXPECTED_REVISION
    assert output.get("schema_errors") == 0
    assert isinstance(records, list) and len(expected) == len(records) == 32

    key = lambda row: (row["prompt_id"], row["language"], row["kind"])
    expected_by_key = {key(row): row for row in expected}
    actual_by_key = {key(row): row for row in records}
    assert len(expected_by_key) == len(actual_by_key) == 32
    assert set(expected_by_key) == set(actual_by_key)
    counts: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    failures = []
    for row_key, source_row in expected_by_key.items():
        actual = actual_by_key[row_key]
        for field in ("source", "translation", "language", "kind"):
            assert actual.get(field) == source_row.get(field), (row_key, field)
        assert "answer" not in actual and "label" not in actual
        assert collections.Counter(MARKER.findall(source_row["source"])) == collections.Counter(MARKER.findall(source_row["translation"]))
        decision = actual.get("decision")
        assert isinstance(decision, dict)
        assert type(decision.get("semantic_pass")) is bool
        assert type(decision.get("fluency_pass")) is bool
        assert isinstance(decision.get("reason"), str)
        cell = counts[(source_row["language"], source_row["kind"])]
        cell["n"] += 1
        cell["semantic_pass"] += int(decision["semantic_pass"])
        cell["fluency_pass"] += int(decision["fluency_pass"])
        cell["both_pass"] += int(decision["semantic_pass"] and decision["fluency_pass"])
        if not (decision["semantic_pass"] and decision["fluency_pass"]):
            failures.append({
                "prompt_id": source_row["prompt_id"],
                "language": source_row["language"],
                "kind": source_row["kind"],
                "semantic_pass": decision["semantic_pass"],
                "fluency_pass": decision["fluency_pass"],
                "reviewer_reason": decision["reason"],
            })

    expected_cells = {(lang, kind) for lang in ("fr", "de", "hi", "th") for kind in ("atomic", "composition")}
    assert set(counts) == expected_cells
    assert all(counts[cell]["n"] == 4 for cell in expected_cells)
    result = {
        "status": "VERIFIED_DIAGNOSTIC_ONLY_NOT_RELEASE",
        "input_sha256": sha(INPUT),
        "review_output_sha256": sha(OUTPUT),
        "kernel_log_sha256": sha(LOG),
        "verifier_sha256": sha(Path(__file__)),
        "reviewer_model": output["model_id"],
        "reviewer_revision": output["model_revision"],
        "device": output.get("gpu"),
        "attention_backend": output.get("attention_backend"),
        "torch_version": output.get("torch"),
        "rows": len(records),
        "unique_rows": len(actual_by_key),
        "schema_errors": output["schema_errors"],
        "answers_included": False,
        "target_markers_preserved": True,
        "per_language_kind": {
            f"{lang}/{kind}": dict(counts[(lang, kind)])
            for lang, kind in sorted(expected_cells)
        },
        "joint_pass_total": sum(cell["both_pass"] for cell in counts.values()),
        "joint_pass_rate": sum(cell["both_pass"] for cell in counts.values()) / len(records),
        "flagged_rows": failures,
        "limitations": [
            "Diagnostic calibration sample only; it does not authorize training.",
            "Reviewer judgments are AI-only and are not human fluency certification.",
            "The reviewer's unsupported or erroneous explanations remain possible; reviewer accuracy was not independently gold-labeled.",
            "The ChatGPT UI translator backend and exact prompt were not recorded, limiting exact reproduction.",
        ],
    }
    (ROOT / "calibration_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "rows": result["rows"],
        "joint_pass_total": result["joint_pass_total"],
        "joint_pass_rate": result["joint_pass_rate"],
        "per_language_kind": result["per_language_kind"],
        "validation_report": str(ROOT / "calibration_validation.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
