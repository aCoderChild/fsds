#!/usr/bin/env python3
"""Verify the completed, diagnostic-only Kaggle Qwen14 review output."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "completed_output"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_raw_input(script_path: Path) -> str:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "raw" for target in node.targets)):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise ValueError("Embedded review input is not a string")
            return value
    raise ValueError("No embedded raw review input found")


def main() -> None:
    selection = read_json(HERE / "selection_manifest.json")
    script = HERE / "review.py"
    raw = extract_raw_input(script)
    inputs = json.loads(raw)
    result_path = OUTPUT / "review.json"
    log_path = OUTPUT / "kernel.log"
    result = read_json(result_path)
    log = read_json(log_path)

    if sha(script) != selection["script_sha256"]:
        raise ValueError("Review script hash differs from selected run")
    input_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if input_sha != selection["input_sha256"] or input_sha != result["input_sha256"]:
        raise ValueError("Embedded, selected, and completed-run input hashes differ")
    if (len(inputs) != selection["rows"] or len(result.get("records", [])) != len(inputs)
            or result.get("status") != "DIAGNOSTIC_AI_REVIEW_ONLY"):
        raise ValueError("Review is incomplete or has an unexpected status")

    identities = []
    summary = defaultdict(lambda: defaultdict(Counter))
    for source, reviewed in zip(inputs, result["records"]):
        expected = (source["language"], source["kind"], source["prompt_id"])
        actual = (reviewed.get("language"), reviewed.get("kind"), reviewed.get("prompt_id"))
        if actual != expected:
            raise ValueError(f"Input/output identity mismatch: {expected} != {actual}")
        if reviewed.get("raw_output") is None:
            raise ValueError(f"Missing raw model response for {expected}")
        decision = reviewed.get("decision", {})
        if (type(decision.get("semantic_pass")) is not bool
                or type(decision.get("fluency_pass")) is not bool
                or not isinstance(decision.get("reason"), str)):
            raise ValueError(f"Invalid decision schema for {expected}")
        identities.append(expected)
        cell = summary[source["language"]][source["kind"]]
        cell["reviewed"] += 1
        cell["both_pass"] += int(decision["semantic_pass"] and decision["fluency_pass"])
        cell["semantic_fail"] += int(not decision["semantic_pass"])
        cell["fluency_fail"] += int(not decision["fluency_pass"])

    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate language/kind/prompt identity in selected cohort")
    stdout = "\n".join(str(item.get("data", "")) for item in log if item.get("stream_name") == "stdout")
    progress = [int(x) for x in re.findall(r"Reviewed\s+(\d+)\s*/\s*147", stdout)]
    if progress != list(range(1, 148)):
        raise ValueError("Kernel log does not show complete ordered review progress")

    warnings = [
        str(item.get("data", "")).strip()
        for item in log
        if "Sliding Window Attention" in str(item.get("data", ""))
        or "offloaded to the cpu" in str(item.get("data", ""))
    ]
    both_pass = sum(
        row["decision"]["semantic_pass"] and row["decision"]["fluency_pass"]
        for row in result["records"]
    )
    audit = {
        "status": "PASS_INTEGRITY_DIAGNOSTIC_ONLY",
        "review_status": result["status"],
        "rows": len(inputs),
        "both_semantic_and_fluency_pass": both_pass,
        "pass_rate": both_pass / len(inputs),
        "by_language_and_kind": {
            language: {kind: dict(counts) for kind, counts in kinds.items()}
            for language, kinds in sorted(summary.items())
        },
        "model_id": result.get("model_id"),
        "model_revision": result.get("model_revision"),
        "generator_revision": result.get("generator_revision"),
        "input_sha256": input_sha,
        "review_script_sha256": sha(script),
        "selection_manifest_sha256": sha(HERE / "selection_manifest.json"),
        "review_output_sha256": sha(result_path),
        "kernel_log_sha256": sha(log_path),
        "ordered_progress_entries": len(progress),
        "runtime_warnings": warnings,
        "limitation": "Adaptive unresolved-row cohort; Qwen-family reviewer; diagnostic only, not a translation release or corpus approval.",
    }
    (OUTPUT / "verification.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
