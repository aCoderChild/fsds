#!/usr/bin/env python3
"""Check exact normalized source-answer mentions in both matched samples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from verify_source_bound_calibration import PROJECT, TRAINING

sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
from assemble_training_release import norm

HERE = Path(__file__).resolve().parent


def main() -> None:
    answers = {}
    for name in ("atomic_reuse.jsonl", "atomic_translation_requests.jsonl", "composition_reuse.jsonl"):
        for line in (TRAINING / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            value = row.get("answer_en") or row.get("answer")
            if value:
                key = row["sft_id"]
                if key in answers and answers[key] != value:
                    raise ValueError(f"Inconsistent source answer: {key}")
                answers[key] = value
    paths = {
        "translategemma": (TRAINING / "translation_repair_diagnostics/source_bound_finalflagged_calibration32/repair_candidates.jsonl", "new_translation"),
        "nllb": (HERE / "evidence/nllb13_matched_calibration32/candidates.jsonl", "text_translation"),
    }
    result = {}
    for label, (path, field) in paths.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing, mentions = [], []
        for row in rows:
            answer = answers.get(row["prompt_id"])
            if not answer:
                missing.append(row["prompt_id"])
            elif norm(answer) in norm(row.get(field, "")):
                mentions.append({"language": row["language"], "prompt_id": row["prompt_id"]})
        if len(rows) != 32 or missing:
            raise ValueError(f"{label}: incomplete sample or missing source answers")
        result[label] = {"rows": len(rows), "exact_normalized_answer_mentions": mentions}
    print(json.dumps({"status": "PASS_EXACT_ANSWER_AUDIT_ONLY", "samples": result,
                      "limitation": "Exact normalized matching does not detect paraphrases, aliases, or indirect disclosure."}, indent=2))


if __name__ == "__main__":
    main()
