#!/usr/bin/env python3
"""Summarize the completed blinded AI-only retain-answer review."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
BLINDED = HERE / "retain_pairwise_review_blinded.csv"
ANNOTATIONS = HERE / "retain_pairwise_review_codex_ai_2026-11-08.csv"
PRIVATE_KEY = HERE / "retain_pairwise_review_key_PRIVATE.csv"
OUTPUT = HERE / "retain_pairwise_review_codex_ai_summary.json"


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["review_id"]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate review IDs in {path.name}")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    blind = read_csv(BLINDED)
    annotations = read_csv(ANNOTATIONS)
    key = read_csv(PRIVATE_KEY)
    if len(blind) != 60 or blind.keys() != annotations.keys() or blind.keys() != key.keys():
        raise ValueError("Blinded sheet, annotations, and private key must align on exactly 60 IDs")
    if any(not row["reviewer"].startswith("Codex GPT-6; AI-only") for row in annotations.values()):
        raise ValueError("Unexpected reviewer label")

    by_stratum: dict[str, dict[str, Counter]] = defaultdict(lambda: {
        "preference": Counter(), "factuality": Counter(), "date_flags": Counter()
    })
    for review_id, key_row in key.items():
        row = annotations[review_id]
        stratum = key_row["stratum"]
        a_condition = key_row["response_A_condition"].lower()
        b_condition = key_row["response_B_condition"].lower()
        preference = row["preferred_overall"]
        if preference == "tie":
            by_stratum[stratum]["preference"]["tie"] += 1
        elif preference == "unclear":
            by_stratum[stratum]["preference"]["unclear"] += 1
        else:
            winner_condition = a_condition if preference == "A" else b_condition
            by_stratum[stratum]["preference"][winner_condition] += 1
        by_stratum[stratum]["factuality"][f"{a_condition}:{row['A_factuality']}"] += 1
        by_stratum[stratum]["factuality"][f"{b_condition}:{row['B_factuality']}"] += 1
        if key_row["adapter_introduced_date_conflict"].lower() == "true":
            by_stratum[stratum]["date_flags"]["adapter_introduced"] += 1
        if key_row["date_conflict_already_present_at_baseline"].lower() == "true":
            by_stratum[stratum]["date_flags"]["already_at_baseline"] += 1

    summary = {
        "review_id": "alpha40_blinded_retain_pairwise_ai_review_2026-11-08",
        "reviewer": "Codex GPT-6; AI-only, one reviewer, one pass",
        "rating_basis": "consistency with the supplied TOFU reference answers; no external fact verification",
        "design": "case-control qualitative diagnostic, not a probability sample",
        "n_pairs": len(blind),
        "strata": {
            stratum: {
                "n_pairs": sum(1 for k in key.values() if k["stratum"] == stratum),
                "preferred_condition": dict(counters["preference"]),
                "reference_consistency_labels": dict(counters["factuality"]),
                "date_conflict_flags": dict(counters["date_flags"]),
            }
            for stratum, counters in sorted(by_stratum.items())
        },
        "limitations": [
            "The 40 answer-string-loss cases were deliberately enriched and the 20 controls were selected; counts do not estimate population prevalence or benchmark accuracy.",
            "This is a single AI reviewer, not fluent-human review or independent inter-rater validation.",
            "Labels judge consistency with the supplied synthetic reference answer, not external factual truth.",
            "No confidence intervals or significance tests are appropriate for this selected qualitative sample.",
        ],
        "input_sha256": {
            "blinded_sheet": digest(BLINDED),
            "ai_annotations": digest(ANNOTATIONS),
        },
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"n_pairs": summary["n_pairs"], "strata": summary["strata"], "limitations": summary["limitations"]}, indent=2, ensure_ascii=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
