#!/usr/bin/env python3
"""Recompute local descriptive retain-generation diagnostics for alpha=40."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "kaggle_output" / "tofu_npo_alpha40_followup_artifacts.zip"
OUTPUT = HERE / "alpha40_retain_generation_audit.json"
MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}\s+\d{{4}}|"
    rf"{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})\b",
    re.IGNORECASE,
)


def member(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one archive member ending in {suffix!r}; got {matches}")
    return matches[0]


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def contains_source_answer(row: dict) -> bool:
    answer = normalized(row["source_answer"])
    return bool(answer) and answer in normalized(row["generation"])


def dates(text: str) -> set[str]:
    return {" ".join(value.lower().replace(",", "").split()) for value in DATE_RE.findall(text)}


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise ValueError(f"Archive CRC check failed at {bad_member}")
        reference = json.loads(zf.read(member(zf, "full_reference_profile_eval.json")))
        baseline_rows = reference["per_example"]["retain90"]
        post_rows = [
            json.loads(line)
            for line in zf.read(member(zf, "dev_alpha_40_examples.jsonl")).decode("utf-8").splitlines()
            if line.strip()
        ]

    baseline = {row["source_row_index"]: row for row in baseline_rows}
    retain_rows = [row for row in post_rows if row["split"] == "retain90"]
    if len(retain_rows) != 400 or len(baseline) != len(baseline_rows):
        raise ValueError("Unexpected retain cohort size or duplicate baseline source-row indices")
    if len({row["source_row_index"] for row in retain_rows}) != len(retain_rows):
        raise ValueError("Duplicate alpha=40 retain source-row indices")

    pairs = []
    for after in retain_rows:
        before = baseline.get(after["source_row_index"])
        if before is None:
            raise ValueError(f"Missing baseline row {after['source_row_index']}")
        if before["profile_block"] != after["profile_block"]:
            raise ValueError(f"Profile-block mismatch at source row {after['source_row_index']}")
        pairs.append((before, after))

    transitions: Counter[tuple[bool, bool]] = Counter()
    rouge_deltas = []
    block_deltas: dict[int, list[float]] = defaultdict(list)
    date_transition_counts: Counter[tuple[bool, bool]] = Counter()
    date_conflicts = []
    for before, after in pairs:
        source_match_before = contains_source_answer(before)
        source_match_after = contains_source_answer(after)
        transitions[(source_match_before, source_match_after)] += 1
        delta = after["rougeL_recall"] - before["rougeL_recall"]
        rouge_deltas.append(delta)
        block_deltas[after["profile_block"]].append(delta)

        expected_dates = dates(before["source_answer"])
        baseline_dates = dates(before["generation"])
        generated_dates = dates(after["generation"])
        baseline_conflict = bool(expected_dates and baseline_dates and not expected_dates.intersection(baseline_dates))
        alpha40_conflict = bool(expected_dates and generated_dates and not expected_dates.intersection(generated_dates))
        date_transition_counts[(baseline_conflict, alpha40_conflict)] += 1
        if alpha40_conflict:
            date_conflicts.append(
                {
                    "source_row_index": after["source_row_index"],
                    "profile_block": after["profile_block"],
                    "source_answer_dates": sorted(expected_dates),
                    "baseline_generation_dates": sorted(baseline_dates),
                    "generated_dates": sorted(generated_dates),
                    "already_conflicted_at_baseline": baseline_conflict,
                }
            )

    lost_exact_answer = [
        (before, after)
        for before, after in pairs
        if contains_source_answer(before) and not contains_source_answer(after)
    ]
    remaining = [
        (before, after)
        for before, after in pairs
        if not (contains_source_answer(before) and not contains_source_answer(after))
    ]

    def mean_generation_length(key: str, unit: str) -> float:
        lengths = []
        for before, after in pairs:
            text = before["generation"] if key == "baseline" else after["generation"]
            lengths.append(len(text) if unit == "characters" else len(text.split()))
        return statistics.mean(lengths)

    block_means = [statistics.mean(values) for values in block_deltas.values()]
    output = {
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "rows": len(pairs),
        "profile_blocks": len(block_deltas),
        "mean_generation_length": {
            "whitespace_words": {
                "baseline": mean_generation_length("baseline", "words"),
                "alpha40": mean_generation_length("alpha40", "words"),
            },
            "characters": {
                "baseline": mean_generation_length("baseline", "characters"),
                "alpha40": mean_generation_length("alpha40", "characters"),
            },
        },
        "normalized_source_answer_substring_transition_counts": {
            "both_present": transitions[(True, True)],
            "baseline_only": transitions[(True, False)],
            "alpha40_only": transitions[(False, True)],
            "neither": transitions[(False, False)],
            "baseline_present_total": transitions[(True, True)] + transitions[(True, False)],
            "alpha40_present_total": transitions[(True, True)] + transitions[(False, True)],
            "warning": "Strict lexical proxy only; not a semantic or factual-accuracy score.",
        },
        "rougeL_change_for_exact_substring_lost_rows_pp": 100
        * statistics.mean(after["rougeL_recall"] - before["rougeL_recall"] for before, after in lost_exact_answer),
        "rougeL_change_for_all_other_rows_pp": 100
        * statistics.mean(after["rougeL_recall"] - before["rougeL_recall"] for before, after in remaining),
        "rougeL_row_deltas": {
            "lower": sum(delta < 0 for delta in rouge_deltas),
            "equal": sum(delta == 0 for delta in rouge_deltas),
            "higher": sum(delta > 0 for delta in rouge_deltas),
        },
        "rougeL_profile_block_deltas_pp": {
            "lower_blocks": sum(delta < 0 for delta in block_means),
            "equal_blocks": sum(delta == 0 for delta in block_means),
            "higher_blocks": sum(delta > 0 for delta in block_means),
            "min": 100 * min(block_means),
            "median": 100 * statistics.median(block_means),
            "max": 100 * max(block_means),
        },
        "explicit_date_conflict_screen": {
            "alpha40_conflicts_vs_reference": date_transition_counts[(False, True)] + date_transition_counts[(True, True)],
            "baseline_conflicts_vs_reference": date_transition_counts[(True, False)] + date_transition_counts[(True, True)],
            "new_conflicts_after_adapter": date_transition_counts[(False, True)],
            "preexisting_conflicts_persisting_after_adapter": date_transition_counts[(True, True)],
            "conflicts_resolved_after_adapter": date_transition_counts[(True, False)],
            "method": "Flag rows where source answer and a generation both contain explicit dates, but no full normalized date string matches.",
            "cases": date_conflicts,
            "warning": "A deterministic screen, not a complete semantic evaluation.",
        },
        "generation_text_identical_rows": sum(before["generation"] == after["generation"] for before, after in pairs),
        "interpretation_limit": "Descriptive matched outputs from one development run. Do not treat row-level counts as independent inferential observations; the registered inferential unit is the profile block.",
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
