#!/usr/bin/env python3
"""Prepare a small blinded semantic review set from the pinned TOFU release.

The review set is enriched for cases where the alpha=40 output lost a strict
source-answer string match. It is for diagnostic human review, not prevalence
estimation or confirmatory inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
TOFU_ROOT = HERE.parents[1]
PROTOCOL_PATH = TOFU_ROOT / "npo_lora_alpha40_followup_protocol.json"
ARCHIVE = HERE / "kaggle_output" / "tofu_npo_alpha40_followup_artifacts.zip"
BLIND_CSV = HERE / "retain_pairwise_review_blinded.csv"
KEY_CSV = HERE / "retain_pairwise_review_key_PRIVATE.csv"
README = HERE / "retain_pairwise_review_instructions.md"
EXPECTED_STRATUM_CASE_N = 40
EXPECTED_CONTROL_N = 20
SEED = 20261108
DATE_CONFLICT_SOURCE_ROWS = {1220, 1344, 1461}
ADAPTER_INTRODUCED_DATE_CONFLICTS = {1220}
PERSISTING_CHANGED_DATE_CONFLICTS = {1344}


def one_member(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive member ending {suffix!r}; found {matches}")
    return matches[0]


def normalized(text: str) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def contains_source_answer(row: dict) -> bool:
    answer = normalized(row["source_answer"])
    return bool(answer) and answer in normalized(row["generation"])


def fetch_pinned_retain(protocol: dict) -> list[dict]:
    repo = protocol["data"]["repo_id"]
    revision = protocol["data"]["revision"]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/retain90.json"
    request = urllib.request.Request(url, headers={"User-Agent": "fsds-alpha40-review-preparation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    expected = protocol["data"]["sha256"]["retain90.json"]
    if digest != expected:
        raise ValueError(f"Pinned retain90.json SHA-256 mismatch: expected {expected}, got {digest}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != protocol["data"]["expected_rows"]["retain90.json"]:
        raise ValueError(f"Expected 3600 pinned retain rows, got {len(rows)}")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw_rows = fetch_pinned_retain(protocol)
    rng = random.Random(SEED)

    with zipfile.ZipFile(ARCHIVE) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Artifact archive CRC failure at {bad}")
        ref_eval = json.loads(zf.read(one_member(zf, "full_reference_profile_eval.json")))
        base_rows = ref_eval["per_example"]["retain90"]
        after_rows = [
            json.loads(line)
            for line in zf.read(one_member(zf, "dev_alpha_40_examples.jsonl")).decode("utf-8").splitlines()
            if line.strip()
        ]

    baseline = {int(row["source_row_index"]): row for row in base_rows}
    dataset_by_index = {index: row for index, row in enumerate(raw_rows)}
    after_retain = [row for row in after_rows if row["split"] == "retain90"]
    pairs: dict[int, dict] = {}
    for after in after_retain:
        index = int(after["source_row_index"])
        before = baseline.get(index)
        source = dataset_by_index.get(index)
        if before is None or source is None:
            raise ValueError(f"Could not align baseline, alpha=40 output, and source data for row {index}")
        question_hash = hashlib.sha256(source["question"].encode("utf-8")).hexdigest()
        if question_hash != before["question_sha256"] or question_hash != after["question_sha256"]:
            raise ValueError(f"Question hash mismatch at source row {index}")
        if source["answer"] != before["source_answer"] or source["answer"] != after["source_answer"]:
            raise ValueError(f"Reference answer mismatch at source row {index}")
        if before["profile_block"] != after["profile_block"]:
            raise ValueError(f"Profile block mismatch at source row {index}")
        pairs[index] = {"before": before, "after": after, "source": source}

    if len(pairs) != 400:
        raise ValueError(f"Expected 400 matched retain rows, got {len(pairs)}")

    cases_by_block: dict[int, list[int]] = defaultdict(list)
    controls_by_block: dict[int, list[int]] = defaultdict(list)
    for index, pair in pairs.items():
        before, after = pair["before"], pair["after"]
        base_has = contains_source_answer(before)
        after_has = contains_source_answer(after)
        block = int(after["profile_block"])
        if base_has and not after_has:
            cases_by_block[block].append(index)
        elif before["generation"] != after["generation"]:
            controls_by_block[block].append(index)

    for block in range(60, 80):
        if not cases_by_block[block] or not controls_by_block[block]:
            raise ValueError(f"Missing a case or changed-output control in profile block {block}")
    if not ADAPTER_INTRODUCED_DATE_CONFLICTS.issubset({i for rows in cases_by_block.values() for i in rows}):
        raise ValueError("Expected adapter-introduced date-conflict row is not in the baseline-only exact-match stratum")

    # Cases: cover every profile block, include the three explicit date conflicts,
    # then sample further cases to reach 40. Controls: one changed-output row per block.
    selected_cases: set[int] = set(ADAPTER_INTRODUCED_DATE_CONFLICTS)
    for block in range(60, 80):
        if not any(int(pairs[index]["after"]["profile_block"]) == block for index in selected_cases):
            selected_cases.add(rng.choice(cases_by_block[block]))
    for block in range(60, 80):
        remaining = [index for index in cases_by_block[block] if index not in selected_cases]
        if remaining:
            selected_cases.add(rng.choice(remaining))
    while len(selected_cases) < EXPECTED_STRATUM_CASE_N:
        remaining = [index for rows in cases_by_block.values() for index in rows if index not in selected_cases]
        selected_cases.add(rng.choice(remaining))
    if len(selected_cases) != EXPECTED_STRATUM_CASE_N:
        raise ValueError(f"Expected {EXPECTED_STRATUM_CASE_N} cases, got {len(selected_cases)}")

    selected_controls = {rng.choice(controls_by_block[block]) for block in range(60, 80)}
    for index in PERSISTING_CHANGED_DATE_CONFLICTS:
        block = int(pairs[index]["after"]["profile_block"])
        if index not in controls_by_block[block]:
            raise ValueError(f"Expected changed-output date-conflict control {index} is not eligible")
        current = next(candidate for candidate in selected_controls if pairs[candidate]["after"]["profile_block"] == block)
        selected_controls.remove(current)
        selected_controls.add(index)
    if len(selected_controls) != EXPECTED_CONTROL_N or selected_controls & selected_cases:
        raise ValueError("Control selection is incomplete or overlaps the case stratum")

    sample = [(index, "answer-string-lost") for index in selected_cases]
    sample += [(index, "changed-output-control") for index in selected_controls]
    rng.shuffle(sample)

    blind_rows = []
    key_rows = []
    for number, (index, stratum) in enumerate(sample, start=1):
        pair = pairs[index]
        before, after, source = pair["before"], pair["after"], pair["source"]
        baseline_as_a = bool(rng.getrandbits(1))
        response_a = before["generation"] if baseline_as_a else after["generation"]
        response_b = after["generation"] if baseline_as_a else before["generation"]
        review_id = f"AR{number:03d}"
        blind_rows.append(
            {
                "review_id": review_id,
                "question": source["question"],
                "reference_answer": source["answer"],
                "response_A": response_a,
                "response_B": response_b,
                "A_factuality": "",
                "B_factuality": "",
                "A_completeness": "",
                "B_completeness": "",
                "preferred_overall": "",
                "confidence_1_to_3": "",
                "reviewer_notes": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "source_row_index": index,
                "profile_block": int(after["profile_block"]),
                "stratum": stratum,
                "response_A_condition": "baseline" if baseline_as_a else "alpha40",
                "response_B_condition": "alpha40" if baseline_as_a else "baseline",
                "baseline_contains_normalized_source_answer": contains_source_answer(before),
                "alpha40_contains_normalized_source_answer": contains_source_answer(after),
                "alpha40_explicit_date_conflict_vs_reference": index in DATE_CONFLICT_SOURCE_ROWS,
                "adapter_introduced_date_conflict": index in ADAPTER_INTRODUCED_DATE_CONFLICTS,
                "date_conflict_already_present_at_baseline": index in PERSISTING_CHANGED_DATE_CONFLICTS,
            }
        )

    write_csv(BLIND_CSV, list(blind_rows[0]), blind_rows)
    write_csv(KEY_CSV, list(key_rows[0]), key_rows)
    README.write_text(
        "# Blinded answer review\n\n"
        "For each question, compare responses A and B with the supplied reference answer. Rate each response independently; do not guess which system produced it. Assess semantic correctness rather than exact wording.\n\n"
        "For factuality, use `correct`, `partly_correct`, `incorrect`, or `unclear`. For completeness, use `complete`, `partial`, or `insufficient`. Then choose A, B, `tie`, or `unclear` for overall quality, assign confidence 1–3, and note unsupported details or contradictions. If the reference answer seems ambiguous or insufficient, use `unclear` and explain.\n\n"
        "The packet is intended for qualitative error analysis and is not designed to estimate benchmark-wide accuracy. The randomized response key is stored separately; do not request or open it until annotations are finished.\n",
        encoding="utf-8",
    )
    print(f"Pinned retain dataset SHA-256: {protocol['data']['sha256']['retain90.json']}")
    print(f"Prepared {len(blind_rows)} blind pairs ({len(selected_cases)} cases, {len(selected_controls)} controls)")
    print(f"Review sheet: {BLIND_CSV}")
    print(f"Private condition key (do not share with reviewer): {KEY_CSV}")
    print(f"Instructions: {README}")


if __name__ == "__main__":
    main()
