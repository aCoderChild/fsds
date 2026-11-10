#!/usr/bin/env python3
"""Verify the quarantined assistant probe against its selected source questions."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_source_bound_calibration import verify, TRAINING  # noqa: E402
from verify_nllb_matched32 import status  # noqa: E402
sys.path.insert(0, str(TRAINING / "scripts"))
from review_training_translations_qwen import deterministic_checks  # noqa: E402
from assemble_training_release import norm  # noqa: E402


def main():
    selected, _, _ = verify()
    source = {(x["language"], x["row"]["kind"], x["row"]["prompt_id"]): x["row"] for x in selected}
    nllb = json.loads((HERE / "evidence/nllb13_matched_roundtrip/checkpoint.json").read_text())["records"]
    tg = json.loads((HERE / "evidence/source_bound_calibration_roundtrip_retry/checkpoint.json").read_text())["records"]
    alternative = json.loads((HERE / "evidence/source_bound_calibration_masking_comparison/checkpoint.json").read_text())["decisions"]
    if len(nllb) != len(tg):
        raise ValueError("Matched review lengths differ")
    expected = set()
    for n, t in zip(nllb, tg):
        key = (t["language"], t["kind"], t["prompt_id"])
        if (n["language"], n["kind"], n["prompt_id"]) != key:
            raise ValueError("Matched review order differs")
        if status(n, n.get("qwen14", {})) == "FLAGGED" and status(t, alternative["|".join(key)]) == "FLAGGED":
            expected.add(key)
    path = HERE / "assistant_candidate_probe.json"
    data = json.loads(path.read_text())
    if data["status"] != "UNREVIEWED_DIAGNOSTIC_ONLY" or data["human_reviewed"] is not False:
        raise ValueError("Probe must remain explicitly unreviewed")
    answers = {}
    for name in ("atomic_reuse.jsonl", "atomic_translation_requests.jsonl", "composition_reuse.jsonl"):
        for line in (TRAINING / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            value = row.get("answer_en") or row.get("answer")
            if value:
                if row["sft_id"] in answers and answers[row["sft_id"]] != value:
                    raise ValueError("Inconsistent source answer")
                answers[row["sft_id"]] = value
    seen = set()
    fields = ("nonempty", "target_script_present", "protected_names_preserved", "protected_numbers_preserved", "no_unrestored_marker")
    for candidate in data["candidates"]:
        key = (candidate["language"], candidate["kind"], candidate["prompt_id"])
        if key in seen or key not in expected or key not in source:
            raise ValueError(f"Unexpected or duplicate candidate: {key}")
        seen.add(key)
        row = {**source[key], "text_translation": candidate["translation"]}
        checks = deterministic_checks(key[0], row)
        if not all(checks.get(field) is True for field in fields):
            raise ValueError(f"Protected-item check failed: {key}: {checks}")
        answer = answers.get(key[2])
        if not answer or norm(answer) in norm(candidate["translation"]):
            raise ValueError(f"Missing answer or exact normalized answer mention: {key}")
    if seen != expected or len(expected) != 15:
        raise ValueError("Probe does not cover exactly the 15 matched double flags")
    print(json.dumps({"status": "DETERMINISTIC_CHECKS_ONLY_NOT_SEMANTIC_REVIEW", "candidates": len(seen),
                      "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "exact_normalized_answer_mentions": 0, "human_reviewed": False,
                      "limitations": ["Exact matching does not detect aliases or indirect disclosure.",
                                      "Run verify_nllb_matched32.py to validate the historical review provenance."]}, indent=2))


if __name__ == "__main__":
    main()
