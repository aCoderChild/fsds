#!/usr/bin/env python3
"""Independently verify the 187-row diagnostic overlay and its source bindings."""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "final_recheck_overlay_2026-11-05"
INPUT = HERE / "kaggle_review/review_input.json"
PARENT = HERE / "kaggle_review/remote_output/repair_review.json"
FIRST = HERE / "marker_safe_retry_2026-11-05/remote_output/marker_safe_retry.json"
QWEN = HERE / "qwen_only_retry_2026-11-05/remote_output/qwen_only_retry.json"
KEY = HERE / "key_constrained_retry_2026-11-05/remote_output/key_constrained_retry.json"
KEY_INPUT = HERE / "key_constrained_retry_2026-11-05/review_input.json"
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(rows) -> str:
    return sha(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode())


def expected_label(record: dict) -> str:
    decision = record.get("decision") or {}
    if record.get("backtranslation_error") or "error" in decision:
        return "BLOCKED"
    if type(decision.get("semantic_pass")) is not bool or type(decision.get("fluency_pass")) is not bool:
        raise ValueError(f"{record['review_key']}: malformed decision booleans")
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        raise ValueError(f"{record['review_key']}: empty decision rationale")
    if decision.get("key") != record["review_key"]:
        raise ValueError(f"{record['review_key']}: decision key mismatch")
    return "PASS" if decision["semantic_pass"] and decision["fluency_pass"] else "FLAG"


def main() -> None:
    source, parent, first, qwen, key_bundle = map(load, (INPUT, PARENT, FIRST, QWEN, KEY))
    overlay = load(OUT / "final_recheck_overlay.json")
    receipt = load(OUT / "build_receipt.json")
    if len(source) != 187 or overlay.get("rows") != 187 or len(overlay.get("records", [])) != 187:
        raise ValueError("Expected exactly 187 source and overlay rows")
    if overlay.get("input_sha256") != canonical_sha(source) or overlay.get("input_sha256") != parent.get("input_sha256"):
        raise ValueError("Frozen input hash mismatch")
    if overlay.get("status") != "COMPLETE_AI_ONLY_THIRD_CANDIDATE_RECHECK_DIAGNOSTIC_ONLY":
        raise ValueError("Overlay status is not diagnostic-only")
    expected_file_hashes = {
        "parent_output_sha256": sha(PARENT.read_bytes()),
        "initial_retry_output_sha256": sha(FIRST.read_bytes()),
        "qwen_only_retry_output_sha256": sha(QWEN.read_bytes()),
        "key_constrained_retry_output_sha256": sha(KEY.read_bytes()),
    }
    for name, value in expected_file_hashes.items():
        if overlay.get(name) != value or receipt.get(name) != value:
            raise ValueError(f"{name} mismatch")

    src_by = {r["review_key"]: r for r in source}
    par_by = {r["review_key"]: r for r in parent["records"]}
    first_by = {r["review_key"]: r for r in first["records"]}
    qwen_by = {r["review_key"]: r for r in qwen["records"]}
    overlay_by = {r["review_key"]: r for r in overlay["records"]}
    key_rec = key_bundle["records"][0]
    if any(len(m) != n for m, n in ((src_by, 187), (par_by, 187), (overlay_by, 187), (first_by, 18), (qwen_by, 15))):
        raise ValueError("Duplicate or missing row key")
    if set(src_by) != set(par_by) or set(src_by) != set(overlay_by):
        raise ValueError("Overlay row keys differ from frozen source")
    if key_rec.get("review_key") != "R0349" or key_rec.get("decision", {}).get("key") != "R0349":
        raise ValueError("The constrained retry is not a valid R0349 decision")
    if key_bundle.get("parent_output_sha256") != expected_file_hashes["qwen_only_retry_output_sha256"]:
        raise ValueError("Constrained retry is not bound to the Qwen-only output")
    if key_bundle.get("input_sha256") != canonical_sha(load(KEY_INPUT)):
        raise ValueError("Constrained retry input hash mismatch")
    if qwen.get("alias_retry_output_sha256") != expected_file_hashes["initial_retry_output_sha256"]:
        raise ValueError("Qwen-only retry is not bound to the saved alias-safe output")

    counts = collections.Counter()
    by_language = collections.defaultdict(collections.Counter)
    by_kind = collections.defaultdict(collections.Counter)
    repaired = 0
    for rid, original in src_by.items():
        old = par_by[rid]
        row = overlay_by[rid]
        source_hash = sha(original["source_question_en"].encode())
        target_hash = sha(original["translated_question"].encode())
        if row.get("source_sha256") != source_hash or row.get("translation_sha256") != target_hash:
            raise ValueError(f"{rid}: source or candidate hash mismatch")
        if row.get("language") != original["language"] or row.get("kind") != original["kind"]:
            raise ValueError(f"{rid}: metadata mismatch")
        failure = old.get("backtranslation_error") is not None or "error" in (old.get("decision") or {})
        if not failure:
            if row.get("decision") != old.get("decision") or row.get("backtranslation_sha256") != old.get("backtranslation_sha256"):
                raise ValueError(f"{rid}: an original valid decision was changed")
        else:
            repaired += 1
            if rid not in first_by:
                raise ValueError(f"{rid}: missing first retry row")
            retry = first_by[rid]
            if retry.get("source_sha256") != source_hash or retry.get("translation_sha256") != target_hash:
                raise ValueError(f"{rid}: retry source/candidate hash mismatch")
            if old.get("backtranslation_error") is not None:
                # The first retry's Qwen call raised a local NameError after
                # producing the BT. Verify the saved BT itself below; the
                # separate Qwen-only receipt binds the completed judgment.
                if not retry.get("backtranslation_en"):
                    raise ValueError(f"{rid}: alias-safe back-translation missing")
                if collections.Counter(MARKER.findall(retry["backtranslation_en"])) != collections.Counter(MARKER.findall(original["source_question_en"])):
                    raise ValueError(f"{rid}: opaque marker multiset was not restored")
                bt_hash = sha(retry["backtranslation_en"].encode())
                if retry.get("backtranslation_sha256") != bt_hash or row.get("backtranslation_sha256") != bt_hash:
                    raise ValueError(f"{rid}: restored back-translation hash mismatch")
                if rid not in qwen_by or qwen_by[rid].get("backtranslation_sha256") != bt_hash:
                    raise ValueError(f"{rid}: Qwen did not review the exact restored back-translation")
                decision_record = key_rec if rid == "R0349" else qwen_by[rid]
                if row.get("decision") != decision_record.get("decision"):
                    raise ValueError(f"{rid}: overlay decision differs from its bound Qwen output")
                if rid == "R0349" and row.get("repair_method") != "alias_safe_backtranslation_qwen_key_constrained_retry":
                    raise ValueError("R0349 did not record its constrained-schema repair")
            else:
                if retry.get("decision", {}).get("key") != rid or retry.get("decision", {}).get("error"):
                    raise ValueError(f"{rid}: malformed-output retry has no valid decision")
                if retry.get("backtranslation_sha256") != old.get("backtranslation_sha256"):
                    raise ValueError(f"{rid}: format retry changed the existing back-translation")
                if row.get("decision") != retry.get("decision"):
                    raise ValueError(f"{rid}: overlay decision differs from its source-bound retry")
        label = expected_label(row)
        if row.get("diagnostic_label") != label:
            raise ValueError(f"{rid}: incorrect diagnostic label")
        counts[label] += 1
        by_language[original["language"]][label] += 1
        by_kind[original["kind"]][label] += 1

    if repaired != 18 or dict(counts) != overlay.get("counts"):
        raise ValueError("Repair count or overall counts mismatch")
    if {k: dict(v) for k, v in sorted(by_language.items())} != overlay.get("counts_by_language"):
        raise ValueError("Per-language counts mismatch")
    if {k: dict(v) for k, v in sorted(by_kind.items())} != overlay.get("counts_by_kind"):
        raise ValueError("Per-kind counts mismatch")
    if counts != {"PASS": 36, "FLAG": 151}:
        raise ValueError(f"Unexpected label distribution: {dict(counts)}")
    if overlay.get("human_translation_reviewed") is not False or overlay.get("release_approved") is not False or overlay.get("training_eligible") is not False:
        raise ValueError("Overlay must remain AI-only and non-releasable")
    if sha((OUT / "final_recheck_overlay.json").read_bytes()) != receipt.get("overlay_sha256"):
        raise ValueError("Overlay receipt hash mismatch")
    if (receipt.get("status") != "BUILT_HASH_BOUND_RECHECK_OVERLAY_NOT_RELEASE_APPROVAL"
            or receipt.get("rows") != 187 or receipt.get("repaired_rows") != 18
            or receipt.get("counts") != overlay.get("counts")
            or receipt.get("input_sha256") != overlay.get("input_sha256")
            or receipt.get("release_approved") or receipt.get("training_eligible")):
        raise ValueError("Build receipt status mismatch")
    print(json.dumps({
        "status": "PASS_HASH_BOUND_DIAGNOSTIC_OVERLAY_ONLY",
        "rows": 187,
        "repaired_rows": repaired,
        "counts": dict(counts),
        "counts_by_language": {k: dict(v) for k, v in sorted(by_language.items())},
        "counts_by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        "overlay_sha256": sha((OUT / "final_recheck_overlay.json").read_bytes()),
        "human_translation_reviewed": False,
        "release_approved": False,
        "training_eligible": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
