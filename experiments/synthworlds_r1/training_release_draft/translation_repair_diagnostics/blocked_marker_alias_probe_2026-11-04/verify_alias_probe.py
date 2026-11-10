#!/usr/bin/env python3
"""Verify a Kaggle alias probe without treating it as translation approval."""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "build_manifest.json"
MODEL = "facebook/nllb-200-distilled-600M"
REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
MARKER = re.compile(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]")
ALIASES = {"[ENTITY_A]": "Alice", "[ENTITY_B]": "Bob"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-dir", type=Path, required=True)
    parser.add_argument("--pulled-script", type=Path, required=True)
    args = parser.parse_args()
    script = args.pulled_script.read_bytes()
    local_script = (HERE / "kaggle_kernel/probe.py").read_bytes()
    if script != local_script:
        raise ValueError("Kaggle-pulled probe differs from the submitted local source")
    source = script.decode("utf-8")
    b64 = re.search(r'^RAW_INPUT_B64 = "([^"]+)"$', source, re.M)
    input_hash = re.search(r'^INPUT_SHA256 = "([^"]+)"$', source, re.M)
    if not b64 or not input_hash:
        raise ValueError("embedded input fields are missing")
    raw = base64.b64decode(b64.group(1))
    if sha(raw) != input_hash.group(1):
        raise ValueError("embedded input hash mismatch")
    inputs = json.loads(raw)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if len(inputs) != 28 or input_hash.group(1) != manifest["input_sha256"]:
        raise ValueError("probe input differs from the 28-row source manifest")
    output_path = args.remote_dir / "alias_probe.json"
    output = json.loads(output_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if (output.get("status") != "DIAGNOSTIC_COMPLETE_NOT_TRANSLATION_APPROVAL"
            or output.get("model_id") != MODEL or output.get("model_revision") != REVISION
            or output.get("input_sha256") != input_hash.group(1)
            or output.get("translation_quality_reviewed") is not False
            or output.get("answers_or_splits_sent") is not False):
        raise ValueError("remote run provenance or scope is invalid")
    rows = output.get("rows", [])
    expected = {row["review_key"]: row for row in inputs}
    if len(rows) != 28 or len({row.get("review_key") for row in rows}) != 28:
        raise ValueError("remote output does not contain exactly 28 unique rows")
    counts = collections.Counter()
    for row in rows:
        key = row["review_key"]
        source_row = expected.get(key)
        if source_row is None:
            raise ValueError(f"unexpected output key {key}")
        if (row.get("source_sha256") != sha(source_row["source_question_en"].encode())
                or row.get("translation_sha256") != sha(source_row["translated_question"].encode())):
            raise ValueError(f"{key}: output is not bound to the source and candidate")
        if row.get("status") == "PASS_ALIAS_COUNT_ONLY":
            aliases = row.get("alias_counts", {})
            target_markers = collections.Counter(MARKER.findall(source_row["translated_question"]))
            source_markers = collections.Counter(MARKER.findall(source_row["source_question_en"]))
            expected_aliases = {ALIASES[marker]: count for marker, count in target_markers.items()}
            if aliases != expected_aliases:
                raise ValueError(f"{key}: alias counts differ from source placeholders")
            raw_counts = {alias: row["backtranslation_raw"].count(alias) for alias in aliases}
            if raw_counts != aliases:
                raise ValueError(f"{key}: successful record has changed alias counts")
            if collections.Counter(MARKER.findall(row["restored_backtranslation_en"])) != source_markers:
                raise ValueError(f"{key}: restored English marker multiset differs")
        elif row.get("status") == "BLOCKED":
            if not row.get("error") or row.get("restored_backtranslation_en"):
                raise ValueError(f"{key}: blocked record lacks an error or contains restored text")
        else:
            raise ValueError(f"{key}: unknown status {row.get('status')}")
        counts[row["status"]] += 1
    receipt = {
        "status": "VERIFIED_ALIAS_MARKER_SURVIVAL_DIAGNOSTIC_ONLY",
        "release_approved": False, "training_eligible": False,
        "rows": len(rows), "counts": dict(counts),
        "model_id": MODEL, "model_revision": REVISION,
        "input_sha256": input_hash.group(1),
        "submitted_script_sha256": sha(local_script),
        "pulled_script_sha256": sha(script),
        "output_sha256": sha(output_path.read_bytes()),
        "human_translation_reviewed": False,
        "answers_or_splits_sent": False,
        "interpretation": "Alias-count preservation only; no translation semantics or fluency decisions were made.",
    }
    (args.remote_dir / "verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
