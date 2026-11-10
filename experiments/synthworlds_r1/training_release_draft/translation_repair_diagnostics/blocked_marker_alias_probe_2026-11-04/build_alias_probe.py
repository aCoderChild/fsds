#!/usr/bin/env python3
"""Build a small Kaggle NLLB marker-survival probe for previously blocked rows."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = HERE / "kaggle_kernel"
WORKBENCH = HERE.parent / "chatgpt_fullflagged_queue_2026-11-04"
MERGED = Path("tmp/r1_review_merge_verification_2026-11-04/merged_full799_diagnostic_reviews.jsonl")
FULL_INPUT = WORKBENCH / "independent_review_inputs_2026-11-04/full799.jsonl"
MODEL = "facebook/nllb-200-distilled-600M"
REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

TEMPLATE = r'''#!/usr/bin/env python3
"""Bounded NLLB marker-alias diagnostic. Never approves translation quality."""
import base64, collections, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
NLLB_CODES = {"fr": "fra_Latn", "de": "deu_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
ALIASES = {"[ENTITY_A]": "Alice", "[ENTITY_B]": "Bob"}
RAW_INPUT_B64 = "__INPUT_B64__"
INPUT_SHA256 = "__INPUT_SHA256__"
OUT = Path("/kaggle/working/alias_probe.json")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def mask(text):
    expected = collections.Counter()
    originals = {}
    def sub(match):
        marker = match.group(0)
        alias = ALIASES.get(marker)
        if alias is None:
            raise ValueError(f"No approved natural alias for {marker}")
        if alias.lower() in text.lower().replace(marker.lower(), ""):
            raise ValueError(f"Alias collision for {marker}")
        expected[alias] += 1
        originals[alias] = marker
        return alias
    masked = re.sub(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", sub, text)
    if not expected:
        raise ValueError("No protected placeholder in candidate")
    return masked, originals, expected

def restore(text, originals, expected):
    found = collections.Counter({alias: text.count(alias) for alias in originals})
    if found != expected:
        raise ValueError(f"alias count changed: expected={dict(expected)}, found={dict(found)}")
    for alias, marker in originals.items():
        text = text.replace(alias, marker)
    if any(alias in text for alias in originals):
        raise ValueError("an alias was not restored")
    return text.strip()

def main():
    raw_input = base64.b64decode(RAW_INPUT_B64)
    if sha(raw_input) != INPUT_SHA256:
        raise ValueError("embedded-input hash mismatch")
    rows = json.loads(raw_input)
    if len(rows) != 28 or len({row["review_key"] for row in rows}) != 28:
        raise ValueError("expected 28 unique, source-bound blocked rows")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing to run a different-device probe")
    runtime = {"status": "RUNNING_MARKER_SURVIVAL_ONLY", "rows": [],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "input_sha256": INPUT_SHA256, "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "translation_quality_reviewed": False, "answers_or_splits_sent": False}
    tok = NllbTokenizerFast.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float16, device_map="auto")
    model.eval()
    for i, row in enumerate(rows, 1):
        result = {"review_key": row["review_key"], "language": row["language"],
            "kind": row["kind"], "source_sha256": sha(row["source_question_en"].encode()),
            "translation_sha256": sha(row["translated_question"].encode()),
            "alias_counts": {}, "backtranslation_raw": "", "restored_backtranslation_en": "",
            "status": "BLOCKED"}
        try:
            src_markers = collections.Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", row["source_question_en"]))
            tgt_markers = collections.Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", row["translated_question"]))
            if src_markers != tgt_markers:
                raise ValueError(f"source/target marker mismatch: {dict(src_markers)} != {dict(tgt_markers)}")
            protected, originals, expected = mask(row["translated_question"])
            result["alias_counts"] = dict(expected)
            tok.src_lang = NLLB_CODES[row["language"]]
            encoded = tok(protected, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.inference_mode():
                output = model.generate(**encoded,
                    forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                    max_new_tokens=128, do_sample=False)
            translated = tok.batch_decode(output, skip_special_tokens=True)[0]
            result["backtranslation_raw"] = translated
            restored = restore(translated, originals, expected)
            if collections.Counter(re.findall(r"\[(?:ENTITY|NUMBER)_[A-Z]+\]", restored)) != src_markers:
                raise ValueError("restored marker multiset does not match English source")
            result["restored_backtranslation_en"] = restored
            result["status"] = "PASS_ALIAS_COUNT_ONLY"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        runtime["rows"].append(result)
        print(f"Processed {i}/{len(rows)}: {result['status']}", flush=True)
    runtime["counts"] = dict(collections.Counter(r["status"] for r in runtime["rows"]))
    runtime["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["status"] = "DIAGNOSTIC_COMPLETE_NOT_TRANSLATION_APPROVAL"
    OUT.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": runtime["status"], "counts": runtime["counts"],
        "output": str(OUT)}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    merged = [json.loads(line) for line in MERGED.read_text(encoding="utf-8").splitlines() if line]
    full = [json.loads(line) for line in FULL_INPUT.read_text(encoding="utf-8").splitlines() if line]
    blocked_keys = {row["review_key"] for row in merged
                    if row.get("review_status") == "AI_ONLY_REPAIR_REVIEW_BLOCKED"
                    and (row.get("decision") or {}).get("error", "").startswith("backtranslation_failed")}
    by_key = {row["review_key"]: row for row in full}
    if len(blocked_keys) != 28 or not blocked_keys <= set(by_key):
        raise ValueError("Expected exactly 28 verified NLLB-placeholder-blocked source rows")
    rows = []
    for key in sorted(blocked_keys):
        source = by_key[key]
        rows.append({field: source[field] for field in
                     ("review_key", "language", "kind", "source_question_en", "translated_question")})
        if "[ENTITY_A]" not in source["translated_question"]:
            raise ValueError(f"{key}: expected ENTITY_A placeholder absent")
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    KERNEL.mkdir(parents=True, exist_ok=True)
    code = TEMPLATE.replace("__INPUT_B64__", base64.b64encode(raw).decode("ascii"))
    code = code.replace("__INPUT_SHA256__", sha(raw))
    (KERNEL / "probe.py").write_text(code, encoding="utf-8")
    metadata = {
        "id": "bunnoob2005/synthworlds-r1-placeholder-alias-probe-2026-11-04",
        "title": "SynthWorlds R1 placeholder alias probe 2026-11-04",
        "code_file": "probe.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    receipt = {"status": "SOURCE_BOUND_28_ROW_ALIAS_PROBE_READY_NOT_APPROVAL", "rows": len(rows),
        "input_sha256": sha(raw), "input_keys_sha256": sha("\n".join(sorted(blocked_keys)).encode()),
        "model_id": MODEL, "model_revision": REVISION, "answers_or_splits_sent": False,
        "aliases": {"[ENTITY_A]": "Alice", "[ENTITY_B]": "Bob"},
        "human_translation_reviewed": False, "training_eligible": False}
    (HERE / "build_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
