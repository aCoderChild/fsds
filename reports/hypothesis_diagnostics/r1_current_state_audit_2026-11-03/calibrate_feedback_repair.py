#!/usr/bin/env python3
"""Try reviewer-feedback repair on two flagged sample questions per language."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from verify_source_bound_calibration import PROJECT, TRAINING, sha
from verify_calibration_masking_comparison import main as verify_comparison

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "evidence/source_bound_calibration_roundtrip_retry"
COMPARE = HERE / "evidence/source_bound_calibration_masking_comparison"
OUT = HERE / "evidence/source_bound_feedback_calibration8"
sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
from review_training_translations_qwen import ask_qwen, deterministic_checks, masked_pair
from repair_flagged_eval_translations_translategemma import protect, restore

LANGS = {"fr": "French", "de": "German", "hi": "Hindi", "th": "Thai"}
REQUIRED = ("nonempty", "target_script_present", "protected_names_preserved", "protected_numbers_preserved", "no_unrestored_marker")
SYSTEM = ("Translate the English question into natural {language}, preserving its relation direction, scope, "
          "tense, and answer type. The prior translation and reviewer feedback may be wrong; the English source "
          "is authoritative. Correct any actual issue without introducing facts. Copy all ZXQ markers exactly "
          "and do not translate, omit, or duplicate them. Return only the new translated question.")


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    verify_comparison()
    rows = json.loads((PRIOR / "checkpoint.json").read_text(encoding="utf-8"))["records"]
    comparison = json.loads((COMPARE / "checkpoint.json").read_text(encoding="utf-8"))["decisions"]
    selected, counts = [], Counter()
    for row in rows:
        key = f"{row['language']}|{row['kind']}|{row['prompt_id']}"
        verdict = comparison[key]
        if (not verdict.get("semantic_pass") or not verdict.get("fluency_pass")) and counts[row["language"]] < 2:
            if not isinstance(verdict.get("reason"), str) or not verdict["reason"].strip():
                raise ValueError("Missing source review reason")
            selected.append((key, row, verdict["reason"]))
            counts[row["language"]] += 1
    if len(selected) != 8 or any(counts[lang] != 2 for lang in LANGS):
        raise ValueError("Not enough flagged items for fixed feedback calibration")
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        models = {r["name"]: r["digest"] for r in json.load(response)["models"]}
    expected = json.loads((PRIOR / "report.json").read_text())["identity"]["model_digests"]
    if any(models[name] != expected[name] for name in ("translategemma:4b", "qwen2.5:7b-instruct")):
        raise ValueError("Translation or reviewer model digest changed")
    identity = {"prior_checkpoint_sha256": sha(PRIOR / "checkpoint.json"),
                "comparison_report_sha256": sha(COMPARE / "report.json"),
                "comparison_checkpoint_sha256": sha(COMPARE / "checkpoint.json"),
                "script_sha256": sha(Path(__file__)),
                "translation_model_digest": models["translategemma:4b"],
                "review_model_digest": models["qwen2.5:7b-instruct"],
                "selected_keys": [key for key, _, _ in selected]}
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["identity"] != identity:
            raise ValueError("Feedback calibration source changed; refusing resume")
    else:
        checkpoint = {"identity": identity, "decisions": {}}
        write(checkpoint_path, checkpoint)
    for key, row, reason in selected:
        if key in checkpoint["decisions"]:
            continue
        item = {"language": row["language"], "kind": row["kind"], "prompt_id": row["prompt_id"],
                "source_sha256": row["source_sha256"], "previous_candidate_sha256": row["candidate_sha256"],
                "feedback": reason, "status": "ERROR"}
        try:
            source, mapping, expected_counts = protect(row["text_en"], row["protected_entity_names"])
            previous, previous_mapping, previous_counts = protect(row["text_translation"], row["protected_entity_names"])
            if mapping != previous_mapping or expected_counts != previous_counts:
                raise ValueError("Question and prior candidate protect different entity markers")
            prompt = {"english_source": source, "previous_translation": previous, "unverified_feedback": reason}
            payload = {"model": "translategemma:4b", "stream": False,
                       "messages": [{"role": "system", "content": SYSTEM.format(language=LANGS[row["language"]])},
                                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
                       "options": {"temperature": 0, "num_predict": 192}, "keep_alive": "10m"}
            request = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=180) as response:
                response_data = json.load(response)
            text = response_data.get("message", {}).get("content", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty translation response")
            translated = restore(text, mapping, expected_counts)
            check = deterministic_checks(row["language"], {**row, "text_translation": translated})
            item.update({"new_translation": translated, "checks": check})
            if not all(check[name] is True for name in REQUIRED):
                item["status"] = "INTEGRITY_FAILED"
            else:
                verdict = ask_qwen(row["language"], [masked_pair({**row, "text_translation": translated})[1]], 180)[0]
                item.update({"qwen7_pass": verdict["decision"], "qwen7_reason": verdict["reason"],
                             "status": "PRELIMINARY_PASS" if verdict["decision"] else "FLAGGED"})
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        checkpoint["decisions"][key] = item
        write(checkpoint_path, checkpoint)
        print(f"Feedback repair {len(checkpoint['decisions'])}/8", flush=True)
    statuses = Counter(item["status"] for item in checkpoint["decisions"].values())
    report = {"status": "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW", "identity": identity,
              "checkpoint_sha256": sha(checkpoint_path), "rows": len(selected), "counts": dict(statuses),
              "human_translation_reviewed": False, "answers_sent_to_reviewer": False,
              "limitations": ["Eight reviewer-selected flagged rows; not a representative quality estimate.",
                              "Feedback and screening are AI-only and may reinforce model errors.",
                              "Candidates have not passed the round-trip and 14B release review."]}
    write(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
