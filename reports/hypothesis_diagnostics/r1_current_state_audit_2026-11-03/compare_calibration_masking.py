#!/usr/bin/env python3
"""Compare 14B diagnostic decisions using collision-free entity placeholders."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from verify_source_bound_calibration import PROJECT, TRAINING, sha
from verify_calibration_roundtrip import main as verify_previous

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "evidence/source_bound_calibration_roundtrip_retry"
OUT = HERE / "evidence/source_bound_calibration_masking_comparison"
sys.path.insert(0, str(TRAINING / "scripts"))
import adjudicate_training_translations_qwen14b as judge
from review_masking_v2 import mask_triplet


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    verify_previous()
    previous = json.loads((PRIOR / "checkpoint.json").read_text(encoding="utf-8"))
    previous_report = json.loads((PRIOR / "report.json").read_text(encoding="utf-8"))
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        models = {row["name"]: row["digest"] for row in json.load(response)["models"]}
    digest = models["qwen2.5:14b"]
    if digest != previous_report["identity"]["model_digests"]["qwen2.5:14b"]:
        raise ValueError("Qwen14 model digest changed")
    identity = {"prior_checkpoint_sha256": sha(PRIOR / "checkpoint.json"),
                "prior_report_sha256": sha(PRIOR / "report.json"),
                "judge_script_sha256": sha(Path(judge.__file__)),
                "masking_script_sha256": sha(TRAINING / "scripts/review_masking_v2.py"),
                "script_sha256": sha(Path(__file__)), "model_digest": digest}
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["identity"] != identity:
            raise ValueError("Comparison checkpoint provenance differs")
    else:
        checkpoint = {"identity": identity, "decisions": {}}
        write(checkpoint_path, checkpoint)
    judge.mask_triplet = mask_triplet
    rows = previous["records"]
    for lang in ("fr", "de", "hi", "th"):
        pending = [row for row in rows if row["language"] == lang
                   and f"{lang}|{row['kind']}|{row['prompt_id']}" not in checkpoint["decisions"]]
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            try:
                decisions = judge.judge_batch(lang, batch, 240)
            except Exception as exc:
                decisions = [{"error": type(exc).__name__, "reason": str(exc)[:500]} for _ in batch]
            for row, decision in zip(batch, decisions):
                checkpoint["decisions"][f"{lang}|{row['kind']}|{row['prompt_id']}"] = decision
            write(checkpoint_path, checkpoint)
            print(f"{lang}: reviewed {min(start + 4, len(pending))}/{len(pending)} pending rows", flush=True)
    counts, by_language, changes = Counter(), {lang: Counter() for lang in ("fr", "de", "hi", "th")}, []
    for row in rows:
        key = f"{row['language']}|{row['kind']}|{row['prompt_id']}"
        decision = checkpoint["decisions"][key]
        q7 = row["qwen7_roundtrip"]
        if type(decision.get("semantic_pass")) is not bool or type(decision.get("fluency_pass")) is not bool:
            status = "BLOCKED_OR_ERROR"
        elif decision["semantic_pass"] and decision["fluency_pass"]:
            status = "TWO_MODEL_PASS" if q7["semantic"]["decision"] and q7["fluency"]["decision"] else "14B_ADJUDICATED_PASS"
        else:
            status = "FLAGGED"
        counts[status] += 1
        by_language[row["language"]][status] += 1
        if status != row["diagnostic_status"]:
            changes.append({"language": row["language"], "kind": row["kind"], "prompt_id": row["prompt_id"],
                            "previous_status": row["diagnostic_status"], "new_status": status})
    report = {"status": "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW", "identity": identity,
              "checkpoint_sha256": sha(checkpoint_path), "rows": len(rows), "counts": dict(counts),
              "by_language": {lang: dict(by_language[lang]) for lang in by_language}, "changed_decisions": changes,
              "human_translation_reviewed": False, "answers_sent_to_reviewer": False,
              "limitations": ["Small fixed sample; candidates are not approved or merged.",
                              "Both 14B judgments use the same model and are not independent reviewers.",
                              "AI-only judgments do not establish human fluency or accuracy."]}
    write(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
