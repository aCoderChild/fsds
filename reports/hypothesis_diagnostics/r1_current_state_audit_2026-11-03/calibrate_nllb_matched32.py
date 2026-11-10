#!/usr/bin/env python3
"""Generate NLLB candidates on the same 32 source-bound calibration questions."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from verify_source_bound_calibration import verify, sha, PROJECT, TRAINING, RUN

sys.path.insert(0, str(TRAINING / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts"))
import calibrate_nllb13_flagged_translations as nllb
from review_training_translations_qwen import deterministic_checks

HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence/nllb13_matched_calibration32"
REQUIRED = ("nonempty", "target_script_present", "protected_names_preserved", "protected_numbers_preserved", "no_unrestored_marker")


def main() -> None:
    selected, _, previous = verify()
    if OUT.exists():
        raise FileExistsError("Refusing to overwrite an existing NLLB calibration")
    from huggingface_hub import snapshot_download
    import torch
    import transformers
    snapshot_download(repo_id=nllb.MODEL_ID, revision=nllb.MODEL_REVISION, local_files_only=True)
    rows = [{**item["row"], "language": item["language"]} for item in selected]
    outputs, metrics = nllb.translate_sample(rows, batch_size=4)
    if len(outputs) != 32:
        raise ValueError("NLLB output does not cover the fixed matched sample")
    decisions = []
    for row, output in zip(rows, outputs):
        text, error = output["text"], output["error"]
        record = {"language": row["language"], "kind": row["kind"], "prompt_id": row["prompt_id"],
                  "source_en_sha256": hashlib.sha256(row["text_en"].encode()).hexdigest(),
                  "final_review_status": row["final_review_status"], "translation_error": error,
                  "translation_model": nllb.MODEL_ID, "model_revision": nllb.MODEL_REVISION,
                  "human_translation_reviewed": False, "answers_sent_to_translator": False}
        if error is None:
            checks = deterministic_checks(row["language"], {**row, "text_translation": text})
            record.update({"text_translation": text, "translation_sha256": hashlib.sha256(text.encode()).hexdigest(),
                           "deterministic_checks": checks,
                           "deterministic_pass": all(checks[name] is True for name in REQUIRED)})
        decisions.append(record)
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / "candidates.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
    report = {"status": "COMPLETE_DIAGNOSTIC_ONLY_NOT_RELEASE_REVIEW", "rows": len(decisions),
              "translation_successes": sum(r.get("translation_error") is None for r in decisions),
              "deterministic_passes": sum(r.get("deterministic_pass") is True for r in decisions),
              "source_calibration_report_sha256": sha(RUN / "report.json"),
              "source_calibration_decisions_sha256": previous["decisions_sha256"],
              "selection_verifier_sha256": sha(HERE / "verify_source_bound_calibration.py"),
              "generator_script_sha256": sha(Path(__file__)), "nllb_helper_sha256": sha(Path(nllb.__file__)),
              "candidates_sha256": sha(path), "model_id": nllb.MODEL_ID, "model_revision": nllb.MODEL_REVISION,
              "torch_version": torch.__version__, "transformers_version": transformers.__version__, "generation": metrics,
              "human_translation_reviewed": False, "answers_sent_to_translator": False,
              "limits": ["Fixed matched sample only; candidates are unreviewed and not approved.",
                         "No human translation review was performed."]}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
