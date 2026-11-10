#!/usr/bin/env python3
"""Create a self-contained Kaggle Dataset folder and zip for the SFT pilot."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "sft_lora_dataset"
ZIP_PATH = ROOT / "kaggle" / "sft_lora_dataset.zip"

FILES = [
    "requirements-sft.txt",
    "scripts/train_sft_lora.py",
    "scripts/unlearn_lora.py",
    "scripts/evaluate_crosslingual_unlearning.py",
    "scripts/import_crosslingual_eval_translations.py",
    "scripts/build_composition_sft.py",
    "split_approval.json",
    "translation_review_policy.json",
    "sft_training/corpus.jsonl",
    "sft_training/manifest.json",
    "sft_training/composition_corpus.jsonl",
    "sft_training/composition_manifest.json",
    "sft_training/build_status.json",
    "sft_translation_ai_review/review_report.json",
    "sft_translation_ai_review/corrections.jsonl",
    "sft_translation_ai_review/repair_report.json",
    "sft_translation_ai_review/backtranslations/report.json",
    "sft_translation_ai_review/backtranslations/fr.jsonl",
    "sft_translation_ai_review/backtranslations/de.jsonl",
    "sft_translation_ai_review/backtranslations/hi.jsonl",
    "sft_translation_ai_review/backtranslations/th.jsonl",
    "sft_translation_candidates/fr.jsonl",
    "sft_translation_candidates/de.jsonl",
    "sft_translation_candidates/hi.jsonl",
    "sft_translation_candidates/th.jsonl",
    "evaluation_dataset/manifest.json",
    "evaluation_dataset/README.md",
    "evaluation_dataset/prompts/en.jsonl",
    "evaluation_dataset/labels/gold_labels.jsonl",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "sft_training/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "AI_ONLY_REVIEWED_FIVE_LANGUAGE_SFT_CORPUS" or manifest.get("do_not_train") is not False:
        raise SystemExit("Refusing to package: SFT corpus is not marked ready")
    if manifest.get("translation_review_mode") != "AI_ONLY" or manifest.get("human_translation_reviewed") is not False:
        raise SystemExit("Refusing to package: translation review disclosure is missing")

    stage = OUT.with_name(OUT.name + ".tmp")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise SystemExit(f"Required package file missing: {relative}")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    readme = """# SynthWorlds-SM Kaggle SFT feasibility pilot\n\nThis package contains the approved-for-pilot five-language atomic corpus, retained-chain composition corpus, and Kaggle runner. It is approved for an AI-only feasibility pilot. It is **not human-validated multilingual data**: no fluent human reviewed these translations, and the draft translator and back-translation audit use the same NLLB-200 distilled 600M model family. Back-translation is a consistency check, not proof of fluency or semantic accuracy.\n\nThe composition corpus adds final-answer examples from retained chains only; it excludes forget and holdout composition chains. Retain prompts intentionally overlap evaluation, so only forget-split performance is used for unseen-chain generalization. The pilot split approval is a delegated AI technical review for this limited 129-case feasibility split; it is not researcher sign-off or a powered main-study design.\n\nIn Kaggle, attach this dataset to `sft_lora_pilot.ipynb` and run it from the Notebook Editor with the `HF_TOKEN` secret attached. Kaggle CLI/API kernel runs do not attach Notebook Secrets. Run preflight first. All training stages remain disabled by default.\n"""
    (stage / "KAGGLE_DATASET_README.md").write_text(readme, encoding="utf-8")
    checksums = {p.relative_to(stage).as_posix(): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    metadata = ROOT / "kaggle" / "kaggle_private_upload" / "dataset-metadata.json"
    if not metadata.is_file():
        raise SystemExit("Private Kaggle dataset metadata is required to publish a version")
    shutil.copy2(metadata, stage / "dataset-metadata.json")
    checksums["dataset-metadata.json"] = sha(stage / "dataset-metadata.json")
    (stage / "package_manifest.json").write_text(json.dumps({
        "status": "KAGGLE_SFT_FEASIBILITY_PILOT_PACKAGE",
        "corpus_sha256": manifest["corpus_sha256"],
        "translation_review_mode": "AI_ONLY",
        "human_translation_reviewed": False,
        "file_count_excluding_package_manifest": len(checksums),
        "files_sha256": checksums,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    backup = OUT.with_name(OUT.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)
    if OUT.exists():
        OUT.rename(backup)
    stage.rename(OUT)
    if backup.exists():
        shutil.rmtree(backup)

    zip_tmp = ZIP_PATH.with_suffix(".zip.tmp")
    with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT).as_posix())
    zip_tmp.replace(ZIP_PATH)
    print(json.dumps({"status": "PACKAGE_READY", "folder": str(OUT), "zip": str(ZIP_PATH), "files": len(list(p for p in OUT.rglob("*") if p.is_file()))}, indent=2))


if __name__ == "__main__":
    main()
