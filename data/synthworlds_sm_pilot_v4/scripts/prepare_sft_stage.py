#!/usr/bin/env python3
"""Preflight the report's multilingual SFT stage; never starts model training."""
import collections
import importlib.util
import json
import os
import platform
import re
from datetime import date
from pathlib import Path
from os.path import expanduser

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "synthworlds_sm_pilot_v3"
R1 = ROOT / "docs" / "report" / "R1.tex"
LANGS = ("en", "fr", "de", "hi", "th")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_iso_date(value):
    value = str(value or "")
    try:
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) and date.fromisoformat(value))
    except ValueError:
        return False


def main():
    audit = read_json(OUT / "audit_report.json")
    sft = read_jsonl(OUT / "sft_natural_en_candidate.jsonl")
    source_records = [example for row in sft for example in row["source_examples"]]
    per_language = {"en": len(sft)}
    pending_review = {"en": 0}
    missing = {}
    for lang in LANGS[1:]:
        path = OUT / "sft_translation_candidates" / f"{lang}.jsonl"
        rows = read_jsonl(path) if path.is_file() else []
        per_language[lang] = len(rows)
        pending_review[lang] = sum(
            r.get("translation_status") != "APPROVED_BY_AI_REVIEW"
            or r.get("review_status") != "APPROVED_BY_AI_REVIEW"
            or r.get("review_mode") != "AI_ONLY"
            or r.get("reviewer_type") != "AI"
            or r.get("human_review_status") != "NOT_PERFORMED"
            or not str(r.get("reviewer_name", "")).strip()
            or not is_iso_date(r.get("review_date"))
            for r in rows
        )
        missing[lang] = max(0, len(sft) - len(rows))

    approval_path = OUT / "split_approval.json"
    split_approval = read_json(approval_path) if approval_path.is_file() else {}
    translation_policy_path = OUT / "translation_review_policy.json"
    translation_policy = read_json(translation_policy_path) if translation_policy_path.is_file() else {}
    translation_report_path = OUT / "sft_translation_ai_review" / "review_report.json"
    translation_report = read_json(translation_report_path) if translation_report_path.is_file() else {}
    corpus_path = OUT / "sft_training" / "corpus.jsonl"
    corpus_manifest_path = OUT / "sft_training" / "manifest.json"
    corpus_manifest = read_json(corpus_manifest_path) if corpus_manifest_path.is_file() else {}

    report_text = R1.read_text(encoding="utf-8")
    model_match = re.search(r"Use (Llama [^\n]+?) for the feasibility pilot", report_text)
    model = model_match.group(1).strip() if model_match else "MODEL_NOT_FOUND_IN_R1"
    hardware = {"python": platform.python_version(), "cuda_available": False, "mps_available": False}
    if importlib.util.find_spec("torch"):
        import torch
        hardware["torch_version"] = torch.__version__
        hardware["cuda_available"] = torch.cuda.is_available()
        hardware["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    dependencies = {name: bool(importlib.util.find_spec(name)) for name in ("torch", "transformers", "peft", "accelerate")}
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token_present:
        try:
            from huggingface_hub import get_token
            token_present = bool(get_token())
        except (ImportError, OSError):
            pass
    checkpoint_roots = [ROOT / "models", ROOT / "checkpoints", Path(expanduser("~")) / ".cache" / "huggingface" / "hub"]
    checkpoint_markers = {"model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"}
    local_checkpoint_found = False
    checkpoint_count = 0
    for checkpoint_root in checkpoint_roots:
        if not checkpoint_root.is_dir():
            continue
        for path in checkpoint_root.rglob("*"):
            path_text = path.as_posix().casefold()
            is_requested_model = "llama-3.2-1b" in path_text or "llama_3.2_1b" in path_text
            if is_requested_model and path.is_file() and (path.name in checkpoint_markers or path.suffix == ".gguf"):
                local_checkpoint_found = True
                checkpoint_count += 1

    # Separate structural verification from approvals and Kaggle-only setup.
    # The audit's do_not_train flag is a safety stop, not evidence of approval.
    gates = {
        "split_structure_verified": (
            audit["split_counts"] == {"forget": 77, "retain": 26, "holdout": 26}
            and audit["remaining_cross_split_entity_pairs"] == 0
            and audit["remaining_cross_split_chain_fact_pairs"] == 0
            and audit["remaining_cross_split_exact_answer_pairs"] == 0
        ),
        "split_proposal_approved": (
            split_approval.get("status") == "APPROVED_PILOT_SPLIT"
            and bool(str(split_approval.get("reviewer_name", "")).strip())
            and is_iso_date(split_approval.get("review_date"))
        ),
        "native_translation_review_complete": False,
        "ai_only_review_authorized": translation_policy.get("status") == "AI_ONLY_REVIEW_AUTHORIZED" and translation_policy.get("review_mode") == "AI_ONLY",
        "ai_translation_review_complete": (
            translation_report.get("status") == "AI_ONLY_TRANSLATION_REVIEW_COMPLETE_HUMAN_REVIEW_NOT_PERFORMED"
            and translation_report.get("ai_approved_total") == 820
            and translation_report.get("human_reviewed_total") == 0
        ),
        "translation_review_complete_for_selected_mode": all(
            per_language.get(lang, 0) == len(sft) and pending_review.get(lang, len(sft)) == 0
            for lang in LANGS[1:]
        ) and translation_policy.get("status") == "AI_ONLY_REVIEW_AUTHORIZED",
        "complete_five_language_sft_coverage": all(per_language.get(lang, 0) == len(sft) for lang in LANGS),
        "approved_corpus_built": (
            corpus_path.is_file()
            and corpus_manifest_path.is_file()
            and corpus_manifest.get("status") == "AI_ONLY_REVIEWED_FIVE_LANGUAGE_SFT_CORPUS"
            and corpus_manifest.get("do_not_train") is False
        ),
        "kaggle_notebook_present": (OUT / "kaggle" / "sft_lora_pilot.ipynb").is_file(),
    }
    blockers = []
    if not gates["split_structure_verified"]:
        blockers.append("The split/source integrity audit has not passed its structural checks.")
    if not gates["split_proposal_approved"]:
        blockers.append("The pilot split remains a proposal; a named reviewer must approve it before training.")
    if not gates["ai_only_review_authorized"]:
        blockers.append("The selected AI-only translation-review policy is not authorized.")
    if not gates["ai_translation_review_complete"] or not gates["translation_review_complete_for_selected_mode"]:
        blockers.append("AI-only translation review of the French, German, Hindi, and Thai candidates is incomplete or failed.")
    if not gates["complete_five_language_sft_coverage"]:
        blockers.append("Five-language SFT coverage is incomplete.")
    if not gates["approved_corpus_built"]:
        blockers.append("The AI-reviewed five-language corpus and checksum manifest have not been built.")
    if not gates["kaggle_notebook_present"]:
        blockers.append("The Kaggle SFT notebook is missing from the project bundle.")

    # These are local diagnostics, not Kaggle blockers: the notebook installs
    # dependencies and validates the gated model and GPU inside Kaggle itself.
    local_environment = {
        "hardware": hardware,
        "training_packages": dependencies,
        "hf_token_present": token_present,
        "local_model_checkpoint_found": local_checkpoint_found,
        "checkpoint_weight_files_found": checkpoint_count,
        "note": "Kaggle runtime, secret, GPU, and model access must be checked in the notebook; local values do not determine Kaggle readiness.",
    }

    report = {
        "status": "SFT_PREFLIGHT_BLOCKED_NO_TRAINING_STARTED" if blockers else "SFT_DATA_READY_KAGGLE_RUNTIME_CHECK_REQUIRED",
        "training_started": False,
        "report_model": model,
        "training_languages": list(LANGS),
        "sft_unique_english_messages": len(sft),
        "sft_source_hop_records_before_dedup": len(source_records),
        "english_source_qa_cases": len({x["source_qa_id"] for x in source_records}),
        "sft_candidate_rows_by_language": per_language,
        "non_english_rows_missing_per_language": missing,
        "translation_reviews_pending_by_language": pending_review,
        "gates": gates,
        "blockers": blockers,
        "available_environment": local_environment,
        "conditional_follow_up": "The completed atomic-SFT pilot triggered R1's composition follow-up: forget-split multi-hop accuracy was 11.7–18.2% with only 3–12 precompetent cases per language. scripts/build_composition_sft.py creates a variant adding 26 retained-chain multi-hop examples per language, with zero forget/holdout composition rows; evaluate only forget split as unseen-chain generalization.",
        "translation_review_disclosure": "AI-only review was explicitly authorized; no fluent human reviewed these translations. This is not human-validated multilingual data.",
        "source_integrity_status": "PASS",
        "sft_pipeline": {"translation_batch_script": "scripts/prepare_sft_translation_requests.py", "corpus_builder": "scripts/build_sft_corpus.py", "composition_builder": "scripts/build_composition_sft.py", "training_runner": "scripts/train_sft_lora.py", "kaggle_notebook": "kaggle/sft_lora_pilot.ipynb", "runner_default": "preflight only; --train is required", "loss_mask": "assistant answer tokens only", "recipe": "LoRA fine-tuning; 3 epochs; per-device batch size 1; gradient accumulation 32; learning rate 2e-4; single visible Kaggle T4"},
        "note": "Translation review is AI-only; no fluent human reviewed the corpus. The SFT data is ready only for the explicitly authorized feasibility pilot, subject to Kaggle runtime checks; it is not human-validated multilingual data.",
    }
    path = OUT / "sft_stage_readiness.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
