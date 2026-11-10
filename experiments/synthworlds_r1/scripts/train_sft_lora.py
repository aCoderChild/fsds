#!/usr/bin/env python3
"""Fail-closed LoRA SFT runner. Preflight is default; training requires --train."""
import argparse
import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

from assistant_masking import encode_assistant_only
from model_profiles import MAIN_PROFILE, PILOT_PROFILE, get_model_profile, model_load_options

MODEL_ID = get_model_profile(MAIN_PROFILE).model_id
LOCKED_TRAINING_STACK = {"transformers": "5.22.1", "peft": "0.21.2", "accelerate": "1.15.0"}
LOCKED_QUANTIZATION_STACK = {"bitsandbytes": "0.50.2"}
SUPPORTED_CORPUS_STATUSES = {"AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha(path):
    path = Path(path)
    if path.is_file():
        return sha(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha(child)))
    return h.hexdigest()


def valid_iso_date(value):
    value = str(value or "")
    try:
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) and date.fromisoformat(value))
    except ValueError:
        return False


def source_integrity_error(corpus_path, manifest_path, manifest):
    """Check that local source reconstruction validated this exact release."""
    corpus_path, manifest_path = Path(corpus_path), Path(manifest_path)
    project_root = corpus_path.parent.parent if corpus_path.parent.name == "sft_training" else corpus_path.parent
    proof_path = corpus_path.parent / "source_integrity_validation.json"
    release_path = corpus_path.parent / "release_validation.json"
    verifier_path = project_root / "scripts/validate_sft_source_integrity.py"
    if not all(path.is_file() for path in (proof_path, release_path, verifier_path)):
        return "Source-bound SFT validation proof, release validation, and verifier script are required"
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        sources = proof.get("source_checks", {})
        valid = (
            proof.get("status") == "PASS_SOURCE_BOUND_SFT_VALIDATION"
            and proof.get("human_translation_reviewed") is False
            and proof.get("corpus_sha256") == sha(corpus_path)
            and proof.get("manifest_sha256") == sha(manifest_path)
            and proof.get("release_validation_sha256") == sha(release_path)
            and proof.get("verifier_sha256") == sha(verifier_path)
            and proof.get("corpus_rows") == manifest.get("total_rows") == 2440
            and proof.get("canonical_sft_ids") == manifest.get("unique_sft_ids_per_language") == 488
            and proof.get("languages") == ["en", "fr", "de", "hi", "th"]
            and sources.get("canonical_sft_ids") == proof.get("canonical_sft_ids")
            and sources.get("source_manifest_sha256") == manifest.get("source_manifest_sha256")
            and sources.get("preparation_report_sha256") == manifest.get("preparation_report_sha256")
            and sources.get("source_inputs_sha256", {}).get("candidate_cases") == manifest.get("candidate_cases_sha256")
        )
    except (OSError, ValueError, TypeError, AttributeError):
        valid = False
    return None if valid else "Source-bound SFT validation proof does not match the corpus, manifest, or verifier"


def preflight(corpus_path, manifest_path, output_dir, train=False, include_splits=("forget", "retain", "train_only"), seed=42, model_revision=None, model_profile=MAIN_PROFILE):
    corpus_path, manifest_path = Path(corpus_path), Path(manifest_path)
    profile = get_model_profile(model_profile)
    errors = []
    if not corpus_path.is_file() or not manifest_path.is_file():
        errors.append("Approved SFT corpus and manifest are required")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in SUPPORTED_CORPUS_STATUSES or manifest.get("do_not_train") is not False:
            errors.append("Corpus manifest is not approved for training")
        if manifest.get("translation_review_mode") != "AI_ONLY" or manifest.get("human_translation_reviewed") is not False:
            errors.append("Corpus review mode is missing or human-review status is misrepresented")
        if (manifest.get("translation_review_authorization") != "AI_ONLY_REVIEW_AUTHORIZED"
                or manifest.get("unique_sft_ids_per_language", 0) <= 0
                or manifest.get("ai_reviewed_rows") != manifest.get("unique_sft_ids_per_language", 0) * 4):
            errors.append("Corpus lacks complete user-authorized AI-only translation review metadata")
        split_approval = manifest.get("split_approval", {})
        project_root = corpus_path.parent.parent if corpus_path.parent.name == "sft_training" else corpus_path.parent
        approval_path = project_root / "split_approval.json"
        if (
            split_approval.get("status") != "PILOT_SPLIT_PIPELINE_VALIDATED"
            or not str(split_approval.get("reviewer_name", "")).strip()
            or not valid_iso_date(split_approval.get("review_date"))
            or split_approval.get("reviewer_type") != "AI_PIPELINE"
            or split_approval.get("human_approval_claimed") is not False
            or not approval_path.is_file()
            or split_approval.get("record_sha256") != sha(approval_path)
        ):
            errors.append("Corpus manifest lacks an intact, explicitly AI-pipeline-validated pilot split record")
        if manifest.get("corpus_sha256") != sha(corpus_path):
            errors.append("Corpus checksum does not match manifest")
        source_error = source_integrity_error(corpus_path, manifest_path, manifest)
        if source_error:
            errors.append(source_error)
        counts = manifest.get("rows_by_language", {})
        if any(counts.get(lang) != manifest.get("unique_sft_ids_per_language") for lang in ("en", "fr", "de", "hi", "th")):
            errors.append("Five-language row counts are incomplete or unequal")
        if manifest.get("holdout_rows") != 0:
            errors.append("Holdout rows are forbidden")
        if manifest.get("total_rows") != manifest.get("unique_sft_ids_per_language", 0) * 5:
            errors.append("Pilot corpus total does not equal the balanced five-language prompt count")
        if not errors:
            with corpus_path.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            actual = {lang: sum(row.get("language") == lang for row in rows) for lang in ("en", "fr", "de", "hi", "th")}
            if actual != counts or len(rows) != manifest.get("total_rows"):
                errors.append("Corpus rows do not match manifest language counts")
            if any(row.get("split") == "holdout" for row in rows):
                errors.append("Corpus contains a holdout row")
            keys = [(row.get("language"), row.get("sft_id")) for row in rows]
            if len(keys) != len(set(keys)):
                errors.append("Corpus contains duplicate language/SFT IDs")
            ids_by_lang = {lang: {row.get("sft_id") for row in rows if row.get("language") == lang} for lang in ("en", "fr", "de", "hi", "th")}
            if any(ids_by_lang[lang] != ids_by_lang["en"] for lang in ("fr", "de", "hi", "th")):
                errors.append("Corpus language rows do not cover the same SFT prompt IDs")
            for row in rows:
                messages = row.get("messages", [])
                if (len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant"
                        or messages[0].get("content") != row.get("question")
                        or messages[1].get("content") != row.get("answer")
                        or not str(row.get("question", "")).strip() or not str(row.get("answer", "")).strip()
                        or row.get("split") not in {"forget", "retain", "train_only"}):
                    errors.append(f"Malformed or forbidden SFT row: {row.get('language')}/{row.get('sft_id')}")
                    break
            actual_split_counts = {lang: {split: sum(r.get("language") == lang and r.get("split") == split for r in rows) for split in ("forget", "retain", "train_only")} for lang in ("en", "fr", "de", "hi", "th")}
            if actual_split_counts != manifest.get("rows_by_split_per_language"):
                errors.append("Corpus split counts do not match the manifest")
    package_names = ("torch", "transformers", "peft", "accelerate", "numpy", *profile.required_packages)
    available = {name: __import__("importlib.util").util.find_spec(name) is not None for name in package_names}
    expected_stack = dict(LOCKED_TRAINING_STACK)
    if profile.quantization == "4bit_nf4":
        expected_stack.update(LOCKED_QUANTIZATION_STACK)
    if all(available.get(name, False) for name in expected_stack):
        from importlib.metadata import version
        actual_stack = {name: version(name) for name in expected_stack}
        if actual_stack != expected_stack:
            errors.append(f"Training stack mismatch; expected {expected_stack}, found {actual_stack}")
    if train and not all(available.values()):
        errors.append("Missing training packages: " + ", ".join(k for k, v in available.items() if not v))
    token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token:
        try:
            from huggingface_hub import get_token
            token = bool(get_token())
        except (ImportError, OSError):
            pass
    if any(s not in {"forget", "retain", "train_only", "holdout"} for s in include_splits) or "holdout" in include_splits:
        errors.append("Training splits may include forget/retain/train_only; holdout is forbidden")
    if train and not token:
        errors.append("No Hugging Face access token found; gated Llama weights cannot be loaded")
    if train and not model_revision:
        errors.append("A pinned Hugging Face model commit SHA is required for reproducible training")
    if train:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append(f"CUDA GPU is required for the selected {profile.role} profile")
        except ImportError:
            pass
    status = "READY_TO_TRAIN" if not errors else "BLOCKED"
    result = {"status": status, "training_started": False, "train_flag": train,
              "model_profile": profile.manifest(), "model_id": profile.model_id,
              "model_revision": model_revision, "errors": errors,
              "translation_review_mode": "AI_ONLY" if not errors else None,
              "human_translation_reviewed": False, "training_packages": available,
              "training_packages_ready": all(available.values()), "hf_token_present": token,
              "note": "Preflight never downloads weights or starts training. Pilot/main-study model and quantization are recorded explicitly; translations were reviewed AI-only with no fluent-human review."}
    if train and not errors:
        result = run_training(corpus_path, manifest_path, output_dir, "AI_ONLY", include_splits, seed, manifest.get("training_variant", "atomic_five_language_sft"), model_revision, model_profile)
    return result


def run_training(corpus_path, manifest_path, output_dir, translation_review_mode, include_splits=("forget", "retain", "train_only"), seed=42, corpus_variant="atomic_five_language_sft", model_revision=None, model_profile=MAIN_PROFILE):
    import random
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    profile = get_model_profile(model_profile)
    if profile.quantization == "4bit_nf4":
        from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(profile.model_id, token=True, revision=model_revision, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset-based assistant-only loss masking")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    load_kwargs = model_load_options(model_profile, torch,
                                     device_map={"": torch.cuda.current_device()} if profile.quantization != "none" else None)
    model = AutoModelForCausalLM.from_pretrained(profile.model_id, token=True, revision=model_revision, **load_kwargs)
    if profile.quantization != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = []
    with Path(corpus_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                if ex.get("split") not in include_splits:
                    continue
                ids, labels = encode_assistant_only(tok, ex["messages"])
                rows.append({"input_ids": ids, "labels": labels})

    class Collator:
        def __call__(self, batch):
            width = max(len(x["input_ids"]) for x in batch)
            ids, masks, labels = [], [], []
            for x in batch:
                n = width - len(x["input_ids"])
                ids.append(x["input_ids"] + [tok.pad_token_id] * n)
                masks.append([1] * len(x["input_ids"]) + [0] * n)
                labels.append(x["labels"] + [-100] * n)
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=1, gradient_accumulation_steps=32, learning_rate=2e-4, logging_steps=10, save_strategy="epoch", report_to="none", remove_unused_columns=False, seed=seed, data_seed=seed, fp16=(device == "cuda"), gradient_checkpointing=(profile.quantization != "none"))
    trainer = Trainer(model=model, args=args, train_dataset=rows, data_collator=Collator())
    trainer.train()
    output_dir = Path(output_dir)
    trainer.save_model(output_dir / "adapter")
    tok.save_pretrained(output_dir / "adapter")
    if corpus_variant == "atomic_five_language_sft":
        training_variant = "retain_only" if set(include_splits) == {"retain"} else "full_sft"
    else:
        training_variant = "composition_retain_only" if "forget" not in set(include_splits) else "composition_sft"
    run_rows = [json.loads(line) for line in Path(corpus_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    run_rows = [row for row in run_rows if row.get("split") in include_splits]
    training_script_path = Path(__file__).resolve()
    assistant_masking_script_path = training_script_path.with_name("assistant_masking.py")
    try:
        import accelerate, peft, platform, transformers
        versions = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                    "accelerate": accelerate.__version__, "numpy": np.__version__, "python": platform.python_version(),
                    "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as exc:
        versions = {"version_capture_error": f"{type(exc).__name__}: {exc}"}
    result = {"status": "TRAINING_COMPLETE", "training_started": True, "model_profile": profile.manifest(), "model_id": profile.model_id, "model_revision": model_revision, "device": device, "seed": seed,
              "rows": len(rows), "rows_by_language": {lang: sum(row.get("language") == lang for row in run_rows) for lang in ("en", "fr", "de", "hi", "th")},
              "rows_by_split": {split: sum(row.get("split") == split for row in run_rows) for split in include_splits},
              "included_splits": list(include_splits), "training_variant": training_variant, "corpus_variant": corpus_variant,
              "corpus_sha256": sha(corpus_path), "corpus_manifest_sha256": sha(manifest_path),
              "source_integrity_validation_sha256": sha(Path(corpus_path).parent / "source_integrity_validation.json"),
              "epochs": 3, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 32,
              "learning_rate": 2e-4, "loss_mask": "assistant answer tokens only", "translation_review_mode": translation_review_mode,
              "human_translation_reviewed": False, "adapter_dir": str(output_dir / "adapter"),
              "adapter_sha256": tree_sha(output_dir / "adapter"),
              "training_script_path": str(training_script_path), "training_script_sha256": sha(training_script_path),
              "assistant_masking_script_path": str(assistant_masking_script_path),
              "assistant_masking_script_sha256": sha(assistant_masking_script_path),
              "software_versions": versions}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parents[1] / "sft_training" / "corpus.jsonl")
    p.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "sft_training" / "manifest.json")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "sft_training" / "llama31_8b_qlora")
    p.add_argument("--train", action="store_true", help="Start training only after all fail-closed checks pass")
    p.add_argument("--include-splits", nargs="+", choices=("forget", "retain", "train_only"), default=("forget", "retain", "train_only"), help="Training splits; holdout is never allowed")
    p.add_argument("--seed", type=int, default=42, help="Seed for reproducible initialization and data order")
    p.add_argument("--model-revision", default=os.environ.get("HF_MODEL_REVISION"), help="Pinned Hugging Face commit SHA from the Kaggle preflight")
    p.add_argument("--model-profile", choices=(PILOT_PROFILE, MAIN_PROFILE), default=MAIN_PROFILE,
                   help="Pilot 1B FP16 or main-study 8B QLoRA-NF4 profile")
    a = p.parse_args()
    result = preflight(a.corpus, a.manifest, a.output_dir, a.train, a.include_splits, a.seed, a.model_revision, a.model_profile)
    print(json.dumps(result, indent=2))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
