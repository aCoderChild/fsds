#!/usr/bin/env python3
"""English-only GA, GD, or NPO unlearning on an approved SFT LoRA adapter.

The default is a fail-closed preflight. Training is only started with --train.
Holdout examples are never loaded. Evaluation labels are never read here.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

from assistant_masking import encode_assistant_only
from evaluate_pilot_three_probe import verify_baseline_target_manifest
from model_profiles import MAIN_PROFILE, PILOT_PROFILE, get_model_profile, model_load_options
from train_sft_lora import source_integrity_error
from unlearning_objectives import npo_loss

MODEL_ID = get_model_profile(MAIN_PROFILE).model_id
LOCKED_TRAINING_STACK = {"transformers": "5.22.1", "peft": "0.21.2", "accelerate": "1.15.0"}
LOCKED_QUANTIZATION_STACK = {"bitsandbytes": "0.50.2"}
METHODS = {"ga", "gd", "npo"}
SUPPORTED_CORPUS_STATUSES = {"AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(path):
    path = Path(path)
    if path.is_file():
        return sha(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha(child)))
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_hop_unlearning_rows(corpus_path):
    """Select English first-hop facts; ignore SFT-only composition practice rows."""
    rows = read_jsonl(corpus_path)
    return [
        row for row in rows
        if row.get("language") == "en"
        and row.get("hop") == "r1"
        and row.get("split") in {"forget", "retain"}
    ]


def retain_schedule_indices(num_steps, num_retain_rows):
    """Cycle over all retained rows before reusing any row."""
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if num_steps and num_retain_rows <= 0:
        raise ValueError("At least one retained row is required")
    if not num_steps:
        return []
    return [step % num_retain_rows for step in range(num_steps)]


def validate_training_parameters(method, epochs, lr, retain_weight, beta, max_length):
    """Reject settings that could produce a nominally completed no-op run."""
    if method not in METHODS:
        raise ValueError(f"Method must be one of {sorted(METHODS)}")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(retain_weight) or retain_weight < 0:
        raise ValueError("retain weight must be finite and non-negative")
    if method == "npo" and (not math.isfinite(beta) or beta <= 0):
        raise ValueError("NPO beta must be finite and positive")
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max length must be a positive integer")


def encode_unlearning_example(tokenizer, messages, max_length):
    """Use the shared offset-based mask for English unlearning examples."""
    input_ids, labels = encode_assistant_only(tokenizer, messages)
    if len(input_ids) > max_length:
        raise ValueError("Encoded unlearning example exceeds max_length")
    return input_ids, labels


def preflight(corpus_path, corpus_manifest_path, sft_run_manifest_path, target_ids_manifest_path, method, baseline_model_name=None, model_profile=MAIN_PROFILE):
    profile = get_model_profile(model_profile)
    corpus_path = Path(corpus_path)
    errors = []
    required = [corpus_path, Path(corpus_manifest_path), Path(sft_run_manifest_path), Path(target_ids_manifest_path)]
    if any(not p.is_file() for p in required):
        return {"status": "BLOCKED", "errors": ["Corpus, corpus manifest, completed SFT run manifest, and baseline-competence target manifest are required"],
                "method": method, "model_profile": profile.manifest(), "model_id": profile.model_id}
    manifest = json.loads(Path(corpus_manifest_path).read_text(encoding="utf-8"))
    sft_run = json.loads(Path(sft_run_manifest_path).read_text(encoding="utf-8"))
    target_manifest = json.loads(Path(target_ids_manifest_path).read_text(encoding="utf-8"))
    try:
        verify_baseline_target_manifest(target_manifest)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"Baseline target manifest failed recomputation: {type(exc).__name__}: {exc}")
    target_ids_list = target_manifest.get("selected_source_qa_ids", [])
    if not isinstance(target_ids_list, list) or any(not isinstance(value, str) or not value for value in target_ids_list):
        errors.append("Target IDs must be a list of non-empty source QA ID strings")
        target_ids_list = []
    target_ids = set(target_ids_list)
    if target_manifest.get("status") != "SFT_PRECOMPETENT_ENGLISH_FORGET_TARGETS":
        errors.append("Target manifest does not identify SFT-precompetent English forget cases")
    if target_manifest.get("model_profile") != profile.manifest():
        errors.append("Baseline competence target profile differs from the requested unlearning model/quantization profile")
    if target_manifest.get("language") != "en" or target_manifest.get("split") != "forget":
        errors.append("Unlearning target selection must be English-only and forget-split only")
    if target_manifest.get("selection_rule") != "English direct_positive, direct_negative_control, and multi_hop all correct in every full-SFT seed":
        errors.append("Target manifest lacks the required pre-unlearning competence rule")
    gate_counts = target_manifest.get("bilingual_counts_by_language", {})
    gate_requirements = target_manifest.get("pilot_gate_thresholds", {})
    required_languages = {"fr", "de", "hi", "th"}
    if target_manifest.get("baseline_gate_status") != "PASS":
        errors.append("The pre-unlearning bilingual competence feasibility gate did not pass")
    if (gate_requirements.get("minimum_matched_cases_per_language") != 15
            or gate_requirements.get("minimum_entity_groups_per_language") != 8
            or gate_requirements.get("all_three_probes") != ["direct_positive", "direct_negative_control", "multi_hop"]
            or len(gate_requirements.get("common_across_full_sft_seeds", [])) != 3):
        errors.append("Target manifest does not record the required three-seed bilingual competence gate thresholds")
    if set(gate_counts) != required_languages or any(
        not isinstance(gate_counts.get(lang), dict)
        or gate_counts[lang].get("bilingual_precompetent_cases", 0) < 15
        or gate_counts[lang].get("entity_groups", 0) < 8
        for lang in required_languages
    ):
        errors.append("Target manifest lacks a passing matched-case/entity-group count for every target language")
    case_metrics_value = target_manifest.get("case_metrics_path")
    case_metrics_path = Path(case_metrics_value) if isinstance(case_metrics_value, str) else Path("__missing_case_metrics__")
    if (not case_metrics_path.is_file()
            or target_manifest.get("case_metrics_sha256") != sha(case_metrics_path)):
        errors.append("Target IDs are not tied to the intact baseline case-metrics file")
    if not target_ids or len(target_ids) != len(target_ids_list):
        errors.append("Target IDs must be a non-empty, unique list")
    artifacts = target_manifest.get("full_sft_artifacts", {})
    target_model_name = None
    adapter_value = sft_run.get("adapter_dir")
    adapter_hash = tree_sha(adapter_value) if adapter_value and Path(adapter_value).is_dir() else None
    if sft_run.get("adapter_sha256") != adapter_hash:
        errors.append("SFT adapter files differ from the hash recorded by the completed SFT run")
    matches = [name for name, artifact in artifacts.items()
               if artifact.get("seed") == sft_run.get("seed")
               and artifact.get("model_id") == profile.model_id
               and artifact.get("model_profile") == profile.manifest()
               and artifact.get("model_revision") == sft_run.get("model_revision")
               and artifact.get("adapter_sha256") == adapter_hash
               and isinstance(artifact.get("checkpoint_sha256"), str)
               and len(artifact["checkpoint_sha256"]) == 64]
    if baseline_model_name:
        if baseline_model_name not in matches:
            errors.append("Named SFT seed/checkpoint does not match the baseline-gate target manifest")
        else:
            target_model_name = baseline_model_name
    elif len(matches) == 1:
        target_model_name = matches[0]
    else:
        target_model_name = None
        errors.append("SFT adapter does not uniquely match a baseline-gate seed; pass --baseline-model-name")
    if manifest.get("corpus_sha256") != sha(corpus_path):
        errors.append("SFT corpus checksum mismatch")
    source_error = source_integrity_error(corpus_path, corpus_manifest_path, manifest)
    if source_error:
        errors.append(source_error)
    source_proof = corpus_path.parent / "source_integrity_validation.json"
    if not source_proof.is_file() or sft_run.get("source_integrity_validation_sha256") != sha(source_proof):
        errors.append("SFT run is not tied to the intact source-integrity validation proof")
    if manifest.get("status") not in SUPPORTED_CORPUS_STATUSES or manifest.get("do_not_train") is not False:
        errors.append("SFT corpus is not explicitly approved for the AI-only feasibility pilot")
    if manifest.get("translation_review_mode") != "AI_ONLY" or manifest.get("human_translation_reviewed") is not False:
        errors.append("SFT corpus review disclosure is missing or inconsistent")
    if manifest.get("translation_review_authorization") != "AI_ONLY_REVIEW_AUTHORIZED":
        errors.append("AI-only translation review was not explicitly authorized")
    approval = manifest.get("split_approval", {})
    project_root = corpus_path.parent.parent if corpus_path.parent.name == "sft_training" else corpus_path.parent
    approval_path = project_root / "split_approval.json"
    if (
        approval.get("status") != "PILOT_SPLIT_PIPELINE_VALIDATED"
        or approval.get("reviewer_type") != "AI_PIPELINE"
        or approval.get("human_approval_claimed") is not False
        or not str(approval.get("reviewer_name", "")).strip()
        or not approval_path.is_file()
        or approval.get("record_sha256") != sha(approval_path)
    ):
        errors.append("SFT corpus lacks an intact AI-pipeline-validated pilot split record")
    if sft_run.get("status") != "TRAINING_COMPLETE" or sft_run.get("training_started") is not True:
        errors.append("A completed SFT adapter is required")
    if (sft_run.get("model_id") != profile.model_id
            or sft_run.get("model_profile") != profile.manifest()):
        errors.append("SFT adapter model/quantization profile does not match the requested experiment profile")
    if not sft_run.get("model_revision"):
        errors.append("SFT run manifest does not record a pinned Hugging Face model commit")
    if sft_run.get("corpus_sha256") != sha(corpus_path):
        errors.append("SFT adapter was not trained from this exact corpus checksum")
    if sft_run.get("corpus_manifest_sha256") != sha(corpus_manifest_path):
        errors.append("SFT adapter was not trained from this exact corpus manifest")
    assistant_masking_path = Path(__file__).resolve().with_name("assistant_masking.py")
    if sft_run.get("assistant_masking_script_sha256") != sha(assistant_masking_path):
        errors.append("SFT run did not record the current shared assistant-mask implementation")
    expected_training_variant = (
        "composition_sft"
        if manifest.get("status") == "AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS"
        else "full_sft"
    )
    if sft_run.get("training_variant") != expected_training_variant:
        errors.append(f"SFT adapter variant must be {expected_training_variant}")
    adapter = Path(sft_run.get("adapter_dir", ""))
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        errors.append("Completed SFT LoRA adapter files are missing")
    if method not in METHODS:
        errors.append("Method must be one of ga, gd, npo")
    rows = read_jsonl(corpus_path)
    en = [r for r in rows if r.get("language") == "en"]
    # The trainer below intentionally consumes only English r1 rows. Other
    # known hop labels are allowed in the five-language SFT corpus because
    # they are filtered out before unlearning. Reject unknown labels so a new
    # data format cannot silently change target selection.
    supported_corpus_hops = {
        "r1", "r2", "composition_2hop", "composition_3hop",
        "composition_support_q1", "composition_support_q1_all",
        "composition_support_q2", "composition_support_q3",
    }
    unsupported_hops = sorted({r.get("hop") for r in en} - supported_corpus_hops)
    if unsupported_hops:
        errors.append("English corpus contains unsupported hop types: " + ", ".join(map(str, unsupported_hops)))
    if any(r.get("split") == "holdout" for r in rows):
        errors.append("Holdout rows are forbidden in SFT/unlearning corpus")
    target_rows = [r for r in en if r.get("hop") == "r1"]
    splits = {s: [r for r in target_rows if r.get("split") == s] for s in ("forget", "retain", "holdout")}
    if not splits["forget"] or not splits["retain"]:
        errors.append("English first-hop forget and retain examples are both required")
    if splits["holdout"]:
        errors.append("First-hop holdout rows are forbidden in unlearning input")
    if len(target_rows) != len(splits["forget"]) + len(splits["retain"]) + len(splits["holdout"]):
        errors.append("English first-hop corpus contains an unexpected split or row")
    available_forget_ids = {r.get("source_qa_id") for r in splits["forget"]}
    if target_ids - available_forget_ids:
        errors.append("Target manifest includes IDs without English first-hop forget rows")
    selected_forget = [r for r in splits["forget"] if r.get("source_qa_id") in target_ids]
    if len(selected_forget) != len(target_ids):
        errors.append("Target ID count does not match English first-hop forget rows")
    unlearning_rows = selected_forget + splits["retain"]
    serialized_target_rows = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unlearning_rows).encode("utf-8")
    target_rows_sha256 = hashlib.sha256(serialized_target_rows).hexdigest()
    package_names = ("torch", "transformers", "peft", "accelerate", "numpy", *profile.required_packages)
    pkgs = {name: importlib.util.find_spec(name) is not None for name in package_names}
    if not all(pkgs.values()):
        errors.append("Missing packages: " + ", ".join(k for k, v in pkgs.items() if not v))
    else:
        from importlib.metadata import version
        expected_stack = dict(LOCKED_TRAINING_STACK)
        if profile.quantization == "4bit_nf4":
            expected_stack.update(LOCKED_QUANTIZATION_STACK)
        actual_stack = {name: version(name) for name in expected_stack}
        if actual_stack != expected_stack:
            errors.append(f"Training stack mismatch; expected {expected_stack}, found {actual_stack}")
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token_present:
        try:
            from huggingface_hub import get_token
            token_present = bool(get_token())
        except (ImportError, OSError):
            pass
    if not token_present:
        errors.append("No Hugging Face token found for gated base model access")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    if not cuda_available:
        errors.append("CUDA GPU is required for the unlearning pilot")
    elif profile.quantization == "4bit_nf4" and method == "npo" and torch.cuda.device_count() < 2:
        errors.append("The 8B QLoRA NPO path requires two CUDA devices so policy and frozen reference weights do not compete for one GPU")
    return {"status": "READY_TO_UNLEARN" if not errors else "BLOCKED", "errors": errors,
            "method": method, "model_profile": profile.manifest(), "model_id": profile.model_id, "language": "en",
            "rows_by_split": {"forget": len(selected_forget), "retain": len(splits["retain"]), "holdout": 0},
            "english_forget_target_ids": sorted(target_ids),
            "english_forget_rows": len(selected_forget),
            "english_retain_rows": len(splits["retain"]),
            "first_hop_training_rows": len(selected_forget) + len(splits["retain"]),
            "non_first_hop_english_rows_ignored": len(en) - len(target_rows),
            "first_hop_rows_sha256": target_rows_sha256, "holdout_rows_used": 0,
            "sft_adapter": str(adapter), "baseline_model_name": target_model_name,
            "model_revision": sft_run.get("model_revision"), "translation_review_mode": manifest.get("translation_review_mode"),
            "target_ids_manifest_sha256": sha(target_ids_manifest_path),
            "target_selection_rule": target_manifest.get("selection_rule"),
            "baseline_gate_status": target_manifest.get("baseline_gate_status"),
            "baseline_gate_thresholds": gate_requirements,
            "baseline_bilingual_counts_by_language": gate_counts,
            "human_translation_reviewed": manifest.get("human_translation_reviewed"),
            "training_packages": pkgs, "hf_token_present": token_present,
            "corpus_sha256": sha(corpus_path)}


def train(corpus_path, corpus_manifest_path, sft_run_manifest_path, target_ids_manifest_path, output_dir, method, epochs, lr, retain_weight, beta, max_length, seed, baseline_model_name=None, model_profile=MAIN_PROFILE):
    # Capture the exact implementation file that this process imports and uses.
    # The path/hash are written to the run manifest so a later bundle can bind
    # adapters and predictions to an auditable source file.
    training_script_path = Path(__file__).resolve()
    training_script_sha256 = sha(training_script_path)
    objective_script_path = training_script_path.with_name("unlearning_objectives.py")
    objective_script_sha256 = sha(objective_script_path)
    assistant_masking_script_path = training_script_path.with_name("assistant_masking.py")
    assistant_masking_script_sha256 = sha(assistant_masking_script_path)
    validate_training_parameters(method, epochs, lr, retain_weight, beta, max_length)
    profile = get_model_profile(model_profile)
    check = preflight(corpus_path, corpus_manifest_path, sft_run_manifest_path, target_ids_manifest_path, method, baseline_model_name, model_profile)
    if check.get("status") != "READY_TO_UNLEARN":
        raise RuntimeError("Unlearning preflight did not authorize training: " + json.dumps(check, ensure_ascii=False))

    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    if profile.quantization != "none":
        from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    run = json.loads(Path(sft_run_manifest_path).read_text(encoding="utf-8"))
    if seed != run.get("seed"):
        raise ValueError("Unlearning seed must match its parent full-SFT seed")
    adapter_dir = Path(run["adapter_dir"])
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=True)
    model_revision = run["model_revision"]
    policy_device = {"": 0} if profile.quantization != "none" else device
    base = AutoModelForCausalLM.from_pretrained(profile.model_id, token=True, revision=model_revision,
                                                **model_load_options(model_profile, torch, device_map=policy_device))
    if profile.quantization != "none":
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    model.train()
    ref = None
    if method == "npo":
        reference_device = {"": 1} if profile.quantization != "none" else device
        ref_base = AutoModelForCausalLM.from_pretrained(profile.model_id, token=True, revision=model_revision,
                                                        **model_load_options(model_profile, torch, device_map=reference_device))
        ref = PeftModel.from_pretrained(ref_base, adapter_dir, is_trainable=False)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    target_manifest = json.loads(Path(target_ids_manifest_path).read_text(encoding="utf-8"))
    target_ids = set(target_manifest["selected_source_qa_ids"])
    rows = first_hop_unlearning_rows(corpus_path)
    forget = [r for r in rows if r["split"] == "forget" and r.get("source_qa_id") in target_ids]
    retain = [r for r in rows if r["split"] == "retain"]
    if not forget or not retain:
        raise RuntimeError("Empty English forget or retain set")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    retain_schedule = (
        retain_schedule_indices(epochs * len(forget), len(retain))
        if method in {"gd", "npo"} else []
    )
    retain_indices_used = sorted(set(retain_schedule))

    def encoded(example, net):
        messages = example["messages"]
        input_ids, labels = encode_unlearning_example(tokenizer, messages, max_length)
        input_device = net.get_input_embeddings().weight.device
        return torch.tensor([input_ids], device=input_device), torch.tensor([labels], device=input_device)

    def forward_stats(net, example, need_logp=False):
        ids, labels = encoded(example, net)
        out = net(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        shift_logits, shift_labels = out.logits[:, :-1].float(), labels[:, 1:]
        token_nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100, reduction="none").reshape_as(shift_labels)
        mask = shift_labels.ne(-100)
        count = mask.sum().clamp_min(1)
        mean_nll = (token_nll * mask).sum() / count
        seq_logp = -(token_nll * mask).sum()
        return mean_nll, seq_logp

    steps = 0
    history = []
    for epoch in range(epochs):
        # Deterministic, alternating forget/retain examples; no holdout examples are sampled.
        for i, frow in enumerate(forget):
            opt.zero_grad(set_to_none=True)
            f_loss, f_logp = forward_stats(model, frow)
            if method == "ga":
                objective = -f_loss
            elif method == "gd":
                rrow = retain[retain_schedule[steps]]
                r_loss, _ = forward_stats(model, rrow)
                objective = -f_loss + retain_weight * r_loss
            else:
                with torch.no_grad():
                    _, ref_logp = forward_stats(ref, frow)
                ref_logp = ref_logp.to(f_logp.device)
                # NPO loss: -(2/beta) * log sigmoid(-beta * log-ratio).
                npo = npo_loss(f_logp, ref_logp, beta)
                rrow = retain[retain_schedule[steps]]
                r_loss, _ = forward_stats(model, rrow)
                objective = npo + retain_weight * r_loss
            objective.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            steps += 1
            history.append(float(objective.detach().cpu()))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    retained_rows_used = [retain[i] for i in retain_indices_used]
    training_rows = forget + retained_rows_used
    subset_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in training_rows).encode("utf-8")
    result = {"status": "UNLEARNING_COMPLETE", "training_started": True, "method": method.upper(),
              "model_profile": profile.manifest(), "model_id": profile.model_id, "model_revision": model_revision, "language": "en", "seed": seed, "epochs": epochs, "optimizer_steps": steps,
              "training_script_path": str(training_script_path),
              "training_script_sha256": training_script_sha256,
              "assistant_masking_script_path": str(assistant_masking_script_path),
              "assistant_masking_script_sha256": assistant_masking_script_sha256,
              "objective_script_path": str(objective_script_path),
              "objective_script_sha256": objective_script_sha256,
              "english_forget_rows": len(forget), "english_retain_rows": len(retain),
              "english_retain_regularization_rows_used": len(retained_rows_used),
              "english_retain_sft_ids_used": sorted(r.get("sft_id", "") for r in retained_rows_used),
              "holdout_rows_used": 0,
              "english_forget_target_ids": sorted(target_ids),
              "target_ids_manifest_sha256": sha(target_ids_manifest_path),
              "target_selection_rule": target_manifest.get("selection_rule"),
              "baseline_case_metrics_sha256": target_manifest.get("case_metrics_sha256"),
              "baseline_checkpoint_sha256": target_manifest.get("full_sft_artifacts", {}).get(check.get("baseline_model_name"), {}).get("checkpoint_sha256"),
              "objective": {"ga": "maximize forget-set answer NLL", "gd": "maximize forget NLL plus retain CE regularizer", "npo": "NPO against frozen SFT reference plus retain CE regularizer"}[method],
              "retain_weight": retain_weight, "beta": beta if method == "npo" else None,
              "sft_adapter": str(adapter_dir), "sft_run_manifest_sha256": sha(sft_run_manifest_path),
              "source_corpus_sha256": sha(corpus_path), "corpus_manifest_sha256": sha(corpus_manifest_path),
              "training_data_sha256": hashlib.sha256(subset_bytes).hexdigest(),
              "training_data_selection": "SFT-precompetent English r1 first-hop forget IDs plus the unique English r1 retain rows actually used for regularization; r2/composition rows ignored",
              "retain_schedule": "round_robin_all_retain_rows" if method in {"gd", "npo"} else "not_used_by_ga",
              "run_mode": "GATE_PASSED_MAIN_STUDY" if profile.role == "main_study" else "GATE_PASSED_PILOT",
              "baseline_gate_status": check.get("baseline_gate_status"),
              "baseline_gate_thresholds": check.get("baseline_gate_thresholds"),
              "baseline_bilingual_counts_by_language": check.get("baseline_bilingual_counts_by_language"),
              "translation_review_mode": "AI_ONLY",
              "human_translation_reviewed": False, "adapter_dir": str(out / "adapter"),
              "adapter_sha256": tree_sha(out / "adapter"),
              "objective_loss_first_last": [history[0], history[-1]] if history else []}
    (out / "run_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--corpus", type=Path, default=root / "sft_training/corpus.jsonl")
    p.add_argument("--corpus-manifest", type=Path, default=root / "sft_training/manifest.json")
    p.add_argument("--sft-run-manifest", type=Path, required=True)
    p.add_argument("--target-ids-manifest", type=Path, required=True,
                   help="Baseline-only case manifest containing English SFT-precompetent forget source IDs")
    p.add_argument("--baseline-model-name", help="Full-SFT model key used in baseline evaluation, e.g. sft_seed13")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", choices=sorted(METHODS), required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--retain-weight", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-profile", choices=(PILOT_PROFILE, MAIN_PROFILE), default=MAIN_PROFILE)
    p.add_argument("--train", action="store_true")
    a = p.parse_args()
    try:
        validate_training_parameters(a.method, a.epochs, a.lr, a.retain_weight, a.beta, a.max_length)
    except ValueError as exc:
        p.error(str(exc))
    result = preflight(a.corpus, a.corpus_manifest, a.sft_run_manifest, a.target_ids_manifest,
                       a.method, a.baseline_model_name, a.model_profile)
    if a.train and result["status"] == "READY_TO_UNLEARN":
        result = train(a.corpus, a.corpus_manifest, a.sft_run_manifest, a.target_ids_manifest,
                       a.output_dir, a.method, a.epochs, a.lr, a.retain_weight, a.beta, a.max_length, a.seed,
                       baseline_model_name=a.baseline_model_name, model_profile=a.model_profile)
    print(json.dumps(result, indent=2))
    if result["status"] in {"BLOCKED", "READY_TO_UNLEARN"} and a.train and result["status"] != "UNLEARNING_COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
