#!/usr/bin/env python3
"""English-only GA, GD, or NPO unlearning on an approved SFT LoRA adapter.

The default is a fail-closed preflight. Training is only started with --train.
Holdout examples are never loaded. Evaluation labels are never read here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
METHODS = {"ga", "gd", "npo"}
SUPPORTED_CORPUS_STATUSES = {
    "AI_ONLY_REVIEWED_FIVE_LANGUAGE_SFT_CORPUS",
    "AI_ONLY_REVIEWED_FIVE_LANGUAGE_COMPOSITION_SFT_CORPUS",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
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


def preflight(corpus_path, corpus_manifest_path, sft_run_manifest_path, target_ids_manifest_path, method):
    corpus_path = Path(corpus_path)
    errors = []
    required = [corpus_path, Path(corpus_manifest_path), Path(sft_run_manifest_path), Path(target_ids_manifest_path)]
    if any(not p.is_file() for p in required):
        return {"status": "BLOCKED", "errors": ["Corpus, corpus manifest, completed SFT run manifest, and baseline-competence target manifest are required"], "method": method}
    manifest = json.loads(Path(corpus_manifest_path).read_text(encoding="utf-8"))
    sft_run = json.loads(Path(sft_run_manifest_path).read_text(encoding="utf-8"))
    target_manifest = json.loads(Path(target_ids_manifest_path).read_text(encoding="utf-8"))
    target_ids_list = target_manifest.get("selected_source_qa_ids", [])
    if not isinstance(target_ids_list, list) or any(not isinstance(value, str) or not value for value in target_ids_list):
        errors.append("Target IDs must be a list of non-empty source QA ID strings")
        target_ids_list = []
    target_ids = set(target_ids_list)
    if target_manifest.get("status") != "SFT_PRECOMPETENT_ENGLISH_FORGET_TARGETS":
        errors.append("Target manifest does not identify SFT-precompetent English forget cases")
    if target_manifest.get("language") != "en" or target_manifest.get("split") != "forget":
        errors.append("Unlearning target selection must be English-only and forget-split only")
    if target_manifest.get("selection_rule") != "SFT direct and full multi-hop answers both correct in English":
        errors.append("Target manifest lacks the required pre-unlearning competence rule")
    gate_counts = target_manifest.get("bilingual_counts_by_language", {})
    gate_requirements = target_manifest.get("pilot_gate_thresholds", {})
    required_languages = {"fr", "de", "hi", "th"}
    if target_manifest.get("baseline_gate_status") != "PASS":
        errors.append("The pre-unlearning bilingual competence feasibility gate did not pass")
    if gate_requirements != {"minimum_matched_cases_per_language": 15, "minimum_entity_groups_per_language": 8}:
        errors.append("Target manifest does not record the required bilingual competence gate thresholds")
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
    if target_manifest.get("source_corpus_sha256") != sha(corpus_path):
        errors.append("Target IDs were not selected against this exact SFT corpus")
    if target_manifest.get("sft_run_manifest_sha256") != sha(sft_run_manifest_path):
        errors.append("Target IDs were not selected from this exact completed SFT run")
    if target_manifest.get("sft_adapter_dir") != sft_run.get("adapter_dir"):
        errors.append("Target IDs name a different SFT adapter")
    if manifest.get("corpus_sha256") != sha(corpus_path):
        errors.append("SFT corpus checksum mismatch")
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
        approval.get("status") != "APPROVED_PILOT_SPLIT"
        or approval.get("reviewer_type") != "AI"
        or not str(approval.get("reviewer_name", "")).strip()
        or not approval_path.is_file()
        or approval.get("record_sha256") != sha(approval_path)
    ):
        errors.append("SFT corpus lacks an intact AI-attributed pilot split approval record")
    if sft_run.get("status") != "TRAINING_COMPLETE" or sft_run.get("training_started") is not True:
        errors.append("A completed SFT adapter is required")
    if sft_run.get("model_id") != MODEL_ID:
        errors.append("SFT adapter model does not match the declared pilot model")
    if sft_run.get("corpus_sha256") != sha(corpus_path):
        errors.append("SFT adapter was not trained from this exact corpus checksum")
    if sft_run.get("corpus_manifest_sha256") != sha(corpus_manifest_path):
        errors.append("SFT adapter was not trained from this exact corpus manifest")
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
    pkgs = {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft", "accelerate", "numpy")}
    if not all(pkgs.values()):
        errors.append("Missing packages: " + ", ".join(k for k, v in pkgs.items() if not v))
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token_present:
        try:
            from huggingface_hub import get_token
            token_present = bool(get_token())
        except (ImportError, OSError):
            pass
    if not token_present:
        errors.append("No Hugging Face token found for gated base model access")
    return {"status": "READY_TO_UNLEARN" if not errors else "BLOCKED", "errors": errors,
            "method": method, "model_id": MODEL_ID, "language": "en",
            "rows_by_split": {"forget": len(selected_forget), "retain": len(splits["retain"]), "holdout": 0},
            "english_forget_target_ids": sorted(target_ids),
            "english_forget_rows": len(selected_forget),
            "english_retain_rows": len(splits["retain"]),
            "first_hop_training_rows": len(selected_forget) + len(splits["retain"]),
            "non_first_hop_english_rows_ignored": len(en) - len(target_rows),
            "first_hop_rows_sha256": target_rows_sha256, "holdout_rows_used": 0,
            "sft_adapter": str(adapter), "translation_review_mode": manifest.get("translation_review_mode"),
            "target_ids_manifest_sha256": sha(target_ids_manifest_path),
            "target_selection_rule": target_manifest.get("selection_rule"),
            "human_translation_reviewed": manifest.get("human_translation_reviewed"),
            "training_packages": pkgs, "hf_token_present": token_present,
            "corpus_sha256": sha(corpus_path)}


def train(corpus_path, corpus_manifest_path, sft_run_manifest_path, target_ids_manifest_path, output_dir, method, epochs, lr, retain_weight, beta, max_length, seed):
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    run = json.loads(Path(sft_run_manifest_path).read_text(encoding="utf-8"))
    adapter_dir = Path(run["adapter_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=True, torch_dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    model.train()
    ref = None
    if method == "npo":
        ref_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=True, torch_dtype=dtype, device_map=device)
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

    def encoded(example):
        messages = example["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if full[:len(prompt)] != prompt or len(full) > max_length:
            raise ValueError("Unsafe chat-template mask or sequence exceeds max_length")
        labels = [-100] * len(prompt) + full[len(prompt):]
        return torch.tensor([full], device=device), torch.tensor([labels], device=device)

    def forward_stats(net, example, need_logp=False):
        ids, labels = encoded(example)
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
                rrow = retain[i % len(retain)]
                r_loss, _ = forward_stats(model, rrow)
                objective = -f_loss + retain_weight * r_loss
            else:
                with torch.no_grad():
                    _, ref_logp = forward_stats(ref, frow)
                # NPO loss: -(2/beta) * log sigmoid(-beta * log-ratio).
                npo = -(2.0 / beta) * F.logsigmoid(-beta * (f_logp - ref_logp))
                rrow = retain[i % len(retain)]
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
    training_rows = forget + retain
    subset_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in training_rows).encode("utf-8")
    result = {"status": "UNLEARNING_COMPLETE", "training_started": True, "method": method.upper(),
              "model_id": MODEL_ID, "language": "en", "seed": seed, "epochs": epochs, "optimizer_steps": steps,
              "english_forget_rows": len(forget), "english_retain_rows": len(retain), "holdout_rows_used": 0,
              "english_forget_target_ids": sorted(target_ids),
              "target_ids_manifest_sha256": sha(target_ids_manifest_path),
              "target_selection_rule": target_manifest.get("selection_rule"),
              "baseline_case_metrics_sha256": target_manifest.get("case_metrics_sha256"),
              "objective": {"ga": "maximize forget-set answer NLL", "gd": "maximize forget NLL plus retain CE regularizer", "npo": "NPO against frozen SFT reference plus retain CE regularizer"}[method],
              "retain_weight": retain_weight, "beta": beta if method == "npo" else None,
              "sft_adapter": str(adapter_dir), "sft_run_manifest_sha256": sha(sft_run_manifest_path),
              "source_corpus_sha256": sha(corpus_path), "corpus_manifest_sha256": sha(corpus_manifest_path),
              "training_data_sha256": hashlib.sha256(subset_bytes).hexdigest(),
              "training_data_selection": "SFT-precompetent English r1 first-hop forget IDs from the baseline target manifest plus all English r1 retain rows; r2/composition rows ignored",
              "translation_review_mode": "AI_ONLY",
              "human_translation_reviewed": False, "adapter_dir": str(out / "adapter"),
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
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", choices=sorted(METHODS), required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--retain-weight", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", action="store_true")
    a = p.parse_args()
    result = preflight(a.corpus, a.corpus_manifest, a.sft_run_manifest, a.target_ids_manifest, a.method)
    if result["status"] == "READY_TO_UNLEARN" and a.train:
        result = train(a.corpus, a.corpus_manifest, a.sft_run_manifest, a.target_ids_manifest,
                       a.output_dir, a.method, a.epochs, a.lr, a.retain_weight, a.beta, a.max_length, a.seed)
    print(json.dumps(result, indent=2))
    if result["status"] in {"BLOCKED", "READY_TO_UNLEARN"} and a.train and result["status"] != "UNLEARNING_COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
