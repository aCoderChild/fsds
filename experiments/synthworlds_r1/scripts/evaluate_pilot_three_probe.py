#!/usr/bin/env python3
"""Leakage-aware, three-probe baseline/effect evaluation for the v5.1 pilot.

Generation never loads gold labels. Scoring is a separate command. Target IDs
are created only from the intersection of three-probe-correct English forget
cases across all full-SFT seeds; the unlearning gate uses matched bilingual
cases and entity groups, not a post-hoc selected seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from model_profiles import MAIN_PROFILE, PILOT_PROFILE, PROFILES, get_model_profile, model_load_options

MODEL_ID = get_model_profile(MAIN_PROFILE).model_id
LOCKED_TRAINING_STACK = {"transformers": "5.22.1", "peft": "0.21.2", "accelerate": "1.15.0"}
LOCKED_QUANTIZATION_STACK = {"bitsandbytes": "0.50.2"}
LANGS = ("en", "fr", "de", "hi", "th")
TARGET_LANGS = LANGS[1:]
PROBES = ("direct_positive", "direct_negative_control", "multi_hop")
MAIN_EFFECT_BOOTSTRAP_MINIMUM = 100000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha(child)))
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join("".join(
        char if char.isalnum() or char == "_" or char.isspace() or unicodedata.category(char).startswith("M") else " "
        for char in text
    ).split())


def invalid_output_reason(answer: str, generation_failed: bool = False) -> str | None:
    if generation_failed:
        return "generation_failed"
    raw = str(answer or "").strip()
    if not raw:
        return "empty"
    if len(raw) > 512 or len(raw.split()) > 64:
        return "overlong"
    if re.search(r"<\|[^>]+\|>|\[/?INST\]|^\s*(assistant|user)\s*:", raw, flags=re.IGNORECASE):
        return "template_artifact"
    tokens = normalize(raw).split()
    for width in range(1, min(5, len(tokens) // 3) + 1):
        for start in range(len(tokens) - 3 * width + 1):
            part = tokens[start:start + width]
            if part == tokens[start + width:start + 2 * width] == tokens[start + 2 * width:start + 3 * width]:
                return "repetitive_ngram"
    compact = "".join(tokens)
    if not any(char.isspace() for char in raw) and not any(char.isdigit() for char in raw) and len(compact) >= 12:
        for width in range(1, min(12, len(compact) // 3) + 1):
            for start in range(len(compact) - 3 * width + 1):
                part = compact[start:start + width]
                if part == compact[start + width:start + 2 * width] == compact[start + 2 * width:start + 3 * width]:
                    return "repetitive_ngram"
    return None


def validate_resume_rows(rows: list[dict], prompts: list[dict], model_key: str, seed: int | None,
                         checkpoint_hash: str, model_revision: str, generation_config: dict,
                         config_hash: str, model_profile: str = MAIN_PROFILE) -> set[tuple[str, str]]:
    profile = get_model_profile(model_profile)
    """Reject stale resumed predictions before reporting completion or loading a model."""
    prompt_by_key = {(row["prompt_id"], row["language"]): row for row in prompts}
    done = set()
    for row in rows:
        if (row.get("checkpoint_sha256") != checkpoint_hash or row.get("model_revision") != model_revision
                or row.get("generation_config_sha256") != config_hash or row.get("generation_config") != generation_config):
            raise ValueError("Existing predictions belong to another checkpoint/config; choose another output file")
        if row.get("model") != model_key:
            continue
        key = (row.get("prompt_id"), row.get("language"))
        prompt = prompt_by_key.get(key)
        if prompt is None:
            raise ValueError(f"Existing prediction file contains a prompt absent from the current source: {key}")
        if (row.get("model_id") != profile.model_id or row.get("model_profile") != profile.manifest()
                or row.get("seed") != seed
                or row.get("prompt_sha256") != hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest()
                or any(row.get(field) != prompt.get(field) for field in ("source_qa_id", "split", "group_id", "probe"))):
            raise ValueError(f"Existing prediction differs from the current prompt, seed, or metadata: {key}")
        if key in done:
            raise ValueError(f"Duplicate pre-existing prediction: {key}")
        done.add(key)
    return done


def generate(args: argparse.Namespace) -> None:
    from importlib.metadata import version
    profile = get_model_profile(args.model_profile)
    expected_stack = dict(LOCKED_TRAINING_STACK)
    if profile.quantization == "4bit_nf4":
        expected_stack.update(LOCKED_QUANTIZATION_STACK)
    actual_stack = {name: version(name) for name in expected_stack}
    if actual_stack != expected_stack:
        raise RuntimeError(f"Generation stack mismatch; expected {expected_stack}, found {actual_stack}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    prompt_path, output_path = Path(args.prompts), Path(args.output)
    prompts = read_jsonl(prompt_path)
    expected = {(r["prompt_id"], r["language"]) for r in prompts}
    if len(expected) != len(prompts) or not prompts:
        raise ValueError("Evaluation prompt IDs/language pairs must be nonempty and unique")
    if any(r.get("language") not in LANGS or r.get("probe") not in PROBES or not r.get("prompt", "").strip() for r in prompts):
        raise ValueError("Evaluation prompt file has an invalid language, probe, or empty prompt")

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    if not args.model_revision:
        raise ValueError("Pin the base model to a Hugging Face commit SHA")
    adapter_hash = tree_sha(adapter_dir) if adapter_dir else None
    checkpoint_identity = {"model_id": profile.model_id, "model_revision": args.model_revision,
                           "adapter_sha256": adapter_hash, "model_profile": profile.manifest()}
    checkpoint_hash = hashlib.sha256(json.dumps(checkpoint_identity, sort_keys=True).encode()).hexdigest()
    generation_config = {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                         "batch_size": args.batch_size, "padding_side": "left"}
    config_hash = hashlib.sha256(json.dumps(generation_config, sort_keys=True).encode()).hexdigest()
    old = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        for row in read_jsonl(output_path):
            key = (row.get("model"), row.get("prompt_id"), row.get("language"))
            if key in old:
                raise ValueError(f"Duplicate pre-existing generation row {key}")
            old[key] = row
    model_key = args.model_name
    done = validate_resume_rows(list(old.values()), prompts, model_key, args.seed, checkpoint_hash,
                                args.model_revision, generation_config, config_hash, args.model_profile)
    if expected <= done:
        print(json.dumps({"status": "ALREADY_COMPLETE", "model": model_key, "predictions": len(done)}))
        return

    token = True  # Hugging Face resolves HF_TOKEN set from Kaggle Secrets by the notebook.
    tokenizer = AutoTokenizer.from_pretrained(profile.model_id, token=token, revision=args.model_revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        profile.model_id, token=token, revision=args.model_revision,
        **model_load_options(args.model_profile, torch,
                             device_map={"": torch.cuda.current_device()} if profile.quantization != "none" else "auto"))
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    device = model.get_input_embeddings().weight.device

    pending = [r for r in prompts if (r["prompt_id"], r["language"]) not in done]
    with output_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            encs = [tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                       tokenize=True, add_generation_prompt=True) for r in batch]
            max_len = max(map(len, encs))
            input_ids = torch.tensor([[tokenizer.pad_token_id] * (max_len - len(ids)) + ids for ids in encs], device=device)
            attention = torch.tensor([[0] * (max_len - len(ids)) + [1] * len(ids) for ids in encs], device=device)
            try:
                with torch.inference_mode():
                    generated = model.generate(input_ids=input_ids, attention_mask=attention,
                        max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.batch_decode(generated[:, max_len:], skip_special_tokens=True)
                errors = [""] * len(batch)
            except Exception as batch_error:
                decoded, errors = [], []
                for row, ids in zip(batch, encs):
                    try:
                        one = torch.tensor([ids], device=device)
                        with torch.inference_mode():
                            gen = model.generate(input_ids=one, attention_mask=torch.ones_like(one),
                                max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
                        decoded.append(tokenizer.decode(gen[0, len(ids):], skip_special_tokens=True)); errors.append("")
                    except Exception as exc:
                        decoded.append(""); errors.append(f"batch={type(batch_error).__name__}; row={type(exc).__name__}: {exc}")
            now = []
            for row, answer, error in zip(batch, decoded, errors):
                record = {
                    "model": model_key, "model_id": profile.model_id, "model_profile": profile.manifest(),
                    "model_revision": args.model_revision,
                    "checkpoint_sha256": checkpoint_hash, "adapter_sha256": adapter_hash,
                    "adapter_dir": str(adapter_dir) if adapter_dir else None,
                    "seed": args.seed, "prompt_id": row["prompt_id"], "source_qa_id": row["source_qa_id"],
                    "split": row["split"], "group_id": row["group_id"], "probe": row["probe"],
                    "language": row["language"], "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    "answer": answer.strip(), "generation_failed": bool(error), "generation_error": error,
                    "generation_config": generation_config, "generation_config_sha256": config_hash,
                }
                now.append(record)
                out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            print(f"{model_key}: {min(start + len(batch), len(pending))}/{len(pending)} remaining generated", flush=True)
    seen = {(r["prompt_id"], r["language"]) for r in read_jsonl(output_path) if r.get("model") == model_key}
    if seen != expected:
        raise RuntimeError(f"Generation incomplete for {model_key}: missing={len(expected-seen)}, extra={len(seen-expected)}")
    print(json.dumps({"status": "GENERATION_COMPLETE", "model": model_key, "checkpoint_sha256": checkpoint_hash,
                      "predictions": len(seen), "prompt_sha256": sha(prompt_path), "output_sha256": sha(output_path)}))


def score(args: argparse.Namespace) -> None:
    predictions = []
    for path in args.predictions:
        predictions.extend(read_jsonl(path))
    prompts = read_jsonl(args.prompts)
    prompt_by_key = {(r["prompt_id"], r["language"]): r for r in prompts}
    prompt_case_ids = {r.get("source_qa_id") for r in prompts}
    if (len(prompt_by_key) != len(prompts) or not prompt_case_ids
            or len(prompts) != len(prompt_case_ids) * len(LANGS) * len(PROBES)):
        raise ValueError("Evaluation prompts must contain three probes for every case in all five languages")
    labels = read_jsonl(args.labels)
    label_by_key = {(r["prompt_id"], r["language"]): r for r in labels}
    if len(label_by_key) != len(labels) or len(labels) != len(prompts):
        raise ValueError("Gold labels must have one unique row for each evaluation prompt/language pair")
    if set(prompt_by_key) != set(label_by_key):
        raise ValueError("Prompt and gold-label keys differ")
    for key, prompt in prompt_by_key.items():
        label = label_by_key[key]
        for field in ("source_qa_id", "split", "group_id", "probe"):
            if prompt.get(field) != label.get(field):
                raise ValueError(f"Prompt/label mismatch for {key}, field {field}")
    models = {r["model"] for r in predictions}
    if not models:
        raise ValueError("No predictions provided")
    seen = set()
    scored = []
    profiles_by_model = {}
    for pred in predictions:
        key = (pred.get("model"), pred.get("prompt_id"), pred.get("language"))
        if key in seen:
            raise ValueError(f"Duplicate prediction {key}")
        seen.add(key)
        label = label_by_key.get((pred.get("prompt_id"), pred.get("language")))
        if label is None:
            raise ValueError(f"Prediction has no label row: {key}")
        prompt = prompt_by_key[(pred["prompt_id"], pred["language"])]
        expected_prompt_hash = hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest()
        if pred.get("prompt_sha256") != expected_prompt_hash:
            raise ValueError(f"Prediction was not generated from the exact evaluation prompt: {key}")
        profile_name = (pred.get("model_profile") or {}).get("name")
        if profile_name is None:
            profile_name = next((name for name, candidate in PROFILES.items()
                                 if candidate.model_id == pred.get("model_id")), None)
        profile = get_model_profile(profile_name) if profile_name else None
        if (profile is None or pred.get("model_id") != profile.model_id
                or pred.get("model_profile") != profile.manifest() or not pred.get("model_revision")
                or not re.fullmatch(r"[0-9a-f]{64}", str(pred.get("checkpoint_sha256", "")))
                or (pred.get("adapter_sha256") is not None and not re.fullmatch(r"[0-9a-f]{64}", str(pred.get("adapter_sha256"))))):
            raise ValueError(f"Prediction lacks a pinned model/checkpoint identity: {key}")
        profiles_by_model[pred["model"]] = profile.manifest()
        identity = hashlib.sha256(json.dumps({"model_id": pred["model_id"],
            "model_revision": pred["model_revision"], "adapter_sha256": pred.get("adapter_sha256"),
            "model_profile": profile.manifest()},
            sort_keys=True).encode()).hexdigest()
        if pred["checkpoint_sha256"] != identity:
            raise ValueError(f"Prediction checkpoint fingerprint is inconsistent: {key}")
        for field in ("source_qa_id", "split", "group_id", "probe"):
            if pred.get(field) != label.get(field):
                raise ValueError(f"Prediction/label mismatch for {key}, field {field}")
        answers = label.get("gold_answers")
        if not isinstance(answers, list) or not answers or any(not isinstance(x, str) for x in answers):
            raise ValueError(f"Invalid gold answer set for {key}")
        if not isinstance(pred.get("answer"), str) or type(pred.get("generation_failed")) is not bool:
            raise ValueError(f"Prediction has an invalid answer or generation-failure flag: {key}")
        correct = not pred["generation_failed"] and any(normalize(pred["answer"]) == normalize(gold) for gold in answers)
        invalid = invalid_output_reason(pred["answer"], pred["generation_failed"])
        scored.append({**pred, "correct": correct, "gold_answers": answers,
                       "output_invalid": invalid is not None, "invalid_output_reason": invalid})
    expected = set(label_by_key)
    for model in models:
        actual = {(r["prompt_id"], r["language"]) for r in scored if r["model"] == model}
        if actual != expected:
            raise ValueError(f"{model} incomplete evaluation: missing={len(expected-actual)}, extra={len(actual-expected)}")
        model_rows = [r for r in scored if r["model"] == model]
        identities = {(r["model_id"], json.dumps(r["model_profile"], sort_keys=True), r["model_revision"],
                       r["checkpoint_sha256"], r.get("adapter_sha256"), r.get("seed")) for r in model_rows}
        configs = {r.get("generation_config_sha256") for r in model_rows}
        if len(identities) != 1 or len(configs) != 1 or None in configs:
            raise ValueError(f"{model} predictions mix checkpoint, seed, or generation settings")

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "scored_predictions.jsonl", scored)
    groups = defaultdict(dict)
    for row in scored:
        groups[(row["model"], row["source_qa_id"], row["language"])][row["probe"]] = row
    if any(set(probes) != set(PROBES) for probes in groups.values()):
        raise ValueError("Each model/case/language must have all three probes")
    case_rows = []
    for (model, case_id, lang), probes in sorted(groups.items()):
        p = next(iter(probes.values()))
        case_rows.append({"model": model, "seed": p.get("seed"), "model_id": p["model_id"],
                          "model_profile": p["model_profile"], "model_revision": p["model_revision"],
                          "checkpoint_sha256": p["checkpoint_sha256"], "adapter_sha256": p.get("adapter_sha256"),
                          "source_qa_id": case_id, "split": p["split"], "group_id": p["group_id"],
                          "language": lang, **{f"{probe}_correct": probes[probe]["correct"] for probe in PROBES},
                          **{f"{probe}_invalid": probes[probe]["output_invalid"] for probe in PROBES}})
    write_jsonl(output / "case_metrics.jsonl", case_rows)
    summary = []
    for model in sorted(models):
        for lang in LANGS:
            for split in ("forget", "retain", "holdout"):
                for probe in PROBES:
                    rows = [r for r in scored if r["model"] == model and r["language"] == lang and r["split"] == split and r["probe"] == probe]
                    if rows:
                        summary.append({"model": model, "language": lang, "split": split, "probe": probe,
                            "n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
                            "correct": sum(r["correct"] for r in rows),
                            "generation_failures": sum(bool(r.get("generation_failed")) for r in rows),
                            "output_invalid": sum(bool(r["output_invalid"]) for r in rows)})
    write_json(output / "summary.json", {"status": "SCORED_COMPLETE", "prompts_sha256": sha(Path(args.prompts)),
        "labels_sha256": sha(Path(args.labels)),
        "prediction_files": {str(Path(p)): sha(Path(p)) for p in args.predictions},
        "translation_review_mode": "AI_ONLY", "human_translation_reviewed": False,
        "model_count": len(models), "models": sorted(models), "model_profiles": profiles_by_model,
        "rows": len(scored)})
    write_jsonl(output / "summary.jsonl", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["model", "language", "split", "probe"])
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps({"status": "SCORED_COMPLETE", "models": sorted(models), "prediction_rows": len(scored),
                      "summary": str(output / 'summary.csv')}))


def checked_case_metadata(rows: list[dict]) -> dict[str, set[tuple[str, str]]]:
    """Require one split and leakage group per case across all models/languages."""
    metadata = defaultdict(set)
    for row in rows:
        case_id, split, group = row.get("source_qa_id"), row.get("split"), row.get("group_id")
        if (not isinstance(case_id, str) or not case_id or split not in {"forget", "retain", "holdout"}
                or not isinstance(group, str) or not group):
            raise ValueError("Case metrics have a missing case ID, split, or leakage group")
        metadata[case_id].add((split, group))
    if any(len(values) != 1 for values in metadata.values()):
        raise ValueError("Case metrics disagree on split or leakage group across models/languages")
    return metadata


def baseline_gate(args: argparse.Namespace) -> None:
    case_rows = read_jsonl(args.case_metrics)
    checked_case_metadata(case_rows)
    models = tuple(args.full_sft_models)
    if args.minimum_cases != 15 or args.minimum_groups != 8:
        raise ValueError("The R1 pilot feasibility gate is fixed at 15 cases and 8 entity groups per target language")
    if len(models) != 3 or len(models) != len(set(models)):
        raise ValueError("Provide exactly three distinct full-SFT seed model names")
    all_models = {r["model"] for r in case_rows}
    if not set(models) <= all_models:
        raise ValueError("Case metrics are missing one or more named full-SFT seed models")
    expected_case_lang = {(r["source_qa_id"], r["language"]) for r in case_rows if r["model"] == models[0]}
    case_ids_for_eval = {case_id for case_id, _ in expected_case_lang}
    eval_languages = {lang for _, lang in expected_case_lang}
    if (len(case_ids_for_eval) != args.expected_cases or eval_languages != set(LANGS)
            or len(expected_case_lang) != args.expected_cases * len(LANGS)):
        raise ValueError(f"Baseline case metrics must cover all {args.expected_cases} frozen evaluation cases in all five languages")
    full_sft_identities = {}
    for model in models:
        actual_case_lang = {(r["source_qa_id"], r["language"]) for r in case_rows if r["model"] == model}
        if actual_case_lang != expected_case_lang or len(actual_case_lang) != sum(r["model"] == model for r in case_rows):
            raise ValueError(f"Case metrics coverage is incomplete or duplicated for {model}")
        model_rows = [r for r in case_rows if r["model"] == model]
        identities = {(r.get("seed"), r.get("model_id"), json.dumps(r.get("model_profile"), sort_keys=True),
                       r.get("model_revision"), r.get("checkpoint_sha256"), r.get("adapter_sha256"))
                      for r in model_rows}
        if len(identities) != 1 or any(value is None or value == "" for value in next(iter(identities))):
            raise ValueError(f"{model} case metrics mix or omit seed/checkpoint identity")
        full_sft_identities[model] = next(iter(identities))
    seeds = [identity[0] for identity in full_sft_identities.values()]
    model_ids = {identity[1] for identity in full_sft_identities.values()}
    profile_json = {identity[2] for identity in full_sft_identities.values()}
    revisions = {identity[3] for identity in full_sft_identities.values()}
    if len(set(seeds)) != 3:
        raise ValueError("Full-SFT seed model names must correspond to three distinct random seeds")
    if len(revisions) != 1:
        raise ValueError("Full-SFT baseline seeds must use the same pinned base-model revision")
    if len(model_ids) != 1 or len(profile_json) != 1:
        raise ValueError("Full-SFT baseline seeds must use one exact model and quantization profile")
    model_profile = json.loads(next(iter(profile_json)))
    profile = get_model_profile(model_profile.get("name"))
    if profile.manifest() != model_profile or profile.model_id != next(iter(model_ids)):
        raise ValueError("Full-SFT baseline metrics use an unsupported or inconsistent model profile")
    by_key = {(r["model"], r["source_qa_id"], r["language"]): r for r in case_rows}
    forget_ids = sorted({r["source_qa_id"] for r in case_rows if r["split"] == "forget"})
    eligible_by_lang = {}
    groups_by_lang = {}
    for lang in TARGET_LANGS:
        eligible = []
        for case_id in forget_ids:
            ok = True
            for model in models:
                for language in ("en", lang):
                    row = by_key.get((model, case_id, language))
                    if row is None or not all(row.get(f"{probe}_correct") is True for probe in PROBES):
                        ok = False; break
                if not ok: break
            if ok:
                eligible.append(case_id)
        eligible_by_lang[lang] = eligible
        groups_by_lang[lang] = sorted({by_key[(models[0], case_id, lang)]["group_id"] for case_id in eligible})
    english_eligible = []
    for case_id in forget_ids:
        if all(by_key[(model, case_id, "en")].get(f"{probe}_correct") is True for model in models for probe in PROBES):
            english_eligible.append(case_id)
    counts = {lang: {"bilingual_precompetent_cases": len(eligible_by_lang[lang]),
                     "entity_groups": len(groups_by_lang[lang]), "source_qa_ids": eligible_by_lang[lang]}
              for lang in TARGET_LANGS}
    passed = all(counts[lang]["bilingual_precompetent_cases"] >= args.minimum_cases and counts[lang]["entity_groups"] >= args.minimum_groups for lang in TARGET_LANGS)
    target_manifest = {
        "status": "SFT_PRECOMPETENT_ENGLISH_FORGET_TARGETS" if passed else "BASELINE_GATE_FAIL",
        "baseline_gate_status": "PASS" if passed else "FAIL", "do_not_unlearn": not passed,
        "language": "en", "split": "forget", "selection_rule": "English direct_positive, direct_negative_control, and multi_hop all correct in every full-SFT seed",
        "model_id": profile.model_id, "model_profile": profile.manifest(),
        "selected_source_qa_ids": english_eligible,
        "selected_source_qa_ids_sha256": hashlib.sha256("\n".join(english_eligible).encode()).hexdigest(),
        "bilingual_counts_by_language": {lang: {"bilingual_precompetent_cases": counts[lang]["bilingual_precompetent_cases"], "entity_groups": counts[lang]["entity_groups"]} for lang in TARGET_LANGS},
        "bilingual_precompetent_source_qa_ids_by_language": eligible_by_lang,
        "english_precompetent_cases": len(english_eligible), "english_precompetent_entity_groups": len({by_key[(models[0], cid, "en")]["group_id"] for cid in english_eligible}),
        "expected_cases": args.expected_cases,
        "pilot_gate_thresholds": {"minimum_matched_cases_per_language": args.minimum_cases, "minimum_entity_groups_per_language": args.minimum_groups,
                                  "all_three_probes": list(PROBES), "common_across_full_sft_seeds": list(models)},
        "full_sft_models": list(models), "case_metrics_path": str(Path(args.case_metrics).resolve()),
        "case_metrics_sha256": sha(Path(args.case_metrics)), "human_translation_reviewed": False,
        "full_sft_artifacts": {model: {"seed": seed, "model_id": model_id,
                                        "model_profile": json.loads(profile_value), "model_revision": revision,
                                        "checkpoint_sha256": checkpoint, "adapter_sha256": adapter}
                                for model, (seed, model_id, profile_value, revision, checkpoint, adapter) in full_sft_identities.items()},
        "status_note": "Feasibility gate only, not a power analysis or human translation validation.",
    }
    write_json(args.output, target_manifest)
    print(json.dumps({"baseline_gate_status": target_manifest["baseline_gate_status"],
                      "english_precompetent_cases": len(english_eligible),
                      "bilingual_counts_by_language": target_manifest["bilingual_counts_by_language"],
                      "target_ids_manifest": str(args.output)}, indent=2))


def verify_baseline_target_manifest(target: dict) -> list[dict]:
    """Recompute the baseline gate from its hash-bound metrics before reuse."""
    metrics_value = target.get("case_metrics_path")
    metrics_path = Path(metrics_value) if isinstance(metrics_value, str) else Path("__missing_case_metrics__")
    if not metrics_path.is_file() or target.get("case_metrics_sha256") != sha(metrics_path):
        raise ValueError("Target selection is not bound to its intact baseline case-metrics file")
    if target.get("expected_cases", 0) <= 0:
        raise ValueError("Target manifest lacks the frozen expected evaluation-case count")
    thresholds = target.get("pilot_gate_thresholds", {})
    if (thresholds.get("minimum_matched_cases_per_language") != 15
            or thresholds.get("minimum_entity_groups_per_language") != 8
            or thresholds.get("all_three_probes") != list(PROBES)):
        raise ValueError("Target manifest does not use the fixed R1 feasibility thresholds")
    models = tuple(target.get("full_sft_models", []))
    if len(models) != 3 or len(set(models)) != 3 or thresholds.get("common_across_full_sft_seeds") != list(models):
        raise ValueError("Target manifest must identify the same three distinct full-SFT seeds in its gate")

    rows = read_jsonl(metrics_path)
    by_key = {}
    for row in rows:
        key = (row.get("model"), row.get("source_qa_id"), row.get("language"))
        if None in key or key in by_key:
            raise ValueError("Baseline case metrics have missing or duplicate model/case/language keys")
        by_key[key] = row

    first_model_rows = [r for r in rows if r["model"] == models[0]]
    expected_case_lang = {(r["source_qa_id"], r["language"]) for r in first_model_rows}
    case_ids = {case_id for case_id, _ in expected_case_lang}
    if (len(case_ids) != target["expected_cases"] or {lang for _, lang in expected_case_lang} != set(LANGS)
            or len(expected_case_lang) != target["expected_cases"] * len(LANGS)):
        raise ValueError("Bound baseline metrics do not cover the declared frozen cases in all five languages")

    identities = {}
    case_metadata = checked_case_metadata(rows)
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        actual = {(r["source_qa_id"], r["language"]) for r in model_rows}
        if actual != expected_case_lang or len(model_rows) != len(expected_case_lang):
            raise ValueError(f"Bound baseline metrics have incomplete or duplicate coverage for {model}")
        model_identities = {(r.get("seed"), r.get("model_id"), json.dumps(r.get("model_profile"), sort_keys=True),
                             r.get("model_revision"), r.get("checkpoint_sha256"), r.get("adapter_sha256"))
                            for r in model_rows}
        if len(model_identities) != 1 or any(value is None or value == "" for value in next(iter(model_identities))):
            raise ValueError(f"Bound baseline metrics mix or omit checkpoint identity for {model}")
        identities[model] = next(iter(model_identities))
    if (len({identity[0] for identity in identities.values()}) != 3
            or len({identity[1] for identity in identities.values()}) != 1
            or len({identity[2] for identity in identities.values()}) != 1
            or len({identity[3] for identity in identities.values()}) != 1):
        raise ValueError("Bound full-SFT metrics do not represent three distinct seeds at one model/profile/revision")
    first_identity = next(iter(identities.values()))
    model_profile = json.loads(first_identity[2])
    profile = get_model_profile(model_profile.get("name"))
    if target.get("model_profile") != profile.manifest() or target.get("model_id") != profile.model_id or first_identity[1] != profile.model_id:
        raise ValueError("Baseline target manifest model/quantization profile differs from bound case metrics")

    forget_ids = sorted(case_id for case_id in case_ids if next(iter(case_metadata[case_id]))[0] == "forget")
    bilingual_ids = {}
    bilingual_counts = {}
    for lang in TARGET_LANGS:
        eligible = []
        for case_id in forget_ids:
            if all(
                all(by_key[(model, case_id, language)].get(f"{probe}_correct") is True for probe in PROBES)
                for model in models for language in ("en", lang)
            ):
                eligible.append(case_id)
        groups = sorted({by_key[(models[0], case_id, lang)]["group_id"] for case_id in eligible})
        bilingual_ids[lang] = eligible
        bilingual_counts[lang] = {"bilingual_precompetent_cases": len(eligible), "entity_groups": len(groups)}
    english_ids = sorted(
        case_id for case_id in forget_ids
        if all(by_key[(model, case_id, "en")].get(f"{probe}_correct") is True for model in models for probe in PROBES)
    )
    english_groups = {by_key[(models[0], case_id, "en")]["group_id"] for case_id in english_ids}
    passed = all(v["bilingual_precompetent_cases"] >= 15 and v["entity_groups"] >= 8 for v in bilingual_counts.values())
    expected = {
        "status": "SFT_PRECOMPETENT_ENGLISH_FORGET_TARGETS" if passed else "BASELINE_GATE_FAIL",
        "baseline_gate_status": "PASS" if passed else "FAIL",
        "do_not_unlearn": not passed,
        "language": "en", "split": "forget",
        "selection_rule": "English direct_positive, direct_negative_control, and multi_hop all correct in every full-SFT seed",
        "selected_source_qa_ids": english_ids,
        "selected_source_qa_ids_sha256": hashlib.sha256("\n".join(english_ids).encode()).hexdigest(),
        "bilingual_counts_by_language": bilingual_counts,
        "bilingual_precompetent_source_qa_ids_by_language": bilingual_ids,
        "english_precompetent_cases": len(english_ids),
        "english_precompetent_entity_groups": len(english_groups),
        "model_id": profile.model_id,
        "model_profile": profile.manifest(),
        "full_sft_artifacts": {model: {"seed": seed, "model_id": model_id,
                                      "model_profile": json.loads(profile_value), "model_revision": revision,
                                      "checkpoint_sha256": checkpoint, "adapter_sha256": adapter}
                                for model, (seed, model_id, profile_value, revision, checkpoint, adapter) in identities.items()},
    }
    for field, value in expected.items():
        if target.get(field) != value:
            raise ValueError(f"Target manifest field {field!r} does not match the recomputed baseline gate")
    return rows


def bootstrap_mean(cluster_values: dict[str, list[float]], seed: int, n_boot: int,
                   ci_alpha: float = 0.05) -> tuple[float | None, list[float] | None]:
    if not 0 < ci_alpha < 1:
        raise ValueError("ci_alpha must be strictly between zero and one")
    clusters = sorted(cluster_values)
    values = [x for group in clusters for x in cluster_values[group]]
    if not values:
        return None, None
    point = sum(values) / len(values)
    if len(clusters) == 1:
        return point, None
    rng = random.Random(seed); estimates = []
    for _ in range(n_boot):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sample = [x for cluster in sampled for x in cluster_values[cluster]]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    lower = int((ci_alpha / 2) * (len(estimates) - 1))
    upper = int((1 - ci_alpha / 2) * (len(estimates) - 1))
    return point, [estimates[lower], estimates[upper]]


def bootstrap_ratio(cluster_counts: dict[str, tuple[int, int]], seed: int, n_boot: int) -> tuple[float | None, list[float] | None]:
    """Cluster-bootstrap a case-level proportion represented as numerator/denominator counts."""
    clusters = sorted(cluster_counts)
    numerator = sum(cluster_counts[c][0] for c in clusters)
    denominator = sum(cluster_counts[c][1] for c in clusters)
    if not denominator:
        return None, None
    point = numerator / denominator
    if len(clusters) == 1:
        return point, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        den = sum(cluster_counts[c][1] for c in sample)
        if den:
            estimates.append(sum(cluster_counts[c][0] for c in sample) / den)
    if not estimates:
        return point, None
    estimates.sort()
    return point, [estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]]


def bootstrap_seed_ratio(seed_cluster_counts: dict[int, dict[str, tuple[int, int]]],
                         seed: int, n_boot: int) -> tuple[float | None, list[float] | None]:
    """Bootstrap the unweighted mean of defined per-seed ratios by entity group.

    A resample with a zero denominator for any originally defined seed makes
    the interval undefined; silently discarding such draws would condition it.
    """
    active = {key: groups for key, groups in seed_cluster_counts.items()
              if sum(denominator for _, denominator in groups.values()) > 0}
    if not active:
        return None, None
    ratios = [sum(n for n, _ in groups.values()) / sum(d for _, d in groups.values())
              for groups in active.values()]
    point = sum(ratios) / len(ratios)
    clusters = sorted({group for groups in active.values() for group in groups})
    if len(clusters) < 2:
        return point, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_boot):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sampled_ratios = []
        for groups in active.values():
            numerator = sum(groups.get(cluster, (0, 0))[0] for cluster in sampled)
            denominator = sum(groups.get(cluster, (0, 0))[1] for cluster in sampled)
            if denominator == 0:
                return point, None
            sampled_ratios.append(numerator / denominator)
        estimates.append(sum(sampled_ratios) / len(sampled_ratios))
    estimates.sort()
    return point, [estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]]


def mean_sd(values: list[float | None]) -> tuple[float | None, float | None]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else None
    return mean, sd


def effects(args: argparse.Namespace) -> None:
    if args.bootstrap < 2:
        raise ValueError("At least two bootstrap replicates are required")
    rows = read_jsonl(args.case_metrics)
    target = read_json(args.target_manifest)
    baseline_rows = verify_baseline_target_manifest(target)
    if target.get("baseline_gate_status") != "PASS" or target.get("do_not_unlearn") is not False:
        raise ValueError("Effects can only be analyzed after a passing baseline competence gate")
    pairs = read_json(args.model_pairs)
    by_key = {(r["model"], r["source_qa_id"], r["language"]): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError("Effects case metrics contain duplicate model/case/language rows")
    identities_by_model = {}
    for model in {r["model"] for r in rows}:
        model_rows = [r for r in rows if r["model"] == model]
        identities = {(r.get("seed"), r.get("model_id"), json.dumps(r.get("model_profile"), sort_keys=True),
                       r.get("model_revision"), r.get("checkpoint_sha256"), r.get("adapter_sha256"))
                      for r in model_rows}
        if len(identities) != 1:
            raise ValueError(f"Case metrics mix checkpoint identity for {model}")
        identities_by_model[model] = next(iter(identities))
    full_models = set(target.get("full_sft_models", []))
    if len(full_models) != 3:
        raise ValueError("The effects analysis requires the three-seed full-SFT baseline")
    frozen_baseline = {
        (r["model"], r["source_qa_id"], r["language"]): r
        for r in baseline_rows if r["model"] in full_models
    }
    current_baseline = {
        (r["model"], r["source_qa_id"], r["language"]): r
        for r in rows if r["model"] in full_models
    }
    if frozen_baseline != current_baseline:
        raise ValueError("Effects metrics differ from the hash-bound full-SFT baseline metrics used to select targets")
    for model, artifact in target["full_sft_artifacts"].items():
        if identities_by_model.get(model) != (
                artifact["seed"], artifact["model_id"], json.dumps(artifact["model_profile"], sort_keys=True),
                artifact["model_revision"], artifact["checkpoint_sha256"], artifact["adapter_sha256"]):
            raise ValueError(f"Effects metrics do not contain the hash-bound baseline checkpoint for {model}")
    methods = ("ga", "gd", "npo")
    if any(pair.get("method") not in methods for pair in pairs):
        raise ValueError("Model pairs may contain only GA, GD, and NPO unlearning runs")
    for method in methods:
        method_pairs = [pair for pair in pairs if pair.get("method") == method]
        refs = [pair.get("reference") for pair in method_pairs]
        if len(method_pairs) != 3 or set(refs) != full_models or len(set(refs)) != 3:
            raise ValueError(f"{method}: effects need one post-unlearning checkpoint for each of the three SFT seeds")
    post_models = [pair.get("post") for pair in pairs]
    if any(not isinstance(model, str) or not model for model in post_models) or len(set(post_models)) != len(post_models) or full_models.intersection(post_models):
        raise ValueError("Each method/seed pair must identify a distinct post-unlearning model")
    out = []
    case_delta_by_seed = defaultdict(dict)
    dap_counts_by_seed = defaultdict(dict)
    joint_outcomes_by_seed = defaultdict(dict)
    for pair in pairs:
        ref_model, post_model, method = pair["reference"], pair["post"], pair["method"]
        if method in {"ga", "gd", "npo"}:
            ref_identity = identities_by_model.get(ref_model)
            post_identity = identities_by_model.get(post_model)
            if (ref_model not in target.get("full_sft_models", []) or ref_identity is None or post_identity is None
                    or ref_identity[0] is None or post_identity[0] != ref_identity[0]
                    or post_identity[1] != ref_identity[1]
                    or post_identity[2] != ref_identity[2]
                    or post_identity[3] != ref_identity[3]
                    or ref_identity[4] == post_identity[4]
                    or not ref_identity[5] or not post_identity[5]):
                raise ValueError(f"{method}: post-unlearning model must be paired to its own full-SFT seed")
        for model in (ref_model, post_model):
            if not any(r["model"] == model for r in rows):
                raise ValueError(f"Model-pair artifact references absent model {model}")
        ref_cases = {(r["source_qa_id"], r["language"]): r for r in rows if r["model"] == ref_model}
        post_cases = {(r["source_qa_id"], r["language"]): r for r in rows if r["model"] == post_model}
        if set(ref_cases) != set(post_cases):
            raise ValueError(f"{post_model}: post-unlearning case/language coverage differs from its baseline")
        if any(ref_cases[key].get(field) != post_cases[key].get(field)
               for key in ref_cases for field in ("split", "group_id")):
            raise ValueError(f"{post_model}: post-unlearning case metadata differs from its baseline")
        for lang in LANGS:
            lang_ids = set(target["bilingual_precompetent_source_qa_ids_by_language"].get(lang, [])) if lang in TARGET_LANGS else set(target["selected_source_qa_ids"])
            for split in ("forget", "retain", "holdout"):
                split_ids = lang_ids if split == "forget" else {r["source_qa_id"] for r in rows if r["split"] == split}
                sub = []
                for case_id in sorted(split_ids):
                    before = by_key.get((ref_model, case_id, lang)); after = by_key.get((post_model, case_id, lang))
                    if before and after and before["split"] == split and after["split"] == split:
                        sub.append((case_id, before, after))
                if not sub:
                    continue
                for probe in PROBES:
                    delta = [int(a[f"{probe}_correct"]) - int(b[f"{probe}_correct"]) for _, b, a in sub]
                    pre_acc = sum(int(b[f"{probe}_correct"]) for _, b, _ in sub) / len(sub)
                    post_acc = sum(int(a[f"{probe}_correct"]) for _, _, a in sub) / len(sub)
                    pre_invalid = sum(int(b[f"{probe}_invalid"]) for _, b, _ in sub) / len(sub)
                    post_invalid = sum(int(a[f"{probe}_invalid"]) for _, _, a in sub) / len(sub)
                    clusters = defaultdict(list)
                    for (cid, b, a), value in zip(sub, delta):
                        clusters[b["group_id"]].append(value)
                        case_delta_by_seed[(method, lang, split, probe)].setdefault(post_identity[0], {})[cid] = {
                            "group_id": b["group_id"], "delta": value,
                        }
                    mean, ci = bootstrap_mean(clusters, int(hashlib.sha256(f"{post_model}|{lang}|{split}|{probe}".encode()).hexdigest()[:8], 16), args.bootstrap)
                    direct_forgetting_ci = (
                        [-100 * ci[1], -100 * ci[0]] if probe == "direct_positive" and ci is not None else None
                    )
                    out.append({"method": method, "reference_model": ref_model, "post_model": post_model, "seed": post_identity[0],
                                "language": lang, "split": split, "probe": probe, "n_cases": len(sub),
                                "entity_groups": len(clusters), "delta_accuracy_pp": 100 * mean if mean is not None else None,
                                "pre_accuracy_pp": 100 * pre_acc, "post_accuracy_pp": 100 * post_acc,
                                "pre_invalid_pct": 100 * pre_invalid, "post_invalid_pct": 100 * post_invalid,
                                "direct_forgetting_pp": (-100 * mean if probe == "direct_positive" and mean is not None else None),
                                "direct_forgetting_cluster_bootstrap_95ci_pp": direct_forgetting_ci,
                                "cluster_bootstrap_95ci_pp": [100 * x for x in ci] if ci else None})
                if split == "forget":
                    joint_outcomes_by_seed[(method, lang)][post_identity[0]] = {
                        case_id: {
                            "group_id": before["group_id"],
                            "b_pass": bool(after["direct_positive_correct"]),
                            "c_pass": bool(after["multi_hop_correct"]),
                        }
                        for case_id, before, after in sub
                    }
                    denom = [(cid, b, a) for cid, b, a in sub if b["direct_positive_correct"] and b["multi_hop_correct"] and not a["direct_positive_correct"]]
                    dap_clusters = defaultdict(lambda: [0, 0])
                    seed_dap = dap_counts_by_seed[(method, lang)].setdefault(post_identity[0], {})
                    for _, before, after in denom:
                        dap_clusters[before["group_id"]][0] += int(after["multi_hop_correct"])
                        dap_clusters[before["group_id"]][1] += 1
                        seed_dap[before["source_qa_id"]] = {
                            "group_id": before["group_id"], "numerator": int(after["multi_hop_correct"]), "denominator": 1,
                        }
                    case_delta_by_seed[(method, lang, split, "downstream_answer_persistence")][post_identity[0]] = {
                        case_id: dict(value) for case_id, value in seed_dap.items()
                    }
                    dap, dap_ci = bootstrap_ratio(
                        {group: tuple(counts) for group, counts in dap_clusters.items()},
                        int(hashlib.sha256(f"{post_model}|{lang}|dap".encode()).hexdigest()[:8], 16), args.bootstrap,
                    )
                    out.append({"method": method, "reference_model": ref_model, "post_model": post_model, "seed": post_identity[0],
                                "language": lang, "split": split, "probe": "downstream_answer_persistence",
                                "n_cases": len(denom), "entity_groups": len(dap_clusters),
                                "delta_accuracy_pp": None, "pre_accuracy_pp": None, "post_accuracy_pp": None,
                                "dap_persistence_pct": 100 * dap if dap is not None else None,
                                "cluster_bootstrap_95ci_pp": [100 * x for x in dap_ci] if dap_ci else None})
        # Paired language-minus-English difference-in-differences on the common
        # precompetent set. Each entity group is resampled as a cluster.
        for lang in TARGET_LANGS:
            case_ids = target["bilingual_precompetent_source_qa_ids_by_language"][lang]
            cluster_values = defaultdict(list)
            cluster_english_values = defaultdict(list)
            cluster_target_values = defaultdict(list)
            case_did = {}
            for case_id in case_ids:
                values = [by_key.get((model, case_id, language)) for model, language in ((ref_model, "en"), (post_model, "en"), (ref_model, lang), (post_model, lang))]
                if any(v is None for v in values):
                    raise ValueError(f"Missing paired SFT/post metrics for {post_model}/{lang}/{case_id}")
                ref_en, post_en, ref_lang, post_lang = values
                if any(r["split"] != "forget" for r in values):
                    raise ValueError("Bilingual contrast includes a non-forget case")
                english_delta = int(post_en["multi_hop_correct"]) - int(ref_en["multi_hop_correct"])
                target_delta = int(post_lang["multi_hop_correct"]) - int(ref_lang["multi_hop_correct"])
                did = target_delta - english_delta
                cluster_values[ref_en["group_id"]].append(did)
                cluster_english_values[ref_en["group_id"]].append(english_delta)
                cluster_target_values[ref_en["group_id"]].append(target_delta)
                case_did[case_id] = {"group_id": ref_en["group_id"], "delta": did,
                                     "english_delta": english_delta, "target_delta": target_delta}
            case_delta_by_seed[(method, lang, "forget", "multi_hop_language_minus_english_DID")][post_identity[0]] = case_did
            mean, ci = bootstrap_mean(cluster_values, int(hashlib.sha256(f"{post_model}|{lang}|did".encode()).hexdigest()[:8], 16), args.bootstrap)
            english_mean, english_ci = bootstrap_mean(
                cluster_english_values,
                int(hashlib.sha256(f"{post_model}|{lang}|matched_english_change".encode()).hexdigest()[:8], 16),
                args.bootstrap,
            )
            target_mean, target_ci = bootstrap_mean(
                cluster_target_values,
                int(hashlib.sha256(f"{post_model}|{lang}|matched_target_change".encode()).hexdigest()[:8], 16),
                args.bootstrap,
            )
            out.append({"method": method, "reference_model": ref_model, "post_model": post_model, "seed": post_identity[0],
                        "language": lang, "split": "forget", "probe": "multi_hop_language_minus_english_DID",
                        "n_cases": sum(map(len, cluster_values.values())), "entity_groups": len(cluster_values),
                        "delta_accuracy_pp": 100 * mean if mean is not None else None,
                        "cluster_bootstrap_95ci_pp": [100 * x for x in ci] if ci else None,
                        "matched_english_delta_accuracy_pp": 100 * english_mean if english_mean is not None else None,
                        "matched_english_cluster_bootstrap_95ci_pp": [100 * x for x in english_ci] if english_ci else None,
                        "matched_target_delta_accuracy_pp": 100 * target_mean if target_mean is not None else None,
                        "matched_target_cluster_bootstrap_95ci_pp": [100 * x for x in target_ci] if target_ci else None})
    multi_seed = []
    expected_seeds = {identities_by_model[model][0] for model in full_models}

    # Four joint outcomes for the original Table 3: post-unlearning recovery
    # of the targeted intermediate fact (B) against final-answer correctness
    # (C), restricted to the same pre-unlearning competent forget cases.
    joint_outcomes = []
    outcome_names = ("b_fail_c_pass", "b_fail_c_fail", "b_pass_c_pass", "b_pass_c_fail")
    for (method, lang), seeds_data in sorted(joint_outcomes_by_seed.items()):
        if set(seeds_data) != expected_seeds:
            raise ValueError(f"{method}/{lang}: joint outcomes are missing one or more SFT seeds")
        case_sets = [{case_id for case_id in values} for values in seeds_data.values()]
        if any(case_set != case_sets[0] for case_set in case_sets[1:]):
            raise ValueError(f"{method}/{lang}: joint-outcome case sets differ across seeds")
        all_groups = {value["group_id"] for values in seeds_data.values() for value in values.values()}
        group_counts = {name: defaultdict(lambda: [0, 0]) for name in outcome_names}
        seed_percentages = {name: [] for name in outcome_names}
        for seed in sorted(expected_seeds):
            values = seeds_data[seed]
            counts = Counter()
            for case_id, value in values.items():
                if value["b_pass"] and value["c_pass"]:
                    category = "b_pass_c_pass"
                elif value["b_pass"]:
                    category = "b_pass_c_fail"
                elif value["c_pass"]:
                    category = "b_fail_c_pass"
                else:
                    category = "b_fail_c_fail"
                counts[category] += 1
                for name in outcome_names:
                    group_counts[name][value["group_id"]][1] += 1
                group_counts[category][value["group_id"]][0] += 1
            for name in outcome_names:
                seed_percentages[name].append(100 * counts.get(name, 0) / len(values))
        row = {"method": method, "language": lang, "n_cases": len(case_sets[0]),
               "n_seed_case_observations": len(case_sets[0]) * len(expected_seeds),
               "entity_groups": len(all_groups), "n_seeds": len(expected_seeds)}
        for name in outcome_names:
            mean_pct, sd_pct = mean_sd(seed_percentages[name])
            group_ratios = {group: tuple(counts) for group, counts in group_counts[name].items()}
            _, ci = bootstrap_ratio(
                group_ratios,
                int(hashlib.sha256(f"{method}|{lang}|{name}|joint-outcomes".encode()).hexdigest()[:8], 16),
                args.bootstrap,
            )
            row[f"{name}_mean_pct"] = mean_pct
            row[f"{name}_sd_across_seeds_pct"] = sd_pct
            row[f"{name}_entity_group_bootstrap_95ci_pct"] = [100 * value for value in ci] if ci else None
        if abs(sum(row[f"{name}_mean_pct"] for name in outcome_names) - 100.0) > 1e-8:
            raise ValueError(f"{method}/{lang}: joint outcome percentages do not sum to 100")
        joint_outcomes.append(row)

    seed_rows_by_key = defaultdict(list)
    for row in out:
        seed_rows_by_key[(row["method"], row["language"], row["split"], row["probe"])].append(row)
    for key, seeds_data in sorted(case_delta_by_seed.items()):
        method, lang, split, probe = key
        if set(seeds_data) != expected_seeds:
            raise ValueError(f"{method}/{lang}/{split}/{probe}: case effects are missing one or more SFT seeds")
        records = seed_rows_by_key[key]
        if {row["seed"] for row in records} != expected_seeds:
            raise ValueError(f"{method}/{lang}/{split}/{probe}: seed summary is incomplete")
        seed_deltas = [row["delta_accuracy_pp"] for row in records]
        mean_delta_seed, sd_delta_seed = mean_sd(seed_deltas)
        mean_pre, sd_pre = mean_sd([row.get("pre_accuracy_pp") for row in records])
        mean_post, sd_post = mean_sd([row.get("post_accuracy_pp") for row in records])
        matched_component_summary = {}
        if probe == "downstream_answer_persistence":
            counts_by_seed_group = {}
            for seed in sorted(expected_seeds):
                per_group = defaultdict(lambda: [0, 0])
                for value in dap_counts_by_seed[(method, lang)].get(seed, {}).values():
                    per_group[value["group_id"]][0] += value["numerator"]
                    per_group[value["group_id"]][1] += value["denominator"]
                counts_by_seed_group[seed] = {group: tuple(counts) for group, counts in per_group.items()}
            point, ci = bootstrap_seed_ratio(counts_by_seed_group,
                int(hashlib.sha256(f"{method}|{lang}|dap|all_seeds".encode()).hexdigest()[:8], 16), args.bootstrap)
            unique_dap_cases = {case_id for seed in expected_seeds for case_id in dap_counts_by_seed[(method, lang)].get(seed, {})}
            case_n = len(unique_dap_cases)
            case_seed_n = sum(counts[1] for groups in counts_by_seed_group.values() for counts in groups.values())
            group_n = len({group for groups in counts_by_seed_group.values() for group in groups})
            mean_dap, sd_dap = mean_sd([row.get("dap_persistence_pct") for row in records])
            defined_dap_seeds = sum(row.get("dap_persistence_pct") is not None for row in records)
            if point is not None and (mean_dap is None or abs(100 * point - mean_dap) > 1e-8):
                raise ValueError("DAP bootstrap point differs from the mean of per-seed DAP")
            mean_delta, sd_delta = None, None
        else:
            case_sets = [{case_id for case_id in values} for values in seeds_data.values()]
            if any(case_set != case_sets[0] for case_set in case_sets[1:]):
                raise ValueError(f"{method}/{lang}/{split}/{probe}: the three seeds do not share the same paired case set")
            pooled_by_group = defaultdict(list)
            pooled_english_by_group = defaultdict(list)
            pooled_target_by_group = defaultdict(list)
            for case_id in sorted(case_sets[0]):
                details = [seeds_data[seed][case_id] for seed in sorted(expected_seeds)]
                group_ids = {value["group_id"] for value in details}
                if len(group_ids) != 1:
                    raise ValueError(f"{case_id}: entity group differs across seed runs")
                group = next(iter(group_ids))
                pooled_by_group[group].append(sum(value["delta"] for value in details) / len(details))
                if probe == "multi_hop_language_minus_english_DID":
                    pooled_english_by_group[group].append(sum(value["english_delta"] for value in details) / len(details))
                    pooled_target_by_group[group].append(sum(value["target_delta"] for value in details) / len(details))
            ci_alpha = 0.05
            interval_scope = "marginal_95_percent"
            multiplicity_adjustment = None
            if split == "forget" and probe == "multi_hop" and lang in TARGET_LANGS:
                ci_alpha = 0.05 / 12
                interval_scope = "primary_12_cell_familywise_95_percent"
                multiplicity_adjustment = "Bonferroni over 3 methods x 4 target languages"
            elif split == "forget" and probe == "multi_hop_language_minus_english_DID":
                ci_alpha = 0.05 / 12
                interval_scope = "secondary_12_cell_familywise_95_percent"
                multiplicity_adjustment = "Bonferroni over 3 methods x 4 target languages"
            point, ci = bootstrap_mean(pooled_by_group,
                int(hashlib.sha256(f"{method}|{lang}|{split}|{probe}|all_seeds".encode()).hexdigest()[:8], 16),
                args.bootstrap, ci_alpha=ci_alpha)
            case_n = sum(map(len, pooled_by_group.values()))
            case_seed_n = case_n * len(expected_seeds)
            group_n = len(pooled_by_group)
            mean_dap, sd_dap = None, None
            defined_dap_seeds = None
            mean_delta, sd_delta = 100 * point if point is not None else None, sd_delta_seed
            if probe == "multi_hop_language_minus_english_DID":
                component_stats = {}
                for component, values_by_group in (("english", pooled_english_by_group), ("target", pooled_target_by_group)):
                    component_point, component_ci = bootstrap_mean(
                        values_by_group,
                        int(hashlib.sha256(f"{method}|{lang}|{split}|matched_{component}|all_seeds".encode()).hexdigest()[:8], 16),
                        args.bootstrap,
                    )
                    seed_means = []
                    for seed in sorted(expected_seeds):
                        per_seed_values = [
                            value[f"{component}_delta"]
                            for value in seeds_data[seed].values()
                        ]
                        seed_means.append(100 * sum(per_seed_values) / len(per_seed_values))
                    component_mean_seed, component_sd_seed = mean_sd(seed_means)
                    component_stats[component] = (component_mean_seed, component_sd_seed,
                        [100 * value for value in component_ci] if component_ci else None)
                matched_component_summary = {
                    "matched_english_mean_delta_accuracy_pp": component_stats["english"][0],
                    "matched_english_sd_delta_across_seeds_pp": component_stats["english"][1],
                    "matched_english_paired_entity_group_bootstrap_95ci_pp": component_stats["english"][2],
                    "matched_target_mean_delta_accuracy_pp": component_stats["target"][0],
                    "matched_target_sd_delta_across_seeds_pp": component_stats["target"][1],
                    "matched_target_paired_entity_group_bootstrap_95ci_pp": component_stats["target"][2],
                }
        ci_pp = [100 * value for value in ci] if ci else None
        multi_seed.append({
            "method": method, "language": lang, "split": split, "probe": probe,
            "n_cases": case_n, "n_seed_case_observations": case_seed_n,
            "entity_groups": group_n, "n_seeds": len(expected_seeds),
            "confidence_interval_scope": interval_scope,
            "bootstrap_tail_alpha": ci_alpha / 2,
            "multiplicity_adjustment": multiplicity_adjustment,
            "mean_pre_accuracy_pp": mean_pre, "sd_pre_accuracy_across_seeds_pp": sd_pre,
            "mean_post_accuracy_pp": mean_post, "sd_post_accuracy_across_seeds_pp": sd_post,
            "mean_delta_accuracy_pp": mean_delta, "sd_delta_accuracy_across_seeds_pp": sd_delta,
            "mean_dap_persistence_pct": mean_dap, "sd_dap_persistence_across_seeds_pp": sd_dap,
            "n_seeds_with_defined_dap": defined_dap_seeds,
            "paired_entity_group_bootstrap_95ci_pp": ci_pp,
            "direct_forgetting_mean_pp": -mean_delta if probe == "direct_positive" and mean_delta is not None else None,
            "direct_forgetting_bootstrap_95ci_pp": [-ci_pp[1], -ci_pp[0]] if probe == "direct_positive" and ci_pp else None,
            **matched_component_summary,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "paired_effects.jsonl", out)
    if out:
        columns = list(dict.fromkeys(field for row in out for field in row))
        with (args.output_dir / "paired_effects.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(out)
    write_jsonl(args.output_dir / "multi_seed_summary.jsonl", multi_seed)
    if multi_seed:
        with (args.output_dir / "multi_seed_summary.csv").open("w", newline="", encoding="utf-8") as f:
            columns = list(dict.fromkeys(field for row in multi_seed for field in row))
            writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(multi_seed)
    write_jsonl(args.output_dir / "intermediate_final_outcomes.jsonl", joint_outcomes)
    if joint_outcomes:
        with (args.output_dir / "intermediate_final_outcomes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(joint_outcomes[0])); writer.writeheader(); writer.writerows(joint_outcomes)
    write_json(args.output_dir / "effect_analysis_manifest.json", {
        "status": "PAIRED_EFFECTS_COMPLETE", "target_manifest_sha256": sha(Path(args.target_manifest)),
        "case_metrics_sha256": sha(Path(args.case_metrics)), "model_pairs_sha256": sha(Path(args.model_pairs)),
        "bootstrap_replicates": args.bootstrap, "cluster": "leakage-isolated entity group",
        "primary_outcome": "Post-minus-pre multi-hop exact-match change in each target language on the frozen forget cases competent in English and that target language before unlearning; 3 methods x 4 languages, pooled as equal-weight mean of the three matched seeds.",
        "primary_confidence_intervals": "Two-sided entity-group percentile bootstrap intervals with Bonferroni familywise 95% coverage over the 12 primary method-language cells; conditional on the three observed seeds. Seed-specific estimates and SDs are reported separately.",
        "secondary_outcome": "Target-language minus English multi-hop change on the same bilingual precompetent cases; separate Bonferroni familywise 95% interval family over the 12 method-language cells.",
        "bootstrap_replicates_minimum_for_kaggle_main_run": MAIN_EFFECT_BOOTSTRAP_MINIMUM,
        "seed_aggregation": "Mean and sample SD of per-seed estimates. Accuracy changes use paired entity-group resampling of per-case effects averaged across seeds. DAP resamples shared entity groups and recomputes the unweighted mean of defined per-seed ratios; its interval is undefined if a resample gives any included seed a zero denominator.",
        "intermediate_final_outcome_definition": "Post-unlearning direct_positive correctness (B) crossed with multi_hop correctness (C) on the same pre-unlearning competent forget cases; per-seed percentages averaged and entity-group bootstrapped.",
        "translation_review_mode": "AI_ONLY", "human_translation_reviewed": False,
        "interpretation": "Exploratory pilot estimates; paired cluster intervals do not establish confirmatory power.",
    })
    print(json.dumps({"status": "PAIRED_EFFECTS_COMPLETE", "rows": len(out), "multi_seed_summary_rows": len(multi_seed),
                      "intermediate_final_outcome_rows": len(joint_outcomes), "output_dir": str(args.output_dir)}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompts", type=Path, required=True); g.add_argument("--output", type=Path, required=True)
    g.add_argument("--model-name", required=True); g.add_argument("--model-revision", required=True)
    g.add_argument("--model-profile", choices=(PILOT_PROFILE, MAIN_PROFILE), default=MAIN_PROFILE)
    g.add_argument("--adapter-dir", type=Path); g.add_argument("--seed", type=int)
    g.add_argument("--batch-size", type=int, default=4); g.add_argument("--max-new-tokens", type=int, default=64)
    s = sub.add_parser("score")
    s.add_argument("--prompts", type=Path, required=True); s.add_argument("--labels", type=Path, required=True); s.add_argument("--predictions", type=Path, nargs="+", required=True); s.add_argument("--output-dir", type=Path, required=True)
    b = sub.add_parser("baseline-gate")
    b.add_argument("--case-metrics", type=Path, required=True); b.add_argument("--full-sft-models", nargs="+", required=True)
    b.add_argument("--output", type=Path, required=True); b.add_argument("--minimum-cases", type=int, default=15); b.add_argument("--minimum-groups", type=int, default=8)
    b.add_argument("--expected-cases", type=int, required=True, help="Frozen evaluation cohort size from the assembled release manifest")
    e = sub.add_parser("effects")
    e.add_argument("--case-metrics", type=Path, required=True); e.add_argument("--target-manifest", type=Path, required=True)
    e.add_argument("--model-pairs", type=Path, required=True); e.add_argument("--output-dir", type=Path, required=True); e.add_argument("--bootstrap", type=int, default=MAIN_EFFECT_BOOTSTRAP_MINIMUM)
    args = p.parse_args()
    if args.command == "generate": generate(args)
    elif args.command == "score": score(args)
    elif args.command == "baseline-gate": baseline_gate(args)
    else:
        if args.bootstrap < MAIN_EFFECT_BOOTSTRAP_MINIMUM:
            p.error(f"Main effects analysis requires at least {MAIN_EFFECT_BOOTSTRAP_MINIMUM:,} bootstrap draws")
        effects(args)


if __name__ == "__main__":
    main()
