#!/usr/bin/env python3
"""Generate and score held-out multilingual direct/MH probes for R1.

Gold labels are loaded only by the scoring process, after generation. The model
receives question-only prompts. Translation readiness is checked fail-closed.
"""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

LANGS = ("en", "fr", "de", "hi", "th")


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def normalize(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def exact_answer(pred, gold):
    return normalize(pred) == normalize(gold)


def invalid_output_reason(answer, generation_failed=False):
    """Return a deterministic format/repetition flag, separate from answer accuracy."""
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
    # Flag a decoding loop only when a short token cycle is repeated three
    # times consecutively. This avoids false positives from common words such
    # as "the" appearing several times in a normal sentence.
    for width in range(1, min(5, len(tokens) // 3) + 1):
        for start in range(len(tokens) - 3 * width + 1):
            phrase = tokens[start:start + width]
            if phrase == tokens[start + width:start + 2 * width] == tokens[start + 2 * width:start + 3 * width]:
                return "repetitive_ngram"
    # Thai output may not use spaces between words. Catch a repeated short
    # character cycle there without counting ordinary repeated tokens.
    compact = "".join(normalize(raw).split())
    if not any(char.isspace() for char in raw) and not any(char.isdigit() for char in raw) and len(compact) >= 12:
        for width in range(1, min(12, len(compact) // 3) + 1):
            for start in range(len(compact) - 3 * width + 1):
                phrase = compact[start:start + width]
                if phrase == compact[start + width:start + 2 * width] == compact[start + 2 * width:start + 3 * width]:
                    return "repetitive_ngram"
    return None


def load_eval(eval_root):
    root = Path(eval_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("evaluation_ready") is not True or manifest.get("ai_review_status") != "AI_ONLY_REVIEWED_COMPLETE":
        raise RuntimeError("Non-English evaluation translations have not passed complete AI-only review")
    prompts = {"en": read_jsonl(root / "prompts/en.jsonl")}
    for lang in LANGS[1:]:
        p = root / "translations/ai_reviewed" / f"{lang}.jsonl"
        rows = read_jsonl(p)
        if len(rows) != 258 or any(r.get("review_status") != "AI_REVIEWED" for r in rows):
            raise RuntimeError(f"AI-reviewed {lang} prompts are incomplete or mislabeled")
        prompts[lang] = rows
    prompt_by_case = {}
    for lang in LANGS:
        for row in prompts[lang]:
            iid = row["instance_id"] if lang == "en" else row["translation_id"].split(":", 1)[0]
            probe = row["probe"] if lang == "en" else row["translation_id"].split(":")[1]
            prompt_by_case[(iid, probe, lang)] = row
    expected_iids = {row["instance_id"] for row in prompts["en"]}
    expected = {(iid, probe, lang) for iid in expected_iids for probe in ("direct", "multi") for lang in LANGS}
    if set(prompt_by_case) != expected:
        raise RuntimeError(f"Prompt coverage mismatch: missing={len(expected-set(prompt_by_case))}, extra={len(set(prompt_by_case)-expected)}")
    return manifest, prompt_by_case


def load_labels(eval_root):
    labels = read_jsonl(Path(eval_root) / "labels/gold_labels.jsonl")
    label_by_id = {r["instance_id"]: r for r in labels}
    if len(labels) != 129 or len(label_by_id) != 129:
        raise RuntimeError("Expected 129 separate gold label rows")
    if any(not row.get("bootstrap_cluster_id") for row in labels):
        raise RuntimeError("Gold labels must include the leakage-isolated entity group for clustered inference")
    return label_by_id


def score(predictions_path, labels, output_dir, bootstrap=2000, contrast_models=None):
    import random
    rows = read_jsonl(predictions_path)
    label_by_id = labels
    seen = set()
    scored = []
    for p in rows:
        key = (p["model"], p["instance_id"], p["probe"], p["language"])
        if key in seen:
            raise RuntimeError(f"Duplicate prediction: {key}")
        seen.add(key)
        if p["instance_id"] not in label_by_id:
            raise RuntimeError(f"Prediction has unknown evaluation instance: {p['instance_id']}")
        if p["probe"] not in {"direct", "multi"} or p["language"] not in LANGS:
            raise RuntimeError(f"Prediction has invalid probe/language: {key}")
        if p.get("split") != label_by_id[p["instance_id"]].get("split"):
            raise RuntimeError(f"Prediction split disagrees with gold-label split: {key}")
        gold_row = label_by_id[p["instance_id"]]
        answers = gold_row["direct_intermediate_answer"] if p["probe"] == "direct" else gold_row["final_answers"]
        if isinstance(answers, str): answers = [answers]
        correct = any(exact_answer(p.get("answer", ""), g) for g in answers)
        invalid_reason = invalid_output_reason(p.get("answer", ""), p.get("generation_failed", False))
        scored.append({**p, "correct": correct, "output_invalid": invalid_reason is not None,
                       "invalid_output_reason": invalid_reason, "gold_answers": answers})
    models = {r["model"] for r in scored}
    if "sft" not in models:
        raise RuntimeError("Complete scoring requires the SFT baseline checkpoint")
    expected_per_model = {
        (iid, probe, lang)
        for iid in label_by_id
        for probe in ("direct", "multi")
        for lang in LANGS
    }
    for model in models:
        actual = {(r["instance_id"], r["probe"], r["language"]) for r in scored if r["model"] == model}
        if actual != expected_per_model:
            raise RuntimeError(f"Incomplete evaluation predictions for {model}: missing={len(expected_per_model-actual)}, extra={len(actual-expected_per_model)}")
    groups = defaultdict(dict)
    for r in scored:
        groups[(r["model"], r["instance_id"], r["language"])][r["probe"]] = r
    summaries = []
    case_rows = []
    def boot_ci(cluster_values, seed):
        clusters = sorted(cluster_values)
        if not clusters: return (None, None)
        point_values = [value for cluster in clusters for value in cluster_values[cluster]]
        if len(clusters) == 1:
            point = sum(point_values) / len(point_values)
            return (point, point)
        rng = random.Random(seed)
        estimates=[]
        for _ in range(bootstrap):
            sampled_clusters=[clusters[rng.randrange(len(clusters))] for _ in clusters]
            sample=[value for cluster in sampled_clusters for value in cluster_values[cluster]]
            estimates.append(sum(sample)/len(sample))
        estimates.sort()
        return estimates[int(0.025*(len(estimates)-1))], estimates[int(0.975*(len(estimates)-1))]
    for (model, iid, lang), probes in groups.items():
        if not {"direct", "multi"} <= probes.keys():
            raise RuntimeError(f"Missing paired direct/multi output: {model}/{iid}/{lang}")
        direct, multi = probes["direct"], probes["multi"]
        case_rows.append({"model": model, "instance_id": iid, "split": direct["split"],
                          "bootstrap_cluster_id": label_by_id[iid]["bootstrap_cluster_id"], "language": lang,
                          "direct_correct": direct["correct"], "multi_correct": multi["correct"],
                          "direct_generation_failed": direct.get("generation_failed", False),
                          "multi_generation_failed": multi.get("generation_failed", False),
                          "direct_output_invalid": direct["output_invalid"],
                          "multi_output_invalid": multi["output_invalid"],
                          "direct_invalid_output_reason": direct["invalid_output_reason"],
                          "multi_invalid_output_reason": multi["invalid_output_reason"]})
    baseline_name = "sft"
    base_lookup = {(r["instance_id"], r["language"]): r for r in case_rows if r["model"] == baseline_name}
    for model in sorted({r["model"] for r in case_rows}):
        for lang in LANGS:
            for split in ("forget", "retain", "holdout"):
                sub = [r for r in case_rows if r["model"] == model and r["language"] == lang and r["split"] == split]
                if not sub: continue
                eligible = [r for r in sub if (base_lookup.get((r["instance_id"],lang), {}).get("direct_correct") and base_lookup.get((r["instance_id"],lang), {}).get("multi_correct"))]
                direct_acc = sum(r["direct_correct"] for r in sub) / len(sub)
                mh_acc = sum(r["multi_correct"] for r in sub) / len(sub)
                retain_base = [r for r in case_rows if r["model"] == baseline_name and r["language"] == lang and r["split"] == split]
                base_mh = sum(r["multi_correct"] for r in retain_base) / len(retain_base) if retain_base else None
                d_direct = (sum(r["direct_correct"] for r in retain_base) / len(retain_base) - direct_acc) if retain_base else None
                delta_mh = mh_acc - base_mh if base_mh is not None else None
                daf = [r for r in eligible if not r["direct_correct"]]
                dap = sum(r["multi_correct"] for r in daf) / len(daf) if daf else None
                eligible_deltas=defaultdict(list)
                for r in eligible:
                    before=base_lookup[(r["instance_id"],lang)]["multi_correct"]
                    eligible_deltas[r["bootstrap_cluster_id"]].append(int(r["multi_correct"])-int(before))
                eligible_delta_values=[v for group in eligible_deltas.values() for v in group]
                precomp_delta=sum(eligible_delta_values)/len(eligible_delta_values) if eligible_delta_values else None
                seed=int(hashlib.sha256(f"{model}|{lang}|{split}".encode()).hexdigest()[:8],16)
                precomp_lo,precomp_hi=boot_ci(eligible_deltas,seed)
                summaries.append({"model": model, "language": lang, "split": split, "n": len(sub),
                                  "precompetent_n": len(eligible), "precompetent_entity_groups": len(eligible_deltas),
                                  "direct_accuracy": direct_acc, "multi_accuracy": mh_acc,
                                  "delta_multi_vs_sft_pp": 100*delta_mh if delta_mh is not None else None,
                                  "eligible_multi_accuracy": sum(r["multi_correct"] for r in eligible)/len(eligible) if eligible else None,
                                  "eligible_multi_delta_vs_sft_pp": 100*precomp_delta if precomp_delta is not None else None,
                                  "eligible_multi_delta_95ci_pp": [100*precomp_lo,100*precomp_hi] if precomp_lo is not None else None,
                                  "direct_forgetting_vs_sft_pp": 100*d_direct if model != baseline_name and d_direct is not None else 0,
                                  "dap": dap, "dap_denominator": len(daf),
                                  "invalid_generation_rate": sum(r["direct_output_invalid"]+r["multi_output_invalid"] for r in sub)/(2*len(sub)),
                                  "invalid_generation_n": sum(r["direct_output_invalid"]+r["multi_output_invalid"] for r in sub),
                                  "repetitive_output_n": sum(r["direct_invalid_output_reason"] == "repetitive_ngram" or r["multi_invalid_output_reason"] == "repetitive_ngram" for r in sub),
                                  "template_artifact_n": sum(r["direct_invalid_output_reason"] == "template_artifact" or r["multi_invalid_output_reason"] == "template_artifact" for r in sub),
                                  "overlong_output_n": sum(r["direct_invalid_output_reason"] == "overlong" or r["multi_invalid_output_reason"] == "overlong" for r in sub)})
    # Primary hypothesis contrast: within each leakage-isolated chain group,
    # compare the post-vs-SFT change in a non-English language with the same
    # change in English. Negative values indicate greater non-English loss.
    contrasts = []
    by_model_case_lang = {(r["model"], r["instance_id"], r["language"]): r for r in case_rows}
    selected_contrast_models = (models - {baseline_name}) if contrast_models is None else (models - {baseline_name}) & set(contrast_models)
    for model in sorted(selected_contrast_models):
        for split in ("forget", "retain", "holdout"):
            for lang in LANGS[1:]:
                paired = []
                for iid, label in label_by_id.items():
                    if label["split"] != split:
                        continue
                    sft_en = by_model_case_lang.get((baseline_name, iid, "en"))
                    sft_l = by_model_case_lang.get((baseline_name, iid, lang))
                    model_en = by_model_case_lang.get((model, iid, "en"))
                    model_l = by_model_case_lang.get((model, iid, lang))
                    if not all((sft_en, sft_l, model_en, model_l)):
                        continue
                    # Restrict to cases learned on both language probes before
                    # unlearning; this avoids counting prior language failure.
                    if not all((sft_en["direct_correct"], sft_en["multi_correct"],
                                sft_l["direct_correct"], sft_l["multi_correct"])):
                        continue
                    en_effect = int(model_en["multi_correct"]) - int(sft_en["multi_correct"])
                    lang_effect = int(model_l["multi_correct"]) - int(sft_l["multi_correct"])
                    paired.append({"instance_id": iid, "cluster": label["bootstrap_cluster_id"],
                                   "en_effect": en_effect, "language_effect": lang_effect,
                                   "did": lang_effect - en_effect})
                clusters = sorted({r["cluster"] for r in paired})
                point_en = sum(r["en_effect"] for r in paired) / len(paired) if paired else None
                point_lang = sum(r["language_effect"] for r in paired) / len(paired) if paired else None
                point_did = sum(r["did"] for r in paired) / len(paired) if paired else None
                lo = hi = None
                if len(clusters) >= 2:
                    seed = int(hashlib.sha256(f"{model}|{lang}|{split}|cluster-did".encode()).hexdigest()[:8], 16)
                    rng = random.Random(seed)
                    members = {cluster: [r for r in paired if r["cluster"] == cluster] for cluster in clusters}
                    estimates = []
                    for _ in range(bootstrap):
                        sampled_clusters = [clusters[rng.randrange(len(clusters))] for _ in clusters]
                        sampled = [r for cluster in sampled_clusters for r in members[cluster]]
                        estimates.append(sum(r["did"] for r in sampled) / len(sampled))
                    estimates.sort()
                    lo = estimates[int(0.025 * (len(estimates) - 1))]
                    hi = estimates[int(0.975 * (len(estimates) - 1))]
                elif paired:
                    lo = hi = point_did
                contrasts.append({"model": model, "split": split, "language": lang,
                                  "n_bilingual_precompetent_cases": len(paired), "n_entity_groups": len(clusters),
                                  "english_multi_delta_vs_sft_pp": 100 * point_en if point_en is not None else None,
                                  "language_multi_delta_vs_sft_pp": 100 * point_lang if point_lang is not None else None,
                                  "language_minus_english_did_pp": 100 * point_did if point_did is not None else None,
                                  "cluster_bootstrap_95ci_pp": [100 * lo, 100 * hi] if lo is not None else None})
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "scored_predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in scored),encoding="utf-8")
    (out / "case_metrics.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in case_rows),encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summaries,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if summaries:
        with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    with (out/"cross_language_contrasts.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(contrasts[0]) if contrasts else ["model","split","language"]);w.writeheader();w.writerows(contrasts)
    (out/"cross_language_contrasts.json").write_text(json.dumps(contrasts,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    invalid_count = sum(bool(r["output_invalid"]) for r in scored)
    invalid_reasons = defaultdict(int)
    for r in scored:
        if r["invalid_output_reason"]:
            invalid_reasons[r["invalid_output_reason"]] += 1
    report=["# R1 cross-lingual unlearning diagnostics", "", "AI-only-reviewed translations; not human-validated. Strict normalized exact gold-answer matching; generation failures count as incorrect. Output-invalid is a separate deterministic heuristic flag for generation failure, empty output, overlong output (>64 words or >512 characters), visible template artifacts, a consecutive repeated cycle of 1–5 whitespace tokens repeated three times, or a repeated letter cycle in a no-whitespace, digit-free string; it is not an answer-correctness label. Eligible MH delta is paired within cases both answered correctly by SFT. Confidence intervals resample leakage-isolated entity groups. Main English-vs-language contrasts use only bilingual precompetent cases. The pilot is exploratory; intervals are not multiplicity-adjusted.", "",
            f"Output-invalid diagnostic: {invalid_count}/{len(scored)} predictions ({100*invalid_count/len(scored):.2f}%). Reasons: {json.dumps(dict(sorted(invalid_reasons.items())), ensure_ascii=False)}.", "",
            "| Model | Language | Split | N | Precompetent | Direct acc. | MH acc. | Full MH Δ (pp) | Eligible MH Δ (95% CI, pp) | DAP | Invalid gen. |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        fmt=lambda x: "N/A" if x is None else f"{100*x:.1f}%"
        mh_delta = "N/A" if r["delta_multi_vs_sft_pp"] is None else f"{r['delta_multi_vs_sft_pp']:+.1f}"
        ci = r["eligible_multi_delta_95ci_pp"]
        e_delta = "N/A" if r["eligible_multi_delta_vs_sft_pp"] is None else f"{r['eligible_multi_delta_vs_sft_pp']:+.1f} ({ci[0]:+.1f},{ci[1]:+.1f})"
        report.append(f"| {r['model']} | {r['language']} | {r['split']} | {r['n']} | {r['precompetent_n']} | {fmt(r['direct_accuracy'])} | {fmt(r['multi_accuracy'])} | {mh_delta} | {e_delta} | {fmt(r['dap'])} | {fmt(r['invalid_generation_rate'])} |")
    report += ["", "## Primary contrast: language effect minus English effect", "",
               "Negative values mean the post-unlearning multi-hop change is more negative in the listed language than in English. Cohort is restricted to cases where SFT answered both direct and multi-hop probes correctly in both languages. Confidence intervals resample entity groups, preserving within-group dependence. Pilot intervals are exploratory.", "",
               "| Model | Split | Language | Bilingual cases | Entity groups | EN Δ (pp) | Language Δ (pp) | Language − EN (95% CI, pp) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in contrasts:
        ci=r["cluster_bootstrap_95ci_pp"]
        did="N/A" if r["language_minus_english_did_pp"] is None else f"{r['language_minus_english_did_pp']:+.1f} ({ci[0]:+.1f},{ci[1]:+.1f})"
        en="N/A" if r["english_multi_delta_vs_sft_pp"] is None else f"{r['english_multi_delta_vs_sft_pp']:+.1f}"
        other="N/A" if r["language_multi_delta_vs_sft_pp"] is None else f"{r['language_multi_delta_vs_sft_pp']:+.1f}"
        report.append(f"| {r['model']} | {r['split']} | {r['language']} | {r['n_bilingual_precompetent_cases']} | {r['n_entity_groups']} | {en} | {other} | {did} |")
    (out/"diagnostic_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")


def generate(manifest, prompts, checkpoints, outpath, max_new_tokens, batch_size=4):
    """Generate question-only predictions, flushing each batch for recovery.

    Each checkpoint has its own append-only JSONL file under
    ``<output-dir>/partial_predictions``. A restarted run skips prompt keys
    already written and resumes at the first missing key. The consolidated
    predictions file is atomically replaced only after all checkpoints finish.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    rows=[]
    english_prompts=read_jsonl(Path(__file__).resolve().parents[1]/"evaluation_dataset/prompts/en.jsonl")
    split_by_id={r["instance_id"]:r["split"] for r in english_prompts}
    outpath=Path(outpath)
    partial_dir=outpath.parent/"partial_predictions"
    partial_dir.mkdir(parents=True,exist_ok=True)
    expected_keys={(iid,probe,lang) for iid,probe,lang in prompts}
    ordered_prompts=[(key,prompts[key]) for lang in LANGS
                     for key in sorted(prompts) if key[2]==lang]
    for checkpoint in checkpoints:
        name, path = checkpoint.split("=",1)
        path=Path(path)
        part_path=partial_dir/f"{name}.jsonl"
        existing=read_jsonl(part_path) if part_path.exists() else []
        seen=set()
        for row in existing:
            key=(row["instance_id"],row["probe"],row["language"])
            if row.get("model")!=name or key not in expected_keys or key in seen:
                raise RuntimeError(f"Invalid or duplicate saved prediction in {part_path}: {key}")
            seen.add(key)
        if seen==expected_keys:
            print(f"{name}: recovered complete checkpoint predictions ({len(seen)} prompts)",flush=True)
            rows.extend(existing)
            continue
        rows.extend(existing)
        is_adapter=path.is_dir() and (path/"adapter_config.json").is_file()
        model_id=manifest.get("model_id","meta-llama/Llama-3.2-1B-Instruct")
        tok=AutoTokenizer.from_pretrained(path if is_adapter else model_id,token=True)
        tok.padding_side="left"
        if tok.pad_token_id is None:
            tok.pad_token=tok.eos_token
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        model=AutoModelForCausalLM.from_pretrained(model_id,token=True,dtype=dtype,device_map="auto")
        if is_adapter: model=PeftModel.from_pretrained(model,path)
        model.eval()
        pending=[(key,p) for key,p in ordered_prompts if key not in seen]
        with part_path.open("a",encoding="utf-8",buffering=1) as partial:
            for offset in range(0,len(pending),batch_size):
                batch=pending[offset:offset+batch_size]
                messages=[[{"role":"user","content":p["question"]+"\nAnswer with the shortest answer possible."}]
                          for _,p in batch]
                rendered=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
                encoded=tok(rendered,padding=True,return_tensors="pt")
                device=next(model.parameters()).device
                encoded={k:v.to(device) for k,v in encoded.items()}
                try:
                    with torch.inference_mode():
                        output=model.generate(**encoded,do_sample=False,max_new_tokens=max_new_tokens,
                                              pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
                    answers=tok.batch_decode(output[:,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
                    failures=[None]*len(batch)
                except Exception as exc:
                    # Recover from a too-large/padded batch by retrying each item.
                    answers=[]; failures=[]
                    for _,p in batch:
                        try:
                            one=tok(rendered[len(answers)],return_tensors="pt")
                            one={k:v.to(device) for k,v in one.items()}
                            with torch.inference_mode():
                                out=model.generate(**one,do_sample=False,max_new_tokens=max_new_tokens,
                                                   pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
                            answers.append(tok.decode(out[0,one["input_ids"].shape[1]:],skip_special_tokens=True))
                            failures.append(None)
                        except Exception as item_exc:
                            answers.append("");failures.append(type(item_exc).__name__)
                batch_rows=[]
                for (key,p),answer,failure_type in zip(batch,answers,failures):
                    iid,probe,lang=key
                    answer=answer.strip()
                    failed=(not bool(answer)) or failure_type is not None
                    row={"model":name,"instance_id":iid,"probe":probe,
                         "split":(p.get("split") or split_by_id[iid]),"language":lang,
                         "answer":answer,"generation_failed":failed}
                    if failure_type: row["failure_type"]=failure_type
                    partial.write(json.dumps(row,ensure_ascii=False)+"\n")
                    batch_rows.append(row);seen.add(key)
                partial.flush();os.fsync(partial.fileno())
                rows.extend(batch_rows)
                print(f"{name}: {len(seen)}/{len(expected_keys)} prompts saved",flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    tmp=outpath.with_suffix(outpath.suffix+".tmp")
    tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    tmp.replace(outpath)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]/"evaluation_dataset"
    p.add_argument("--eval-root",type=Path,default=root)
    p.add_argument("--checkpoint",action="append",required=True,help="name=base-or-adapter-path; include sft=...")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--predictions",type=Path)
    p.add_argument("--max-new-tokens",type=int,default=48)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--score-only",action="store_true")
    p.add_argument("--baseline-only",action="store_true",help="Generate the pretrained, SFT, and retain-only reference checkpoints without unlearning models")
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--contrast-model",action="append",help="Restrict primary language-vs-English contrasts to these checkpoint names; repeat per model")
    a=p.parse_args()
    manifest, prompts=load_eval(a.eval_root)
    if not a.score_only:
        checkpoint_names=[item.split("=",1)[0] for item in a.checkpoint]
        baseline_required={"pretrained","sft","retain_only"}
        full_required=baseline_required|{"GA","GD","NPO"}
        required=baseline_required if a.baseline_only else full_required
        allowed_sets={frozenset(required)}
        if not a.baseline_only:
            allowed_sets.add(frozenset(full_required|{"atomic_sft"}))
        if len(checkpoint_names)!=len(set(checkpoint_names)) or frozenset(checkpoint_names) not in allowed_sets:
            raise RuntimeError(f"{'Baseline' if a.baseline_only else 'Full'} R1 evaluation requires exactly {sorted(required)}")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pred=a.predictions or a.output_dir/"predictions.jsonl"
    if not a.score_only:
        if a.batch_size<1: raise RuntimeError("--batch-size must be positive")
        generate(manifest,prompts,a.checkpoint,pred,a.max_new_tokens,a.batch_size)
    labels=load_labels(a.eval_root)
    contrast_models = a.contrast_model
    if a.baseline_only and contrast_models is None:
        contrast_models = []
    score(pred,labels,a.output_dir,a.bootstrap,contrast_models)
    print(json.dumps({"status":"EVALUATION_COMPLETE","checkpoints":a.checkpoint,"predictions":str(pred),"report":str(a.output_dir/"diagnostic_report.md"),"translation_review_mode":"AI_ONLY","human_translation_reviewed":False},indent=2))


if __name__=="__main__": main()
