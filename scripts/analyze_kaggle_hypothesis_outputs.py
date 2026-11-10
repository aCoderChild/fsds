#!/usr/bin/env python3
"""Recompute paired diagnostics from the saved three-stage Kaggle MQuAKE run.

This analysis is intentionally separate from the SynthWorlds-SM experiment in
docs/report/R1.tex. It consumes saved Kaggle outputs only; it does not train or
modify model artifacts.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/hypothesis_diagnostics/kaggle_existing_outputs"
OUT = ROOT / "reports/hypothesis_diagnostics/recomputed"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261102
BOOTSTRAPS = 10000

source = json.loads((INPUT / "01/source_manifest.json").read_text())
unlearning = json.loads((INPUT / "02/unlearning_manifest.json").read_text())
df = pd.read_csv(INPUT / "03/paired_fact_analysis.csv")
training = pd.read_csv(INPUT / "02/unlearning_training_log.csv")
assert source["source_sha256"] == unlearning["source_sha256"]
assert df.fact_id.notna().all() and df.language.notna().all()
assert not df.duplicated(["checkpoint", "fact_id", "language"]).any()
assert set(df.checkpoint) == {"retain_only", "ga", "gd", "npo"}
assert set(df.language) == set(source["evaluation_languages"])
assert set(df.split) == {"forget", "retain", "holdout"}
assert df.groupby(["checkpoint", "fact_id"]).language.nunique().eq(6).all()
assert training.groupby("method").step.max().to_dict() == {"ga": 512, "gd": 512, "npo": 512}

def boot_mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    ix = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[ix].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))

rng = np.random.default_rng(SEED)
primary = []
for method in ["retain_only", "ga", "gd", "npo"]:
    for split in ["forget", "retain", "holdout"]:
        for lang in source["evaluation_languages"]:
            block = df[(df.checkpoint == method) & (df.split == split) & (df.language == lang)]
            eligible = block[block.eligible_both_correct == 1]
            losses = 1 - eligible.post_multi.to_numpy(dtype=float)
            lo, hi = boot_mean_ci(eligible.post_multi, rng)
            primary.append({
                "checkpoint": method, "split": split, "language": lang,
                "n_cases": len(block), "n_precompetent": len(eligible),
                "pre_multi_accuracy_all": block.sft_multi.mean(),
                "post_multi_accuracy_all": block.post_multi.mean(),
                "post_multi_accuracy_precompetent": eligible.post_multi.mean(),
                "post_multi_ci95_low": lo, "post_multi_ci95_high": hi,
                "precompetent_multi_loss_n": int(losses.sum()),
                "precompetent_multi_loss_rate": float(losses.mean()) if len(losses) else np.nan,
                "precompetent_direct_lost_n": int(eligible.direct_lost.sum()),
                "downstream_answer_persistence_after_direct_loss": (
                    float(eligible.loc[eligible.direct_lost == 1, "post_multi"].mean())
                    if eligible.direct_lost.sum() else np.nan
                ),
            })
primary = pd.DataFrame(primary)
primary.to_csv(OUT / "paired_accuracy_by_language.csv", index=False)

# Same-case comparator: effect of the unlearning adapter relative to retain-only
# on forget cases where full SFT answered both probes correctly in that language.
effect = []
for method in ["ga", "gd", "npo"]:
    for lang in source["evaluation_languages"]:
        a = df[(df.checkpoint == method) & (df.split == "forget") & (df.language == lang)]
        b = df[(df.checkpoint == "retain_only") & (df.split == "forget") & (df.language == lang)]
        a = a.set_index("fact_id").sort_index()
        b = b.set_index("fact_id").sort_index()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        mask = (a.eligible_both_correct == 1) & (b.eligible_both_correct == 1)
        d = (a.loc[mask, "post_multi"] - b.loc[mask, "post_multi"]).to_numpy(dtype=float)
        lo, hi = boot_mean_ci(d, rng)
        effect.append({
            "checkpoint": method, "language": lang, "n_paired_precompetent": len(d),
            "post_multi_difference_vs_retain_only": float(d.mean()) if len(d) else np.nan,
            "ci95_low": lo, "ci95_high": hi,
            "lost_cases_vs_retain_only": int((d < 0).sum()),
            "gained_cases_vs_retain_only": int((d > 0).sum()),
        })
effect = pd.DataFrame(effect)
effect.to_csv(OUT / "forget_effect_vs_retain_only.csv", index=False)

# Cross-language contrast, matched on the exact same source facts that were
# precompetent in both English and the comparison language.
cross = []
for method in ["retain_only", "ga", "gd", "npo"]:
    for lang in [x for x in source["evaluation_languages"] if x != "en"]:
        a = df[(df.checkpoint == method) & (df.split == "forget") & (df.language == lang)].set_index("fact_id")
        e = df[(df.checkpoint == method) & (df.split == "forget") & (df.language == "en")].set_index("fact_id")
        common = a.index.intersection(e.index)
        a, e = a.loc[common], e.loc[common]
        mask = (a.eligible_both_correct == 1) & (e.eligible_both_correct == 1)
        d = (a.loc[mask, "post_multi"] - a.loc[mask, "sft_multi"]
             - e.loc[mask, "post_multi"] + e.loc[mask, "sft_multi"]).to_numpy(dtype=float)
        lo, hi = boot_mean_ci(d, rng)
        cross.append({
            "checkpoint": method, "language": lang, "n_same_precompetent_facts": len(d),
            "nonenglish_minus_english_change": float(d.mean()) if len(d) else np.nan,
            "ci95_low": lo, "ci95_high": hi,
        })
cross = pd.DataFrame(cross)
cross.to_csv(OUT / "forget_cross_language_difference_in_differences.csv", index=False)

# Summaries for training-loss / catastrophic-degradation diagnostics.
train_diag = training.groupby("method").agg(
    optimizer_steps=("step", "max"), final_logged_loss=("loss", "last"),
    min_loss=("loss", "min"), max_abs_loss=("loss", lambda s: float(s.abs().max())),
).reset_index()
all_post = df[df.split.isin(["retain", "holdout"])].groupby("checkpoint").agg(
    nonforget_post_direct=("post_direct", "mean"), nonforget_post_multi=("post_multi", "mean"),
    nonforget_baseline_multi=("sft_multi", "mean"),
).reset_index()
train_diag.to_csv(OUT / "training_diagnostics.csv", index=False)

report = [
    "# Recomputed diagnostics from saved Kaggle runs",
    "",
    "## Scope and provenance",
    "",
    f"These outputs are for `{source['original_dataset']}` with source SHA-256 `{source['source_sha256']}`. The saved Kaggle notebook source matched the current local source cell-for-cell for all three stages. This is the MQuAKE-T/MQuAKE-CF, Qwen2.5-7B-Instruct study (EN+DE SFT; EN/DE/RU/ZH/TR/VI evaluation), not the SynthWorlds-SM, Llama, EN/FR/DE/HI/TH study in `docs/report/R1.tex`.",
    "",
    f"Evaluation contains {df[df.checkpoint == 'ga'].fact_id.nunique()} distinct cases: 32 per split/depth across depths 2–4, evaluated in six languages. This is one seed. The primary precompetence subset requires full-SFT direct and multi-hop answers both correct in that same language. CIs below are percentile bootstrap intervals over matched source facts ({BOOTSTRAPS:,} resamples); they do not account for translation uncertainty, seed variation, or benchmark selection.",
    "",
    "## Forget-set multi-hop result",
    "",
    "Post-unlearning multi-hop accuracy among precompetent cases (correct after full SFT on both direct and multi-hop probes):",
    "",
    "| Checkpoint | EN | DE | RU | ZH | TR | VI |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for method in ["retain_only", "ga", "gd", "npo"]:
    bits=[]
    for lang in source["evaluation_languages"]:
        row=primary[(primary.checkpoint==method)&(primary.split=="forget")&(primary.language==lang)].iloc[0]
        bits.append(f"{row.post_multi_accuracy_precompetent:.3f} ({int(row.n_precompetent)})")
    report.append("| "+method+" | "+" | ".join(bits)+" |")
report += [
    "",
    "The `n` in each cell is the number of precompetent source facts; rates are paired within those same facts. The unlearning-minus-retain-only paired effects for the forget set are in `recomputed/forget_effect_vs_retain_only.csv`. English-versus-other-language difference-in-differences, matched on cases precompetent in both languages, are in `recomputed/forget_cross_language_difference_in_differences.csv`.",
    "",
    "## Diagnostic interpretation",
    "",
    "- GA is unusable in this run: it reduced post multi-hop accuracy to zero on forget, retain, and holdout cases across all six languages, indicating model-wide generation collapse rather than selective forgetting.",
    "- GD shows little forget-set multi-hop loss relative to the retain-only reference; direct target forgetting is weak. It is not evidence of a large cross-language effect.",
    "- NPO produces stronger direct forgetting, while multi-hop answers are often retained. There is some degradation, including in non-English languages, but the one-seed, small held-out evaluation cannot establish a reliable language-specific transfer effect. The matched comparison CSV is the best diagnostic estimate in these saved outputs.",
    "- In the matched forget-set comparison with retain-only, NPO's Chinese multi-hop difference is -0.277 (95% case-bootstrap CI [-0.426, -0.128]); Turkish is -0.132 ([-0.235, -0.044]) and Vietnamese -0.160 ([-0.253, -0.067]). The Chinese-minus-English difference-in-differences is -0.222 ([-0.378, -0.067]); German, Russian, Turkish, and Vietnamese intervals include zero. These are unadjusted exploratory intervals among five language contrasts, on one seed and small eligible subsets; treat as a signal to replicate, not a confirmed language-specific effect.",
    "- An answer to the final multi-hop question is not a direct measurement that the intermediate fact was erased or recovered. The current experiment includes direct probes, but the final answer can be generated from a shortcut or other association.",
    "",
    "## What this says about the requested R1 hypothesis",
    "",
    "It does not directly answer R1. R1 specifies SynthWorlds-SM, Llama 3.2 1B pilot / Llama 3.1 8B main, five languages EN/FR/DE/HI/TH, and English-only unlearning. The saved runs use another dataset, another model, six different evaluation languages, and only EN+DE SFT. Do not cite these numbers as results for R1.",
    "",
    "The R1 SynthWorlds pilot has no completed SFT/unlearning/evaluation result. Its corpus has 1,025 single-hop SFT rows (205 per language; forget and retain only). The earlier translation-pilot files had only 3 evaluation cases per non-English language. I prepared a complete English evaluation set (258 direct/multi-hop prompts for 129 assigned chains), separate hidden labels, and 258 translation requests per non-English language in `data/synthworlds_sm_pilot_v3/evaluation_dataset/`. NLLB drafts now exist for all 1,032 non-English prompts. The automated audit flagged 73 records (FR 66, DE 5, HI 1, TH 1), mostly protected-name/placeholder failures; all 1,032 remain pending semantic AI review, and the evaluation-readiness gate is false. The Kaggle SFT run reached corpus preflight but could not retrieve the `HF_TOKEN` secret, so training did not start. The corpus is AI-only reviewed by explicit authorization; this is not human language validation. Translation quality and target/path validity remain limitations.",
    "",
    "## Next experimental action",
    "",
    "For a diagnostic answer to R1, translate and AI-review the prepared evaluation questions, then rerun the actual SynthWorlds pipeline after resolving the Kaggle HF secret access. Implement and execute: (1) pre-SFT competence checks, (2) five-language factual SFT with multi-hop QA excluded, (3) English-only unlearning with GA/GD/NPO and retain-only control, and (4) paired pre/post final-answer and direct-fact evaluation on the same leakage-isolated eligible chains. Use at least three seeds before making a confirmatory claim. Do not use GA if a pre-registered retain/holdout generation-quality gate fails.",
]
(OUT / "diagnostic_report.md").write_text("\n".join(report)+"\n")
print("PASS: provenance, schema, pairing, and 512 steps/method validated")
print("Wrote", OUT)
print("\nForget set post multi-hop among precompetent:")
print(primary[primary.split == "forget"].pivot(index="checkpoint", columns="language", values="post_multi_accuracy_precompetent").round(3).to_string())
print("\nUnlearning minus retain-only, matched forget cases:")
print(effect.pivot(index="checkpoint", columns="language", values="post_multi_difference_vs_retain_only").round(3).to_string())
