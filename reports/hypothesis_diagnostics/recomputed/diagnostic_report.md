# Recomputed diagnostics from saved Kaggle runs

## Scope and provenance

These outputs are for `MQuAKE-original-mixed` with source SHA-256 `679b7cc3cc953e4be5f964fc10dea468ad442595a0572cb5c53fe8823e629295`. The saved Kaggle notebook source matched the current local source cell-for-cell for all three stages. This is the MQuAKE-T/MQuAKE-CF, Qwen2.5-7B-Instruct study (EN+DE SFT; EN/DE/RU/ZH/TR/VI evaluation), not the SynthWorlds-SM, Llama, EN/FR/DE/HI/TH study in `docs/report/R1.tex`.

Evaluation contains 192 distinct cases: 32 per split/depth across depths 2–4, evaluated in six languages. This is one seed. The primary precompetence subset requires full-SFT direct and multi-hop answers both correct in that same language. CIs below are percentile bootstrap intervals over matched source facts (10,000 resamples); they do not account for translation uncertainty, seed variation, or benchmark selection.

## Forget-set multi-hop result

Post-unlearning multi-hop accuracy among precompetent cases (correct after full SFT on both direct and multi-hop probes):

| Checkpoint | EN | DE | RU | ZH | TR | VI |
|---|---:|---:|---:|---:|---:|---:|
| retain_only | 0.917 (84) | 0.927 (82) | 0.944 (54) | 0.915 (47) | 0.926 (68) | 0.920 (75) |
| ga | 0.000 (84) | 0.000 (82) | 0.000 (54) | 0.000 (47) | 0.000 (68) | 0.000 (75) |
| gd | 0.929 (84) | 0.963 (82) | 1.000 (54) | 0.936 (47) | 0.956 (68) | 0.947 (75) |
| npo | 0.833 (84) | 0.841 (82) | 0.889 (54) | 0.638 (47) | 0.794 (68) | 0.760 (75) |

The `n` in each cell is the number of precompetent source facts; rates are paired within those same facts. The unlearning-minus-retain-only paired effects for the forget set are in `recomputed/forget_effect_vs_retain_only.csv`. English-versus-other-language difference-in-differences, matched on cases precompetent in both languages, are in `recomputed/forget_cross_language_difference_in_differences.csv`.

## Diagnostic interpretation

- GA is unusable in this run: it reduced post multi-hop accuracy to zero on forget, retain, and holdout cases across all six languages, indicating model-wide generation collapse rather than selective forgetting.
- GD shows little forget-set multi-hop loss relative to the retain-only reference; direct target forgetting is weak. It is not evidence of a large cross-language effect.
- NPO produces stronger direct forgetting, while multi-hop answers are often retained. There is some degradation, including in non-English languages, but the one-seed, small held-out evaluation cannot establish a reliable language-specific transfer effect. The matched comparison CSV is the best diagnostic estimate in these saved outputs.
- In the matched forget-set comparison with retain-only, NPO's Chinese multi-hop difference is -0.277 (95% case-bootstrap CI [-0.426, -0.128]); Turkish is -0.132 ([-0.235, -0.044]) and Vietnamese -0.160 ([-0.253, -0.067]). The Chinese-minus-English difference-in-differences is -0.222 ([-0.378, -0.067]); German, Russian, Turkish, and Vietnamese intervals include zero. These are unadjusted exploratory intervals among five language contrasts, on one seed and small eligible subsets; treat as a signal to replicate, not a confirmed language-specific effect.
- An answer to the final multi-hop question is not a direct measurement that the intermediate fact was erased or recovered. The current experiment includes direct probes, but the final answer can be generated from a shortcut or other association.

## What this says about the requested R1 hypothesis

It does not directly answer R1. R1 specifies SynthWorlds-SM, Llama 3.2 1B pilot / Llama 3.1 8B main, five languages EN/FR/DE/HI/TH, and English-only unlearning. The saved runs use another dataset, another model, six different evaluation languages, and only EN+DE SFT. Do not cite these numbers as results for R1.

The R1 SynthWorlds pilot has no completed SFT/unlearning/evaluation result. Its corpus has 1,025 single-hop SFT rows (205 per language; forget and retain only). The earlier translation-pilot files had only 3 evaluation cases per non-English language. I prepared a complete English evaluation set (258 direct/multi-hop prompts for 129 assigned chains), separate hidden labels, and 258 translation requests per non-English language in `data/synthworlds_sm_pilot_v3/evaluation_dataset/`. NLLB drafts now exist for all 1,032 non-English prompts. The automated audit flagged 73 records (FR 66, DE 5, HI 1, TH 1), mostly protected-name/placeholder failures; all 1,032 remain pending semantic AI review, and the evaluation-readiness gate is false. The Kaggle SFT run reached corpus preflight but could not retrieve the `HF_TOKEN` secret, so training did not start. The corpus is AI-only reviewed by explicit authorization; this is not human language validation. Translation quality and target/path validity remain limitations.

## Next experimental action

For a diagnostic answer to R1, translate and AI-review the prepared evaluation questions, then rerun the actual SynthWorlds pipeline after resolving the Kaggle HF secret access. Implement and execute: (1) pre-SFT competence checks, (2) five-language factual SFT with multi-hop QA excluded, (3) English-only unlearning with GA/GD/NPO and retain-only control, and (4) paired pre/post final-answer and direct-fact evaluation on the same leakage-isolated eligible chains. Use at least three seeds before making a confirmatory claim. Do not use GA if a pre-registered retain/holdout generation-quality gate fails.
