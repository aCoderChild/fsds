# What the completed pilot says about the R1 hypothesis

**Status: exploratory diagnostic only; not a confirmatory result.**

## Question and available run

R1 asks whether unlearning an intermediate fact using English examples changes a model's downstream multi-hop answer accuracy when it is asked in French, German, Hindi, or Thai. The only completed SynthWorlds run available for a direct diagnostic is the earlier Llama 3.2 1B pilot: one seed (42), 129 evaluation cases (77 forget, 26 retain, 26 holdout), and three English-only unlearning methods (GA, GD, NPO). The primary R1 plan instead requires Llama 3.1 8B, three seeds, and a passing pre-unlearning competence gate.

The run generated 7,740 predictions: 1,290 for each of pretrained, SFT, retain-only, GA, GD, and NPO. There were no recorded generation failures or deterministic invalid-output flags. Translations were reviewed by AI only, not fluent human reviewers.

## Baseline competence and matched cohort

| Language | SFT direct accuracy, forget | SFT multi-hop accuracy, forget | SFT-precompetent cases | English-matched cases / groups |
|---|---:|---:|---:|---:|
| English | 94.8% | 20.8% | 15 | — |
| French | 90.9% | 18.2% | 13 | 10 / 6 |
| German | 87.0% | 19.5% | 12 | 9 / 5 |
| Hindi | 87.0% | 18.2% | 11 | 9 / 6 |
| Thai | 94.8% | 19.5% | 15 | 11 / 7 |

The R1 baseline gate requires at least 15 English-and-target-language matched cases across 8 leakage-isolated groups in each target language. Every target language misses that floor. Also, SFT multi-hop accuracy is only about 18–20% over the full forget set, so most questions were not answerable before unlearning. This makes any post-unlearning effect difficult to interpret.

## Primary matched cross-language contrast

For each target language, the analysis keeps only the same forget cases that SFT answered correctly on both direct and multi-hop probes in English and in that target language. It computes:

`target-language multi-hop change − English multi-hop change`

Positive values mean the target-language accuracy fell less than English accuracy. Confidence intervals are 95% entity-group bootstrap intervals (2,000 resamples); with only 5–7 groups, these intervals are necessarily coarse. They are exploratory and not adjusted for the 12 comparisons.

| Method | Language | Matched cases / groups | English change (pp) | Target change (pp) | Target − English (pp, 95% CI) |
|---|---|---:|---:|---:|---:|
| GA | French | 10 / 6 | −10.0 | 0.0 | +10.0 [0.0, 50.0] |
| GA | German | 9 / 5 | −33.3 | −11.1 | +22.2 [0.0, 80.0] |
| GA | Hindi | 9 / 6 | −11.1 | −11.1 | 0.0 [−33.3, 33.3] |
| GA | Thai | 11 / 7 | −27.3 | −18.2 | +9.1 [−28.6, 57.1] |
| GD | French | 10 / 6 | −10.0 | 0.0 | +10.0 [0.0, 50.0] |
| GD | German | 9 / 5 | −22.2 | −11.1 | +11.1 [−33.3, 80.0] |
| GD | Hindi | 9 / 6 | −22.2 | −11.1 | +11.1 [0.0, 50.0] |
| GD | Thai | 11 / 7 | −27.3 | −18.2 | +9.1 [−42.9, 71.4] |
| NPO | French | 10 / 6 | −10.0 | 0.0 | +10.0 [0.0, 50.0] |
| NPO | German | 9 / 5 | −22.2 | −11.1 | +11.1 [−33.3, 80.0] |
| NPO | Hindi | 9 / 6 | −22.2 | −11.1 | +11.1 [0.0, 50.0] |
| NPO | Thai | 11 / 7 | −27.3 | −18.2 | +9.1 [−27.3, 57.1] |

All 12 intervals include zero or touch zero. In this small sample, the estimates do **not** show greater multi-hop damage in non-English than in English. GD and NPO show losses in some target-language matched cases, but those losses are generally smaller than their English changes. The uncertainty is too wide to rule out meaningful harm or to conclude there is no transfer.

## Direct intermediate-fact probe versus final answer

As a manipulation check, I recomputed post-unlearning direct-probe accuracy for the intermediate fact (B) and final two-hop answer accuracy (C) on the **same** SFT-precompetent, bilingual forget cases used for the paired language contrasts. Before unlearning, each selected case had both B and C correct in English and the target language. Results below are case counts; each target-language cell contains only 5–7 leakage groups.

| Method | Language | Matched cases / groups | English B correct | Target B correct | English C correct | Target C correct | Target B wrong / C correct |
|---|---|---:|---:|---:|---:|---:|---:|
| GA | French | 10 / 6 | 0/10 | 0/10 | 9/10 | 10/10 | 10/10 |
| GA | German | 9 / 5 | 0/9 | 0/9 | 6/9 | 8/9 | 8/9 |
| GA | Hindi | 9 / 6 | 0/9 | 0/9 | 8/9 | 8/9 | 8/9 |
| GA | Thai | 11 / 7 | 0/11 | 0/11 | 8/11 | 9/11 | 9/11 |
| GD | French | 10 / 6 | 0/10 | 2/10 | 9/10 | 10/10 | 8/10 |
| GD | German | 9 / 5 | 0/9 | 1/9 | 7/9 | 8/9 | 7/9 |
| GD | Hindi | 9 / 6 | 0/9 | 0/9 | 7/9 | 8/9 | 8/9 |
| GD | Thai | 11 / 7 | 0/11 | 0/11 | 8/11 | 9/11 | 9/11 |
| NPO | French | 10 / 6 | 0/10 | 1/10 | 9/10 | 10/10 | 9/10 |
| NPO | German | 9 / 5 | 0/9 | 0/9 | 7/9 | 8/9 | 8/9 |
| NPO | Hindi | 9 / 6 | 0/9 | 0/9 | 7/9 | 8/9 | 8/9 |
| NPO | Thai | 11 / 7 | 0/11 | 0/11 | 8/11 | 9/11 | 9/11 |

English direct recall of B is zero in every method/language-matched subset, while target-language direct recall is zero or one case (two for French GD). Yet target-language C remains correct on 8–10 of the 9–11 cases, and in 7–10 cases per cell B is wrong while C is right. This is evidence that direct-probe forgetting did not reliably remove the measured final-answer capability in this pilot. Residual associations, prompt cues, or limits of the direct probe are possible explanations; these data cannot distinguish them. Because the baseline competence floor failed and the sample is tiny, this is a diagnostic pattern, not evidence that the R1 hypothesis is false or that “deep unlearning” was achieved.

The table is reproducible from the archived, independently rescored case metrics using [the recomputation script](evidence/recompute_v4_paired_probe_outcomes.py); the hash-bound output is [here](evidence/v4_paired_probe_outcomes_2026-11-04.json).

## Reliability audit

| Verification | Result |
|---|---|
| Result archive ZIP CRC | Pass |
| Indexed archive files | 77/77 sizes and SHA-256 hashes match both the index and extracted files |
| Prediction records | 7,740 unique model/case/probe/language keys; 1,290 per checkpoint |
| Independent rescore | All 7 derived score and report files byte-identical to the archived results |
| Generation failures / invalid-output flags | 0 / 0 |
| GD/NPO schedule | Manifests record 45 steps, all 26 retain IDs used, and zero holdout rows |
| Code-to-training provenance | Reconciled after the run; raw Kaggle manifests did not include the unlearning script hash |
| Seeds / model | One seed; Llama 3.2 1B feasibility model, not the planned 8B main model |
| Translation validation | AI-only; no fluent-human audit |

The rescore used the archived `evaluate_crosslingual_unlearning.py` with `--score-only`, the archived evaluation prompts/labels, the saved predictions, 2,000 bootstrap resamples, and contrasts restricted to GA/GD/NPO. The exact command, output hashes, source hashes, and matching checks are recorded in [the independent rescore receipt](evidence/v4_corrected_independent_rescore_2026-11-04.json). This verifies scoring reproducibility and artifact integrity. It does not prove the archived training process loaded the bundled unlearning source because that hash was not recorded in the raw run manifest.

## Conclusion and decision

This run provides a preliminary signal that English-only unlearning can coincide with changed final-answer performance in other languages, but it does **not** establish the R1 hypothesis. The relative cross-language estimates are small-sample, imprecise, and do not show greater target-language loss than English; the competence gate failed before unlearning; there is only one 1B seed; and translation review was AI-only. Treat this as a debugging/feasibility diagnostic, not evidence for the thesis claim.

The current full v5.1 training release still fails the source-bound translation coverage gate, so no new R1 SFT/unlearning job is running. Do not lower the competence threshold or use the current candidate translations as approved training data. The next sound experiment requires an approved, leakage-isolated multilingual corpus and baseline competence on the frozen target cases before unlearning. Until that exists, the completed v4 pilot is the project's available hypothesis diagnostic.
