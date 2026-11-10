# Rechecked R1 pilot diagnostic: numbers and uncertainty

**Status: verified exploratory diagnostic only; not confirmatory evidence.** The authoritative source is the hash-verified corrected v4 run bundle. The older `r1_pilot_recomputed_v2` predictions are treated as a separate, unverified output because their answers differ and they are not bound to the v4 archive.

## Run and scoring checks

| Item | Result |
|---|---:|
| Model / seed | Llama 3.2 1B / 42 |
| Evaluation cases | 129 (77 forget, 26 retain, 26 holdout) |
| Predictions | 7,740 (1,290 per checkpoint; 6 checkpoints) |
| Generation failures / output-invalid flags | 0 / 0 (0.00%) |
| Archive | ZIP CRC pass; 77/77 indexed file hashes pass |
| Independent rescoring | 7/7 derived outputs byte-identical |
| Translation review | AI-only; no fluent-human review |

## Pre-unlearning baseline and competence gate

| Language | Direct accuracy | Final multi-hop accuracy | SFT-precompetent cases | Bilingual matched cases / groups | Gate |
|---|---:|---:|---:|---:|---|
| English | 94.8% | 20.8% | 15/77 | — | n/a |
| French | 90.9% | 18.2% | 13/77 | 10/6 | FAIL |
| German | 87.0% | 19.5% | 12/77 | 9/5 | FAIL |
| Hindi | 87.0% | 18.2% | 11/77 | 9/6 | FAIL |
| Thai | 94.8% | 19.5% | 15/77 | 11/7 | FAIL |

The documented target-language gate requires at least 15 bilingual cases across 8 leakage-isolated groups. All four target languages miss it. Full-set SFT multi-hop accuracy is 18.2–20.8%.

## Primary target-language outcome

The R1 primary estimand is target-language post-minus-pre multi-hop exact-match change on the same bilingual SFT-precompetent forget cases. The table gives the one-seed 1B pilot point estimates and fully enumerated percentile entity-group bootstrap intervals, with nominal Bonferroni familywise 95% coverage over 12 method-language cells. English change is shown for context; the target-language change is the primary outcome.

| Method | Language | Cases / groups | English change (pp) | Target change (pp; familywise 95% CI) |
|---|---|---:|---:|---:|
| GA | French | 10/6 | -10.0 | +0.0 [0.0, 0.0] |
| GA | German | 9/5 | -33.3 | -11.1 [-80.0, 0.0] |
| GA | Hindi | 9/6 | -11.1 | -11.1 [-66.7, 0.0] |
| GA | Thai | 11/7 | -27.3 | -18.2 [-85.7, 0.0] |
| GD | French | 10/6 | -10.0 | +0.0 [0.0, 0.0] |
| GD | German | 9/5 | -22.2 | -11.1 [-80.0, 0.0] |
| GD | Hindi | 9/6 | -22.2 | -11.1 [-66.7, 0.0] |
| GD | Thai | 11/7 | -27.3 | -18.2 [-85.7, 0.0] |
| NPO | French | 10/6 | -10.0 | +0.0 [0.0, 0.0] |
| NPO | German | 9/5 | -22.2 | -11.1 [-80.0, 0.0] |
| NPO | Hindi | 9/6 | -22.2 | -11.1 [-66.7, 0.0] |
| NPO | Thai | 11/7 | -27.3 | -18.2 [-85.7, 0.0] |

All 12 adjusted intervals include or touch zero. The French interval is degenerate because none of its matched cases changed; it does not indicate high precision. These are post-hoc estimates from one 1B seed, and all four competence gates failed, so they cannot answer the hypothesis.

## Secondary target-minus-English contrast

The paired DID asks whether a target language changed differently from English. Positive values mean less damage in the target language. These are separate fully enumerated entity-group bootstrap intervals with nominal Bonferroni familywise 95% coverage over the same 12 cells.

| Method | Language | Cases / groups | DID (pp; familywise 95% CI) |
|---|---|---:|---:|
| GA | French | 10/6 | +10.0 [0.0, +66.7] |
| GA | German | 9/5 | +22.2 [0.0, +100.0] |
| GA | Hindi | 9/6 | +0.0 [-66.7, +66.7] |
| GA | Thai | 11/7 | +9.1 [-42.9, +71.4] |
| GD | French | 10/6 | +10.0 [0.0, +66.7] |
| GD | German | 9/5 | +11.1 [-60.0, +100.0] |
| GD | Hindi | 9/6 | +11.1 [0.0, +66.7] |
| GD | Thai | 11/7 | +9.1 [-71.4, +100.0] |
| NPO | French | 10/6 | +10.0 [0.0, +66.7] |
| NPO | German | 9/5 | +11.1 [-60.0, +100.0] |
| NPO | Hindi | 9/6 | +11.1 [0.0, +66.7] |
| NPO | Thai | 11/7 | +9.1 [-42.9, +71.4] |

All 12 DID intervals include or touch zero. Exact enumeration reproduced all endpoints from the earlier 100,000-draw sensitivity. The intervals are very wide because each contrast has only 5--7 groups; nominal bootstrap coverage is approximate and unstable at this cluster count. No confirmatory p-values are reported.

## Direct fact probe versus final answer

On the same bilingual precompetent cases after unlearning, B is the direct intermediate-fact answer and C is the final multi-hop answer.

| Method | Language | Cases / groups | B correct | Final C correct | B wrong, C correct |
|---|---|---:|---:|---:|---:|
| GA | French | 10/6 | 0/10 | 10/10 | 10/10 |
| GA | German | 9/5 | 0/9 | 8/9 | 8/9 |
| GA | Hindi | 9/6 | 0/9 | 8/9 | 8/9 |
| GA | Thai | 11/7 | 0/11 | 9/11 | 9/11 |
| GD | French | 10/6 | 2/10 | 10/10 | 8/10 |
| GD | German | 9/5 | 1/9 | 8/9 | 7/9 |
| GD | Hindi | 9/6 | 0/9 | 8/9 | 8/9 |
| GD | Thai | 11/7 | 0/11 | 9/11 | 9/11 |
| NPO | French | 10/6 | 1/10 | 10/10 | 9/10 |
| NPO | German | 9/5 | 0/9 | 8/9 | 8/9 |
| NPO | Hindi | 9/6 | 0/9 | 8/9 | 8/9 |
| NPO | Thai | 11/7 | 0/11 | 9/11 | 9/11 |

In these selected cases, direct recall is correct in 0–2 of 9–11 cases while the final answer is correct in 8–10. This shows that measured final-answer performance often remained after direct-probe failure; it does not establish why or prove deep unlearning.

## Prediction provenance discrepancy

The older `r1_pilot_recomputed_v2` file has the same 7,740 composite row keys as the verified v4 output, but 5,906 answer records differ: GA 1,290, GD 1,269, NPO 1,258, retain_only 1,232, sft 857. Its report flags 1,385/7,740 outputs invalid; the verified v4 archive and independent rescore flag 0/7,740. The older file is not bound to a run manifest in that folder, so its metrics are excluded from the authoritative result. The cause of this alternate-output difference remains unresolved.

The v4 archive is reproducible at the scoring layer, but raw Kaggle unlearning manifests omitted the unlearning-script hash. The local V3 snapshot verifier also finds one notebook hash mismatch: 182/183 listed files match, while the working Kaggle notebook differs from the frozen snapshot. The archive’s code/output association is post-run reconciled, not proof that the raw job loaded that exact file.

## Decision

This is a useful debugging diagnostic, not a reliable test of the R1 hypothesis. Limits are one 1B seed, failed baseline competence, low multi-hop baseline accuracy, AI-only translation review, and incomplete training-code provenance. The planned 8B study remains blocked by multilingual data release.

Reproduce the scoring using the command in `pilot_diagnostic_verification.json`; source hashes, values, and the older-output reconciliation are recorded there.
