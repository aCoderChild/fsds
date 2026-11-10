# SynthWorlds-SM v4 exploratory diagnostic: run accuracy and current status

**Generated:** 2026-11-03  
**Interpretation:** exploratory only; the bilingual competence gate failed. The archive’s post-unlearning outputs are reproducible, but GD/NPO used a retain-row sampling schedule that has since been corrected. The corrected Kaggle run is pending.

## What was executed

The archived Kaggle run used `meta-llama/Llama-3.2-1B-Instruct`, seed 42, three epochs, and the 3,355-row five-language corpus (671 rows per language; 765 forget, 390 retain, 2,200 training-only). Full SFT and retain-only SFT completed. The 129-case evaluation comprised 77 forget, 26 retain, and 26 holdout cases. SFT and retain-only manifests bind to corpus SHA `2b2119939fe133286d2a70de1f405dc0227332f9615d87bb92c1e65b76f3d95c`.

The baseline generated 3,870 predictions (pretrained, full SFT, retain-only); the post-unlearning run generated 7,740 (SFT, retain-only, GA, GD, NPO, and pretrained). The manifests report zero generation failures and zero output-invalid flags. Holdout facts were not used for training/unlearning. Holdout accuracy is a split-control measure, not evidence of forgetting.

## Baseline competence on forget cases

| Language | Cases | Direct accuracy | Multi-hop accuracy | Precompetent cases |
|---|---:|---:|---:|---:|
| EN | 77 | 94.8% | 20.8% | 15 |
| FR | 77 | 90.9% | 18.2% | 13 |
| DE | 77 | 87.0% | 19.5% | 12 |
| HI | 77 | 87.0% | 18.2% | 11 |
| TH | 77 | 94.8% | 19.5% | 15 |

The English target-selection rule required the SFT model to answer both direct and full multi-hop prompts correctly. It selected 15 of 77 forget cases. Bilingual pairs and leakage-isolated groups were FR 10/6, DE 9/5, HI 9/6, TH 11/7. The declared gate floor was 15 matched cases and 8 groups per language; **all four target languages failed**. Standard unlearning must remain blocked at this gate.

## Archived exploratory primary contrasts

The contrast is change in target-language multi-hop accuracy minus change in English accuracy, on cases SFT answered correctly in both languages. Negative values indicate additional loss in the target language. Confidence intervals bootstrap leakage-isolated entity groups; they are not multiplicity-adjusted.

| Language | Cases/groups | GA DID [95% CI], pp | GD DID [95% CI], pp | NPO DID [95% CI], pp |
|---|---:|---:|---:|---:|
| FR | 10/6 | +10.0 [0.0, 50.0] | +10.0 [0.0, 50.0] | +10.0 [0.0, 50.0] |
| DE | 9/5 | +22.2 [0.0, 80.0] | +11.1 [-33.3, 80.0] | +11.1 [-33.3, 80.0] |
| HI | 9/6 | +0.0 [-33.3, 33.3] | +11.1 [0.0, 50.0] | +11.1 [0.0, 50.0] |
| TH | 11/7 | +9.1 [-28.6, 57.1] | +9.1 [-42.9, 71.4] | +9.1 [-27.3, 57.1] |

Every interval includes zero, and the samples are small. These diagnostics do not establish a cross-language effect or its absence. They are not confirmatory evidence.

## Retain utility diagnostic

Post-unlearning minus SFT multi-hop accuracy on the 26 retain cases (percentage points):

| Method | Language | Change (pp) |
|---|---|---:|
| GA | EN | -19.2 |
| GA | FR | -15.4 |
| GA | DE | -23.1 |
| GA | HI | -7.7 |
| GA | TH | -11.5 |
| GD | EN | -19.2 |
| GD | FR | -19.2 |
| GD | DE | -15.4 |
| GD | HI | -11.5 |
| GD | TH | -15.4 |
| NPO | EN | -19.2 |
| NPO | FR | -19.2 |
| NPO | DE | -15.4 |
| NPO | HI | -11.5 |
| NPO | TH | -11.5 |

Retain prompts overlap SFT training by design, so these values are utility diagnostics, not held-out generalization estimates.

## Code audit and correction

The archived GA/GD/NPO runs each used 15 English forget rows, 26 available English retain rows, and zero holdout rows. The archived GD/NPO code reset its retain-row index at each epoch. With 15 forget steps per epoch and 26 retain examples, it reused the same first 15 retain rows in all three epochs; the remaining 11 rows were never used. GA has no retain regularizer and is unaffected by this specific sampler defect.

The current `scripts/unlearn_lora.py` instead cycles across all 26 rows before reuse over the 45 GD/NPO steps. It records the unique retain row IDs used in the run manifest. The standard path still rejects a failed competence gate. An opt-in exploratory path allows the exact gate-only failure and rejects any additional blocker. The Kaggle cell checks that the expected unique retain IDs were consumed and writes to new `retainfix` output paths.

The updated package is `data/synthworlds_sm_pilot_v4/kaggle/r1_retainfix_dataset.zip`; its upload-ready directory is `data/synthworlds_sm_pilot_v4/kaggle/r1_retainfix_upload/`. A fresh corrected GD/NPO run has **not** been executed. The Kaggle Notebook Editor session is required because `HF_TOKEN` is available from Kaggle Notebook Secrets there; API/CLI-launched kernels do not receive the secret.

## Verification performed locally

- ZIP CRC: passed; 82 archive entries.
- Artifact index: all 81 listed files matched recorded byte counts and SHA-256 values.
- Full-SFT and retain-only run manifests: `TRAINING_COMPLETE`; corpus and manifest checksums matched the archived inputs.
- GA/GD/NPO manifests: `UNLEARNING_COMPLETE`, one seed, explicitly exploratory after failed gate; zero holdout rows.
- Baseline/post prediction rescoring: current evaluator reproduced all scored predictions, summary JSON, and contrast JSON **byte-for-byte**.
- Type-B source verifier: 85 chains and 340 English prompts, no errors.
- R1 experiment fixtures: 14 tests pass; SFT pipeline fixtures: 7 tests pass.
- Notebook JSON parsed and all code cells compiled as Python syntax.

These checks validate the artifacts, deterministic scoring, and the corrected implementation fixtures. They do not prove that the corrected code has run on Kaggle or that the translations are fluent/semantically correct; translation review is AI-only and no native-speaker review was performed.
