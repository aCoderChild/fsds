# R1 results summary — 2026-11-03

## Corrected v4 exploratory unlearning pilot

The experiment asks whether English-only unlearning of a first-hop fact changes downstream multi-hop answer accuracy in other languages. The corrected v4 run is exploratory: one seed, a 1B model, and a failed bilingual-competence gate.

### SFT baseline on the 77 forget cases

| Language | Direct accuracy | Multi-hop accuracy | Correct on both before unlearning |
|---|---:|---:|---:|
| English | 94.8% | 20.8% | 15 |
| French | 90.9% | 18.2% | 13 |
| German | 87.0% | 19.5% | 12 |
| Hindi | 87.0% | 18.2% | 11 |
| Thai | 94.8% | 19.5% | 15 |

### Target-language minus English change in multi-hop accuracy

Positive values mean the target-language change was less negative than the English change. Intervals are 95% entity-group bootstrap intervals, on the bilingual precompetent cases only.

| Unlearning method | French | German | Hindi | Thai |
|---|---:|---:|---:|---:|
| GA | +10.0 [0.0, 50.0] pp | +22.2 [0.0, 80.0] pp | 0.0 [−33.3, 33.3] pp | +9.1 [−28.6, 57.1] pp |
| GD | +10.0 [0.0, 50.0] pp | +11.1 [−33.3, 80.0] pp | +11.1 [0.0, 50.0] pp | +9.1 [−42.9, 71.4] pp |
| NPO | +10.0 [0.0, 50.0] pp | +11.1 [−33.3, 80.0] pp | +11.1 [0.0, 50.0] pp | +9.1 [−27.3, 57.1] pp |

All 12 confidence intervals include zero. The point estimates do not show more damage in the non-English languages, but the intervals are too wide to rule out meaningful effects.

### Execution and validity checks

| Check | Result |
|---|---:|
| Prediction rows rescored | 7,740 |
| Generation failures / invalid-output flags | 0 / 0 |
| Bilingual precompetent forget cases, FR / DE / HI / TH | 10 / 9 / 9 / 11 |
| Corresponding entity groups, FR / DE / HI / TH | 6 / 5 / 6 / 7 |
| Required cases and groups per language | 15 and 8 |
| Competence gate | Failed |
| Retain-set multi-hop change | −7.7 to −23.1 pp |
| Seeds / model size | 1 / 1B |

The results ZIP passed CRC and indexed SHA-256 checks. Kaggle’s unlearning manifests did not record the source-script hash, so code/output provenance is reconciled but not cryptographically attested.

## v5.1 source split and evaluation-prompt readiness

The source-backed v5.1 split contains 195 unique cases allocated exactly as planned: 117 forget, 39 retain, and 39 holdout. The independent validator rechecked source R1/R2 facts, unique final-answer support, negative controls, leakage isolation, and all draft hashes. The split is structurally valid but remains quarantined for training.

The current evaluation path is `translategemma_repair_v1_full`, not the older `qwen_blinded_v1` audit. Recomputing the evaluation cohort directly from its hash-bound requests, translations, and cross-model review yields 80 cases, 1,200 prompts, and 1,200 separate labels. The stored files and the recomputed files match. The split counts are:

| Split | Cases | Entity groups | Minimum for later competence gate |
|---|---:|---:|---:|
| Forget | 43 | 24 | 15 cases / 8 groups |
| Retain | 20 | 12 | control set |
| Holdout | 17 | 11 | untouched |

Thus the **translation-screened evaluation cohort** clears the planned size floor before model competence is measured. It does not establish that Llama has learned these facts. The translation review is AI-only, with no fluent-human certification; the screen report records 2,137 passes, 156 flagged rows, and 47 failed translation attempts among 2,340 requested prompts. The frozen 80-case cohort contains only rows that pass the cross-model inclusion rule.

## v5.1 training-translation release and bounded repair diagnostic

Training translations remain the release blocker. Each target language has 488 training questions to screen (312 atomic and 176 composition rows). The Qwen2.5 14B second pass is complete for all 488 questions in each target language. It records 324 French, 309 German, 288 Hindi, and 232 Thai passes; the remaining 164, 179, 200, and 256 rows are flagged, respectively. Recomputing the assembler's strict all-four-language, all-atomic-facts criterion leaves only 5/117 forget cases and 3/39 retain cases eligible. None of the 44 training-only composition chains passes. The release assembler therefore blocks before writing an approved SFT corpus or manifest. The review was AI-only, as authorized for this pilot; no fluent-human review was performed. That limitation is disclosed, but it is not the immediate release-gate failure: insufficient complete translation coverage is.

A separate, diagnostic-only repair calibration sampled eight already-flagged atomic rows and eight already-flagged composition rows per target language (64 total), using stable SHA-256 ordering. TranslateGemma 4B translated all 64; all passed the required deterministic integrity checks, while Qwen2.5 7B accepted 48/64 (75%) for semantic equivalence and target-language fluency:

| Language | Sampled | Deterministic checks passed | Qwen semantic/fluency pass |
|---|---:|---:|---:|
| French | 16 | 16 | 12 |
| German | 16 | 16 | 14 |
| Hindi | 16 | 16 | 13 |
| Thai | 16 | 16 | 9 |
| **Total** | **64** | **64** | **48** |

This is a calibration result, not a corpus-wide acceptance estimate or release decision. The sample remains small, and the 16 rejected examples include relation-direction, scope, answer-type, and fluency issues; Thai had the lowest observed acceptance (9/16). No candidate was copied into the source translations, first-stage reviews, or release manifest. The hash-bound output is `experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/translategemma4b_qwen7b_calibration64_2026-11-03/`.

## Full flagged-row repair diagnostic

All 948 rows flagged by the first-stage training-translation screen received a TranslateGemma 4B repair candidate. All 948 passed deterministic integrity checks. Qwen2.5 7B returned valid semantic/fluency judgments for 947: 683 passed and 264 were flagged. One Thai atomic item produced an invalid Qwen response (a decision with an empty reason); it is counted as an error, not a pass or flag.

| Language | Atomic Qwen pass / valid | Composition Qwen pass / valid | Overall Qwen pass / valid |
|---|---:|---:|---:|
| French | 103 / 131 (78.6%) | 37 / 64 (57.8%) | 140 / 195 (71.8%) |
| German | 150 / 166 (90.4%) | 51 / 77 (66.2%) | 201 / 243 (82.7%) |
| Hindi | 105 / 127 (82.7%) | 58 / 100 (58.0%) | 163 / 227 (71.8%) |
| Thai | 126 / 184 (68.5%) | 53 / 98 (54.1%) | 179 / 282 (63.5%) |
| **Total** | **484 / 608 (79.6%)** | **199 / 339 (58.7%)** | **683 / 947 (72.1%)** |

The Thai atomic denominator excludes the one malformed review response; 185 rows were selected in that cell. The 264 Qwen-flagged candidates and the one malformed review remain unresolved. These are AI-only screens, not independent human judgments: the Qwen reviewer is in the same model family as the first-stage reviewer. The outputs are quarantined diagnostic candidates; no source translation or release manifest was changed. The v5.1 training release remains `NOT_APPROVED_FOR_TRAINING`, and no v5.1 SFT or unlearning run has started. The hash-bound run is `experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03/`.

## Feedback-guided repair diagnostic

A second, separate calibration sampled eight previously Qwen-flagged atomic rows and eight previously Qwen-flagged composition rows in each target language (64 rows total). TranslateGemma 4B received the earlier Qwen critique as explicitly unverified feedback; the English source remained authoritative. All 64 new candidates passed deterministic integrity checks, but Qwen2.5 7B accepted only 16/64 (25%); 48 remained flagged and there were no generation or review errors.

| Language | Atomic Qwen pass | Composition Qwen pass | Total Qwen pass | Deterministic checks |
|---|---:|---:|---:|---:|
| French | 4 / 8 | 2 / 8 | 6 / 16 | 16 / 16 |
| German | 2 / 8 | 2 / 8 | 4 / 16 | 16 / 16 |
| Hindi | 2 / 8 | 1 / 8 | 3 / 16 | 16 / 16 |
| Thai | 2 / 8 | 1 / 8 | 3 / 16 | 16 / 16 |
| **Total** | **10 / 32** | **6 / 32** | **16 / 64** | **64 / 64** |

This is a stratified diagnostic of difficult rows, not a corpus-wide estimate. The earlier candidates for these rows were Qwen-flagged by selection, so the 16 accepted replacements are same-reviewer rescues, not independent confirmation of translation quality. The reviewer is also in the same Qwen model family as the original screen. The new candidates were not merged; source translations, release manifests, and SFT status are unchanged. Given the 25% acceptance rate, scaling this exact critique-guided repair route to all remaining flags is not justified. The hash-bound output and resume integrity check are in `experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/translategemma4b_qwen7b_feedback64_2026-11-03/`.

## Short interpretation and next step

The corrected v4 model run is a real but exploratory result: the pre-unlearning bilingual-competence gate failed, and all 12 cross-language confidence intervals include zero. It does not show that unlearning has no cross-language effect. The v5.1 evaluation prompts now have a hash-verified 80-case cohort above the size floor, but training translations do not yet provide complete five-language training coverage. Starting SFT now would violate the training design and spend compute on an unapproved corpus.

Next, freeze a revised answer-blind translation-resolution policy before spending more compute. The feedback-guided calibration accepted only 16/64 difficult rows under the same Qwen reviewer, so do not scale that exact route. The full Qwen2.5 14B second pass is now complete, but it leaves only 5/117 forget cases, 3/39 retain cases, and 0/44 composition chains eligible under the assembler's strict complete-case rule. Resolve or explicitly exclude the remaining rows under a predeclared policy, then rerun the release validator. AI-only review was authorized for this pilot, but it does not certify fluent-human translation quality; describe this as a limitation and use fluent reviewers for a confirmatory release if resources permit. Only after a valid release should the three-seed SFT and retain-only references run; unlearning remains gated on the baseline competence check. Keep the 15-case/8-group threshold unchanged; it is a feasibility gate, not a power calculation.

## Source records

- Current source/data-gate audit: `reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/diagnostic_report.md`
- Current audit evidence: `reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verification.json`
- Corrected v4 detailed report: `reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/extracted/corrected_evaluation/diagnostic_report.md`
- Corrected v4 archive verification: `reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/LOCAL_VERIFICATION.json`
- v5.1 source/split verification: `experiments/synthworlds_r1/LOCAL_VALIDATION.json`
- v5.1 translation merge: `experiments/synthworlds_r1/translation_drafts/merged_v1/report.json`
- v5.1 evaluation translations: `experiments/synthworlds_r1/translation_reviews/translategemma_repair_v1_full/report.json` and `summary.json`
- v5.1 recomputed evaluation cohort and quarantined training plan: `experiments/synthworlds_r1/training_release_draft/PREPARATION_REPORT.json`
- v5.1 stopped training-translation adjudication: `experiments/synthworlds_r1/training_release_draft/translation_reviews_roundtrip_qwen14b/report.json`
- v5.1 64-row repair calibration: `experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/translategemma4b_qwen7b_calibration64_2026-11-03/report.json`
- v5.1 full flagged-row repair diagnostic: `experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/translategemma4b_qwen7b_all_flagged_2026-11-03/report.json`
- v5.1 critique-guided feedback repair diagnostic: `experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/translategemma4b_qwen7b_feedback64_2026-11-03/report.json`
- v5.1 offline pipeline fixtures: `experiments/synthworlds_r1/scripts/test_r1_pilot_pipeline.py` (12 tests; synthetic fixtures, no model training)
