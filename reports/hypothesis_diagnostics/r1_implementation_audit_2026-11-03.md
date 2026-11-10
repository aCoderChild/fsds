# R1 v5.1 implementation and readiness audit

Audit date: 2026-11-03  
Scope: `docs/report/R1.tex`, the v5.1 release assembler, SFT/unlearning/evaluation scripts, Kaggle notebook, current release reports, and local fixture checks.

## Assessment

The v5.1 pilot implementation encodes the main causal comparison in R1 reasonably faithfully. It trains a five-language composition-SFT model, requires a same-seed retain-only reference, selects only English forget cases learned across three full-SFT seeds, unlearns the English first-hop relation, and evaluates direct, negative-control, and multi-hop probes in five languages. The effect script computes paired target-language-minus-English changes and cluster-bootstrap intervals; it also reports per-seed variation and retain-set changes.

This is an implementation/design assessment, not evidence that the actual v5.1 experiment ran. The v5.1 corpus release is blocked, so its model stages have not run. The older v4 exploratory outputs are real diagnostics but remain one-seed and below the predeclared competence gate.

## Continuation update: complete translation screen and verified release gate

On 2026-11-03, the resumable Qwen2.5 14B second pass was completed for all 1,952 target-language training questions (488 each in FR/DE/HI/TH). The hash-bound verifier accepts the complete review files, but the corpus builder correctly rejects the source translations as insufficiently complete:

| Language | Final AI-screen passes | Flagged |
|---|---:|---:|
| FR | 324/488 | 164 |
| DE | 309/488 | 179 |
| HI | 288/488 | 200 |
| TH | 232/488 | 256 |

Across the same strict final-status rule, only 5/117 forget cases, 3/39 retain cases, and 0/44 composition chains have complete passing translations in all four target languages. The required counts remain 117, 39, and 44. The assembled corpus, manifest, and release-validation file were not written.

The first execution of `checked_translation_reviews()` also exposed an implementation bug: it treated a legitimate Qwen2.5 14B downgrade of a first-stage pass as a corrupt review record. The validator now accepts such a row as flagged/excluded, while still rejecting inconsistent final passes or flags. A regression fixture covers the downgrade rule. The actual verifier then passed all 1,952 hash-bound rows; `build_corpus()` stopped at the incomplete atomic-case gate. The fixture suite passes 12/12 and the assembler compiles. A twelfth regression test also verifies that a Kaggle SFT resume rejects a run manifest made by a different training-script revision.

The screen remains AI-only. Qwen2.5 7B and 14B share a model family, and their judgments disagree often; this is not fluent-human certification. Completing the second pass did not make the corpus trainable.

## Bounded NLLB repair calibration (diagnostic only)

To avoid spending GPU time on a full repair pass, a deterministic, answer-blind sample of 64 difficult flagged rows was selected: 8 each for every language-by-type cell (FR/DE/HI/TH × atomic/composition). Each row was assigned an NLLB candidate if it passed the paired Qwen7 screen, otherwise the exact hash-matched TranslateGemma repair when that had passed Qwen7. The selected translation then went through the existing TranslateGemma back-translation checks and Qwen14B semantic/fluency adjudication. This is a deliberately stratified diagnostic sample from unresolved rows, not a random or representative sample of the corpus.

| Language | NLLB selected | TranslateGemma fallback | No Qwen7-passing candidate | Complete AI screen pass |
|---|---:|---:|---:|---:|
| FR | 14 | 1 | 1 | 9/16 |
| DE | 12 | 4 | 0 | 5/16 |
| HI | 15 | 1 | 0 | 8/16 |
| TH | 14 | 2 | 0 | 6/16 |
| **Total** | **55** | **8** | **1** | **28/64** |

All 63 selected candidates passed the deterministic target checks and back-translation checks; Qwen14B then flagged 35, passed 28, and returned no review errors. No answer fields were sent to any translator or reviewer. The result is AI-only and does not constitute fluent-human validation. Since the sample was enriched for difficult flagged rows, 28/64 is not an estimate of expected full-dataset coverage. It is still a weak yield for this repair recipe, so the resource-conscious decision is **do not scale this NLLB/TranslateGemma recipe to all flagged rows and do not train from these candidates**. The original strict release gate remains unchanged and failed.

The calibration harness's dry run verified all 64 unique rows and matched each fallback to the exact source-hash-bound TranslateGemma repair record before loading models. Independent post-run checks recomputed all language/type counts, verified the selected-row SHA-256, confirmed answer-blindness and `do_not_train=true`, and confirmed that no corpus, manifest, or release-validation artifact was created. The diagnostic outputs are in `experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/nllb13_cal64_roundtrip_qwen14_2026-11-03/`; the runner is `experiments/synthworlds_r1/training_release_draft/scripts/adjudicate_nllb13_cal64.py`.

## Requirement-to-code checks

| R1 requirement | Implementation evidence | Assessment |
|---|---|---|
| English-only unlearning of the first-hop relation | `scripts/unlearn_lora.py` filters to English `r1` rows and forget/retain splits; it rejects holdout data. | Matches the target definition. |
| Confirm learned facts before unlearning | `scripts/evaluate_pilot_three_probe.py baseline-gate` requires three distinct full-SFT seeds and all three probes correct on matched English/target-language cases. | Strong safeguard against interpreting never-learned facts as forgotten. |
| Direct, negative-control, and multi-hop evaluation | The frozen prompt set and evaluator use `direct_positive`, `direct_negative_control`, and `multi_hop`; generation reads prompts and scoring reads labels separately. | Matches the planned controls. |
| Compare unlearning methods and preserve unrelated knowledge | GA, GD, and NPO are implemented; GD/NPO use a deterministic retain-row schedule. Effects include forget, retain, and holdout splits. | Structurally aligned; actual convergence and utility effects remain unverified. |
| Multiple seeds and uncertainty | Kaggle notebook schedules seeds 42, 43, and 44; effects use entity-group bootstrap and summarize seed mean/SD. | Matches the pilot plan; no real v5.1 run evidence yet. |
| Control for prior knowledge and general degradation | Notebook scores pretrained, full-SFT, and retain-only checkpoints; the R1 analysis is restricted to full-SFT-precompetent cases and separately reports retain changes. | Appropriate controls are present. The already-published v4 results still have wide intervals and a failed competence gate. |
| Reasoning depth | The frozen v5.1 cohort contains only two-hop cases, which the report explicitly discloses. | Three- and four-hop effects are not estimable in this pilot. |

The design choices are consistent with the cited motivation: cross-lingual forgetting needs pre-learning and retain controls; multi-hop outcomes can diverge from direct-fact outcomes; and shortcut-resistant evaluation must avoid easy head-to-answer associations. The implementation also records that translation review is AI-only, not human language certification.

## Current release and compute state

| Check | Current evidence | Result |
|---|---|---|
| Source split validation | `LOCAL_VALIDATION.json` | Passes source/split validation, but explicitly says not approved for training or evaluation. |
| Translation preparation snapshot | `training_release_draft/PREPARATION_REPORT.json` | Historical status says review pending; current review status is recorded in the Qwen14B report below. `do_not_train=true`. |
| Second-pass translation screen | `translation_reviews_roundtrip_qwen14b/report.json` | Complete: 488/488 for each target language; hash-bound and answer-blind. Final passes: FR 324, DE 309, HI 288, TH 232. AI-only, not human-certified. |
| Strict complete-case gate | `assemble_training_release.py` / `build_corpus()` | 5/117 forget, 3/39 retain, 0/44 composition chains; required counts are 117/39/44. Fails closed before writing a corpus. |
| Repair diagnostic upper bound | `optimistic_coverage_report.json` | First-stage passes OR Qwen7 repair passes: 52/117 forget, 17/39 retain, 1/44 composition chains. Repaired rows have not passed frozen release review. |
| Approved training artifacts | `sft_training/corpus.jsonl`, `manifest.json`, `release_validation.json` | All absent. |
| Local SFT preflight | `scripts/train_sft_lora.py` | Blocked on missing approved corpus/manifest; local environment also lacks `peft`, `accelerate`, and an HF token. No model download or training occurred. |
| Fixture suite and syntax | `scripts/test_r1_pilot_pipeline.py`; assembler `py_compile` | 12/12 fixture tests pass; the adjudication and assembler scripts compile. These do not validate real SFT behavior or Kaggle execution. |

The assembler intentionally refuses partial coverage: it requires all 117 forget cases, 39 retain cases, and all 44 training-only composition chains to be complete across the four target languages. Relaxing that gate would change the planned dataset and could leave the model without the intended multi-hop training signal, so this audit does not recommend doing so.

## Risks before spending GPU time

1. **Data is the decisive blocker.** The 14B screen is complete, but strict eligible coverage is only 5/117 forget, 3/39 retain, and 0/44 composition chains. The current repair upper bound is also far below the frozen release requirements. More review of these same unchanged translations cannot produce a release; new, substantially better reviewed translations are needed.
2. **AI-only translation review has a known ceiling.** The reviewer models are in the same Qwen family; deterministic marker/entity checks do not establish translation equivalence or naturalness. This remains explicitly disclosed and is not human fluency validation.
3. **The current implementation is the 1B feasibility pilot.** R1 reserves Llama 3.1 8B for the main study. A successful 1B pilot would not by itself establish the main-study claim.
4. **Interruption recovery is incomplete for active training.** The notebook snapshots completed models/stages and generation can resume from prediction rows, but SFT refuses partial run directories and the custom unlearning loop writes its final adapter only after all steps. A mid-run interruption can require repeating that model/method run.
5. **No real v5.1 outcomes exist.** Existing v4 tables are exploratory; the v4 competence gate failed and the one-seed paired intervals are wide. They do not support a conclusion for or against the hypothesis.

## Current-state continuation audit (2026-11-03)

The separate 2026-11-02 composition-baseline artifact was verified independently from raw predictions. All files listed in its `artifact_index.json` match their recorded byte sizes and SHA-256 values. Re-running the archived evaluator in score-only mode against its 3,870 prediction rows and the matching v4 evaluation set reproduced `summary.json`, `cross_language_contrasts.json`, `diagnostic_report.md`, `scored_predictions.jsonl`, and `case_metrics.jsonl` byte-for-byte. The matching training-corpus SHA-256 is present in the archived run manifests and resolves to the local v3 corpus snapshot. This verifies the scoring/report derivation; it does not prove the original GPU generation or training was free of implementation/runtime errors.

The baseline gate calculation from `case_metrics.jsonl` yields only 2, 2, 4, and 3 bilingual SFT-precompetent forget cases in FR, DE, HI, and TH, respectively, spanning 1, 2, 2, and 2 entity groups. This is well below the declared 15-case/8-group floor. Accordingly, `cross_language_contrasts.json` is empty because no unlearning was run. The archived baseline SFT multi-hop exact-match rates on forget cases are EN 13.0%, FR 7.8%, DE 18.2%, HI 15.6%, and TH 15.6%; these low rates explain why a cross-lingual forgetting effect is not estimable from this run.

The latest Kaggle kernel state was also checked through Kaggle's read-only CLI and saved output log. The kernel is `COMPLETE`, but that run uses the older v4 notebook plus v3/v4 inputs. Its T4 was visible and the package smoke test passed, but the notebook-secret lookup failed in that API-launched job. Both training switches and all downstream stage switches were false, and the combined access/training gate blocked. The log explicitly records `training_started=false`; no SFT, unlearning, or evaluation stage ran in that job. This is separate from the archived 2026-11-02 baseline outputs above.

The current local v5.1 notebook has no executed cells or outputs. Its data assembler still fails closed at 5/117 forget cases and 3/39 retain cases, with 0/44 composition chains passing; the approved corpus and manifest are absent. The local SFT preflight returns `BLOCKED` before model download. Thus the software gates are functioning as intended, but the new v5.1 experiment is not running and has no model results. Full raw evidence and reproduction checks are in `reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/`.

## Resource-conscious next gate

Do not start SFT or unlearning on the current data. The full review is complete, and its actual strict case/chain coverage has been checked. Replace the rejected translations with a better method, run the frozen checks and independent AI screen again, and require the assembler plus SFT preflight to pass before using Kaggle GPU time. If those data gates cannot be cleared, report a data-quality no-go rather than training on a weakened cohort. After release, run the six SFT adapters, baseline evaluation/gate, nine unlearning runs only if the baseline gate passes, then post-evaluation and paired analysis.

## Literature checked

- Xiang et al., [Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility](https://proceedings.mlr.press/v306/xiang26b.html).
- Waseem and Fawaz, [A False Positive in Cross-Lingual Unlearning Evaluation](https://aclanthology.org/2026.mrl-1.43/).
- Yang et al., [Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](https://aclanthology.org/2025.findings-acl.205/).
- Zhong et al., [MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions](https://aclanthology.org/2023.emnlp-main.971/).
- Hwang et al., [Knowledge Beyond Language: Bridging the Gap in Multilingual Machine Unlearning Evaluation](https://aclanthology.org/2026.acl-long.1105/).
- Qian et al., [Leak-Resistant Unlearning: A New Benchmark for Evaluating Multi-Hop Reasoning Consistency and Recovery Robustness](https://aclanthology.org/2026.findings-emnlp.2004/).
