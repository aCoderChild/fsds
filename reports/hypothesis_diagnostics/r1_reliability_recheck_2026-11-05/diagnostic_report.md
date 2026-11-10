# R1 reliability recheck — 2026-11-05

**Stage:** multilingual training-data release. The planned Llama 3.1 8B SFT, baseline, and unlearning runs have not started because the source-verified corpus is not releasable. The checked workspace contains no standalone timestamped preregistration artifact; this report does not label the protocol formally preregistered.

## Current diagnostic numbers

| Check | Result | Interpretation |
|---|---:|---|
| Frozen source split | 195 cases; 117 forget / 39 retain / 39 holdout; 102 leakage groups | Source structure passes; split remains a draft, not approval |
| Eligible release coverage | 5/117 forget; 3/39 retain; 0/44 composition chains | Release assembler correctly blocks; no training corpus emitted |
| Primary targeted translation queue | 36/187 pass (19.3%); 151 flagged after 18 retries | Candidate-only AI screen; selected gap queue, not a population rate |
| 1B pilot paired contrasts | 12/12 95% group-bootstrap intervals include or touch zero | Exploratory only; baseline competence failed and there was one 1B seed |
| 8B runtime smoke | One synthetic LoRA step and one synthetic NPO update on 2×T4 | Runtime compatibility only; no research data |
| Scoped Kaggle jobs | 5/5 complete | Includes the legacy 1B pilot alias, three translation diagnostics, and the synthetic 8B smoke; completion status alone does not mean training ran |
| Full related-kernel inventory | 38 kernels in the visible account inventory; 36 complete, 2 error | Naming-filtered point-in-time inventory; no related job is currently running |
| Legacy notebook execution log | PREFLIGHT_ONLY_NO_TRAINING | The published alias log is for the legacy Llama 3.2 1B pilot dataset, records training_started=false, and is not a v5.1 main-study run. |

The latest 187-row review passed 36/187 (19.3%); by language: FR 8/21, DE 3/19, HI 14/62, TH 11/85. The independent, hash-bound candidate-union audit adds 49 unique prompt-language rows to the previous 1,628, for 1,677 candidate rows total. It estimates 42/117 forget and 16/39 retain cases with all four target-language prompts, spanning 22 and 12 leakage groups; the optimistic relation-aware template ceiling is 47/117 and 19/39. Only 2/44 composition chains are complete, with 114 translation rows missing. These are quarantined AI-screened coverage diagnostics, not released training data or quality estimates.

| Follow-up screen | Reviewed / blocked | Joint passes | Net new unique rows |
|---|---:|---:|---:|
| Retry of 17 prior blocked rows | 17 / 0 | 3 | 0 |
| 148-row NLLB alternate batch | 23 / 125 | 3 | 1 |
| Marker-preserving segmented retry | 124 / 1 | 0 | 0 |

The batches overlap earlier candidate rows; raw pass counts must not be summed. A cross-review disagreed with the initial Qwen labels on 154/187 rows. Constructed mismatch checks passed 2/15 for Mistral and 1/16 for Qwen14; these are small probes, not reviewer error-rate estimates. There are no human-gold controls. Every output remains AI-only and quarantined. The candidate-union audit records row-level deduplication, model/script hashes, and screening receipts.

A **separate** full flagged-question repair union combines ChatGPT and TranslateGemma alternatives: 445 and 397 diagnostic passes, with 230 overlapping rows and 612 unique rows with at least one candidate pass. If combined optimistically with the original production passes, it reaches 60/117 four-language forget cases, 19/39 retain cases, and 6/44 complete composition chains. This is an availability ceiling, not a release: the candidate screens use different AI-only criteria, have no human-gold control, and every selected translation must pass the frozen production review. These rows are not added to the separate targeted-gap counts above.

A second, deliberately optimistic ceiling combines that 799-row union with source-bound targeted-gap candidates and the later retries. The batches are deduplicated by source question and language before case counts are recomputed:

| Cumulative candidate set | Forget cases | Retain cases | Composition chains |
|---|---:|---:|---:|
| Full flagged-question union | 60/117 | 19/39 | 6/44 |
| + targeted-gap candidates | 69/117 | 19/39 | 7/44 |
| + blocked-row retry | 69/117 | 20/39 | 7/44 |
| + NLLB alternatives | 69/117 | 21/39 | 7/44 |
| + marker-preserving retry | 69/117 | 21/39 | 7/44 |

The broadest union reaches 69/117 forget, 21/39 retain, and 7/44 composition cases; the last segmented retry added no passing rows. This remains a candidate-availability ceiling: screens are heterogeneous and AI-only, and no candidate has passed the full frozen production review. The authoritative release gate remains 5/117 forget, 3/39 retain, and 0/44 composition chains.

For the 1B pilot, a post-hoc Bonferroni sensitivity over 12 contrasts with 100,000 entity-group bootstrap draws left every interval covering zero; the simultaneous interval extremes were -71.4 to +100.0 percentage points. With only 5--7 groups per contrast, this is evidence of low precision, not evidence of no cross-language transfer.

An exact enumeration of every ordered cluster-bootstrap draw (3,125--823,543 resamples per contrast) reproduced all 12 marginal and Bonferroni-adjusted interval endpoints from the 100,000-draw simulation exactly (maximum endpoint difference: 0.0 pp). This removes Monte Carlo resampling noise; it does not reduce the underlying uncertainty from only 5--7 groups per contrast.

A separate exact entity-group sign sensitivity gives no familywise signal: all 24 adjusted sign-test p-values (12 primary and 12 secondary) are 1.00. There are only 5--7 groups per pilot contrast, with many tied zero-change groups. At the design's minimum of eight groups, even eight non-tied groups all moving in one direction yield a two-sided sign p-value of 0.0078, above the 0.00417 per-cell threshold for 12 tests; nine non-tied groups are the mathematical minimum for this test to reach that threshold. This is a small-sample sensitivity result, not a power recommendation. See the [exact group sign sensitivity](../r1_reliability_tracker_2026-11-05/pilot_exact_cluster_sign_test.md).

A separate post-hoc downstream-answer-persistence analysis finds final answers correct in 7--10 of 8--12 post-unlearning direct-recall failures per method-language cell (DAP 81.8--100.0%). It exactly resamples leakage groups. Some intervals are degenerate at 100% because all observed failures in those cells retained a correct final answer; this boundary behavior is a small-sample limitation, not evidence of certainty. See the [DAP results and limits](../r1_reliability_tracker_2026-11-05/pilot_dap_exact_cluster_bootstrap.md). This result does not establish reasoning recovery or override the failed competence gate.

## Verification run

- Offline implementation fixtures: **36/36 passed**.
- Main Kaggle notebook: frozen-label SHA-256 is checked before baseline/post scoring, after generation; notebook SHA-256 is recorded in `verification.json`.
- Fresh pilot rescore: **all 7 derived score/report files byte-identical** to the archived outputs; 7,740 prediction rows.
- Exact pilot bootstrap sensitivity: **12/12 intervals reproduced by exhaustive resampling**, with 0.0 pp endpoint difference from the 100,000-draw analysis; the output is post-hoc and exploratory.
- Exact entity-group sign sensitivity: **24/24 Bonferroni-adjusted p-values are 1.00**; the eight-group feasibility floor cannot reach the 12-cell threshold even under unanimous non-tied directions.
- Pilot downstream-answer-persistence diagnostic: **12/12 cells recomputed** from hash-bound case metrics with exact entity-group resampling; still exploratory and one-seed.
- 187-row overlay: hash, row coverage, retry lineage, and counts verified; status is diagnostic-only.
- Latest alias, NLLB alternate, and segmented NLLB receipts: all three verifiers reproduce; their net new contribution is 1, not 6, after deduplication.
- Candidate-union audit: 1,677 quarantined prompt-language candidates; source-bound release counts remain unchanged.
- Live Kaggle status query (2026-11-05T15:35:24.124880+00:00): 5/5 scoped diagnostic/smoke kernels complete.
- Full related-kernel inventory (2026-11-05T15:36:24.419920+00:00): 38 matching jobs across 106 visible kernels; 36 complete, 2 error, and none running. See the [full inventory](../r1_reliability_tracker_2026-11-05/kaggle_related_job_inventory_2026-11-05.md).
- 8B runtime receipt: exact model revision and runner verified; `research_data_loaded=false`.
- Main source assembler: **blocked at the expected translation-coverage gate**. The command wrote no training release.
- Local SFT preflight: blocked because no approved corpus exists. Missing local packages are not evidence about Kaggle's environment.

The detailed historical pilot tables and confidence intervals are in [the completed v4 diagnostic](../r1_current_state_audit_2026-11-04/completed_v4_pilot_hypothesis_diagnostic.md). The complete machine-readable receipts and file hashes are in [verification.json](verification.json).

## Conclusion

There are reproducible diagnostic results, but no valid estimate from the planned main experiment yet. Do not interpret the small 1B pilot or candidate-screen percentages as confirmation. The next experiment gate is a fully source-bound multilingual release, followed by three-seed baseline competence checks; unlearning must remain off until that baseline gate passes.
