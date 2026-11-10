# R1 reliability tracking — 5 November 2026

**Observed:** 2026-11-05 14:31 UTC. **Stage:** training-data release; no main training run active.

## Current status

- The 13:50 UTC audit found all 5 scoped Kaggle diagnostic/smoke kernels `COMPLETE`; a 14:21 UTC refresh confirms the pilot and 8B runtime-smoke kernels remain `COMPLETE`. No main R1 training or unlearning job is running.
- The 36 offline pipeline fixtures pass. Prepared-source alignment also passes for 488 canonical SFT IDs. These are code/data-integrity checks, not training results.
- The V3 snapshot verifier still fails on one working-file hash: 182/183 manifest-listed files match, but `kaggle/sft_lora_pilot.ipynb` differs from its snapshot hash. The checksum index matches the manifest; archived pilot scoring reproduces separately.
- The source-bound assembler still fails closed: 5/117 forget cases, 3/39 retain cases, and 0/44 training-only composition chains meet the current translation gate. It wrote neither `corpus.jsonl` nor `manifest.json`.
- The pinned Llama 3.1 8B runtime-only smoke receipt verifies one synthetic LoRA optimizer step and one synthetic NPO update. It loaded no research data and is not an experiment.
- The latest quarantined candidate union has 1,677 passing prompt-language rows, up 49 from the prior 1,628-row union. It completes 42/117 forget cases (22 leakage groups), 16/39 retain cases (12 groups), and 2/44 composition chains; the optimistic template-reuse ceiling is 47/117, 19/39, and 2/44. None changes source-bound release eligibility.

## Pilot statistics — exact cluster sensitivity

The only hypothesis-adjacent result remains the one-seed Llama 3.2 1B pilot. The target-language baseline-competence gate failed in FR, DE, HI, and TH (15-case/8-group minimum; observed 9–11 cases across 5–7 groups). The primary outcome is target-language post-minus-pre multi-hop exact-match change on the same bilingual precompetent cases; target-minus-English DID is secondary.

| Language | Cases / groups | Primary change (pp; familywise 95% CI) | Secondary DID (pp; familywise 95% CI range across methods) |
|---|---:|---:|---:|
| French | 10 / 6 | 0.0 [0.0, 0.0] | +10.0 [0.0, +66.7] |
| German | 9 / 5 | -11.1 [-80.0, 0.0] | +11.1 to +22.2; CI envelope [-60.0, +100.0] |
| Hindi | 9 / 6 | -11.1 [-66.7, 0.0] | 0.0 to +11.1; CI envelope [-66.7, +66.7] |
| Thai | 11 / 7 | -18.2 [-85.7, 0.0] | +9.1; CI envelope [-71.4, +100.0] |

All 12 primary and all 12 secondary Bonferroni-adjusted exact cluster-bootstrap intervals include or touch zero. The French primary interval is degenerate because no matched cases changed. Exact enumeration removes Monte Carlo resampling noise, but 5–7 groups leave coverage unstable; this is post-hoc diagnostic evidence, not a reliable hypothesis test. Full method-by-language values and hashes are in [`pilot_exact_cluster_sensitivity.md`](pilot_exact_cluster_sensitivity.md).

## Latest translation diagnostics

| Candidate path | Outcome | Release effect |
|---|---:|---|
| Targeted 187-row screen | 36 joint passes / 170 reviewed; 17 blocked | Quarantined; none approved |
| NLLB 148-row screen | 3 joint passes / 23 reviewed; 125 blocked | Quarantined |
| Segmented retry of 125 | 0 semantic passes / 124 reviewed; 1 blocked | Quarantined |
| Marker-safe 18-row retry | 1 pass, 2 flags, 15 blocked | Quarantined |
| Qwen retry of 15 | 5 passes, 9 flags, 1 blocked | Quarantined |
| Key-constrained one-row retry | 1 AI-only pass | Quarantined |

All these checks are AI-only; the pass labels do not certify translation fluency. No candidate was written into the canonical translation source or SFT release.

## Template-reuse audit — 5 November 2026

I checked whether reviewed translations could cover the same frozen cohort by reusing a translation across prompts with the same entity-slot pattern and source relation signature. The audit verified all 1,952 original review rows against their hash-bound source files and all 799 flagged-repair rows against their source maps and candidate-text hashes. The repair review produced 445 AI-only passes; none has human fluency certification.

| Coverage view | Forget cases | Retain cases | Composition chains |
|---|---:|---:|---:|
| Existing source-screen passes only | 5/117 | 3/39 | 0/44 |
| Union with hash-verified repair candidates | 29/117 | 10/39 | 1/44 |
| Optimistic relation-aware template-reuse ceiling | 32/117 | 12/39 | 1/44 |

The ceiling assumes one passing translation can be reused for every prompt with the same normalized entity slots and relation signature, without re-reviewing each rendering. It adds at most 12 atomic prompt rows across the four languages. Relation-aware grouping finds 274 templates among 312 atomic prompts and 176 among 176 composition prompts; text-only grouping would merge 12 groups with different relation signatures. This route cannot supply the frozen R1 corpus, and it does not justify weakening the release gate.

The audit is reproducible with [`template_reuse_ceiling_audit.py`](template_reuse_ceiling_audit.py); machine-readable counts and input hashes are in [`template_reuse_ceiling.json`](template_reuse_ceiling.json). Both artifacts are diagnostic only. No corpus was assembled and training remains stopped.

## Reliability conclusion

The implementation and synthetic 8B runtime path are currently verifiable. The full R1 experiment is **not running** and is **not ready to run**: translation coverage remains far below the frozen release requirement. The prior one-seed Llama 3.2 1B analysis remains the only hypothesis diagnostic, and its baseline competence gate failed in all four target languages. Starting the 8B main study now would not produce a reliable answer.

Reproduction details and SHA-256 bindings are in [`verification.json`](verification.json).


## Implementation recheck — 5 November 2026, 08:01 UTC

A code audit found that SFT used the verified character-offset assistant mask, but English unlearning still used separately tokenized prompt/full-sequence prefix matching. This is the same class of tokenizer-boundary failure that had already appeared in the Llama runtime smoke. I corrected unlearning to use the shared offset-mask helper. SFT and unlearning manifests now record the helper SHA-256, the unlearning preflight requires the same helper version used for SFT, and the Kaggle notebook verifies and exports it.

The full offline fixture suite passes **35/35**. It now exercises the unlearning path with a tokenizer fixture whose prompt-tokenization is deliberately not a prefix of the full conversation. Prepared source alignment also passes for 488 canonical SFT IDs. These checks validate the pipeline and source alignment; they do not constitute a real model run.

A fresh status check found all nine checked Kaggle kernels complete, with no job running. The 8B receipt remains runtime-only. The source-bound training release remains blocked at 5/117 forget, 3/39 retain, and 0/44 composition chains; even the optimistic relation-aware template-reuse ceiling is 32/117, 12/39, and 1/44. Therefore no main SFT or unlearning experiment has started.

Exact modified-source hashes and command results are recorded in [`verification.json`](verification.json).

## Translation diagnostic recheck — 5 November 2026, 08:09 UTC

I reran all three hash-bound verifiers for the 16-question Qwen3 translation pilot. The generator reproduced 16/16 rows and the downloaded Kaggle source matched the local reviewed source. The initial Qwen14 + NLLB round-trip screen produced 4 pass, 9 flag, and 3 blocked rows. A separate blinded direct screen on the same 16 selected questions reviewed 32 candidates plus 16 seeded mismatch controls: it passed 11/16 Qwen3 candidates and 10/16 alternative candidates, and rejected 15/16 controls while accepting one known error. The missed control was a Hindi composition question where “editor” had been changed to “director.” The predefined 15/16 control threshold therefore passed, but the false acceptance remains material.

Direct and round-trip labels differed for 8/16 Qwen3 candidates (all flag-to-pass) and 7/16 alternative candidates (4 flag-to-pass, 3 pass-to-flag). The review protocols and runtime settings differed, so this does not isolate NLLB as the cause. These results show that AI-only review labels are unstable on this selected pilot; they do not identify which translation is correct. No candidate was promoted. The full source-bound coverage remains 5/117 forget, 3/39 retain, and 0/44 composition chains, with no SFT or unlearning run started.

The offline fixture suite still passes **35/35**, and prepared-source alignment still passes for **488** canonical IDs. A fresh authenticated Kaggle status check found all **12** checked kernels complete and none running; completion of the translation diagnostics and synthetic 8B runtime smoke is not evidence that the R1 study ran. Exact inputs, output hashes, counts, and caveats are in [`verification.json`](verification.json).

## Matched direct-screen coverage extension — 5 November 2026, 08:31 UTC

I re-ran the direct-screen verifier and extended the read-only coverage audit to include its 30 PASS decisions among 31 candidates. The verifier confirmed the exact source, candidate-text, input, model, and script bindings. This screen had no human-gold controls, and only 13/31 overall pass/flag labels agreed with the earlier back-translation screen. Therefore these rows remain quarantined AI-only candidates.

| Candidate-union coverage | Forget | Retain | Composition train-only |
|---|---:|---:|---:|
| Exact complete cases/chains across all four target languages | 36/117 | 15/39 | 1/44 |
| Leakage groups for complete atomic cases | 18 | 12 | -- |
| Optimistic relation-aware template-reuse ceiling | 40/117 | 17/39 | 1/44 |

This extension raises the diagnostic candidate union from 1,598 to 1,628 passing rows. It does **not** change the authoritative release counts: 5/117 forget, 3/39 retain, and 0/44 composition chains. The 36/18 and 15/12 candidate case/group counts are not evidence that the model is precompetent; that is measured only after a valid corpus is trained and baseline predictions are scored. The full R1 release still fails closed.

Among the 44 composition chains, one is complete in the candidate union, 11 have exactly one unpassed translation row, and 138 rows remain unpassed in total. The 11 near-complete rows are prioritized in the machine-readable audit for further translation work. Do not train on them until each row passes the release review and the complete frozen data gate passes.

The extension is reproducible with [`direct31_extended_coverage_audit.py`](direct31_extended_coverage_audit.py); its hash-bound counts and the 11 near-complete chain gaps are in [`direct31_extended_coverage_audit.json`](direct31_extended_coverage_audit.json). Neither artifact approves data or changes the experiment design. The main research SFT/unlearning run has not started.

## Reliability recheck — 5 November 2026, 08:44 UTC

- Revalidated the Llama 3.1 8B receipt against the pinned model revision and runner hash. The corrected T4 x2 smoke loaded the 8B weights and completed one synthetic LoRA step and one synthetic NPO update; it loaded no research data. The mask preflight and safe-to-publish notebook checks also pass.
- Re-ran the frozen source-bound assembler. It failed closed at the same gate: 5/117 forget and 3/39 retain cases passed, versus required 117 and 39. No corpus or manifest was created.
- Rechecked all 12 tracked Kaggle kernels; each is `COMPLETE`, with none running or queued. These include diagnostics and the runtime smoke; they do not indicate an active R1 training run.
- Reproduced the quarantined coverage audit: 36 forget cases / 18 groups, 15 retain cases / 12 groups, and 1/44 complete composition chain. The 11 one-gap chains remain diagnostic candidates only.

The 8B smoke therefore clears runtime compatibility, but the experiment is still blocked at multilingual data release. The historical 4 November access-failure note now links to the superseding successful runtime-only receipt. Machine-readable recheck details and hashes are in [`verification.json`](verification.json).

## Statistical diagnostic and reproducibility recheck — 5 November 2026, 08:59 UTC

The corrected v4 one-seed Llama 3.2 1B pilot was rescored again from the hash-verified prediction file. All **7/7 derived score and report files were byte-identical** to the archived outputs. The fresh Kaggle status queries returned **12/12 COMPLETE**, with no running or queued jobs. The 35 offline pipeline fixtures passed, and prepared-source alignment passed for 488 canonical SFT IDs.

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Predictions / checkpoints | 7,740 / 6 | 1,290 rows per checkpoint; one seed (42) |
| Generation failures / invalid-output flags | 0 / 0 | Scoring integrity signal; not evidence of model competence |
| Baseline multi-hop accuracy, full forget set | 18.2–20.8% | Low baseline performance limits interpretation |
| Bilingual matched cases / leakage groups, FR / DE / HI / TH | 10/6 / 9/5 / 9/6 / 11/7 | Every target-language gate fails the required 15 cases and 8 groups |
| Target-minus-English contrasts with 95% group-bootstrap intervals spanning or touching 0 | 12/12 | Exploratory; very small group counts and no multiplicity adjustment |
| Latest source-bound release gate | 5/117 forget; 3/39 retain; 0/44 composition | Failed closed; no SFT corpus or release manifest written |

The interval estimates show no clear cross-language amplification, but their width and the failed competence gate mean they cannot establish absence of transfer. The direct-probe/final-answer diagnostic found B correct in 0–2 of 9–11 matched cases while final C was correct in 8–10; this pattern does not identify why final answering persisted.

An older recomputed folder contains a conflicting alternate prediction set: the row keys match, but 5,906/7,740 answers differ; its report marks 1,385 predictions invalid, versus 0 in corrected v4. Its generating provenance is unresolved, so its metrics are excluded. See [`r1_pilot_recomputed_v2/README.md`](../r1_pilot_recomputed_v2/README.md).

Reproduction values, file hashes, command, and statistical limitations are recorded in [`pilot_diagnostic_verification.json`](pilot_diagnostic_verification.json); the readable tables are in [`pilot_diagnostic_results.md`](pilot_diagnostic_results.md). This remains an exploratory 1B diagnostic, not a completed R1 experiment. No research-data SFT or unlearning job is running.


## Fresh reliability recheck — 2026-11-05T09:17:40Z

I reran the checks below against the current workspace. No research-data training was launched.

| Check | Fresh result | Meaning |
|---|---:|---|
| Offline pipeline fixtures | 35/35 passed | Synthetic fixtures only; no model result. |
| Prepared-source alignment | 488/488 canonical SFT IDs passed | Source reconstruction passed; translation review/release did not. |
| Source-bound SFT release assembler | Blocked: 5/117 forget, 3/39 retain, 0/44 composition chains | It exited before writing a corpus or manifest. |
| Independent pilot rescoring | 7/7 derived outputs byte-identical | The archived 1B pilot scores reproduce. |
| 8B Kaggle runtime receipt | Valid; one synthetic LoRA step and one synthetic NPO step | Runtime compatibility only; no research corpus loaded. |
| Tracked Kaggle kernels | 12/12 complete; 0 running/queued | No active research job. |

The corrected v4 pilot still has one seed (Llama 3.2 1B). Baseline final multi-hop accuracy is EN 20.8%, FR 18.2%, DE 19.5%, HI 18.2%, and TH 19.5%. The four target-language competence gates all fail: matched cases/groups are FR 10/6, DE 9/5, HI 9/6, and TH 11/7, below the required 15 cases and 8 groups. All 12 target-minus-English estimates have 95% entity-group bootstrap intervals that include or touch zero; their small cohorts and failed gate make them exploratory, not evidence of no transfer. Full estimates and caveats are in [`pilot_diagnostic_results.md`](pilot_diagnostic_results.md).

The runtime receipt is for the exact Llama 3.1 8B pinned revision on 2×T4. It reports finite synthetic losses (LoRA 0.02476; NPO 13.86294) and peak allocations of 10.47/5.61 GiB. This confirms only that a minimal synthetic update works. The safe-publish verifier separately checks the v6 access-preflight artifact; it is not proof that the v8 interactive smoke notebook is the current published source.

**Current stage: data release.** The research SFT/unlearning experiment is not running and is not ready: the frozen multilingual release gate still fails, and no corpus or manifest exists. The current verified numbers support an exploratory 1B diagnostic and engineering checks, not a confirmatory R1 conclusion. Exact command results and SHA-256 bindings for this refresh are recorded in [`verification.json`](verification.json).

## Post-hoc multiplicity sensitivity — 2026-11-05

Recomputed the corrected v4 pilot’s 12 method-by-language contrasts from the archived case metrics using 100,000 entity-group bootstrap replicates per contrast. A Bonferroni adjustment gives nominal simultaneous 95% intervals from **−71.4 to +100.0 percentage points** across the 12 comparisons; **all 12 intervals include zero**. The only matched sets contain 9–11 cases across 5–7 groups. This reinforces the limited precision of the pilot; it does not establish that transfer is absent. Because the cluster count is very small, the adjusted bootstrap coverage is approximate. This post-hoc sensitivity is not confirmatory and does not replace the primary analysis.

Readable table: [`pilot_multiplicity_sensitivity.md`](pilot_multiplicity_sensitivity.md). Reproduction script: [`pilot_multiplicity_sensitivity.py`](pilot_multiplicity_sensitivity.py). Input archive SHA-256: `48b800312c8c43dee14fdf8d0384a2153b4317c253d7de1228d4c81f2697ab43`; case-metrics SHA-256: `786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9`. Analysis script SHA-256: `4150611d2bf8ace1c2a9e4917d06c6f23014fc688107cd79cd90b98c4e988769`.

## Live Kaggle status refresh — 2026-11-05T09:38:00Z

Authenticated status recheck: **12/12 tracked kernels are COMPLETE; 0 are running or queued.** This confirms there is no active R1 research-training job. The completed 8B notebook remains a synthetic runtime smoke, not a research experiment.

## Targeted translation-gap union audit — 5 November 2026, 10:01 UTC

I reran the primary targeted-gap verifier and the alias retry, NLLB alternate-candidate, and segmented-retry verifiers in temporary copies. Their results matched their archived receipts. I also rechecked the underlying source, original review, repair review, and matched direct-screen bindings. The audit is reproducible from 76 recorded input/code hashes; reruns matched on every result field apart from the generation timestamp.

| Screen | Joint passes | Incremental unique prompt-language rows after ordered deduplication |
|---|---:|---:|
| Primary targeted gap (187 rows) | 36 | 35 |
| Retry of 17 blocked rows | 3 | 0 |
| NLLB alternate candidates | 3 | 1 |
| Segmented NLLB retry | 0 | 0 |
| **Deduplicated increase** | -- | **36** |

| Candidate coverage diagnostic | Before | After | Frozen release requirement |
|---|---:|---:|---:|
| Exact complete four-language forget cases / groups | 36 / 18 | 39 / 20 | 117 cases |
| Exact complete four-language retain cases / groups | 15 / 12 | 15 / 12 | 39 cases |
| Optimistic relation-aware forget ceiling | 40 | 43 | 117 |
| Optimistic relation-aware retain ceiling | 17 | 18 | 39 |
| Complete training-only composition chains | 1/44 | 2/44 | 44 |
| Missing composition translation rows | 138 | 119 | 0 |

The candidate union rises from **1,628 to 1,664** passing prompt-language rows, but it is still only an AI-screened coverage estimate. The candidate screens had no fluent-human review or human-gold controls; the primary Kaggle runtime also logged a sliding-window-attention/SDPA warning. The unchanged authoritative release still has only **5/117 forget, 3/39 retain, and 0/44 composition chains**. Therefore, no research training corpus exists, no pre-unlearning competence result is available, and no 8B SFT or unlearning run can start reliably yet.

The exact row-level joins, screen overlap counts, source/model/script hashes, and caveats are in [`targeted_gap_union_audit.json`](targeted_gap_union_audit.json). Reproduction command: `python -B targeted_gap_union_audit.py`. Script SHA-256: `dbc8f8fe6bc2d5a5b0d32a04431e2263b7f38912fa3f11cf997ead4df1b83313`; JSON SHA-256: `57ca91d98e733a7f3e56332ab47841379efe36013f8a4e5d1ea3326f498c2c4f`; Markdown SHA-256: `e0378bdea094a627472c0a2104cf744c018c3caa0b04f764d99267cc90e52873`.

## Live reliability recheck — 5 November 2026, 10:11 UTC

- Kaggle status queries for all 12 tracked kernels completed successfully: 12 are complete and none is running or queued. The installed CLI emits an outdated-version warning, but each query exited successfully.
- The exact pinned Llama 3.1 8B runtime receipt passes local validation. It proves only one synthetic LoRA update and one synthetic NPO update on 2×T4; it loaded no research data.
- The R1 offline fixture suite passes **35/35**. These are synthetic pipeline tests, not model results.
- The current research stage remains **training-data release**. The frozen source-bound gate is still 5/117 forget, 3/39 retain, and 0/44 train-only composition cases; no approved corpus exists, so no 8B research SFT or unlearning is running.

The machine-readable status and scope limits are recorded in [`verification.json`](verification.json).

## Full flagged-repair review and live status — 5 November 2026, 10:30 UTC

The first 799-row Kaggle review stopped with an error after processing 733 rows. Its hash-bound retry reviewed the 97 unresolved rows, and the merge verifier now confirms exact coverage of all 799 source-bound candidate rows. The freshly retrieved retry output matched the archived output hash; the merged rows and receipt also matched the archived hashes.

| AI-only review result | Rows | Share |
|---|---:|---:|
| Pass | 445 | 55.7% |
| Flagged | 323 | 40.4% |
| Blocked | 31 | 3.9% |

The 97-row retry produced 22 passes, 44 flags, and 31 blocks. The remaining blocks comprise 28 NLLB back-translation placeholder failures and 3 Qwen schema/inference errors. They remain blocked rather than being counted as passes. Across the 445 ChatGPT and 397 TranslateGemma candidate passes, 230 rows overlap; 612/799 rows have at least one candidate pass and 187 have none.

The optimistic mixed-source candidate union at that time completed 60/117 forget cases, 19/39 retain cases, and 6/44 composition chains. The frozen source-bound release still had only 5/117 forget, 3/39 retain, and 0/44 composition chains eligible. Because this estimate combined non-equivalent sources and screens, it is not directly comparable with the later exact-row audit below and should not be used as the current coverage estimate. It was a translation diagnostic, not a model result or release approval; the candidates were quarantined and had no fluent-human review.

The 10:23 UTC live status check found 13 of 14 Kaggle kernels complete, the original interrupted review kernel in error, and no running or queued jobs; its retry kernel is complete. The fixture suite passed 35/35 and prepared-source alignment passed for 488 canonical IDs. The only hypothesis result remains the exploratory one-seed 1B pilot; no research SFT or unlearning run is active.

## Fresh source-bound recheck — 5 November 2026, 11:03 UTC

I reran the targeted candidate-union audit, all 35 offline pipeline fixtures, the prepared-source alignment check, the source-bound release assembler, and Kaggle status queries. The audit's JSON and Markdown were regenerated from the hash-bound source maps and independent screen verifiers.

| Diagnostic | Fresh result | Interpretation |
|---|---:|---|
| Candidate union | 1,628 → 1,677 rows (+49 unique) | AI-screened candidate coverage only; all candidates quarantined |
| Exact four-language forget cases / groups | 42/117 / 22 | Below the frozen 117-case release requirement |
| Exact four-language retain cases / groups | 16/39 / 12 | Below the frozen 39-case release requirement |
| Optimistic relation-aware template ceiling | 47/117 forget; 19/39 retain | Reuse assumption; not exact row coverage or release evidence |
| Complete training-only composition chains | 2/44; 114 translation rows missing | Far below requirement |
| Source-bound release gate | 5/117 forget; 3/39 retain; 0/44 composition | Expected fail-closed; no corpus or manifest written |
| Offline pipeline fixtures | 35/35 passed | Synthetic tests only |
| Prepared-source alignment | 488 canonical IDs passed | Source alignment only; does not approve translations |
| Kaggle status | 15/15 complete; 0 active | No main R1 training or unlearning run is active |

Reviewer diagnostics remain a reason to keep the translations quarantined: direct Mistral review disagreed with initial Qwen labels on 154/187 rows; Mistral passed 2/15 and Qwen14 passed 1/16 deliberately mismatched source/candidate controls. These small constructed probes are not estimates of reviewer accuracy. No fluent-human review or human-gold set exists. The 8B runtime smoke is still runtime-only and used no research data. The only hypothesis result remains the exploratory one-seed Llama 3.2 1B pilot, which fails the target-language baseline competence gate.

The current row-level joins, screen overlap, reviewer-control receipts, and SHA-256 provenance are in [`targeted_gap_union_audit.json`](targeted_gap_union_audit.json). Reproduction: `python -B targeted_gap_union_audit.py`. The release assembler is expected to exit nonzero until the frozen gate passes.
