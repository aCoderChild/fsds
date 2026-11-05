# R1 reliability recheck — 2026-11-05

**Last verified: 2026-11-05 06:58 UTC. Current stage: training-data release. No main-study SFT corpus or local training run exists.** Kaggle reports all 20 kernels returned by `kaggle kernels list --mine`, plus the `experiments` notebook checked separately, as `COMPLETE`. This covers the 21 checked kernels, not every historical kernel in the account.

## Current checks

| Check | Result | Meaning |
|---|---|---|
| Offline R1 fixture suite | 35/35 passed at 06:58 UTC | Validates software paths and fail-closed guards on synthetic fixtures; not a model result. |
| 8B Kaggle runtime smoke | Passed; receipt verifies | Exact pinned Llama 3.1 8B loaded on T4 × 2; one synthetic LoRA step and one synthetic NPO update completed. No research data was loaded. |
| Kaggle smoke editor | Session off; all run switches false | No active smoke session was visible in the open Kaggle editor. This check covers that notebook only. |
| Prepared 8B notebook | Hash and safe-default verifier passed | GPU and Internet default off; no research dataset attached; assistant mask is checked before weight loading. |
| Source-bound data reconstruction | Passed at 06:58 UTC | Prepared-only verifier rebuilt 488 canonical SFT IDs and matched all prepared inputs to current source hashes; this does not approve a training release. |
| Main SFT release gate | **Blocked as designed** | 5/117 forget cases, 3/39 retain cases, and 0/44 composition chains meet the complete five-language review gate. |
| Main SFT corpus/manifest | Absent | The assembler stopped before writing either file. |
| Kaggle kernel activity | All 21 checked kernels report `COMPLETE` | No SFT or hypothesis run is active among the checked kernels. |

## Available hypothesis result

The available result remains the one-seed Llama 3.2 1B feasibility pilot. Its independent rescore verified the result archive and reproduced all seven derived outputs byte-for-byte from 7,740 predictions. All four target languages failed the preregistered baseline floor of 15 matched cases across 8 entity groups (FR 10/6, DE 9/5, HI 9/6, TH 11/7). The pilot is therefore exploratory and does not establish or rule out cross-language transfer.

The earlier 2026-11-05 Kaggle editor snapshot that showed `ERROR` predates the later successful v8 runtime-only receipt. The receipt verifies infrastructure compatibility only; it does not advance the research experiment past data preparation.

## Translation-candidate follow-up

I rechecked the 187-row targeted retranslation batch against its exact blinded input and source map. The AI-only screen verified 170 decisions, blocked 17 rows, and passed 36 candidates (FR 9, DE 7, HI 8, TH 12). Among the 170 completed decisions, 134 were semantic flags and one was a fluency flag. The candidate generator's exact model revision is unknown, and these are not production approvals.

| Coverage view | Forget cases | Retain cases | Composition chains |
|---|---:|---:|---:|
| Existing candidate-union ceiling | 60/117 | 19/39 | 6/44 |
| After adding the 36 targeted AI-only passes | 69/117 | 19/39 | 7/44 |
| Authoritative release gate | **5/117** | **3/39** | **0/44** |

The first two rows are optimistic planning ceilings: they combine different candidate texts and review criteria. They do not change the authoritative release gate. One Thai candidate for “godparent” was flagged after NLLB changed its back-translation to “decapitated parents.” This is evidence that the screen can reject a candidate because of a back-translation error; it does not prove the Thai candidate is correct. The batch remains quarantined, and the frozen production review still governs release.

### Bounded retry (complete; AI-only diagnostic)

I reran the 17 previously blocked rows on Kaggle T4 × 2: 15 used collision-checked natural aliases for the placeholder back-translation, and 2 reused their prior hash-verified back-translations. The run completed in 415 seconds and produced valid judge decisions for all 17 rows. The pinned NLLB and Qwen revisions, script, retry input, output hash, exact row coverage, and restored placeholder multisets verify locally. The batch contains questions only, without answers or split labels.

| Language | Retried | Semantic pass | Fluency pass | Joint pass |
|---|---:|---:|---:|---:|
| French | 2 | 0 | 2 | 0 |
| German | 3 | 1 | 3 | 1 |
| Hindi | 10 | 3 | 9 | 2 |
| Thai | 2 | 1 | 1 | 0 |
| **Total** | **17** | **5** | **15** | **3** |

Adding the three joint passes to the optimistic candidate union completes one more retain case. It does not complete any additional forget case or composition chain. These AI decisions remain diagnostic; they are not fluent-human certification.

| Coverage view | Forget cases | Retain cases | Composition chains |
|---|---:|---:|---:|
| Existing union + 36 targeted passes | 69/117 | 19/39 | 7/44 |
| After the 17-row retry | 69/117 | 20/39 | 7/44 |
| After the first 148-row NLLB candidate screen | 69/117 | 21/39 | 7/44 |
| After the 125-row marker-preserving retry | 69/117 | 21/39 | 7/44 |
| Authoritative release gate | **5/117** | **3/39** | **0/44** |

The Kaggle log also records unrelated preinstalled-package dependency warnings and an Abseil CUDA-stub warning; the job's PyTorch preflight passed, it reported two Tesla T4 devices, and all 17 GPU-gated review operations completed. Transformers warned that temperature/top-p/top-k were ignored with deterministic decoding (`do_sample=false`). These warnings are recorded as runtime caveats, not hidden. The local verifier accepted a synthetic valid receipt fixture and rejected a tampered source hash before the remote result was accepted.

### First alternate NLLB candidate screen (complete; AI-only diagnostic)

The answer-blind 148-row Kaggle screen completed on two Tesla T4 GPUs in about 8 minutes 10 seconds. The local verifier matched the submitted script hash, input hash, exact 148-row coverage, pinned NLLB/Qwen revisions, marker multisets, decisions, and output receipt.

| Language | Selected | Candidate generated | Judged | Semantic pass | Fluency pass | Joint pass | Blocked |
|---|---:|---:|---:|---:|---:|---:|---:|
| French | 12 | 12 | 12 | 0 | 12 | 0 | 0 |
| German | 11 | 10 | 9 | 4 | 8 | 3 | 2 |
| Hindi | 52 | 0 | 0 | 0 | 0 | 0 | 52 |
| Thai | 73 | 2 | 2 | 0 | 2 | 0 | 71 |
| **Total** | **148** | **24** | **23** | **4** | **22** | **3** | **125** |

The principal failure was in the candidate-generation safeguard: NLLB changed or dropped temporary “Alice”/“Bob” aliases used to protect entity markers. This caused 123 alias-preservation failures (117 one-marker and 6 two-marker rows); one source question had no protected marker, and one more row lost its alias during back-translation. The pipeline correctly blocked all such rows instead of admitting unverified text. Of the 23 judged rows, three German candidates passed both AI checks. Adding them raises the optimistic candidate ceiling to 21/39 retain cases; forget and composition coverage do not change. These candidates remain quarantined, and no human review is claimed.

### Marker-preserving retry (complete; AI-only diagnostic)

The retry completed on Kaggle T4 × 2 in about 19 minutes. Its exact input, output, submitted/pulled script, model revisions, 125-row coverage, marker multisets, and Qwen decision schemas pass local verification.

| Language | Selected | Candidate generated | Judged | Semantic pass | Fluency pass | Joint pass | Technical block |
|---|---:|---:|---:|---:|---:|---:|---:|
| German | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| Hindi | 52 | 51 | 51 | 0 | 40 | 0 | 1 |
| Thai | 71 | 71 | 71 | 0 | 59 | 0 | 0 |
| **Total** | **125** | **124** | **124** | **0** | **101** | **0** | **1** |

The retry preserved all markers, but segmentation damaged the surrounding question meaning: the Qwen judge rejected semantic fidelity for all 124 generated candidates. It accepted fluency for 101. The only technical block was the Hindi source question that contained no protected entity marker. This method does not produce usable translations. The optimistic ceiling remains 69/117 forget, 21/39 retain, and 7/44 composition cases; the authoritative release gate remains 5/117, 3/39, and 0/44.

The two alternative translation methods now tested have different failure modes: aliases are changed or dropped in ordinary NLLB translation, while translating around markers preserves them but fails semantic review. Neither method changes the frozen production translations or makes SFT eligible. The retry is recorded here: [Kaggle kernel](https://www.kaggle.com/code/bunnoob2005/sw-gap187-nllb-segmented-retry-2026-11-05).

## Reliability limits and next gate

Translation review is AI-only, with no fluent-human language audit. A new source-bound candidate audit still falls well short of the full release requirement; the translation reviewer also has demonstrated back-translation failure modes and has not been validated against fluent-human labels. The pilot has one seed, and its raw unlearning manifests omitted the unlearning-script hash; post-run provenance reconciliation is not cryptographic proof of which source ran. These limits are recorded in the independent-rescore receipt and the recheck JSON.

Do not start SFT or relax the competence threshold. First produce and verify the complete source-bound multilingual corpus. Then run the pinned 8B pre-unlearning baseline; proceed to the three-seed unlearning comparison only if every language clears the preregistered competence gate. The GPU runtime smoke does not satisfy either research gate.

## Reproduction pointers

- Fixture suite: `python -B experiments/synthworlds_r1/scripts/test_r1_pilot_pipeline.py`
- Runtime notebook safe-default check: `python -B experiments/synthworlds_r1/kaggle_8b_runtime_smoke/verify_prepared_notebook.py`
- Runtime receipt check: `python -B experiments/synthworlds_r1/kaggle_8b_runtime_smoke/verify_receipt.py experiments/synthworlds_r1/kaggle_8b_runtime_smoke/remote_output/interactive_smoke_v8_passed_runtime_only_2026-11-05/r1_8b_runtime_smoke.json`
- Source-bound assembler: `python -B experiments/synthworlds_r1/training_release_draft/scripts/assemble_sft_source_verified.py` (currently exits at the expected coverage gate; it writes no corpus or manifest).
- Pilot rescore record: `reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/v4_corrected_independent_rescore_2026-11-04.json`
