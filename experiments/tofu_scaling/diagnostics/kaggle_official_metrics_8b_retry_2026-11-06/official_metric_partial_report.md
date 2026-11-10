# Historical TOFU official-metric baseline: version 5 (partial)

> Historical receipt for the version-5 attempt. The isolated version-6 retry completed the 8B retain90 oracle and all four checkpoints now pass independent verification. See the [completed baseline report](official_metric_baseline_report.md) for current results.

**Status: PARTIAL.** Both 3B checkpoints and the 8B full checkpoint pass independent row-level and metric verification. The 8B retain90 oracle remains outstanding, so an 8B forget-quality KS comparison is not yet available. No NPO or other unlearning training has started.

## Verified full-checkpoint results

The evaluation used the pinned TOFU revision, 400 questions each for `forget10`, canonical `retain90`, and `holdout10`, plus 100 `real_authors` and 117 `world_facts` rows. The 8B output contains all 1,417 expected rows. Confidence intervals are 95% profile-block bootstrap intervals over 20 groups with 5,000 resamples and seed 42.

| Checkpoint / split | Questions | Mean answer probability (95% profile CI) | Mean ROUGE-L recall (95% profile CI) | Mean Truth Ratio |
|---|---:|---:|---:|---:|
| 3B full — forget10 | 400 | 0.9449 [0.9356, 0.9527] | 0.9141 [0.8730, 0.9474] | 0.4984 |
| 8B full — forget10 | 400 | 0.9912 [0.9892, 0.9929] | 0.9888 [0.9818, 0.9949] | 0.5204 |
| 3B full — retain90 | 400 | 0.9367 [0.9293, 0.9437] | 0.9028 [0.8783, 0.9261] | 0.4669 |
| 8B full — retain90 | 400 | 0.9910 [0.9892, 0.9925] | 0.9954 [0.9902, 0.9993] | 0.4851 |
| 3B full — holdout10 | 400 | 0.1461 [0.1280, 0.1646] | 0.3774 [0.3511, 0.4017] | — |
| 8B full — holdout10 | 400 | 0.1226 [0.1065, 0.1394] | 0.3689 [0.3436, 0.3921] | — |
| 3B retain90 oracle — forget10 | 400 | 0.1252 [0.1023, 0.1510] | — | 0.8541 [0.7965, 0.9138] |

The 3B full-versus-retain90-oracle forget-quality statistic is **KS = 0.3775, p = 9.00 × 10⁻²⁶**, recomputed independently from the preserved per-question truth-ratio rows. The 8B full checkpoint assigns high likelihood to forget and retain answers and lower likelihood to holdout answers. This is a pre-unlearning baseline result; answer probability is a likelihood metric, and ROUGE-L is generated-text overlap, not exact-answer accuracy. The 3B and 8B checkpoint contrast is descriptive because the models differ in Llama generation and training history.

## Verification and provenance

- 8B checkpoint: `open-unlearning/tofu_Llama-3.1-8B-Instruct_full@1a5c5b1a557f8c99bdadecd5168ebd03f640b00e`.
- Runtime: Kaggle T4×2, FP16 + SDPA, `balanced` placement, no CPU/disk offload; four-token placement smoke passed; 4,208.96 seconds.
- The independent verifier recomputed per-row answer probabilities, ROUGE-L, Truth Ratio, bootstrap summaries, ordered source rows, profile groups, pinned dataset hashes, model revision, and runtime safeguards. It reports `PARTIAL`, with the 3B full, 3B oracle, and 8B full models passing, and only the 8B oracle missing.
- The raw Kaggle version 5 archive is preserved byte-for-byte under [`version5_2026-11-07/raw_kaggle_output/`](version5_2026-11-07/raw_kaggle_output/); ZIP CRC passed. Its status records that the 8B full output completed before CUDA OOM occurred while loading the separate 8B oracle. The raw archive is not edited.
- The reconciled verification view is [`version5_2026-11-07/reconciled_partial_3b_8bfull_verified/`](version5_2026-11-07/reconciled_partial_3b_8bfull_verified/). It carries forward the separately documented 3B KS calculation and copies the 8B full JSON and JSONL byte-identically from the raw archive.

## Current retry

Kaggle notebook version 6 is running only `llama_3_1_8b_retain90`, in a fresh runtime. It restores the 3B full, 3B oracle, and 8B full outputs from a private resume dataset, checks its upload receipt and every file SHA-256, re-fetches and verifies the pinned TOFU data, and runs the 8B oracle with the same FP16 + SDPA, balanced T4×2 placement, 13 GiB per-GPU cap, offload guard, and four-token smoke. This isolates the remaining checkpoint load and avoids repeating the completed 8B full run.

If the oracle result and independent verifier pass, the four-checkpoint pre-unlearning baseline will be complete. NPO remains a separate next stage and is not launched by this notebook.

## Reproduction receipts

- [Combined partial verification JSON](version5_2026-11-07/reconciled_partial_3b_8bfull_verified/official_eval_verification.json)
- [Combined partial verification report](version5_2026-11-07/reconciled_partial_3b_8bfull_verified/official_eval_verification.md)
- [Version 5 attempt manifest](version5_2026-11-07/attempt_manifest.json)
- [Version 6 attempt manifest](version6_2026-11-07/attempt_manifest.json)
- [Version 6 private resume dataset](https://www.kaggle.com/datasets/bunnoob2005/tofu-3b-8bfull-resume-2026-11-07)
- [Pinned protocol](../../protocol.json)
