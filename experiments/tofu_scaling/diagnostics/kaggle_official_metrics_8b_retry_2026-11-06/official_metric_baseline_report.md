# TOFU official-metric baseline: complete

**Status: PASS.** The two full checkpoints and their matching `retain90` oracle checkpoints all passed the independent verifier. This is an inference-only pre-unlearning baseline; no NPO or SFT training was run.

## Evaluation and verification

- Dataset: `locuslab/TOFU`, revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`; all seven evaluation-file SHA-256 values match the pinned audit.
- Evaluator: Open-Unlearning commit `2d98027677f3468d52b6baa3fc4f2b9f2dc08820`, using its prompt and metric definitions.
- Checkpoints: Llama 3.2 3B full/oracle and Llama 3.1 8B full/oracle, each pinned to an immutable Hugging Face commit in the per-model result JSON.
- Runtime: Kaggle T4×2, FP16 + SDPA, greedy generation (200-token cap), no input truncation. This differs numerically from the upstream BF16 + FlashAttention 2 runtime.
- Full checkpoints: 1,417 rows each across forget10, retain90, holdout10, real-author, and world-fact data. Each oracle scores the same 400 ordered forget10 rows. Bootstrap intervals use 20 profile blocks, 5,000 draws, seed 42.
- Verification: Kaggle verification and a separate local row-level verifier both report `PASS` for all four pinned checkpoints. The local check recomputed row-level likelihoods, ROUGE-L, Truth Ratio, profile bootstrap intervals, source ordering, model and data revisions, and both KS statistics. The downloaded output ZIP passed CRC; SHA-256: `b4df8407e556050c905f6a648602783c6dbcc96093ed4df7a2a9e8bbb5fee6a5`.

## Main results

| Checkpoint / split | Questions | Mean answer probability (95% profile CI) | ROUGE-L recall (95% profile CI) | Mean Truth Ratio |
|---|---:|---:|---:|---:|
| 3B full — forget10 | 400 | 0.9449 [0.9356, 0.9527] | 0.9141 [0.8730, 0.9474] | 0.4984 |
| 3B retain90 oracle — forget10 | 400 | 0.1252 [0.1023, 0.1510] | — | 0.8541 |
| 3B full — retain90 | 400 | 0.9367 [0.9293, 0.9437] | 0.9028 [0.8783, 0.9261] | 0.4669 |
| 3B full — holdout10 | 400 | 0.1461 [0.1280, 0.1646] | 0.3774 [0.3511, 0.4017] | — |
| 8B full — forget10 | 400 | 0.9912 [0.9892, 0.9929] | 0.9888 [0.9818, 0.9949] | 0.5204 |
| 8B retain90 oracle — forget10 | 400 | 0.1045 [0.0851, 0.1257] | — | 0.9175 |
| 8B full — retain90 | 400 | 0.9910 [0.9892, 0.9925] | 0.9954 [0.9902, 0.9993] | 0.4851 |
| 8B full — holdout10 | 400 | 0.1226 [0.1065, 0.1394] | 0.3689 [0.3436, 0.3921] | — |

| Full checkpoint | Matched oracle | Forget-quality KS statistic | KS p-value | Interpretation |
|---|---|---:|---:|---|
| Llama 3.2 3B | 3B retain90 | 0.3775 | 9.00 × 10⁻²⁶ | Forget and oracle Truth Ratio distributions are distinguishable |
| Llama 3.1 8B | 8B retain90 | 0.3900 | 1.59 × 10⁻²⁷ | Forget and oracle Truth Ratio distributions are distinguishable |

The primary matched, profile-block-bootstrap contrasts are:

| Model | Full minus oracle answer probability (95% matched profile CI) | Full minus oracle Truth Ratio (95% matched profile CI) |
|---|---:|---:|
| Llama 3.2 3B | +0.8197 [0.7956, 0.8408] | −0.3557 [−0.4119, −0.3007] |
| Llama 3.1 8B | +0.8867 [0.8653, 0.9058] | −0.3971 [−0.4678, −0.3302] |

The full models assign much higher likelihood to forget-set reference answers than their retain-only oracles. Both matched bootstrap intervals exclude zero, consistent with the full checkpoints retaining target information relative to the retain-only controls. The row-level KS `p`-values above are conventional two-sample values and do not adjust for dependence among questions within the same profile; treat them as descriptive. These are **not** results for an unlearning method and do not show successful deletion.

The full checkpoint composite-utility harmonic means are 0.6702 (3B) and 0.6370 (8B). The 8B checkpoint has higher forget-set likelihood, but the models differ in generation and training history, so this comparison is descriptive and cannot isolate a parameter-count effect. Answer probability is a likelihood measure, and ROUGE-L is text overlap; neither is exact-answer accuracy. TOFU profiles are fictional and these results do not establish deletion of real personal data or real-world privacy.

## Reproduction artifacts

- Kaggle run: [private 8B oracle retry notebook](https://www.kaggle.com/code/bunnoob2005/tofu-official-metrics-8b-retry-2026-11-06), version 6.
- Complete reconciled results and local verification: [`version6_2026-11-07/reconciled_complete_3b_8b_verified/`](version6_2026-11-07/reconciled_complete_3b_8b_verified/).
- Byte-preserved Kaggle outputs and run log: [`version6_2026-11-07/raw_kaggle_output/`](version6_2026-11-07/raw_kaggle_output/).
- Notebook source snapshot, remote notebook metadata, attempt receipt, evaluator, verifier, and exact per-question outputs are retained alongside these results.

**Next stage:** independently lock an NPO protocol and resource-bounded real-data smoke, with explicit forget, retain, and holdout criteria. Start multi-seed training only after the smoke and objective-equivalence checks pass.
