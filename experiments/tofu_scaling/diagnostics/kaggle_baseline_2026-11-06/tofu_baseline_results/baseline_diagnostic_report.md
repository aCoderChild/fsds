# TOFU full-checkpoint baseline diagnostic

**Status:** per-question receipts and profile-clustered intervals independently verified. This is a pre-unlearning baseline, not an unlearning result.

Metric: mean per-example `exp(-mean assistant-token cross-entropy)`, following the pinned Open-Unlearning probability metric. It measures likelihood assigned to reference answers; it is not generation accuracy.

## Per-model scores

| Split | Llama 3.2 3B mean [95% profile-bootstrap CI] | Llama 3.1 8B mean [95% profile-bootstrap CI] | Questions / profile blocks |
|---|---:|---:|---:|
| forget10 | 0.9449 [0.9356, 0.9527] | 0.9912 [0.9893, 0.9929] | 400 / 20 |
| retain90 | 0.9454 [0.9377, 0.9519] | 0.9904 [0.9889, 0.9918] | 400 / 20 |
| holdout10 | 0.1461 [0.1280, 0.1646] | 0.1226 [0.1065, 0.1394] | 400 / 20 |
| retain90, conflicting question rows excluded | 0.9457 [0.9380, 0.9524] | 0.9909 [0.9894, 0.9923] | 398 / 20 |

## Matched 8B minus 3B differences

Intervals bootstrap 20 matched profile blocks, not individual questions. These differences are descriptive, not causal evidence for a parameter-size effect.

| Split | Mean paired difference | 95% profile-bootstrap CI | Blocks where 8B > 3B |
|---|---:|---:|---:|
| forget10 | +0.0463 | [+0.0393, +0.0547] | 20/20 |
| retain90 | +0.0451 | [+0.0387, +0.0526] | 20/20 |
| holdout10 | -0.0235 | [-0.0293, -0.0179] | 1/20 |
| retain90, collision rows excluded | +0.0452 | [+0.0388, +0.0526] | 20/20 |

## Short interpretation

Both full checkpoints assign high likelihood to answers for forget10 and retain90 and substantially lower likelihood to the separate holdout10 split. This shows a useful pre-unlearning signal on the benchmark, though likelihood alone does not prove exact answer generation. The 8B checkpoint has higher answer likelihood on forget and retain but lower on holdout in this sample. Since the pair spans Llama 3.2 and Llama 3.1, this cannot be attributed to parameter count alone.

Excluding the two retain rows with the identical question and conflicting answers changes the mean only slightly. Keep the official split for primary comparability, and retain the sensitivity result.

## Run and integrity details

- Kaggle: [private baseline diagnostic kernel](https://www.kaggle.com/code/bunnoob2005/tofu-llama-3b-8b-baseline-diagnostic-2026-11-06).
- Models: `open-unlearning/tofu_Llama-3.2-3B-Instruct_full@24f31ca19f6966dcb6f6b29abc511cce71222d4a` and `open-unlearning/tofu_Llama-3.1-8B-Instruct_full@1a5c5b1a557f8c99bdadecd5168ebd03f640b00e`.
- Dataset: pinned TOFU revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`.
- Each model scored 1,200 QAs: all 400 forget10, all 400 holdout10, and 400 retain90 from 20 profile blocks; both runs used identical retain profile blocks.
- 3B runtime: 99.22 seconds; 8B runtime: 226.27 seconds; 2 Tesla T4 GPUs; FP16; no training or adapter creation.
- Bootstrap: 5,000 profile-cluster resamples, seed 42. These intervals describe variation across the 20 profile blocks; they do not account for training-seed variation because no unlearning was run.
- Data audit and score receipt verification: `PASS`.
