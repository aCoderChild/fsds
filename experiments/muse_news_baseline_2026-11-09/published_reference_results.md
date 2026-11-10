# Published MUSE-News reference results

These are reported in the original MUSE paper, not produced by this repository or the Kaggle run. The paper reports 95% percentile-bootstrap intervals from 9,999 resamples for C1, C2, and C4. See [Tables 3 and 6 of the MUSE paper](https://arxiv.org/abs/2407.06460).

| Model | C1 VerbMem forget ↓ | C2 KnowMem forget ↓ | C3 privacy leak (ideal −5 to +5) | C4 KnowMem retain ↑ |
|---|---:|---:|---:|---:|
| Published target | 58.4 [54.1, 62.9] | 63.9 [58.7, 69.0] | −99.8 | 55.2 [50.7, 59.9] |
| Retain-only retrain oracle | 20.8 [18.5, 23.7] | 33.1 [26.8, 39.5] | 0.0 | 55.0 [50.3, 59.8] |
| GA | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 5.2 | 0.0 [0.0, 0.0] |
| NPO | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 24.4 | 0.0 [0.0, 0.0] |

## Short reading

The published target retains substantial forget-set verbatim and factual recall. The retain-only oracle lowers both forget metrics while keeping retain QA utility close to the target. Unregularized GA and NPO drive the reported forget metrics to zero but also drive retain QA utility to zero; their privacy-leak scores are outside the benchmark's ideal interval. In this benchmark, forgetting scores alone would therefore give a misleadingly positive picture.

These values are literature context, not results from this project's run. The target-model Hub card does not document its training provenance, so exact identity between the paper checkpoint and the currently pinned Hub revision is not independently established. The current Kaggle baseline still needs its access preflight and fresh inference evaluation.
