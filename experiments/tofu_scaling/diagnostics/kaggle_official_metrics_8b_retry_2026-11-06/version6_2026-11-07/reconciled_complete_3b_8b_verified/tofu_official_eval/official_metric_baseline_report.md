# TOFU official-metric pre-unlearning baseline

**Status:** complete and inference-only. It evaluates the published full checkpoints and paired retain90 oracle checkpoints. No NPO training or research adapter was run.

The metric formulas, chat preprocessing, and generated-reference extraction follow the pinned Open-Unlearning implementation. `answer_probability` is mean per-answer-token probability; `rougeL_recall` scores greedy generated answers; Truth Ratio compares the geometric mean likelihood of perturbed answers with the paraphrased correct answer (or source correct answer for real-author/world-fact utility). Kaggle T4 uses FP16 + SDPA here, while upstream configs specify BF16 + FlashAttention 2; formulas and prompts match, but numerical generations are not bitwise-identical to the upstream runtime.

## Full-checkpoint metrics

| Split | Model | Questions / profile blocks | Mean answer probability (95% profile CI) | Mean ROUGE-L recall (95% profile CI) | Mean truth ratio | TOFU truth-ratio score | Normalized option probability |
|---|---|---:|---:|---:|---:|---:|---:|
| Forget10 | Llama 3.2 3B full | 400 / 20 | 0.9449 [0.9356, 0.9527] | 0.9141 [0.8730, 0.9474] | 0.4984 | 0.4728 | — |
| Forget10 | Llama 3.1 8B full | 400 / 20 | 0.9912 [0.9892, 0.9929] | 0.9888 [0.9818, 0.9949] | 0.5204 | 0.4734 | — |
| Canonical retain eval | Llama 3.2 3B full | 400 / 20 | 0.9367 [0.9293, 0.9437] | 0.9028 [0.8783, 0.9261] | 0.4669 | 0.5413 | — |
| Canonical retain eval | Llama 3.1 8B full | 400 / 20 | 0.9910 [0.9892, 0.9925] | 0.9954 [0.9902, 0.9993] | 0.4851 | 0.5304 | — |
| Holdout10 | Llama 3.2 3B full | 400 / 20 | 0.1461 [0.1280, 0.1646] | 0.3774 [0.3511, 0.4017] | — | — | — |
| Holdout10 | Llama 3.1 8B full | 400 / 20 | 0.1226 [0.1065, 0.1394] | 0.3689 [0.3436, 0.3921] | — | — | — |
| Real authors | Llama 3.2 3B full | 100 / — | 0.0197 | 0.8620 | 1.1882 | 0.6425 | 0.4983 |
| Real authors | Llama 3.1 8B full | 100 / — | 0.0189 | 0.9490 | 4.5620 | 0.5124 | 0.4079 |
| World facts | Llama 3.2 3B full | 117 / — | 0.0067 | 0.8621 | 2.5589 | 0.6643 | 0.4955 |
| World facts | Llama 3.1 8B full | 117 / — | 0.0075 | 0.9117 | 9.2802 | 0.6276 | 0.4803 |

## Composite utility and forget-quality comparison

| Model | TOFU model-utility harmonic mean | Forget-quality KS p-value vs retain90 oracle |
|---|---:|---:|
| Llama 3.2 3B full | 0.670163 | 8.99931e-26
| Llama 3.1 8B full | 0.637019 | 1.58532e-27

## Retain-question collision sensitivity

The pinned `retain90` data repeats one generic question with different answers. The canonical 400-row retain evaluation includes one of the two conflicting rows. Primary TOFU metrics retain the official row; the sensitivity row below excludes it.

| Model | Retain probability, primary → excluding collision | Retain ROUGE-L recall, primary → excluding collision | Retain truth-ratio score, primary → excluding collision |
|---|---:|---:|---:|
| Llama 3.2 3B full | 0.9367 → 0.9369 | 0.9028 → 0.9033 | 0.5413 → 0.5414 |
| Llama 3.1 8B full | 0.9910 → 0.9912 | 0.9954 → 0.9954 | 0.5304 → 0.5306 |

## Interpretation limits

- Confidence intervals resample the 20 inferred profile blocks for forget10, canonical retain, and holdout10. The JSON contains question-level outputs; real-author and world-fact rows have no explicit cluster IDs, so they are reported without profile-cluster intervals.
- Inference uses FP16 + SDPA because Kaggle T4 does not provide the upstream BF16 + FlashAttention 2 runtime. Small numeric and generation differences from an upstream evaluation are possible.
- The 3B and 8B checkpoints come from different Llama generations and training histories. Their paired differences are descriptive checkpoint comparisons, not causal evidence about parameter count.
- A small/large ROUGE or likelihood value alone does not prove forgetting. The forget-quality p-value compares distributions with a retain-only oracle and is not a standalone proof of equivalence or privacy.
- TOFU profiles are fictional. These results cannot establish deletion of real personal data or real-world privacy protection.
- The earlier 6 Nov likelihood-only run used a custom retain sample; this report uses the canonical `retain_perturbed` evaluation set from the pinned benchmark revision.

## Reproduction metadata

Dataset revision: `324592d84ae4f482ac7249b9285c2ecdb53e3a68`. Open-Unlearning code revision: `2d98027677f3468d52b6baa3fc4f2b9f2dc08820`. Bootstrap: 5,000 profile-cluster draws, seed 42. Greedy generation: max 200 new tokens.
