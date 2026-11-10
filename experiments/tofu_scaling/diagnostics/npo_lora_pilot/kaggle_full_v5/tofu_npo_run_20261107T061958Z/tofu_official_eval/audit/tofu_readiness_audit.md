# TOFU 3B/8B readiness diagnostic

Generated: 2026-11-06T04:05:21.481986+00:00

## Data integrity

Status: **PASS; one known data ambiguity is reported below**

| File | Rows | SHA-256 | Empty required fields | Exact duplicate QA rows | Repeated questions |
|---|---:|---|---:|---:|---:|
| `full.json` | 4,000 | `667baef2f26f781f1328701d8ea54bf25123ff22fb2d98574c96cd991cb54086` | 0 | 0 | 1 |
| `forget10.json` | 400 | `0044c8c2e70a38be93f62ec6cb1c1cc2a1f55a8df2fb549a4da2da8dde9d92f6` | 0 | 0 | 0 |
| `retain90.json` | 3,600 | `debcf019c41db7b30b62ca6ad41fd8a4bcd29a5b321c21d2961974e308dfca55` | 0 | 0 | 1 |
| `holdout10.json` | 400 | `efec32c7a7f66bfefaa90b8a4cc583296b0bbf882495db61076dec2d6edf44dd` | 0 | 0 | 0 |
| `forget10_perturbed.json` | 400 | `6fbcb946c57ea1d7b2124cea0e61bf3b5409d1bc10d02368f090109450ed73c7` | 0 | 0 | 0 |
| `retain_perturbed.json` | 400 | `fc69f33bad70d3dca65920bdb54380039e3e8872dd16bbae20c1a839b99533fb` | 0 | 0 | 0 |
| `real_authors.json` | 100 | `5ed86f884ecbefc3a5db0b9dfe84ab3db710690c74d9402c0acc14bc3dd5a4b7` | 0 | 0 | 0 |
| `real_authors_perturbed.json` | 100 | `a23255383ac75a5fbe5acd57622a03fb9c368954fd7b2428afa152b784d9533a` | 0 | 0 | 0 |
| `world_facts.json` | 117 | `5547b07340f46755b868a525941341c3d93814b5370ae7fa10143f653a3c7033` | 0 | 0 | 0 |
| `world_facts_perturbed.json` | 117 | `0e171838040d0ec94b1ce248d891508e85e3b77344370077caa27b61898eb3e1` | 0 | 0 | 0 |

Forget10 and retain90 form the full 4,000-row training set exactly. Forget10, retain90, and holdout10 have no exact question overlap with one another. The official evaluation perturbation files are checked against their source rows, field schemas, and expected wrong-answer counts. The 400-row `retain_perturbed` evaluation set maps exactly to the first 20 complete retain90 profile blocks; the evaluator must use this canonical set rather than substitute a custom sample.

### Data warning

- `retain90` contains the same question 2 times with 2 different answers: What is the full name of the author? (The full name of the author is Aurelio Beltrán. / The full name of the author is Isabel Martínez.). This is one ambiguous QA collision; preserve the official split for comparability and report a sensitivity analysis excluding both rows.

The published design gives 20 QA pairs per fictitious author. Split sizes therefore correspond to 20 forget profiles, 180 retain profiles, and 20 holdout profiles. The released JSON has no explicit author ID, so this audit does not claim an independent structured-ID leakage proof.

## Checkpoint availability and resource facts

| Checkpoint | Immutable revision | Weight shards | Weight size (decimal GB) | Config | Tokenizer files | Weights downloaded? |
|---|---|---:|---:|---|---|
| Llama 3.2 3B full | `24f31ca19f6966dcb6f6b29abc511cce71222d4a` | 2 | 6.43 | PASS | PASS | no |
| Llama 3.2 3B retain90 oracle | `2c35819e9a731d096df9abd6efbcc361870c5333` | 2 | 6.43 | PASS | PASS | no |
| Llama 3.1 8B full | `1a5c5b1a557f8c99bdadecd5168ebd03f640b00e` | 4 | 16.06 | PASS | PASS | no |
| Llama 3.1 8B retain90 oracle | `f02956a9f25a18f38ad7808fbefb2f957347273e` | 4 | 16.06 | PASS | PASS | no |

Local runtime: `Linux-6.18.55+-x86_64-with-glibc2.39`; CUDA available: `True`; CUDA devices: `2`; MPS available: `False`.

An earlier Kaggle T4×2 smoke passed one synthetic 8B LoRA step and one synthetic NPO update. That verifies only the synthetic runtime path; it is not a TOFU score or an unlearning result. This local diagnostic did not download model weights or start training.

## Interpretation and next gate

TOFU is a workable public benchmark for a controlled pilot: the file-level splits are internally consistent and disjoint, and published 3B/8B full and retain90 checkpoints are reachable by immutable revision. The repeated generic retain question is a small but real label ambiguity. The author-group structure has no explicit IDs in the downloaded schema, which limits local leakage verification.

TOFU uses synthetic fictional author profiles. Results on it do not establish deletion of real personal data or real-world privacy protection.

An earlier Kaggle run measured answer likelihood on 1,200 examples per checkpoint, but it used a custom retain sample and did not include generation, Truth Ratio, model utility, or the retain-only oracle comparison; treat it as exploratory. The new pinned official-metric evaluator is separate and has not yet run. Run it on Kaggle before NPO, archive all question-level outputs and per-profile uncertainty, then review the results before choosing a training recipe. Treat 3B vs 8B as a comparison of two released model checkpoints, not a pure parameter-count effect: they are Llama 3.2 and Llama 3.1 respectively.

## Provenance

- TOFU dataset revision: `324592d84ae4f482ac7249b9285c2ecdb53e3a68`
- Open-Unlearning source revision for configs/metric semantics: `2d98027677f3468d52b6baa3fc4f2b9f2dc08820`
- Dataset audit does not include or download model weights.
