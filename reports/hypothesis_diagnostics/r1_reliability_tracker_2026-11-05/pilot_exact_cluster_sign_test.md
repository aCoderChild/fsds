# V4 pilot: exact entity-group sign sensitivity

**Post-hoc exploratory diagnostic only.** The planned primary estimand remains the target-language post-minus-pre multi-hop change. This sign test is a conservative direction check at the leakage-group level, not a replacement analysis.

Within each language/method cell, case-level paired changes were averaged within each independent entity group. Positive, negative, and exactly-zero group means were counted; exact-zero groups were omitted from the exact two-sided binomial sign test. Bonferroni correction is applied separately to the 12 primary cells and 12 secondary DID cells.

| Family | Method | Language | Cases / groups | Positive / negative / tied groups | Exact p | Adjusted p |
|---|---|---|---:|---:|---:|---:|
| primary | GA | FR | 10 / 6 | 0 / 0 / 6 | 1.0000 | 1.0000 |
| primary | GA | DE | 9 / 5 | 0 / 1 / 4 | 1.0000 | 1.0000 |
| primary | GA | HI | 9 / 6 | 0 / 1 / 5 | 1.0000 | 1.0000 |
| primary | GA | TH | 11 / 7 | 0 / 2 / 5 | 0.5000 | 1.0000 |
| primary | GD | FR | 10 / 6 | 0 / 0 / 6 | 1.0000 | 1.0000 |
| primary | GD | DE | 9 / 5 | 0 / 1 / 4 | 1.0000 | 1.0000 |
| primary | GD | HI | 9 / 6 | 0 / 1 / 5 | 1.0000 | 1.0000 |
| primary | GD | TH | 11 / 7 | 0 / 2 / 5 | 0.5000 | 1.0000 |
| primary | NPO | FR | 10 / 6 | 0 / 0 / 6 | 1.0000 | 1.0000 |
| primary | NPO | DE | 9 / 5 | 0 / 1 / 4 | 1.0000 | 1.0000 |
| primary | NPO | HI | 9 / 6 | 0 / 1 / 5 | 1.0000 | 1.0000 |
| primary | NPO | TH | 11 / 7 | 0 / 2 / 5 | 0.5000 | 1.0000 |
| secondary_did | GA | FR | 10 / 6 | 1 / 0 / 5 | 1.0000 | 1.0000 |
| secondary_did | GA | DE | 9 / 5 | 2 / 0 / 3 | 0.5000 | 1.0000 |
| secondary_did | GA | HI | 9 / 6 | 1 / 1 / 4 | 1.0000 | 1.0000 |
| secondary_did | GA | TH | 11 / 7 | 2 / 1 / 4 | 1.0000 | 1.0000 |
| secondary_did | GD | FR | 10 / 6 | 1 / 0 / 5 | 1.0000 | 1.0000 |
| secondary_did | GD | DE | 9 / 5 | 2 / 1 / 2 | 1.0000 | 1.0000 |
| secondary_did | GD | HI | 9 / 6 | 1 / 0 / 5 | 1.0000 | 1.0000 |
| secondary_did | GD | TH | 11 / 7 | 3 / 2 / 2 | 1.0000 | 1.0000 |
| secondary_did | NPO | FR | 10 / 6 | 1 / 0 / 5 | 1.0000 | 1.0000 |
| secondary_did | NPO | DE | 9 / 5 | 2 / 1 / 2 | 1.0000 | 1.0000 |
| secondary_did | NPO | HI | 9 / 6 | 1 / 0 / 5 | 1.0000 | 1.0000 |
| secondary_did | NPO | TH | 11 / 7 | 2 / 1 / 4 | 1.0000 | 1.0000 |

No cell reaches the per-cell familywise threshold (0.004167). At the design's minimum of eight groups, even a hypothetical outcome where all eight non-tied groups move in the same direction has minimum two-sided sign p = 0.007812, above that threshold. Nine non-tied groups are the mathematical minimum for this sign test to reach the threshold; that is not a sample-size or power recommendation.

This reinforces that the current 15-case/8-group gate is a feasibility screen only. The 1B pilot has just 5-7 groups, many tied at zero; it cannot establish transfer or no transfer. This sensitivity remains post-hoc, one-seed, and below the competence gate.

Reproduce with `python -B pilot_exact_cluster_sign_test.py`. Input metrics SHA-256: `786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9`.
