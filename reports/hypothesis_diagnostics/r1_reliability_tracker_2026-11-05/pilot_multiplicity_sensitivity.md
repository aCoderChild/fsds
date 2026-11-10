# V4 pilot: multiplicity sensitivity for cross-language contrasts

**Status: post-hoc exploratory sensitivity only.** These results do not turn the failed-competence pilot into a confirmatory test.

I recomputed the 12 method-by-language contrasts from the hash-verified archived case metrics and added a Bonferroni adjustment. Each interval resamples leakage-isolated entity groups, preserving all cases within each sampled group. The adjusted intervals target nominal 95% simultaneous coverage across 12 comparisons.

| Method | Language | Matched cases / groups | DID (pp) | Marginal 95% cluster-bootstrap CI (pp) | Bonferroni familywise 95% CI (pp) |
|---|---|---:|---:|---:|---:|
| GA | FR | 10/6 | +10.0 | [+0.0, +50.0] | [+0.0, +66.7] |
| GA | DE | 9/5 | +22.2 | [+0.0, +80.0] | [+0.0, +100.0] |
| GA | HI | 9/6 | +0.0 | [-33.3, +33.3] | [-66.7, +66.7] |
| GA | TH | 11/7 | +9.1 | [-28.6, +57.1] | [-42.9, +71.4] |
| GD | FR | 10/6 | +10.0 | [+0.0, +50.0] | [+0.0, +66.7] |
| GD | DE | 9/5 | +11.1 | [-40.0, +80.0] | [-60.0, +100.0] |
| GD | HI | 9/6 | +11.1 | [+0.0, +50.0] | [+0.0, +66.7] |
| GD | TH | 11/7 | +9.1 | [-36.4, +71.4] | [-71.4, +100.0] |
| NPO | FR | 10/6 | +10.0 | [+0.0, +50.0] | [+0.0, +66.7] |
| NPO | DE | 9/5 | +11.1 | [-40.0, +80.0] | [-60.0, +100.0] |
| NPO | HI | 9/6 | +11.1 | [+0.0, +50.0] | [+0.0, +66.7] |
| NPO | TH | 11/7 | +9.1 | [-28.6, +57.1] | [-42.9, +71.4] |

All 12 multiplicity-adjusted intervals include zero. The intervals are extremely wide: the pilot has only 5-7 independent entity groups per contrast, so effects of meaningful size in either direction remain compatible with the data. This is evidence of inadequate precision, not evidence that cross-language transfer is absent.

The adjustment is nominal because cluster-bootstrap coverage is itself approximate with so few groups. No confirmatory p-values are reported. The original marginal intervals and competence-gate results remain in [`pilot_diagnostic_results.md`](pilot_diagnostic_results.md).

Reproduce with `python -B pilot_multiplicity_sensitivity.py`. Bootstrap replicates per contrast: 100,000. Input case-metric SHA-256: `786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9`. Analysis source SHA-256 is recorded in `pilot_multiplicity_sensitivity.json`.
