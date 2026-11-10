# V4 pilot: exact cluster-bootstrap sensitivity

**Status: post-hoc exploratory sensitivity only.** Exact resampling removes Monte Carlo noise from the intervals; it does not turn the failed-competence, one-seed pilot into confirmatory evidence.

For each method-language contrast, this enumerates every ordered bootstrap draw of the 5-7 leakage-isolated entity groups. Each draw keeps all cases in a selected group together. The weighted quantiles use the same order-statistic convention as the primary scorer. The Bonferroni intervals cover the 12 method-by-language contrasts at nominal familywise 95%.

| Method | Language | Cases / groups | Primary target change (pp; familywise 95% CI) | Secondary DID (pp; familywise 95% CI) | Resample sequences |
|---|---|---:|---:|---:|---:|
| GA | FR | 10/6 | +0.0 [+0.0, +0.0] | +10.0 [+0.0, +66.7] | 46,656 |
| GA | DE | 9/5 | -11.1 [-80.0, +0.0] | +22.2 [+0.0, +100.0] | 3,125 |
| GA | HI | 9/6 | -11.1 [-66.7, +0.0] | +0.0 [-66.7, +66.7] | 46,656 |
| GA | TH | 11/7 | -18.2 [-85.7, +0.0] | +9.1 [-42.9, +71.4] | 823,543 |
| GD | FR | 10/6 | +0.0 [+0.0, +0.0] | +10.0 [+0.0, +66.7] | 46,656 |
| GD | DE | 9/5 | -11.1 [-80.0, +0.0] | +11.1 [-60.0, +100.0] | 3,125 |
| GD | HI | 9/6 | -11.1 [-66.7, +0.0] | +11.1 [+0.0, +66.7] | 46,656 |
| GD | TH | 11/7 | -18.2 [-85.7, +0.0] | +9.1 [-71.4, +100.0] | 823,543 |
| NPO | FR | 10/6 | +0.0 [+0.0, +0.0] | +10.0 [+0.0, +66.7] | 46,656 |
| NPO | DE | 9/5 | -11.1 [-80.0, +0.0] | +11.1 [-60.0, +100.0] | 3,125 |
| NPO | HI | 9/6 | -11.1 [-66.7, +0.0] | +11.1 [+0.0, +66.7] | 46,656 |
| NPO | TH | 11/7 | -18.2 [-85.7, +0.0] | +9.1 [-42.9, +71.4] | 823,543 |

The exact familywise DID intervals include zero in 12/12 cells; the primary target-change intervals include zero in 12/12 cells. Against the earlier 100,000-draw DID simulation, the largest endpoint difference is 0.0 pp for marginal intervals and 0.0 pp for familywise intervals. Intervals remain wide because only 5-7 independent groups support each contrast.

These intervals quantify uncertainty in an exploratory pilot whose baseline competence gate failed in every target language. They do not establish no transfer and do not replace the required 8B, three-seed study.

Reproduce with `python -B pilot_exact_cluster_sensitivity.py`. Inputs are bound to archive SHA-256 `48b800312c8c43dee14fdf8d0384a2153b4317c253d7de1228d4c81f2697ab43` and case-metric SHA-256 `786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9`. The script enumerates finite cluster resamples exactly; no random seed or Monte Carlo replicate count is involved.
