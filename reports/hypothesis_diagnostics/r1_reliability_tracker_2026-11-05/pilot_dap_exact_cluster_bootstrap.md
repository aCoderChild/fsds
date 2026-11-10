# Pilot diagnostic: downstream-answer persistence with exact group-bootstrap intervals

**Status: post-hoc exploratory diagnostic, not a confirmatory experiment.**

DAP is the share of post-unlearning direct-recall failures for which the final multi-hop answer remains correct. The denominator includes only cases where the target-language full-SFT checkpoint answered both direct and multi-hop probes correctly before unlearning. Whole leakage groups are resampled; all exact ordered draws are enumerated.

| Method | Language | SFT-competent cases / groups | Direct failures | Final answer persists | DAP | Exact group-bootstrap 95% CI | Undefined bootstrap draws |
|---|---|---:|---:|---:|---:|---:|---:|
| GA | FR | 13 / 9 | 10 | 10 | 100.0% | [100.0%, 100.0%] | 0.005%
| GA | DE | 12 / 8 | 9 | 8 | 88.9% | [40.0%, 100.0%] | 0.039%
| GA | HI | 11 / 7 | 10 | 9 | 90.0% | [50.0%, 100.0%] | 0.000%
| GA | TH | 15 / 11 | 12 | 10 | 83.3% | [44.4%, 100.0%] | 0.000%
| GD | FR | 13 / 9 | 8 | 8 | 100.0% | [100.0%, 100.0%] | 0.068%
| GD | DE | 12 / 8 | 8 | 7 | 87.5% | [25.0%, 100.0%] | 0.391%
| GD | HI | 11 / 7 | 10 | 9 | 90.0% | [50.0%, 100.0%] | 0.000%
| GD | TH | 15 / 11 | 11 | 9 | 81.8% | [37.5%, 100.0%] | 0.001%
| NPO | FR | 13 / 9 | 9 | 9 | 100.0% | [100.0%, 100.0%] | 0.005%
| NPO | DE | 12 / 8 | 9 | 8 | 88.9% | [40.0%, 100.0%] | 0.039%
| NPO | HI | 11 / 7 | 10 | 9 | 90.0% | [50.0%, 100.0%] | 0.000%
| NPO | TH | 15 / 11 | 12 | 10 | 83.3% | [44.4%, 100.0%] | 0.000%

The point estimates show final-answer persistence in 83.3–100.0% of observed direct-recall failures (8–12 failures per method-language cell). The intervals and group counts show substantial uncertainty. This pattern does not prove that the model recovered the intermediate fact or that the final answer was reached by valid reasoning.

The bootstrap intervals condition on resamples with at least one direct-failure denominator event; the fraction of undefined resamples is shown. They do not account for seed variation. This analysis cannot bypass the failed bilingual competence gate or support a cross-language unlearning claim.

Reproduce with `python -B pilot_dap_exact_cluster_bootstrap.py`. Input case-metrics SHA-256: `786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9`. Analysis script SHA-256: `40ec667b5ea36bd5915575724e0401fa4616e75209df9fa7cf28f0a98f10add6`.
