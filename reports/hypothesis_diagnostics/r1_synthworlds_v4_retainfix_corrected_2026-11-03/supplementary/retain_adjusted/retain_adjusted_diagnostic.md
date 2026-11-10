# Retain-adjusted cross-language diagnostic

This is a secondary, exploratory analysis of the corrected SynthWorlds-SM v4 pilot. It does not replace the report's preregistered primary contrast.

For each split, D = (post-unlearning minus SFT multi-hop correctness in the target language) minus the same change in English. The adjusted estimate is D(forget) minus D(retain). A negative adjusted estimate means the target-language change is more negative on forget than expected from its language-specific change on retain. Cases are restricted to those SFT-precompetent in English and the target language (both direct and multi-hop probes correct). Entity groups are resampled independently within forget and retain; intervals are percentile cluster-bootstrap intervals and are not multiplicity-adjusted.

| Method | Language | Forget cases/groups | Retain cases/groups | D forget (pp) | D retain (pp) | Adjusted estimate (95% CI, pp) |
|---|---:|---:|---:|---:|---:|---:|
| GA | fr | 10/6 | 21/17 | +10.0 | +4.8 | +5.2 ([-22.2, 45.3]) |
| GA | de | 9/5 | 21/17 | +22.2 | -4.8 | +27.0 ([-0.7, 89.5]) |
| GA | hi | 9/6 | 19/13 | +0.0 | +5.3 | -5.3 ([-42.0, 33.3]) |
| GA | th | 11/7 | 21/16 | +9.1 | +9.5 | -0.4 ([-35.3, 44.6]) |
| GD | fr | 10/6 | 21/17 | +10.0 | +4.8 | +5.2 ([-17.8, 44.9]) |
| GD | de | 9/5 | 21/17 | +11.1 | +4.8 | +6.3 ([-40.0, 74.1]) |
| GD | hi | 9/6 | 19/13 | +11.1 | +5.3 | +5.8 ([-8.7, 44.4]) |
| GD | th | 11/7 | 21/16 | +9.1 | +4.8 | +4.3 ([-44.1, 63.0]) |
| NPO | fr | 10/6 | 21/17 | +10.0 | +4.8 | +5.2 ([-18.9, 43.3]) |
| NPO | de | 9/5 | 21/17 | +11.1 | +4.8 | +6.3 ([-43.5, 74.1]) |
| NPO | hi | 9/6 | 19/13 | +11.1 | +5.3 | +5.8 ([-8.7, 44.4]) |
| NPO | th | 11/7 | 21/16 | +9.1 | +9.5 | -0.4 ([-36.0, 46.0]) |

## Provenance and limits

Input: `reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/extracted/corrected_evaluation/case_metrics.jsonl` (SHA-256 `786a8fc9646e6ec3ef1bb1a42cb62eed197164e8bda8a2537d75e7249661dcb9`). Original primary contrasts: `reports/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03/extracted/corrected_evaluation/cross_language_contrasts.csv` (SHA-256 `a9d78e99cb47d9c653e3c0709622ba4d19b2ae04936af6c3a55ef8a00c0217a7`); all 24 component counts and point estimates reproduced exactly. This analysis reuses the evaluator's case-level correctness labels and bilingual precompetence rule. It is based on one seed, small eligible cohorts, AI-only-reviewed translations, and a single 1B model. The adjustment is exploratory and cannot establish causal language transfer by itself.
