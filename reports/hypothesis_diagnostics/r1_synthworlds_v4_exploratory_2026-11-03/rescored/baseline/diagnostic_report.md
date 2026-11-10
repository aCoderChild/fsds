# R1 cross-lingual unlearning diagnostics

AI-only-reviewed translations; not human-validated. Strict normalized exact gold-answer matching; generation failures count as incorrect. Output-invalid is a separate deterministic heuristic flag for generation failure, empty output, overlong output (>64 words or >512 characters), visible template artifacts, a consecutive repeated cycle of 1–5 whitespace tokens repeated three times, or a repeated letter cycle in a no-whitespace, digit-free string; it is not an answer-correctness label. Eligible MH delta is paired within cases both answered correctly by SFT. Confidence intervals resample leakage-isolated entity groups. Main English-vs-language contrasts use only bilingual precompetent cases. The pilot is exploratory; intervals are not multiplicity-adjusted.

Output-invalid diagnostic: 0/3870 predictions (0.00%). Reasons: {}.

| Model | Language | Split | N | Precompetent | Direct acc. | MH acc. | Full MH Δ (pp) | Eligible MH Δ (95% CI, pp) | DAP | Invalid gen. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained | en | forget | 77 | 15 | 0.0% | 0.0% | -20.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | en | retain | 26 | 23 | 0.0% | 0.0% | -92.3 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | fr | forget | 77 | 13 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | fr | retain | 26 | 24 | 0.0% | 0.0% | -96.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | de | forget | 77 | 12 | 0.0% | 0.0% | -19.5 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | de | retain | 26 | 22 | 0.0% | 0.0% | -96.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | hi | forget | 77 | 11 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | hi | retain | 26 | 21 | 0.0% | 0.0% | -96.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | th | forget | 77 | 15 | 0.0% | 0.0% | -19.5 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | th | retain | 26 | 24 | 0.0% | 0.0% | -100.0 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | en | forget | 77 | 15 | 0.0% | 0.0% | -20.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | en | retain | 26 | 23 | 100.0% | 92.3% | +0.0 | -4.3 (-16.7,+0.0) | N/A | 0.0% |
| retain_only | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | fr | forget | 77 | 13 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | fr | retain | 26 | 24 | 96.2% | 100.0% | +3.8 | +0.0 (+0.0,+0.0) | 100.0% | 0.0% |
| retain_only | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | de | forget | 77 | 12 | 0.0% | 0.0% | -19.5 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | de | retain | 26 | 22 | 88.5% | 96.2% | +0.0 | -4.5 (-10.0,+0.0) | 100.0% | 0.0% |
| retain_only | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | hi | forget | 77 | 11 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | hi | retain | 26 | 21 | 88.5% | 88.5% | -7.7 | -9.5 (-28.6,+0.0) | N/A | 0.0% |
| retain_only | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | th | forget | 77 | 15 | 0.0% | 0.0% | -19.5 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | th | retain | 26 | 24 | 100.0% | 92.3% | -7.7 | -4.2 (-15.8,+0.0) | N/A | 0.0% |
| retain_only | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | en | forget | 77 | 15 | 94.8% | 20.8% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | en | retain | 26 | 23 | 96.2% | 92.3% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | fr | forget | 77 | 13 | 90.9% | 18.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | fr | retain | 26 | 24 | 96.2% | 96.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | de | forget | 77 | 12 | 87.0% | 19.5% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | de | retain | 26 | 22 | 88.5% | 96.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | hi | forget | 77 | 11 | 87.0% | 18.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | hi | retain | 26 | 21 | 84.6% | 96.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | th | forget | 77 | 15 | 94.8% | 19.5% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | th | retain | 26 | 24 | 92.3% | 100.0% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |

## Primary contrast: language effect minus English effect

Negative values mean the post-unlearning multi-hop change is more negative in the listed language than in English. Cohort is restricted to cases where SFT answered both direct and multi-hop probes correctly in both languages. Confidence intervals resample entity groups, preserving within-group dependence. Pilot intervals are exploratory.

| Model | Split | Language | Bilingual cases | Entity groups | EN Δ (pp) | Language Δ (pp) | Language − EN (95% CI, pp) |
|---|---|---:|---:|---:|---:|---:|---:|
