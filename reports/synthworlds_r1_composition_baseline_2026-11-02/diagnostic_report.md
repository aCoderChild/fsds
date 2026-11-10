# R1 cross-lingual unlearning diagnostics

AI-only-reviewed translations; not human-validated. Strict normalized exact gold-answer matching; generation failures count as incorrect. Output-invalid is a separate deterministic heuristic flag for generation failure, empty output, overlong output (>64 words or >512 characters), visible template artifacts, a consecutive repeated cycle of 1–5 whitespace tokens repeated three times, or a repeated letter cycle in a no-whitespace, digit-free string; it is not an answer-correctness label. Eligible MH delta is paired within cases both answered correctly by SFT. Confidence intervals resample leakage-isolated entity groups. Main English-vs-language contrasts use only bilingual precompetent cases. The pilot is exploratory; intervals are not multiplicity-adjusted.

Output-invalid diagnostic: 0/3870 predictions (0.00%). Reasons: {}.

| Model | Language | Split | N | Precompetent | Direct acc. | MH acc. | Full MH Δ (pp) | Eligible MH Δ (95% CI, pp) | DAP | Invalid gen. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained | en | forget | 77 | 8 | 0.0% | 0.0% | -13.0 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | en | retain | 26 | 15 | 0.0% | 0.0% | -80.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | fr | forget | 77 | 4 | 0.0% | 0.0% | -7.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | fr | retain | 26 | 12 | 0.0% | 0.0% | -76.9 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | de | forget | 77 | 9 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | de | retain | 26 | 14 | 0.0% | 0.0% | -84.6 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | hi | forget | 77 | 10 | 0.0% | 0.0% | -15.6 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | hi | retain | 26 | 16 | 0.0% | 0.0% | -88.5 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | th | forget | 77 | 7 | 0.0% | 0.0% | -15.6 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | th | retain | 26 | 13 | 0.0% | 0.0% | -76.9 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | en | forget | 77 | 8 | 0.0% | 0.0% | -13.0 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | en | retain | 26 | 15 | 19.2% | 53.8% | -26.9 | -40.0 (-69.2,-15.4) | 45.5% | 0.0% |
| retain_only | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | fr | forget | 77 | 4 | 0.0% | 0.0% | -7.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | fr | retain | 26 | 12 | 26.9% | 69.2% | -7.7 | -33.3 (-63.6,-7.7) | 55.6% | 0.0% |
| retain_only | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | de | forget | 77 | 9 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | de | retain | 26 | 14 | 19.2% | 57.7% | -26.9 | -42.9 (-69.2,-20.0) | 50.0% | 0.0% |
| retain_only | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | hi | forget | 77 | 10 | 0.0% | 0.0% | -15.6 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | hi | retain | 26 | 16 | 11.5% | 65.4% | -23.1 | -43.8 (-76.9,-18.8) | 53.3% | 0.0% |
| retain_only | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | th | forget | 77 | 7 | 0.0% | 0.0% | -15.6 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | th | retain | 26 | 13 | 19.2% | 57.7% | -19.2 | -30.8 (-70.0,-7.7) | 60.0% | 0.0% |
| retain_only | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | en | forget | 77 | 8 | 79.2% | 13.0% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | en | retain | 26 | 15 | 73.1% | 80.8% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | fr | forget | 77 | 4 | 74.0% | 7.8% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | fr | retain | 26 | 12 | 65.4% | 76.9% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | de | forget | 77 | 9 | 71.4% | 18.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | de | retain | 26 | 14 | 69.2% | 84.6% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | hi | forget | 77 | 10 | 67.5% | 15.6% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | hi | retain | 26 | 16 | 73.1% | 88.5% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | th | forget | 77 | 7 | 74.0% | 15.6% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | th | retain | 26 | 13 | 73.1% | 76.9% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |

## Primary contrast: language effect minus English effect

Negative values mean the post-unlearning multi-hop change is more negative in the listed language than in English. Cohort is restricted to cases where SFT answered both direct and multi-hop probes correctly in both languages. Confidence intervals resample entity groups, preserving within-group dependence. Pilot intervals are exploratory.

| Model | Split | Language | Bilingual cases | Entity groups | EN Δ (pp) | Language Δ (pp) | Language − EN (95% CI, pp) |
|---|---|---:|---:|---:|---:|---:|---:|
