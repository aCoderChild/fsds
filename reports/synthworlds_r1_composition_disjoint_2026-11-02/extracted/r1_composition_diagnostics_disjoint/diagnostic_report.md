# R1 cross-lingual unlearning diagnostics

AI-only-reviewed translations; not human-validated. Strict normalized exact gold-answer matching; generation failures count as incorrect. Output-invalid is a separate deterministic heuristic flag for generation failure, empty output, overlong output (>64 words or >512 characters), visible template artifacts, a consecutive repeated cycle of 1–5 whitespace tokens repeated three times, or a repeated letter cycle in a no-whitespace, digit-free string; it is not an answer-correctness label. Eligible MH delta is paired within cases both answered correctly by SFT. Confidence intervals resample leakage-isolated entity groups. Main English-vs-language contrasts use only bilingual precompetent cases. The pilot is exploratory; intervals are not multiplicity-adjusted.

Output-invalid diagnostic: 0/3870 predictions (0.00%). Reasons: {}.

| Model | Language | Split | N | Precompetent | Direct acc. | MH acc. | Full MH Δ (pp) | Eligible MH Δ (95% CI, pp) | DAP | Invalid gen. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained | en | forget | 77 | 8 | 0.0% | 0.0% | -10.4 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | en | retain | 26 | 21 | 0.0% | 0.0% | -92.3 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | fr | forget | 77 | 6 | 0.0% | 0.0% | -9.1 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | fr | retain | 26 | 21 | 0.0% | 0.0% | -96.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | de | forget | 77 | 11 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | de | retain | 26 | 19 | 0.0% | 0.0% | -96.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | hi | forget | 77 | 9 | 0.0% | 0.0% | -14.3 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | hi | retain | 26 | 18 | 0.0% | 0.0% | -88.5 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | th | forget | 77 | 8 | 0.0% | 0.0% | -15.6 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | th | retain | 26 | 18 | 0.0% | 0.0% | -88.5 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | en | forget | 77 | 8 | 0.0% | 0.0% | -10.4 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | en | retain | 26 | 21 | 76.9% | 76.9% | -15.4 | -19.0 (-37.5,-6.2) | 80.0% | 0.0% |
| retain_only | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | fr | forget | 77 | 6 | 0.0% | 0.0% | -9.1 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | fr | retain | 26 | 21 | 65.4% | 80.8% | -15.4 | -23.8 (-43.8,-9.1) | 83.3% | 0.0% |
| retain_only | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | de | forget | 77 | 11 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | de | retain | 26 | 19 | 57.7% | 65.4% | -30.8 | -42.1 (-64.3,-25.0) | 83.3% | 0.0% |
| retain_only | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | hi | forget | 77 | 9 | 0.0% | 0.0% | -14.3 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | hi | retain | 26 | 18 | 61.5% | 76.9% | -11.5 | -22.2 (-47.1,-5.0) | 83.3% | 0.0% |
| retain_only | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | th | forget | 77 | 8 | 0.0% | 0.0% | -15.6 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | th | retain | 26 | 18 | 69.2% | 88.5% | +0.0 | -11.1 (-23.8,+0.0) | 80.0% | 0.0% |
| retain_only | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | en | forget | 77 | 8 | 88.3% | 10.4% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | en | retain | 26 | 21 | 88.5% | 92.3% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | fr | forget | 77 | 6 | 84.4% | 9.1% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | fr | retain | 26 | 21 | 84.6% | 96.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | de | forget | 77 | 11 | 77.9% | 18.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | de | retain | 26 | 19 | 76.9% | 96.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | hi | forget | 77 | 9 | 83.1% | 14.3% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | hi | retain | 26 | 18 | 80.8% | 88.5% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | th | forget | 77 | 8 | 83.1% | 15.6% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | th | retain | 26 | 18 | 80.8% | 88.5% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |

## Primary contrast: language effect minus English effect

Negative values mean the post-unlearning multi-hop change is more negative in the listed language than in English. Cohort is restricted to cases where SFT answered both direct and multi-hop probes correctly in both languages. Confidence intervals resample entity groups, preserving within-group dependence. Pilot intervals are exploratory.

| Model | Split | Language | Bilingual cases | Entity groups | EN Δ (pp) | Language Δ (pp) | Language − EN (95% CI, pp) |
|---|---|---:|---:|---:|---:|---:|---:|
