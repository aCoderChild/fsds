# R1 cross-lingual unlearning diagnostics

AI-only-reviewed translations; not human-validated. Strict normalized exact gold-answer matching; generation failures count as incorrect. Output-invalid is a separate deterministic heuristic flag for generation failure, empty output, overlong output (>64 words or >512 characters), visible template artifacts, a consecutive repeated cycle of 1–5 whitespace tokens repeated three times, or a repeated letter cycle in a no-whitespace, digit-free string; it is not an answer-correctness label. Eligible MH delta is paired within cases both answered correctly by SFT. Confidence intervals resample leakage-isolated entity groups. Main English-vs-language contrasts use only bilingual precompetent cases. The pilot is exploratory; intervals are not multiplicity-adjusted.

Output-invalid diagnostic: 1385/7740 predictions (17.89%). Reasons: {"repetitive_ngram": 1385}.

| Model | Language | Split | N | Precompetent | Direct acc. | MH acc. | Full MH Δ (pp) | Eligible MH Δ (95% CI, pp) | DAP | Invalid gen. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GA | en | forget | 77 | 8 | 0.0% | 0.0% | -14.3 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | en | retain | 26 | 2 | 0.0% | 0.0% | -7.7 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 100.0% |
| GA | fr | forget | 77 | 6 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | fr | retain | 26 | 0 | 0.0% | 0.0% | -3.8 | N/A | N/A | 100.0% |
| GA | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 100.0% |
| GA | de | forget | 77 | 5 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | de | retain | 26 | 2 | 0.0% | 0.0% | -7.7 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 100.0% |
| GA | hi | forget | 77 | 12 | 0.0% | 0.0% | -16.9 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | hi | retain | 26 | 1 | 0.0% | 0.0% | -3.8 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 100.0% |
| GA | th | forget | 77 | 10 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | th | retain | 26 | 1 | 0.0% | 0.0% | -3.8 | -100.0 (-100.0,-100.0) | 0.0% | 100.0% |
| GA | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 100.0% |
| GD | en | forget | 77 | 8 | 0.0% | 0.0% | -14.3 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | en | retain | 26 | 2 | 23.1% | 0.0% | -7.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | fr | forget | 77 | 6 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | fr | retain | 26 | 0 | 11.5% | 0.0% | -3.8 | N/A | N/A | 0.0% |
| GD | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | de | forget | 77 | 5 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | de | retain | 26 | 2 | 3.8% | 0.0% | -7.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | hi | forget | 77 | 12 | 0.0% | 0.0% | -16.9 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | hi | retain | 26 | 1 | 7.7% | 0.0% | -3.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | th | forget | 77 | 10 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | th | retain | 26 | 1 | 7.7% | 0.0% | -3.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| GD | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| NPO | en | forget | 77 | 8 | 0.0% | 0.0% | -14.3 | -100.0 (-100.0,-100.0) | 0.0% | 4.5% |
| NPO | en | retain | 26 | 2 | 19.2% | 0.0% | -7.7 | -100.0 (-100.0,-100.0) | 0.0% | 1.9% |
| NPO | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 5.8% |
| NPO | fr | forget | 77 | 6 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 4.5% |
| NPO | fr | retain | 26 | 0 | 23.1% | 0.0% | -3.8 | N/A | N/A | 1.9% |
| NPO | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 7.7% |
| NPO | de | forget | 77 | 5 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 7.1% |
| NPO | de | retain | 26 | 2 | 11.5% | 0.0% | -7.7 | -100.0 (-100.0,-100.0) | 0.0% | 9.6% |
| NPO | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 3.8% |
| NPO | hi | forget | 77 | 12 | 0.0% | 1.3% | -15.6 | -100.0 (-100.0,-100.0) | 0.0% | 10.4% |
| NPO | hi | retain | 26 | 1 | 11.5% | 0.0% | -3.8 | -100.0 (-100.0,-100.0) | 0.0% | 3.8% |
| NPO | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 13.5% |
| NPO | th | forget | 77 | 10 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 9.1% |
| NPO | th | retain | 26 | 1 | 15.4% | 0.0% | -3.8 | -100.0 (-100.0,-100.0) | N/A | 3.8% |
| NPO | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 15.4% |
| pretrained | en | forget | 77 | 8 | 0.0% | 0.0% | -14.3 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | en | retain | 26 | 2 | 0.0% | 0.0% | -7.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | fr | forget | 77 | 6 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | fr | retain | 26 | 0 | 0.0% | 0.0% | -3.8 | N/A | N/A | 0.0% |
| pretrained | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | de | forget | 77 | 5 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | de | retain | 26 | 2 | 0.0% | 0.0% | -7.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | hi | forget | 77 | 12 | 0.0% | 0.0% | -16.9 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | hi | retain | 26 | 1 | 0.0% | 0.0% | -3.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| pretrained | th | forget | 77 | 10 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | th | retain | 26 | 1 | 0.0% | 0.0% | -3.8 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| pretrained | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | en | forget | 77 | 8 | 0.0% | 0.0% | -14.3 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | en | retain | 26 | 2 | 3.8% | 11.5% | +3.8 | -50.0 (-100.0,+0.0) | 50.0% | 0.0% |
| retain_only | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | fr | forget | 77 | 6 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 1.3% |
| retain_only | fr | retain | 26 | 0 | 11.5% | 15.4% | +11.5 | N/A | N/A | 0.0% |
| retain_only | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | de | forget | 77 | 5 | 0.0% | 0.0% | -11.7 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | de | retain | 26 | 2 | 19.2% | 15.4% | +7.7 | +0.0 (+0.0,+0.0) | 100.0% | 0.0% |
| retain_only | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | hi | forget | 77 | 12 | 0.0% | 0.0% | -16.9 | -100.0 (-100.0,-100.0) | 0.0% | 1.3% |
| retain_only | hi | retain | 26 | 1 | 7.7% | 11.5% | +7.7 | +0.0 (+0.0,+0.0) | 100.0% | 0.0% |
| retain_only | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| retain_only | th | forget | 77 | 10 | 0.0% | 0.0% | -18.2 | -100.0 (-100.0,-100.0) | 0.0% | 0.6% |
| retain_only | th | retain | 26 | 1 | 11.5% | 3.8% | +0.0 | -100.0 (-100.0,-100.0) | 0.0% | 0.0% |
| retain_only | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | en | forget | 77 | 8 | 75.3% | 14.3% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | en | retain | 26 | 2 | 69.2% | 7.7% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | fr | forget | 77 | 6 | 70.1% | 11.7% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | fr | retain | 26 | 0 | 65.4% | 3.8% | +0.0 | N/A | N/A | 0.0% |
| sft | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | de | forget | 77 | 5 | 59.7% | 11.7% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | de | retain | 26 | 2 | 57.7% | 7.7% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | hi | forget | 77 | 12 | 70.1% | 16.9% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | hi | retain | 26 | 1 | 73.1% | 3.8% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| sft | th | forget | 77 | 10 | 59.7% | 18.2% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | th | retain | 26 | 1 | 65.4% | 3.8% | +0.0 | +0.0 (+0.0,+0.0) | N/A | 0.0% |
| sft | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |

## Primary contrast: language effect minus English effect

Negative values mean the post-unlearning multi-hop change is more negative in the listed language than in English. Cohort is restricted to cases where SFT answered both direct and multi-hop probes correctly in both languages. Confidence intervals resample entity groups, preserving within-group dependence. Pilot intervals are exploratory.

| Model | Split | Language | Bilingual cases | Entity groups | EN Δ (pp) | Language Δ (pp) | Language − EN (95% CI, pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| GA | forget | fr | 4 | 2 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GA | forget | de | 3 | 2 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GA | forget | hi | 6 | 3 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GA | forget | th | 4 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GA | retain | fr | 0 | 0 | N/A | N/A | N/A |
| GA | retain | de | 1 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GA | retain | hi | 1 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GA | retain | th | 0 | 0 | N/A | N/A | N/A |
| GA | holdout | fr | 0 | 0 | N/A | N/A | N/A |
| GA | holdout | de | 0 | 0 | N/A | N/A | N/A |
| GA | holdout | hi | 0 | 0 | N/A | N/A | N/A |
| GA | holdout | th | 0 | 0 | N/A | N/A | N/A |
| GD | forget | fr | 4 | 2 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GD | forget | de | 3 | 2 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GD | forget | hi | 6 | 3 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GD | forget | th | 4 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GD | retain | fr | 0 | 0 | N/A | N/A | N/A |
| GD | retain | de | 1 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GD | retain | hi | 1 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| GD | retain | th | 0 | 0 | N/A | N/A | N/A |
| GD | holdout | fr | 0 | 0 | N/A | N/A | N/A |
| GD | holdout | de | 0 | 0 | N/A | N/A | N/A |
| GD | holdout | hi | 0 | 0 | N/A | N/A | N/A |
| GD | holdout | th | 0 | 0 | N/A | N/A | N/A |
| NPO | forget | fr | 4 | 2 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| NPO | forget | de | 3 | 2 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| NPO | forget | hi | 6 | 3 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| NPO | forget | th | 4 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| NPO | retain | fr | 0 | 0 | N/A | N/A | N/A |
| NPO | retain | de | 1 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| NPO | retain | hi | 1 | 1 | -100.0 | -100.0 | +0.0 (+0.0,+0.0) |
| NPO | retain | th | 0 | 0 | N/A | N/A | N/A |
| NPO | holdout | fr | 0 | 0 | N/A | N/A | N/A |
| NPO | holdout | de | 0 | 0 | N/A | N/A | N/A |
| NPO | holdout | hi | 0 | 0 | N/A | N/A | N/A |
| NPO | holdout | th | 0 | 0 | N/A | N/A | N/A |
