# R1 cross-lingual unlearning diagnostics

AI-only-reviewed translations; not human-validated. Strict normalized exact gold-answer matching; generation failures count as incorrect. Output-invalid is a separate deterministic heuristic flag for generation failure, empty output, overlong output (>64 words or >512 characters), visible template artifacts, a consecutive repeated cycle of 1–5 whitespace tokens repeated three times, or a repeated letter cycle in a no-whitespace, digit-free string; it is not an answer-correctness label. Eligible MH delta is paired within cases both answered correctly by SFT. Confidence intervals resample leakage-isolated entity groups. Main English-vs-language contrasts use only bilingual precompetent cases. The pilot is exploratory; intervals are not multiplicity-adjusted.

Output-invalid diagnostic: 0/7740 predictions (0.00%). Reasons: {}.

| Model | Language | Split | N | Precompetent | Direct acc. | MH acc. | Full MH Δ (pp) | Eligible MH Δ (95% CI, pp) | DAP | Invalid gen. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GA | en | forget | 77 | 15 | 68.8% | 16.9% | -3.9 | -26.7 (-60.0,-7.1) | 73.3% | 0.0% |
| GA | en | retain | 26 | 23 | 96.2% | 73.1% | -19.2 | -21.7 (-41.2,-5.9) | N/A | 0.0% |
| GA | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GA | fr | forget | 77 | 13 | 64.9% | 16.9% | -1.3 | -7.7 (-33.3,+0.0) | 100.0% | 0.0% |
| GA | fr | retain | 26 | 24 | 96.2% | 80.8% | -15.4 | -16.7 (-34.8,-4.0) | N/A | 0.0% |
| GA | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GA | de | forget | 77 | 12 | 67.5% | 16.9% | -2.6 | -8.3 (-37.5,+0.0) | 88.9% | 0.0% |
| GA | de | retain | 26 | 22 | 88.5% | 73.1% | -23.1 | -27.3 (-45.5,-11.1) | N/A | 0.0% |
| GA | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GA | hi | forget | 77 | 11 | 67.5% | 15.6% | -2.6 | -9.1 (-42.9,+0.0) | 90.0% | 0.0% |
| GA | hi | retain | 26 | 21 | 92.3% | 88.5% | -7.7 | -9.5 (-26.7,+0.0) | N/A | 0.0% |
| GA | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GA | th | forget | 77 | 15 | 62.3% | 16.9% | -2.6 | -20.0 (-45.5,+0.0) | 83.3% | 0.0% |
| GA | th | retain | 26 | 24 | 88.5% | 88.5% | -11.5 | -12.5 (-30.0,+0.0) | 100.0% | 0.0% |
| GA | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | en | forget | 77 | 15 | 64.9% | 16.9% | -3.9 | -26.7 (-60.0,-5.6) | 73.3% | 0.0% |
| GD | en | retain | 26 | 23 | 96.2% | 73.1% | -19.2 | -21.7 (-41.2,-8.0) | N/A | 0.0% |
| GD | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | fr | forget | 77 | 13 | 67.5% | 16.9% | -1.3 | -7.7 (-33.3,+0.0) | 100.0% | 0.0% |
| GD | fr | retain | 26 | 24 | 92.3% | 76.9% | -19.2 | -20.8 (-40.0,-6.7) | 100.0% | 0.0% |
| GD | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | de | forget | 77 | 12 | 64.9% | 16.9% | -2.6 | -8.3 (-37.5,+0.0) | 87.5% | 0.0% |
| GD | de | retain | 26 | 22 | 92.3% | 80.8% | -15.4 | -18.2 (-38.9,-3.8) | N/A | 0.0% |
| GD | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | hi | forget | 77 | 11 | 66.2% | 16.9% | -1.3 | -9.1 (-42.9,+0.0) | 90.0% | 0.0% |
| GD | hi | retain | 26 | 21 | 88.5% | 84.6% | -11.5 | -14.3 (-37.5,+0.0) | 100.0% | 0.0% |
| GD | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| GD | th | forget | 77 | 15 | 68.8% | 18.2% | -1.3 | -20.0 (-54.5,+0.0) | 81.8% | 0.0% |
| GD | th | retain | 26 | 24 | 88.5% | 84.6% | -15.4 | -16.7 (-33.3,-3.7) | 100.0% | 0.0% |
| GD | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| NPO | en | forget | 77 | 15 | 68.8% | 16.9% | -3.9 | -26.7 (-58.3,-5.6) | 73.3% | 0.0% |
| NPO | en | retain | 26 | 23 | 96.2% | 73.1% | -19.2 | -21.7 (-41.2,-5.9) | N/A | 0.0% |
| NPO | en | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| NPO | fr | forget | 77 | 13 | 68.8% | 16.9% | -1.3 | -7.7 (-33.3,+0.0) | 100.0% | 0.0% |
| NPO | fr | retain | 26 | 24 | 96.2% | 76.9% | -19.2 | -20.8 (-40.0,-5.9) | N/A | 0.0% |
| NPO | fr | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| NPO | de | forget | 77 | 12 | 63.6% | 16.9% | -2.6 | -8.3 (-37.5,+0.0) | 88.9% | 0.0% |
| NPO | de | retain | 26 | 22 | 88.5% | 80.8% | -15.4 | -18.2 (-38.9,-3.8) | N/A | 0.0% |
| NPO | de | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| NPO | hi | forget | 77 | 11 | 66.2% | 16.9% | -1.3 | -9.1 (-42.9,+0.0) | 90.0% | 0.0% |
| NPO | hi | retain | 26 | 21 | 96.2% | 84.6% | -11.5 | -14.3 (-37.5,+0.0) | N/A | 0.0% |
| NPO | hi | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
| NPO | th | forget | 77 | 15 | 64.9% | 16.9% | -2.6 | -20.0 (-54.5,+0.0) | 83.3% | 0.0% |
| NPO | th | retain | 26 | 24 | 88.5% | 88.5% | -11.5 | -12.5 (-31.6,+0.0) | 100.0% | 0.0% |
| NPO | th | holdout | 26 | 0 | 0.0% | 0.0% | +0.0 | N/A | N/A | 0.0% |
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
| GA | forget | fr | 10 | 6 | -10.0 | +0.0 | +10.0 (+0.0,+50.0) |
| GA | forget | de | 9 | 5 | -33.3 | -11.1 | +22.2 (+0.0,+80.0) |
| GA | forget | hi | 9 | 6 | -11.1 | -11.1 | +0.0 (-33.3,+33.3) |
| GA | forget | th | 11 | 7 | -27.3 | -18.2 | +9.1 (-28.6,+57.1) |
| GA | retain | fr | 21 | 17 | -19.0 | -14.3 | +4.8 (-15.8,+28.6) |
| GA | retain | de | 21 | 17 | -19.0 | -23.8 | -4.8 (-17.6,+11.8) |
| GA | retain | hi | 19 | 13 | -15.8 | -10.5 | +5.3 (+0.0,+10.7) |
| GA | retain | th | 21 | 16 | -19.0 | -9.5 | +9.5 (+0.0,+20.8) |
| GA | holdout | fr | 0 | 0 | N/A | N/A | N/A |
| GA | holdout | de | 0 | 0 | N/A | N/A | N/A |
| GA | holdout | hi | 0 | 0 | N/A | N/A | N/A |
| GA | holdout | th | 0 | 0 | N/A | N/A | N/A |
| GD | forget | fr | 10 | 6 | -10.0 | +0.0 | +10.0 (+0.0,+50.0) |
| GD | forget | de | 9 | 5 | -22.2 | -11.1 | +11.1 (-33.3,+80.0) |
| GD | forget | hi | 9 | 6 | -22.2 | -11.1 | +11.1 (+0.0,+50.0) |
| GD | forget | th | 11 | 7 | -27.3 | -18.2 | +9.1 (-42.9,+71.4) |
| GD | retain | fr | 21 | 17 | -19.0 | -14.3 | +4.8 (-10.5,+23.5) |
| GD | retain | de | 21 | 17 | -19.0 | -14.3 | +4.8 (+0.0,+17.6) |
| GD | retain | hi | 19 | 13 | -15.8 | -10.5 | +5.3 (+0.0,+10.7) |
| GD | retain | th | 21 | 16 | -19.0 | -14.3 | +4.8 (-11.1,+18.8) |
| GD | holdout | fr | 0 | 0 | N/A | N/A | N/A |
| GD | holdout | de | 0 | 0 | N/A | N/A | N/A |
| GD | holdout | hi | 0 | 0 | N/A | N/A | N/A |
| GD | holdout | th | 0 | 0 | N/A | N/A | N/A |
| NPO | forget | fr | 10 | 6 | -10.0 | +0.0 | +10.0 (+0.0,+50.0) |
| NPO | forget | de | 9 | 5 | -22.2 | -11.1 | +11.1 (-33.3,+80.0) |
| NPO | forget | hi | 9 | 6 | -22.2 | -11.1 | +11.1 (+0.0,+50.0) |
| NPO | forget | th | 11 | 7 | -27.3 | -18.2 | +9.1 (-27.3,+57.1) |
| NPO | retain | fr | 21 | 17 | -19.0 | -14.3 | +4.8 (-10.0,+23.5) |
| NPO | retain | de | 21 | 17 | -19.0 | -14.3 | +4.8 (-11.8,+23.5) |
| NPO | retain | hi | 19 | 13 | -15.8 | -10.5 | +5.3 (+0.0,+10.7) |
| NPO | retain | th | 21 | 16 | -19.0 | -9.5 | +9.5 (+0.0,+20.0) |
| NPO | holdout | fr | 0 | 0 | N/A | N/A | N/A |
| NPO | holdout | de | 0 | 0 | N/A | N/A | N/A |
| NPO | holdout | hi | 0 | 0 | N/A | N/A | N/A |
| NPO | holdout | th | 0 | 0 | N/A | N/A | N/A |
