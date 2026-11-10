# Evaluation implementation reliability audit

**Scope:** the current v5.1 three-probe evaluator and synthetic offline fixtures. No model was loaded and no SFT or unlearning experiment was run. The approved training corpus remains absent.

| Finding | Correction |
|---|---|
| Hindi vowel marks and Thai tone/vowel marks were removed by answer normalization, allowing distinct strings to collapse to the same normalized form. | Unicode combining marks are retained. Tests cover distinct Hindi and Thai examples and canonically equivalent accented text. None of the 1,200 current gold-answer entries changed under the new normalization; generated answers have not been assessed by this check. |
| A prediction marked as a generation failure could score correct if it also contained a matching answer. | Generation-failure flags must be Boolean and failed generations score incorrect. Invalid-output flags remain separately reported. |
| A resumed prediction file could be declared complete when prompt text, case metadata or the seed had changed. | Resume validation now compares the prompt hash, case metadata, seed and full generation configuration before accepting prior rows. |
| The effects code could silently drop the same missing post-unlearning case across all seeds. It also did not compare post-run split/group metadata to the baseline, and the initial baseline gate did not check group consistency across seeds. | Every post model must cover the exact baseline case/language keys and match split/group metadata. The baseline gate rejects inconsistent case groups across models and languages. A post model cannot be reused for different method/seed pairs. |
| A single entity group produced an artificial confidence interval with identical endpoints. | The point estimate is retained but the interval is undefined (`null`) with one group. |
| The reported downstream-answer-persistence (DAP) mean was an unweighted mean across seeds, while its interval used pooled case counts, a different estimand. Per-seed DAP was also placed in an accuracy-change field. | DAP uses its own percentage field. Its interval resamples entity groups and recomputes the mean of defined seed ratios. If a resample gives an included seed a zero denominator, the interval is reported as undefined rather than discarding that draw. The output reports how many seeds had defined DAP. |

The complete offline fixture suite passed **21/21** at low process priority after the subsequent source-integrity additions. It includes tests for multilingual normalization, stale resume data, incomplete post-run coverage, changed group metadata, reused post models, failed-generation scoring, one-group intervals and DAP aggregation. The fixtures demonstrate code behavior; they do not establish translation quality, model competence or unlearning effects.

```sh
nice -n 15 python -B experiments/synthworlds_r1/scripts/test_r1_pilot_pipeline.py
```

At verification, the evaluator SHA-256 was `10d0d1f708ec617d46d3c3ce5a8e000ae1b610b69e20d9f8be3b85a36738c2f2` and the fixture script SHA-256 was `8929ee4b2af112616c1555744409e5c512043be17e1febc95956b43b26d7042c`. The source-verified release entry point was rerun and correctly stopped at 5/117 forget and 3/39 retain atomic cases. These changes apply to future v5.1 evaluation; archived exploratory outputs were not rescored by this audit.
