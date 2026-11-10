# R1 multilingual unlearning pilot: diagnostic summary

## Main result

The corrected v4 pilot produced exploratory estimates, but it does **not** resolve whether English-only unlearning changes downstream multi-hop accuracy in other languages. The primary bilingual-competence gate failed, and the confidence intervals are wide.

### SFT baseline on the 77 forget cases

| Language | Direct accuracy | Multi-hop accuracy | SFT-precompetent cases* |
|---|---:|---:|---:|
| English | 94.8% | 20.8% | 15 |
| French | 90.9% | 18.2% | 13 |
| German | 87.0% | 19.5% | 12 |
| Hindi | 87.0% | 18.2% | 11 |
| Thai | 94.8% | 19.5% | 15 |

\* Correct on both direct and multi-hop probes before unlearning.

### Target-language minus English multi-hop change

Values are percentage-point contrasts; positive means the target-language change was less negative than the English change. Intervals are 95% entity-group bootstrap intervals.

| Method | French | German | Hindi | Thai |
|---|---:|---:|---:|---:|
| GA | +10.0 [0.0, 50.0] | +22.2 [0.0, 80.0] | 0.0 [−33.3, 33.3] | +9.1 [−28.6, 57.1] |
| GD | +10.0 [0.0, 50.0] | +11.1 [−33.3, 80.0] | +11.1 [0.0, 50.0] | +9.1 [−42.9, 71.4] |
| NPO | +10.0 [0.0, 50.0] | +11.1 [−33.3, 80.0] | +11.1 [0.0, 50.0] | +9.1 [−27.3, 57.1] |

All 12 intervals include zero. The estimates do not show greater damage in the non-English languages, but they also cannot rule out meaningful effects.

## Execution and validity checks

| Check | Result |
|---|---:|
| Evaluated predictions | 7,740 |
| Generation failures / invalid-output flags | 0 / 0 |
| Forget-set bilingual cases by language | FR 10, DE 9, HI 9, TH 11 |
| Leakage-isolated groups for those comparisons | FR 6, DE 5, HI 6, TH 7 |
| Required minimum per language | 15 cases and 8 groups |
| Competence gate | **Failed** |
| Translation review | AI-only; no fluent-human review |
| SFT-to-unlearning multi-hop change on retain cases | −7.7 to −23.1 pp across reported conditions |

The corrected run archive passed ZIP CRC and all indexed SHA-256 checks. The saved predictions were rescored against the current evaluator. Kaggle's raw unlearning manifests did not include the source-script hash, so code-to-run provenance is reconciled after the run rather than cryptographically attested.

## Larger-cohort readiness

| Audit | Finding |
|---|---:|
| Source-backed Type-A candidate pool | 195 cases / 104 internal groups |
| Exact group-preserving 60/20/20 allocation | Feasible: 117 / 39 / 39 |
| Candidate cases already used as current train-only Type-A SFT examples | 50 / 195 |
| Type-B train-only rows with candidate QID overlap | 27 / 85 |
| Candidate groups touched by current SFT overlap screens | 28 / 104 |
| Candidate cases after excluding touched groups | 94 / 195 |
| Existing Type-B composition chains passing source/path and candidate-isolation checks | 44 |
| Complete AI-only-reviewed Type-B training rows available from those chains | 880 (176 per language) |

The 195-case split is only structurally feasible inside the candidate pool. The current SFT checkpoint cannot be reused for it. A new split and corpus are needed, followed by a fresh baseline and the same competence gate before unlearning. A stronger source audit found 44 existing Type-B chains that pass exact three-edge path checks, candidate-QID/surface isolation, and complete five-language translation-row checks; their 880 rows could supply independent composition practice in a rebuilt corpus. This pool remains AI-only reviewed and still needs passage screening against the finalized split. The expansion also changes the direct probe to a fact-specific yes/no question for formerly ambiguous first-hop labels; that probe and its multilingual translations need validation before use.

## Interpretation

This is a technically completed, one-seed, 1B-model exploratory run, not confirmatory evidence. Low SFT multi-hop accuracy and too few bilingual precompetent cases make the primary contrast underpowered. Retain-set degradation is also large enough to warrant monitoring in the next run. The next experiment should use a fresh, leakage-audited split and SFT corpus, validate the revised direct probe, then run baseline evaluation before unlearning.

## Reproducibility references

- Detailed metrics: `extracted/corrected_evaluation/diagnostic_report.md`
- Archive/hash verification: `LOCAL_VERIFICATION.json`
- Candidate overlap audit: `../r1_candidate_sft_overlap_audit_2026-11-03/training_overlap_audit.md`
- Source-answer/cardinality audit: `../r1_answer_cardinality_audit_2026-11-03/answer_cardinality_audit.md`
