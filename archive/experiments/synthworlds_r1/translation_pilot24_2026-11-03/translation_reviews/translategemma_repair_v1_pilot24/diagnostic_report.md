# v5.1 evaluation translation audit

## Status

This is a blinded AI-only translation screen, not fluent-human certification. Draft models: facebook/nllb-200-distilled-600M, ollama:qwen2.5:7b-instruct, ollama:translategemma:4b. Reviewer: ollama:qwen2.5:7b-instruct. Rows translated and screened by the same model count as self-checks, not independent-model review.

Audit status: **AI_ONLY_REVIEW_COMPLETE_WITH_BLOCKED_TRANSLATIONS**. Coverage: 2,330 AI-reviewed rows, 10 translation failures, 0 unresolved reviewer rows, and 0 duplicate-verdict conflicts.

## Results by language

| Language | Rows | AI reviewed | Pass | Flagged | Translation failed | Pass rate / all rows | Cross-model | Same-model self-check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FR | 585 | 576 | 455 | 121 | 9 | 77.8% | 559 | 17 |
| DE | 585 | 585 | 473 | 112 | 0 | 80.9% | 564 | 21 |
| HI | 585 | 585 | 429 | 156 | 0 | 73.3% | 555 | 30 |
| TH | 585 | 584 | 417 | 167 | 1 | 71.3% | 558 | 26 |
| **Total** | **2340** | **2330** | **1774** | **556** | **10** | **75.8%** | **2236** | **94** |

Rows marked as translation failures are not included in AI-reviewed counts. They remain blocked and are not interpreted as review passes or flags.

## Results by probe

| Probe | Rows | AI pass | AI flagged | Translation failed | Pass rate / all rows |
|---|---:|---:|---:|---:|---:|
| Direct positive | 780 | 686 | 90 | 4 | 87.9% |
| Direct negative control | 780 | 653 | 122 | 5 | 83.7% |
| Multi-hop | 780 | 435 | 344 | 1 | 55.8% |

## All-four-language case intersections

A case counts only if every translation for that probe passes in French, German, Hindi, and Thai. Counts are prompt-screen results, not model competence results.

| Probe | All-four cases | Forget cases / groups | Retain cases / groups | Holdout cases / groups |
|---|---:|---:|---:|---:|
| Direct positive | 124 | 72 / 41 | 26 / 17 | 26 / 14 |
| Direct negative control | 112 | 65 / 38 | 21 / 14 | 26 / 14 |
| Multi-hop | 35 | 18 / 13 | 9 / 6 | 8 / 5 |

### Primary-probe joint eligibility

| Joint screen | Forget cases / groups | Retain cases / groups | Holdout cases / groups |
|---|---:|---:|---:|
| Direct positive + multi-hop pass in all four target languages | 10 / 8 | 5 / 4 | 6 / 4 |
| All three probes pass in all four target languages | 10 / 8 | 4 / 3 | 4 / 3 |

The forget-set primary-probe intersection is below the planned 15-case/8-group floor. The translations are therefore not ready to establish the pre-unlearning competence cohort. No SFT or unlearning was started from this v5.1 draft.

## Integrity and limits

- Request SHA-256: `8fef3fb458676b6b7c24b53dbdc8b77124d80ab20534d2caffba3176c9dcebf2`
- Review report SHA-256: `1faa1728b8bb567e0823a6c1936fbab22ae0dca6a33309ba49806dbb669319b6`
- Each language draft and review file is SHA-256-bound in `report.json`; all prompt IDs and source metadata matched the English request file.
- The screen reviewed 1,660 unique masked source/translation payloads; identical payloads had no conflicting verdicts.
- AI judgments may miss semantic or pragmatic translation errors. No fluent-human review was performed.
- The source question wording itself can be awkward; a screen pass is not a guarantee that a prompt is scientifically well-posed.
- Failed placeholder recovery and AI flags require repair or retranslation before training/evaluation approval.

## Reproduction

Run `python data/synthworlds_sm_pilot_v5_1/scripts/summarize_translation_audit.py --draft-dir /Users/maianhpham/Documents/fsds/data/synthworlds_sm_pilot_v5_1/translation_drafts/translategemma_repair_v1_pilot24 --review-dir /Users/maianhpham/Documents/fsds/data/synthworlds_sm_pilot_v5_1/translation_reviews/translategemma_repair_v1_pilot24` from the repository root. The script verifies request, translation, review, provenance, coverage, and masked-pair hashes before writing this report and `summary.json`.
