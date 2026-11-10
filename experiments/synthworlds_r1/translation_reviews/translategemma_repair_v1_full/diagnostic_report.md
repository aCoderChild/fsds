# v5.1 evaluation translation audit

## Status

This is a blinded AI-only translation screen, not fluent-human certification. Draft models: facebook/nllb-200-distilled-600M, ollama:qwen2.5:7b-instruct, ollama:translategemma:4b. Reviewer: ollama:qwen2.5:7b-instruct. Rows translated and screened by the same model count as self-checks, not independent-model review.

Audit status: **AI_ONLY_REVIEW_COMPLETE_WITH_BLOCKED_TRANSLATIONS**. Coverage: 2,293 AI-reviewed rows, 47 translation failures, 0 unresolved reviewer rows, and 0 duplicate-verdict conflicts.

## Results by language

| Language | Rows | AI reviewed | Pass | Flagged | Translation failed | Pass rate / all rows | Cross-model | Same-model self-check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FR | 585 | 542 | 509 | 33 | 43 | 87.0% | 535 | 7 |
| DE | 585 | 585 | 559 | 26 | 0 | 95.6% | 572 | 13 |
| HI | 585 | 583 | 548 | 35 | 2 | 93.7% | 573 | 10 |
| TH | 585 | 583 | 521 | 62 | 2 | 89.1% | 569 | 14 |
| **Total** | **2340** | **2293** | **2137** | **156** | **47** | **91.3%** | **2249** | **44** |

Rows marked as translation failures are not included in AI-reviewed counts. They remain blocked and are not interpreted as review passes or flags.

## Results by probe

| Probe | Rows | AI pass | AI flagged | Translation failed | Pass rate / all rows |
|---|---:|---:|---:|---:|---:|
| Direct positive | 780 | 747 | 11 | 22 | 95.8% |
| Direct negative control | 780 | 742 | 17 | 21 | 95.1% |
| Multi-hop | 780 | 648 | 128 | 4 | 83.1% |

## All-four-language case intersections

A case counts only if every translation for that probe passes in French, German, Hindi, and Thai. Counts are prompt-screen results, not model competence results.

| Probe | All-four cases | Forget cases / groups | Retain cases / groups | Holdout cases / groups |
|---|---:|---:|---:|---:|
| Direct positive | 164 | 94 / 51 | 34 / 19 | 36 / 17 |
| Direct negative control | 160 | 95 / 51 | 29 / 16 | 36 / 17 |
| Multi-hop | 110 | 61 / 34 | 28 / 17 | 21 / 14 |

### Primary-probe joint eligibility

| Joint screen | Forget cases / groups | Retain cases / groups | Holdout cases / groups |
|---|---:|---:|---:|
| Direct positive + multi-hop pass in all four target languages | 46 / 26 | 25 / 16 | 19 / 12 |
| All three probes pass in all four target languages | 44 / 25 | 21 / 13 | 18 / 11 |

The forget-set primary-probe intersection is below the planned 15-case/8-group floor. The translations are therefore not ready to establish the pre-unlearning competence cohort. No SFT or unlearning was started from this v5.1 draft.

## Integrity and limits

- Request SHA-256: `8fef3fb458676b6b7c24b53dbdc8b77124d80ab20534d2caffba3176c9dcebf2`
- Review report SHA-256: `fff853181ce058d68c392c0292e3740f501a8f0207c80be78b547630c818d3e3`
- Each language draft and review file is SHA-256-bound in `report.json`; all prompt IDs and source metadata matched the English request file.
- The screen reviewed 1,621 unique masked source/translation payloads; identical payloads had no conflicting verdicts.
- AI judgments may miss semantic or pragmatic translation errors. No fluent-human review was performed.
- The source question wording itself can be awkward; a screen pass is not a guarantee that a prompt is scientifically well-posed.
- Failed placeholder recovery and AI flags require repair or retranslation before training/evaluation approval.

## Reproduction

Run `python experiments/synthworlds_r1/scripts/summarize_translation_audit.py --draft-dir experiments/synthworlds_r1/translation_drafts/translategemma_repair_v1_full --review-dir experiments/synthworlds_r1/translation_reviews/translategemma_repair_v1_full` from the repository root. The script verifies request, translation, review, provenance, coverage, and masked-pair hashes before writing this report and `summary.json`.
