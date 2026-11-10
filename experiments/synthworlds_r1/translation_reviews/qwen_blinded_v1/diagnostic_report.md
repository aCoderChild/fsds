# v5.1 evaluation translation audit

## Status

This is a blinded AI-only translation screen, not fluent-human certification. Draft models: facebook/nllb-200-distilled-600M, ollama:qwen2.5:7b-instruct. Reviewer: ollama:qwen2.5:7b-instruct. Rows translated and screened by the same model count as self-checks, not independent-model review.

Audit status: **AI_ONLY_REVIEW_COMPLETE_WITH_BLOCKED_TRANSLATIONS**. Coverage: 2,337 AI-reviewed rows, 3 translation failures, 0 unresolved reviewer rows, and 0 duplicate-verdict conflicts.

## Results by language

| Language | Rows | AI reviewed | Pass | Flagged | Translation failed | Pass rate / all rows | Cross-model | Same-model self-check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FR | 585 | 585 | 446 | 139 | 0 | 76.2% | 568 | 17 |
| DE | 585 | 585 | 457 | 128 | 0 | 78.1% | 563 | 22 |
| HI | 585 | 582 | 414 | 168 | 3 | 70.8% | 547 | 35 |
| TH | 585 | 585 | 404 | 181 | 0 | 69.1% | 557 | 28 |
| **Total** | **2340** | **2337** | **1721** | **616** | **3** | **73.5%** | **2235** | **102** |

Rows marked as translation failures are not included in AI-reviewed counts. They remain blocked and are not interpreted as review passes or flags.

## Results by probe

| Probe | Rows | AI pass | AI flagged | Translation failed | Pass rate / all rows |
|---|---:|---:|---:|---:|---:|
| Direct positive | 780 | 674 | 105 | 1 | 86.4% |
| Direct negative control | 780 | 637 | 143 | 0 | 81.7% |
| Multi-hop | 780 | 410 | 368 | 2 | 52.6% |

## All-four-language case intersections

A case counts only if every translation for that probe passes in French, German, Hindi, and Thai. Counts are prompt-screen results, not model competence results.

| Probe | All-four cases | Forget cases / groups | Retain cases / groups | Holdout cases / groups |
|---|---:|---:|---:|---:|
| Direct positive | 117 | 67 / 41 | 26 / 17 | 24 / 14 |
| Direct negative control | 106 | 61 / 38 | 21 / 14 | 24 / 13 |
| Multi-hop | 31 | 16 / 12 | 9 / 6 | 6 / 4 |

### Primary-probe joint eligibility

| Joint screen | Forget cases / groups | Retain cases / groups | Holdout cases / groups |
|---|---:|---:|---:|
| Direct positive + multi-hop pass in all four target languages | 9 / 7 | 5 / 4 | 5 / 4 |
| All three probes pass in all four target languages | 8 / 7 | 4 / 3 | 2 / 2 |

The forget-set primary-probe intersection is below the planned 15-case/8-group floor. The translations are therefore not ready to establish the pre-unlearning competence cohort. No SFT or unlearning was started from this v5.1 draft.

## Integrity and limits

- Request SHA-256: `8fef3fb458676b6b7c24b53dbdc8b77124d80ab20534d2caffba3176c9dcebf2`
- Review report SHA-256: `26c5936e5c8fed9b7f80d6f2ae7a2c2f88171cb7e6476aa93610b6bdde249a2e`
- Each language draft and review file is SHA-256-bound in `report.json`; all prompt IDs and source metadata matched the English request file.
- The screen reviewed 1,660 unique masked source/translation payloads; identical payloads had no conflicting verdicts.
- AI judgments may miss semantic or pragmatic translation errors. No fluent-human review was performed.
- The source question wording itself can be awkward; a screen pass is not a guarantee that a prompt is scientifically well-posed.
- Failed placeholder recovery and AI flags require repair or retranslation before training/evaluation approval.

## Reproduction

Run `python experiments/synthworlds_r1/scripts/summarize_translation_audit.py --draft-dir experiments/synthworlds_r1/translation_drafts/merged_v1 --review-dir experiments/synthworlds_r1/translation_reviews/qwen_blinded_v1` from the repository root. The script verifies request, translation, review, provenance, coverage, and masked-pair hashes before writing this report and `summary.json`.
