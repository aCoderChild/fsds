# v5 evaluation-prompt translation screen

**Status: AI-only screening complete; repairs required. The v5 draft remains blocked from training and evaluation.**

## Scope and verification

The screen covered 2,340 translated evaluation prompts: 195 source-backed cases, three probes per case, and four target languages (French, German, Hindi, and Thai). Qwen 2.5 7B Instruct assessed the masked English/translation pairs for meaning and scope, naturalness, protected entity preservation, and response-contract preservation. Entity names and direct-probe labels were hidden from the reviewer.

The review contains 2,340 decisions over 1,661 globally deduplicated masked text pairs. There were no conflicting decisions for duplicate pairs, no model-call failures, and no unresolved rows. All deterministic checks passed: prompt text was nonempty; protected names/numbers, target script, and YES/NO contract were preserved; and no masking markers remained. SHA-256 values for each merged translation and review file match the review manifest. No fluent-human review was performed.

Of the reviewed rows, 2,242 used NLLB translations screened by Qwen (cross-model checks); 98 Qwen-fallback translations were screened by Qwen itself (same-model consistency checks). Those 98 are not independent verification.

## Results by language

| Language | Prompts | AI-screen pass | Flagged | Pass rate |
|---|---:|---:|---:|---:|
| French | 585 | 408 | 177 | 69.7% |
| German | 585 | 431 | 154 | 73.7% |
| Hindi | 585 | 375 | 210 | 64.1% |
| Thai | 585 | 373 | 212 | 63.8% |
| **Total** | **2,340** | **1,587** | **753** | **67.8%** |

## Results by probe

| Probe | Prompts | AI-screen pass | Flagged | Pass rate |
|---|---:|---:|---:|---:|
| Direct positive | 780 | 767 | 13 | 98.3% |
| Direct negative control | 780 | 665 | 115 | 85.3% |
| Multi-hop final question | 780 | 155 | 625 | 19.9% |

The low multi-hop pass rate is the main problem for the primary outcome. Examples include French translating “given name” literally as *nom donné*, and Hindi rendering it as a surname (*उपनाम*). These are concrete translation defects, while other flags may be conservative model judgments and need adjudication.

## Cross-language usable-prompt intersection

| Inclusion rule | Cases passing in all four languages | Forget | Retain | Holdout |
|---|---:|---:|---:|---:|
| Multi-hop prompt passes in FR, DE, HI, and TH | 12 | 5 | 4 | 3 |
| All three probes pass in all four languages | 9 | 3 | 3 | 3 |

The first row is the relevant translation-screen intersection for the primary multi-hop outcome. Only five forget cases/groups remain. This is below the planned floor of 15 bilingual-precompetent forget cases and eight leakage groups per language even before measuring model competence after SFT.

## Interpretation and action

The screen verifies file completeness and detects a clear concentration of concerns in multi-hop translations; it does not certify translation accuracy or human fluency. Do not use the currently flagged rows. Repair or regenerate the multi-hop prompts (and flagged negative controls), rerun the blinded screen with the same hash-bound protocol, then recompute the all-language intersection. Keep the split and files marked draft; do not train or evaluate until repairs, source/passage isolation, and the baseline competence gate have been independently checked.

Machine-readable results and file hashes are in `report.json`. The source translations are in `../../translation_drafts/merged/`.
