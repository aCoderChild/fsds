# Targeted translation-candidate union audit

Generated: 2026-11-05T15:35:13+00:00 (UTC)

This is an AI-only, source-bound coverage diagnostic. It does not approve translations or create training data.

| Screen | Joint-pass review rows | New unique rows after ordered deduplication |
|---|---:|---:|
| initial_187 | 36 | 35 |
| alias_retry_17 | 3 | 0 |
| nllb_alternates | 3 | 1 |
| segmented_nllb_retry | 0 | 0 |
| third_candidate_overlay | 36 | 13 |

| Coverage measure | Before added candidates | After all deduplicated candidates | Required |
|---|---:|---:|---:|
| Exact four-language forget cases | 36 | 42 | 117 |
| Exact four-language retain cases | 15 | 16 | 39 |
| Optimistic relation-aware forget ceiling | 40 | 47 | 117 |
| Optimistic relation-aware retain ceiling | 17 | 19 | 39 |
| Complete composition training chains | 1 | 2 | 44 |
| Unpassed composition translation rows | 138 | 114 | 0 |

| Language | New atomic rows | New composition rows |
|---|---:|---:|
| FR | 7 | 3 |
| DE | 0 | 8 |
| HI | 7 | 9 |
| TH | 11 | 4 |

All rows above remain quarantined. The AI screen had no human-gold controls and logged a sliding-window/SDPA attention caveat. Translation coverage is not model competence or evidence for the unlearning hypothesis.

The third-candidate overlay ended at 36 pass rows and 151 flags after 18 retries; 13 rows were new to the existing candidate union. Direct Mistral review marked 178/187 pass but disagreed with the initial Qwen labels on 154/187 rows. On constructed mismatch probes Mistral passed 2/15 and Qwen14 passed 1/16; neither small control estimates real-world error rates. No fluent-human or human-gold review exists, so all rows remain quarantined.

The frozen full-release gate remains unchanged. No training corpus was written and no training or unlearning run started.

Machine-readable provenance: `targeted_gap_union_audit.json`.
