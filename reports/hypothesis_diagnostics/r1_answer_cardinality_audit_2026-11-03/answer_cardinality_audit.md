# SynthWorlds-SM source answer-cardinality audit

Status: read-only audit; the frozen v4 split and experiment outputs were not changed.

## Findings

| Check | Result |
|---|---:|
| Type-A source questions | 200 |
| Q1 questions with one source-supported answer | 130 |
| Q1 questions with multiple source-supported answers | 70 |
| Q2 labels matching the complete source-supported answer set | 196 / 200 |
| Q2 questions with multiple or mismatched source outputs | 4 / 200 |
| Current v4 forget/retain cases with a unique source-supported final answer | 103 / 103 |
| Current v4 holdout cases with a source-answer cardinality issue | 1 |
| Exact matches for that holdout case with original / complete source labels | 0 / 0 of 30 |

For the 70 ambiguous Q1 cases, the graph-selected bridge is the only Q1 source candidate with an R2 continuation. A total of 67 of those 70 have a unique source-supported final-answer set. The remaining cases have an ambiguous or mismatched R2 answer set and are excluded from the proposed expansion.

## Candidate expansion (not approved)

A source-backed candidate pool contains 195 cases. Under conservative grouping by graph/document entities, all candidate Q1 fact IDs, chain fact IDs, and answer surfaces, it forms 104 indivisible groups; the largest has 23 cases. An exact 117/39/39 group-preserving split is structurally feasible. No split was created.

The 67 recovered cases have multiple valid answers to their original open-ended Q1. They require a fact-specific yes/no probe that asks whether the answer set to Q1 includes the designated bridge. The same direct-probe format must be used for every case in a new experiment. The candidate set also needs fresh translation/review, a new split approval, and a new baseline gate before unlearning.

The current 129-case v4 results remain frozen. Its sole final-answer cardinality issue is case `0038-qa-sm`, which is in holdout; all 103 forget/retain cases have unique source-supported final answers. Rescoring its 30 saved final-answer predictions against both source-supported answers still yields zero exact matches, so the issue does not change the reported holdout score or enter the primary forget-set cross-language contrast.

## Provenance

Source hashes are checked against `data/synthworlds_sm_pilot_v4/source_manifest.json`. The audit manifest is `reports/hypothesis_diagnostics/r1_answer_cardinality_audit_2026-11-03/answer_cardinality_audit.json`.
