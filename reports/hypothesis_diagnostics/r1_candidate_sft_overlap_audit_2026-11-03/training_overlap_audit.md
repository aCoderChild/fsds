# Candidate-pool overlap with the completed v4 SFT corpus

Status: read-only audit. The existing candidate split, SFT checkpoints, translations, and experiment outputs were not changed.

## Findings

| Check | Result |
|---|---:|
| Source-backed Type-A candidate pool | 195 cases in 104 internal entity/fact/answer groups |
| Candidate-only exact 60/20/20 allocation | 117 / 39 / 39; feasible: true |
| Current SFT train-only source QA IDs | 135 (50 Type A, 85 Type B) |
| Candidate cases already present as train-only Type A | 50 |
| Type-B train-only QA IDs sharing a candidate graph/entity QID | 27 / 85 |
| Candidate cases with a Type-B train-only QID overlap | 23 |
| Type-B train-only QA IDs with an exact candidate answer-surface overlap | 31 / 85 |
| English Type-B SFT output rows with an exact candidate answer-surface overlap | 58 |
| Candidate groups touched by these train-only overlap screens | 28 |
| Candidate rows remaining after excluding those whole groups | 94 in 76 groups |

## Candidate source pool for a rebuilt SFT corpus

This first-pass screen checks non-candidate QA rows for exact graph/entity QID and normalized entity/answer-surface overlap with the full 195-case pool. A row passing it is only a training-pool candidate; source-path and passage audits are still required.

| Graph type | Non-candidate rows | Entity-QID and surface disjoint rows | Entity-QID-only overlap | Surface-only overlap | Both overlaps |
|---|---:|---:|---:|---:|---:|
| A | 5 | 1 | 1 | 2 | 1 |
| B | 200 | 62 | 8 | 15 | 115 |
| C | 200 | 59 | 2 | 16 | 123 |
| D | 200 | 60 | 4 | 13 | 123 |
| E | 200 | 36 | 3 | 10 | 151 |
| F | 200 | 44 | 2 | 11 | 143 |
| **Total disjoint rows** | -- | **262** | -- | -- | -- |

The disjoint-row total is not a verified replacement corpus. Each selected chain still needs exact fact-path validation, composition-question/answer checks, passage-disclosure screening, and translation review.

A stronger check within the already translated Type-B training-only rows validates exact three-edge source paths, final-node labels, no candidate QID/entity/answer-surface overlap, and all 20 prompt rows per chain (four prompt types in five languages). It leaves **44 chains** (176 rows per language). These are useful replacement-corpus candidates, not an approved SFT corpus; they remain AI-only reviewed.

## Interpretation

The candidate-only split calculation is not enough to make this expansion safe with the existing checkpoint. Fifty of the 195 candidate questions were explicitly included as Type-A `train_only` SFT examples. The Type-B train-only set also shares graph/entity QIDs and exact answer strings with some candidates. QID overlap and answer-surface overlap are conservative exposure warnings, not proof that a specific target triple was taught, but they require a fresh corpus design before this set can be evaluated.

The 117/39/39 feasibility result means only that the candidate set can be partitioned internally by its connected groups. It does not account for SFT exposure. The current SFT checkpoint must not be reused for this 195-case expansion. A new version should reserve candidate forget/retain/holdout groups first, remove or replace train-only rows that overlap those groups, build the direct-probe and multilingual evaluation data, train a new baseline, and re-run the competence gate before unlearning.

The answer-surface screen matches candidate Q1 possible-answer and final-answer strings against English Type-B SFT outputs after the audit's normalized exact-answer transform. It is a conservative screen; no claim-level human adjudication was performed.

## Provenance

The source JSONL hashes are rechecked against `data/synthworlds_sm_pilot_v4/source_manifest.json`. The SFT corpus hash is rechecked against its completed run manifest. Candidate membership is reproduced using `data/synthworlds_sm_pilot_v4/scripts/audit_source_answer_cardinality.py` helpers and the same exact source-answer/path conditions. Machine-readable results: `reports/hypothesis_diagnostics/r1_candidate_sft_overlap_audit_2026-11-03/training_overlap_audit.json`.
