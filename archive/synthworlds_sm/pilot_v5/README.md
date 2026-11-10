# SynthWorlds-SM v5 cohort and probe draft

**Current status: draft only. Do not train or evaluate with these files.**

The v4 exploratory run did not meet the bilingual competence gate, and the v4
SFT checkpoint overlaps the proposed larger cohort. This v5 folder starts a
fresh design; it does not modify the frozen v4 data or results.

## Source-backed cohort and split

The candidate selection contains 195 Type-A cases. Each case has one exact
source-supported target first-hop fact, one exact source-supported second-hop
fact, a unique source-supported final-answer set matching its gold label, and
no alternate source-supported Q1 bridge with an R2 continuation. The known
0080 source/graph conflict and three multi-answer final-path cases are excluded.

Grouping connects cases that share a graph/entity QID, any source fact used by
the candidate path, a normalized answer or graph-node surface, or an entity
alias. This stronger grouping has 102 groups; the largest has 24 cases. The
mixed-integer allocation preserves every group while producing exactly 117
forget, 39 retain, and 39 holdout cases. It also balances group count,
multi-answer Q1 frequency, and date/year-shaped final answers.

## Draft direct probes

Some Q1 prompts have multiple valid source-supported answers. The draft asks
whether the graph-selected bridge is a valid answer to Q1 and requires the
model to return exactly `YES` or `NO`. A paired source-derived control uses an
entity outside the complete source-supported Q1 answer set. A control is
called unsupported relative to this dataset's fact inventory; it is not a
claim about facts outside the synthetic source data.

There are 185 controls from another subject under the same relation and 10
from another relation recorded for the same subject. Their QIDs and surfaces
are disjoint from the candidate cohort, the old v4 train-only composition
entities, and controls in the other split. They are included as a direct-probe
specificity check. The primary outcome remains multi-hop final-answer accuracy
on the original composed question.

## Files

- `candidate_cases.draft.jsonl`: source-checked case metadata and English probes.
- `assignments.draft.jsonl`: exact group-preserving split assignment.
- `leakage_groups.draft.jsonl`: group members and split-balancing counts.
- `labels/gold_labels.draft.jsonl`: separate direct and multi-hop gold labels.
- `translation_requests/en.draft.jsonl`: English prompt text only, without gold labels.
- `manifest.draft.json`: source hashes, split details, status gates, and artifact hashes.
- `LOCAL_VALIDATION.json`: independent source, label, hash, and cross-split checks.
- `scripts/build_candidate_draft.py`: reproducible source selection and draft build.
- `scripts/validate_candidate_draft.py`: independent fail-closed verifier.

Machine-draft translations of the evaluation prompts now exist for all four
target languages. A blinded AI-only screen is complete: 1,587 of 2,340 rows
passed and 753 were flagged, with zero unresolved rows. The review is not
fluent-human certification. Only 12 cases pass the multi-hop translation
screen in all four languages, including five forget cases, so this draft is
not ready for the baseline competence gate. See
`translation_reviews/qwen_blinded_v4/diagnostic_report.md` and `report.json`.

Repair or regenerate the flagged evaluation prompts and rerun the screen
before using them. Then build a fresh multilingual SFT corpus, run the baseline,
and recheck source/passage isolation and competence. The v5 artifacts remain
draft-only and marked `do_not_train` and `do_not_evaluate`; the required gate
remains 15 bilingual-precompetent forget cases and 8 leakage groups per
non-English language.
