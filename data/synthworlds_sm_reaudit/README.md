# SynthWorlds-SM | Type-A fact-level Stage 1 re-audit

Status: **PROPOSED_FOR_REVIEW_NOT_APPROVED**. No training files or translations are issued. Source files are unchanged.

- Parsed 1,200 QA rows; Type A: 200.
- Graph-valid exact r1 first-hop facts: 200.
- Q1 uniquely selects n2: 130; 70 Type-A instances excluded due to multiple factual Q1 answers; B–F excluded (1,000).
- Within the strict 130: same graph QID pairs 41; QID-only pairs 18; same chain fact pairs 2; repeated target fact pairs 0; same complete reasoning path 0; same gold answer 22.
- Exact fact groups: {1: 126, 2: 2}.
- Exact-fact + path + answer groups: {1: 98, 2: 12, 4: 2}.
- Provisional counts (never row-split): {'forget': 78, 'retain': 26, 'holdout': 26}. Seed 20261102.
- Candidate forget+retain source-gold-QID document pool: 203; 10 flagged unsafe by holdout source-entity or subject/object co-mention; 193 not-certified-safe candidates.
- Fact rows in candidate pool with holdout chain source QIDs: 3, definitely unsafe.
- Across entire doc corpus, 239 documents flagged for possible holdout-chain disclosure. **Do not interpret remaining documents as proven safe.**

Files:
- `stage1_reaudit_report.json`: full counts, definitions, disclosures, limitations.
- `candidate_split_manifest.json`: source SHA-256 hashes, grouped provisional assignments, target fact IDs.
- `type_a_target_eligibility.jsonl`: exact target matches, Q1 mapping, all Type-A eligibility reasons.
- `excluded_examples.jsonl`: all excluded 1,070 QA IDs and reasons.
- `overlap_pairs_type_a.jsonl`: example-level overlap for each sharing pair.
- `candidate_fact_groups.jsonl`: indivisible groups and target IDs.
- `proposed_split_assignments.jsonl`: proposed labels ONLY, not an approved split or train data.
- `candidate_training_document_risks.jsonl`: all candidate gold QID passages and safety flags.
- `full_corpus_holdout_document_risks.jsonl`: broader entire-corpus natural text risk flags.
- `candidate_training_fact_risks.jsonl`: definite risk in source fact-row training.
- `full_corpus_holdout_fact_exposures.jsonl`: exact source fact-row QIDs for holdout chain.
- `retain_only_document_risks.jsonl`: forget-chain disclosure in retain-only passages.
- `split_specific_document_risks.jsonl`: partition-wise passage exposure and pending verification.
- `split_specific_fact_risks.jsonl`: partition-wise whole-fact-row exposure.
- `excluded_graph_type_source_exposure.jsonl`: excluded B–F source-doc reuse risks.
- `outputs_sha256.json`: generated file hashes, separately from original checksums.
- `build_reaudit.py`: rerunnable Python analysis.

**Interpretation**: A factual split is mathematically feasible, but no natural-language passage-level isolation is fully verified. The policy is ready for researcher review, **not ready for training/deployment**. A conservative pipeline should use fact-level records scoped to verified chain facts, and only admit document passages after sentence-level claim verification/redaction and leakage tests.

To reproduce this audit from the repository root, run:

```bash
python3 data/synthworlds_sm_reaudit/build_reaudit.py
```

The script reads the unchanged source files from `data/synthworlds_sm/` and rewrites the proposal and audit outputs in this folder. The proposed split is not approved training data.
