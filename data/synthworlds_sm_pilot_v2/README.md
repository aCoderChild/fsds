# SynthWorlds-SM multilingual unlearning pilot — v2 (review-only)

**Status:** `PROPOSED_SPLIT_NOT_APPROVED` / `SFT_CANDIDATE_NOT_APPROVED`. No model training performed. All translated text has `human_review_status: "pending"`. Structural checks do not establish linguistic correctness.

## What was independently verified

The original sources are never overwritten. Their hashes, bytes, and 1,200 QA / 6,290 documents / 6,290 fact-record counts match both `synthworlds_sm_reaudit/candidate_split_manifest.json` and the v1 `progress_manifest.json` (see `source_manifest_crosscheck.json`). The pilot retains precisely 130 strict Type-A eligible examples, excluding 70 ambiguous Type A and 1,000 non-Type-A examples. The existing 78 forget / 26 retain / 26 holdout assignment and all 112 indivisible exact-fact/shared-answer groups are unchanged. Exact target and chain facts do not cross split boundaries. **Entities are not disjoint across splits:** 41 pairs of eligible QA share a graph/document QID, including 9 cross-split pairs.

## Newly discovered SFT issue

`0080-qa-sm`: the decomposed Q2 and `graph_qa.r2.relation_str` ask about **Irvin Oathen**; the proposed exact r2 source-fact text refers to **Irvin Thorne (Q131981)**. This is *not a translation issue*. The full QA example is quarantined from SFT (both r1 and r2) rather than silently repairing it; its original split and all audit metadata remain unchanged. See `sft_quarantined_cases.jsonl`.

The English single-hop SFT candidate is **206 QA–hop records** from **103 usable forget/retain QA cases** and 204 distinct exact fact IDs. There are 154 forget records, 52 retain records, and no holdout records. Each SFT example is a **natural-language question and exact source answer**. r1 uses the original Q1 and answers with n2's exact surface; r2 uses Q2 with `<bridge>` tags removed and answers with n3's exact surface or literal. The separate `source_fact_arrow_en` is metadata, never the only SFT text. The SFT export contains no composed multihop evaluation query, `gold_answers` field, or full source document or fact row. Qualified r2 fact provenance and relation IDs are retained. QID-null literals remain `null`, never invented.

## Translation progress

| Category | Completed per `fr/de/hi/th` | Partial per language | Pending per language | Expected |
| --- | ---: | ---: | ---: | ---: |
| QA full record (including translated canonical `gold_docs`) | 1 | 4 | 1,199 (includes 4 partial) | 1,200 |
| Canonical documents | 2 | 0 | 6,288 | 6,290 |
| Canonical fact rows (relation wording only) | 1 | 0 | 6,289 | 6,290 |
| Single-hop SFT questions | 10 | 0 | 196 | 206 |

**Batch 0001** (imported/validated from the earlier v1 seed) includes complete QA `0074-qa-sm`, documents `Q25989` and `Q41329`, fact row `Q25989`, and SFT uses `0074-qa-sm:r1/r2` in all four languages. Gold passages are populated only by joining the translated document QID.

**Batch 0002** includes translated `query` and Q1/Q2 for QA `0000-qa-sm`, `0007-qa-sm`, `0019-qa-sm`, `0020-qa-sm`, and eight translated single-hop SFT questions per language. Their canonical supporting passages are still pending, so QA translations are **partial**, `gold_docs_{lang}` has empty strings with `translation_notes`, and these QA IDs remain among the pending full-record IDs.

`progress_manifest.json` contains **exact completed, partially completed, and pending ID arrays** for every file and language, a per-ID map of all proposed split/group/fact IDs, input hashes, and per-output hashes. `merged_partial/` provides per-language merged files for current inspection, without asserting full-dataset coverage. Batch files remain individually available in `batches/`.

## Portable repository layout

Install this folder under `data/synthworlds_sm_pilot_v2/` in a repository containing the two **existing, unmodified** input directories:

```text
<repo>/
  data/
    synthworlds_sm/
      qa-sm.jsonl
      qa-sm-docs.jsonl
      qa-sm-facts.jsonl
    synthworlds_sm_reaudit/
      candidate_split_manifest.json
      type_a_target_eligibility.jsonl
      proposed_split_assignments.jsonl
      stage1_reaudit_report.json
    synthworlds_sm_pilot_v2/
      scripts/
      translation_payloads/
      batches/
      merged_partial/
      progress_manifest.json
      ...
```

No script depends on a hard-coded working directory or absolute `/mnt/data` source path. They derive `<repo>` from their own file path. If a required input file is missing or mismatched, processing exits with a clear error and **does not create replacement data**.

```bash
python data/synthworlds_sm_pilot_v2/scripts/build_v2.py
python data/synthworlds_sm_pilot_v2/scripts/ingest_translation_batch.py 0001
python data/synthworlds_sm_pilot_v2/scripts/ingest_translation_batch.py 0002
python data/synthworlds_sm_pilot_v2/scripts/merge_v2.py
python data/synthworlds_sm_pilot_v2/scripts/verify_v2.py
```

The existing bundle **already contains** both ingested batches. Do **not** re-ingest them as that would produce a duplicate-batch error. The command sequence above is for an empty v2 installation or reconstruction. For the next translation batch, author an explicit payload in `translation_payloads/batch_0003.json` with complete entries for all four target languages under `qa`, `documents`, and `facts` dictionaries (empty dictionaries permitted), then run the ingest script with `0003`, run `merge_v2.py`, and run `verify_v2.py`. SFT question translations are deterministically derived from the translated QA Q1/Q2, not translated independently. Later completion of previously partial QA cases is permitted through a new batch.

## Important limitations

- Exactly **one** QA has all required multilingual canonical source passages translated; the full-source translation is incomplete. Pending IDs in the manifest are not translated and should not be treated as such.
- `source_fact_arrow_en` and the unmodified source text preserve entity/QID strings and qualifiers. Linguistic quality, nuanced relation direction, and source paraphrase equivalence have **not** been certified by native speakers; translations require review.
- Automatic checks compare numerical tokens, dates, structure, source English fields, graph-linked names and bridge markup. They cannot prove full semantic equivalence, especially for multilingual questions with complex temporal qualifiers.
- Relation PIDs are copied from `graph_qa`; original free-text facts do not independently encode PIDs.
- No split, SFT candidate, or translation has been approved for model training.
