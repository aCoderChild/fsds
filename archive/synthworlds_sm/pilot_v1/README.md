# SynthWorlds-SM multilingual unlearning pilot — partial deliverable

**Status:** UNAPPROVED PROPOSED SPLIT / UNAPPROVED FACT-LEVEL SFT CANDIDATES / PARTIAL TRANSLATION. This archive does not contain trained models, nor does it authorize training.

## Source and split validation

`build_pilot.py` reads the three sources and four earlier audit artifacts from `/mnt/data` without modifying them, checks the documented SHA-256 digests, byte sizes and row counts, independently rechecks the Type-A graph target/bridge matching and recomputes all derived fact hashes from `(source_fact_record_qid + NUL + exact_original_fact_text)`. It confirms the proposed 78/26/26 assignment, 112 indivisible groups and zero cross-split chain-fact or gold-answer matches.

Strict eligible: 130 Type-A cases; excluded: 70 ambiguous Type-A cases and 1,000 other graph types. No split is approved.

## SFT candidate records

`pilot_fact_sft_en.jsonl` holds 206 unique factual statements, linked to 208 r1/r2 uses across 104 forget/retain QA cases. A repeated exact fact has one record and multiple `uses` entries. Each use identifies the source QA, target ID, group, proposed split, relation PID, hop, and canonical entity QIDs where they exist. If a graph node is literal, its QID is `null` rather than invented. Full documents, full per-QID fact lists, the composed evaluation question, and separately labeled gold answers are excluded. The r2 fact necessarily names the final answer entity because its object is the supported relationship; that entity is not repeated in an evaluation-answer field.

`verified_type_a_qa_index.jsonl` gives all 130 candidate QA mappings, including holdout mappings for auditing (not training). `excluded_ambiguous_type_a.jsonl` lists the 70 ineligible Type-A examples.

## Multilingual seed batch 0001

Files under `batch_0001/` contain one fully translated QA (ID `0074-qa-sm`), both canonical supporting documents (`Q41329`, `Q25989`), one complete source fact record (`Q25989`, exactly one fact), and the two corresponding eligible pilot SFT fact records. The batch covers French, German, Hindi and Thai.

Every localized QA retains its original `mhop_qa` string, `query`, `gold_docs`, gold answer and other source metadata; it adds `query_<lang>`, `decomposed_questions_<lang>` (same `Q1`/`Q2` keys), and `gold_docs_<lang>`. The translated gold passages come **only** from the corresponding translated canonical document records in this batch and are verified against source QID-to-document alignment.

Each localized document retains `doc` and adds `doc_<lang>`. Each localized fact row retains `facts` and adds `facts_<lang>` with original subject and object texts/IDs and relation order unchanged. Each SFT fact retains the English statement and adds `fact_<lang>`, `relation_<lang>` and `training_text_<lang>`.

## Pending work and limitations

`progress_manifest.json` enumerates **exact completed and pending IDs** for each of the four languages and four data categories, records the translation method, source hashes, split/group mappings and output hashes. No records are counted as completed unless localized fields have been written and structural checks passed. The first batch's language translations were manually generated in this session using GPT-6, *not* from an installed large-scale translation system. No full-dataset translation is implied.

The final translated texts need human linguistic review for semantic equivalence, especially the meaning of relation descriptions and organizational terms. Structural Python checks preserve synthetic names, dates, numbers, IDs and bridge markup but cannot guarantee translation accuracy. Remaining IDs must be processed in later batches using the manifest to avoid duplication. Neither original source files nor prior audit artifacts were overwritten.

Rebuild/revalidate source-level candidate data with `python build_pilot.py`, then regenerate the initial translation batch with `python translate_batch_0001.py`; both scripts assume the same input location `/mnt/data`. Do not run these scripts if the files listed in the manifest are not accessible or have changed.
