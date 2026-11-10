# SFT source-integrity audit

**Status:** prepared source alignment passes; training release remains blocked by translation coverage. No model was loaded and no real release was generated during this audit.

## Finding

The frozen legacy assembler verifies translation review provenance, language balance and matching answers across languages. It did not independently compare every composition answer with the original verified Type-B source at assembly time. In an isolated in-memory fixture, review flags were artificially set to pass and the same incorrect answer was inserted for one composition prompt in all five languages. The legacy `build_corpus` accepted those five rows. This was a synthetic test of a possible data-integrity failure; it did **not** modify source files or show that the current prepared answers are wrong.

The legacy assembler is retained byte-for-byte because completed diagnostic records bind its hash. It must not be used as the sole release validation.

## New gate

`scripts/validate_sft_source_integrity.py` reconstructs atomic rows and the 44 disjoint composition chains from the original source snapshots and the draft case manifest. It checks source and preparation hashes and compares all four prepared inputs with that reconstruction. The present prepared inputs match, yielding 488 canonical SFT IDs.

For an actual release, the validator also checks every answer, English source question, split, hop and fact lineage against those canonical rows; rebuilds the complete corpus from the hash-bound final translation reviews; recomputes the evaluation and split checks; and checks the release manifest. Only then does it write `sft_training/source_integrity_validation.json`, binding the exact corpus, manifest, release validation and verifier script to the source checks. This proof records local validation and is not a cryptographic signature against someone deliberately rewriting files.

`training_release_draft/scripts/assemble_sft_source_verified.py` is the required release entry point. It checks source alignment before invoking the frozen assembler and creates the proof only after validating the complete release. The SFT preflight requires that proof and binds its hash into the SFT run manifest. Unlearning preflight requires the same intact proof. The Kaggle notebook checks and exports the proof; the proof must be generated locally with the original source snapshots present before uploading the project package.

The new entry point currently stops at the existing review-coverage gate: 5/117 forget and 3/39 retain atomic cases pass. No `corpus.jsonl` or source-integrity proof exists yet.

## Verification

The complete low-priority offline suite passed **21/21** locally. It includes an in-memory regression with synthetic passing review statuses: the source validator accepted an unmodified reconstructed corpus and rejected the same wrong answer across all five languages even after corpus and manifest hashes were updated to match. It also tests that the verified assembler stops before release when source reconstruction fails, that SFT preflight requires an intact proof, that a changed corpus invalidates the proof, and that the notebook rejects a stale proof on resume. The two tests requiring the original source snapshots are skipped when those snapshots are not attached, such as a minimal Kaggle package.

```sh
nice -n 15 python -B experiments/synthworlds_r1/scripts/test_r1_pilot_pipeline.py
python -B experiments/synthworlds_r1/scripts/validate_sft_source_integrity.py --prepared-only
python -B experiments/synthworlds_r1/training_release_draft/scripts/assemble_sft_source_verified.py
```

The last command is expected to stop at the translation gate until the review inputs are complete. The verifier SHA-256 at this audit was `b3c58354e2ca4c222ccb4ae2404a169a8ecc2c88e2383ff4e0fdb6a733f333d6`; the source-verified assembler SHA-256 was `afb34f4abad6772c4aab984e520e7345b23ed6e57b4dd92b45d0ac39fcc29beb`. The full fixture script SHA-256 was `8929ee4b2af112616c1555744409e5c512043be17e1febc95956b43b26d7042c`.

The checks establish source alignment and recorded AI-review provenance. They do not certify translation semantics or fluency, and the user's authorization remains AI-only review with no fluent-human audit.
