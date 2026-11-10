# Targeted gap translation candidates

These files contain 187 assistant-generated translation **candidates** for the masked questions in `../requests/`: 21 French, 19 German, 62 Hindi, and 85 Thai.

`audit_candidates.py` checks exact row-ID coverage, answer-blind schemas, protected-marker multiplicities, target-script presence, and question punctuation. Its result is recorded in `candidate_audit.json` with source and candidate SHA-256 hashes.

The structural audit is not a semantic or fluency review. A subsequent answer-blind NLLB + Qwen2.5-14B AI-only screen produced 36 joint semantic/fluency passes from 170 completed decisions; 17 rows were blocked (15 marker-preservation failures and 2 invalid judge outputs). This single AI screen is provisional, and its runtime logged an attention-backend warning. The assistant's exact translation-generation model revision is unknown, and no fluent human has reviewed these candidates. They are not approved translations and are not eligible for the canonical data or SFT corpus. Do not copy them into the `responses/` directory or run the release assembler with them. See `../candidate_review_kaggle/README.md` and its hash-bound artifact for details.
