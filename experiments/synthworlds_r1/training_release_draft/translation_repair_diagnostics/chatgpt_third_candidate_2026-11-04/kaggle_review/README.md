# Third-candidate AI-only review (187 masked rows)

This Kaggle job is a bounded diagnostic screen of the 187 unresolved repair candidates. The input contains only masked English source questions, candidate translations, language, and atomic/composition category. It contains no answer, hidden fact, or split assignment.

The runner is copied from the previously used, revision-pinned NLLB-200 + Qwen2.5-14B review pipeline. It back-translates each candidate and asks Qwen for independent semantic/fluency flags. It checkpoints after every row. Output is **AI-only evidence**, not fluent-human certification and not approval for training.

## Verify before use

After downloading Kaggle output, run:

```bash
python verify_output.py path/to/repair_review.json path/to/verification.json
```

The verifier requires exact 187-row coverage/order, input hash, pinned model revisions, per-row source/candidate hashes, placeholder preservation, valid decision schema, and explicit AI-only / answer-blind labels. Any blocked row keeps the review status blocked. The result never edits or releases the training corpus.

## Kaggle settings

GPU and Internet are enabled. The reviewer loads public model checkpoints; no Hugging Face token is needed. No dataset sources are attached because the masked input is embedded and hash-bound in `review.py`.
