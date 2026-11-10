# Marker-safe retry of blocked third-candidate rows

This bounded diagnostic retries only the 18 rows blocked by the 187-row Qwen/NLLB screen. It does not modify the candidate translations, the frozen 187-row review, source data, or the training release.

| Retry path | Rows | Purpose |
|---|---:|---|
| Alias-safe back-translation plus Qwen re-review | 15 | Avoid opaque marker loss by temporarily substituting `Alice`/`Bob`, then restore markers only after exact counts match |
| Qwen format retry with existing verified back-translation | 3 | Retry unusable Qwen decision outputs using a short rationale and strict JSON parsing |

All rows are bound to exact source and candidate hashes. The private Kaggle kernel includes only these masked questions and uses pinned NLLB/Qwen revisions. It sends no answers, facts, split labels, or row-to-source mapping. The result remains AI-only diagnostic evidence; alias counts verify integrity, not meaning or fluency.

## Run and verify

From the repository root, run `python -B prepare_retry.py`, push `kaggle_kernel/` with the Kaggle CLI, then download its `marker_safe_retry.json` output into `remote_output/`. Compare the downloaded source with `kaggle kernels pull` and run:

```bash
python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_third_candidate_2026-11-04/marker_safe_retry_2026-11-05/verify_retry.py
```

The verifier enforces exact row order/coverage, parent and input hashes, pinned model/runtime details, source/candidate/back-translation hashes, exact placeholder multisets, decision schema, and explicit non-release labels. No output here can promote translations into the SFT corpus.
