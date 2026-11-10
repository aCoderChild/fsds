# Retry the 17 quarantined translation-screen rows

This is a bounded AI-only diagnostic retry for 15 rows where NLLB changed opaque placeholder counts and 2 rows where the prior Qwen judge emitted invalid output. It reads the exact source-bound rows from the previous review, uses collision-checked `Alice`/`Bob` aliases for the 15 marker round-trips, reuses the two prior hash-verified back-translations, and retries the pinned Qwen judge with a short schema-constrained response.

The retry contains only question text, language, row kind, and hashes. It includes no answer, split label, or entity facts. It does not modify canonical translations, the frozen candidate merge, release manifest, or SFT corpus. No output from this job can approve a translation or make training eligible.

## Run

From the repository root:

```bash
python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_retry_2026-11-05/prepare_retry.py
kaggle kernels push -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_retry_2026-11-05 --accelerator NvidiaTeslaT4
```

Choose GPU T4 x2 in Kaggle Notebook settings before running if Kaggle does not retain the accelerator selection. The script fails closed without a GPU. After completion, download the kernel output into `remote_output/`, then verify the pulled source and receipt:

```bash
kaggle kernels output bunnoob2005/sw-gap187-alias-retry-2026-11-05 -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_retry_2026-11-05/remote_output
kaggle kernels pull bunnoob2005/sw-gap187-alias-retry-2026-11-05 -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_retry_2026-11-05/remote_source
python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_retry_2026-11-05/verify_retry.py --remote-dir experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_retry_2026-11-05/remote_output --pulled-script experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_retry_2026-11-05/remote_source/retry.py
```

The local verifier rejects mismatched inputs, model revisions, source code, GPU type, missing/duplicate rows, broken marker multisets, invalid decisions, or inconsistent prior back-translations. A passing receipt remains a quarantine-only AI diagnostic; human review and release gates remain unchanged.
