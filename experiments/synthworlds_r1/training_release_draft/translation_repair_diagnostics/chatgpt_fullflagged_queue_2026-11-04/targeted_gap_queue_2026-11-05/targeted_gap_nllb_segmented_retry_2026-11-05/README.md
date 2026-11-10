# Marker-preserving NLLB retry

This bounded AI-only retry covers only the 125 rows that the prior 148-row NLLB candidate screen blocked. That screen showed its alias placeholders were being translated or dropped, especially in Hindi and Thai. This version translates text spans around the original `[ENTITY_*]` and `[NUMBER_*]` markers and copies each marker verbatim into the result. Rows without a protected marker remain blocked.

NLLB generates each target translation and its English back-translation. The pinned Qwen2.5-14B model independently screens eligible candidates for semantic fidelity and target-language fluency. The round-trip uses the same translation model, so back-translation is supporting evidence rather than an independent translation validation. Review is AI-only; no fluent-human review is claimed.

The input includes only source questions, language, row kind, and hashes. It excludes answers, entity facts, and split labels. Outputs remain quarantined: this retry cannot edit canonical translations, approve the source-bound SFT release, or make training eligible.

## Verify before upload

From the repository root:

```bash
python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_segmented_retry_2026-11-05/prepare_segmented_retry.py
```

The offline fixture should confirm the helper copies marker tokens verbatim, translates only text spans, and fails closed when no marker exists. The generated input manifest must report 125 unique blocked rows, 52 Hindi, 71 Thai, and 2 German, with no French rows.

## Run on Kaggle

```bash
kaggle kernels push -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_segmented_retry_2026-11-05 --accelerator NvidiaTeslaT4
```

Use a private kernel, enable Internet, attach the existing Hugging Face token only if required to fetch the pinned models, and select GPU T4 x2. The notebook refuses CPU fallback and requires two T4 GPUs.

After Kaggle completes:

```bash
kaggle kernels output bunnoob2005/sw-gap187-nllb-segmented-retry-2026-11-05 -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_segmented_retry_2026-11-05/remote_output
kaggle kernels pull bunnoob2005/sw-gap187-nllb-segmented-retry-2026-11-05 -p experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_segmented_retry_2026-11-05/remote_source -m
python -B experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_segmented_retry_2026-11-05/verify_segmented_retry.py --remote-dir experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_segmented_retry_2026-11-05/remote_output --pulled-script experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/targeted_gap_queue_2026-11-05/targeted_gap_nllb_segmented_retry_2026-11-05/remote_source/sw-gap187-nllb-segmented-retry-2026-11-05.py
```

The local verifier checks the exact failed-row selection, source/code/output hashes, model revisions, GPU, row coverage, marker multisets, decision schema, and quarantine flags. A passing verifier is still an AI-only diagnostic receipt, not translation approval.
