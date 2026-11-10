# Model scale and translation-review addendum — 2026-11-04

## Model-size decision

Meta's published Llama lineup lists Llama 3.2 text models at 1B and 3B, and Llama 3.1 text models at 8B and 405B. It does not list an official 4B Llama checkpoint. The closest small Llama option to a requested 4B comparison is Llama 3.2 3B; the R1 report-aligned main model remains Llama 3.1 8B.

Kaggle's current notebook documentation describes a T4 x2 accelerator as two T4 GPUs with 16 GB each. This makes 8B NF4 QLoRA a plausible setup to validate, but the local R1 smoke evidence does not establish that it works end to end: the gated-model configuration request was denied before any 8B weights loaded. The existing notebook and training/evaluation/unlearning profiles target 8B. Do not interpret the two-GPU specification as a completed runtime check.

## Translation-review reconciliation

The full flagged-candidate review has a verified exact 799-row merge. The first run produced 702 verified decisions and 31 blocked attempts before its Kaggle worker was killed after logging 733/799; a retry reviewed the exact 97 unresolved rows. The merged receipt confirms 799 unique source-bound rows:

| AI-only candidate review status | Rows |
|---|---:|
| Pass | 445 |
| Flagged | 323 |
| Blocked | 31 |
| **Total** | **799** |

This is a candidate diagnostic. The receipt explicitly keeps `release_approved=false` and `training_eligible=false`; it is not human fluency certification and does not alter the frozen production translation review. The full R1 release remains blocked at 5/117 eligible forget cases, 3/39 retain cases, and 0/44 composition chains under the production gate.

## Consequence for the experiment

No 4B Llama run is available under Meta's published model lineup. A Llama 3.2 3B comparison could be added only as a separately labeled feasibility experiment; it would not replace the R1 8B main study. The next valid main-study run still requires (1) Hugging Face access to Llama 3.1 8B for the account behind the Kaggle secret, (2) a source-verified five-language training release, and (3) a successful bounded 8B Kaggle runtime smoke before research training.

## Sources and local evidence

- Meta, [Llama model and library downloads](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) (official model-family sizes).
- Kaggle, [Notebook documentation](https://www.kaggle.com/docs/notebooks) and [T4 x2 specification announcement](https://www.kaggle.com/product-announcements/735239) (accelerator configuration).
- Local 8B smoke record: [`kaggle_8b_runtime_smoke_2026-11-04.md`](kaggle_8b_runtime_smoke_2026-11-04.md).
- Local 799-row merge receipt: [`merge_verification.json`](../../../../experiments/synthworlds_r1/training_release_draft/translation_repair_diagnostics/chatgpt_fullflagged_queue_2026-11-04/remote_output/merged_full799/version_1/merge_verification.json).
