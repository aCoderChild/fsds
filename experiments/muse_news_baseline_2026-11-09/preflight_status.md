# MUSE-News baseline status

**No fresh MUSE-News evaluation results yet.** Kaggle version 2 completed safe staging only; its receipt confirms that the access check and baseline were skipped. Version 3 then attempted the no-GPU access preflight but failed at `UserSecretsClient.get_secret("HF_TOKEN")` with HTTP 400, before contacting Hugging Face. No tokenizer files, model weights, or evaluation rows were downloaded or scored, so there is no evaluation archive or metric receipt. The response does not distinguish an unattached/disabled secret from a Kaggle secret-service error.

The [original MUSE paper's published News table](published_reference_results.md) gives useful prior results, but these are literature references rather than a fresh run or verified reproduction of the currently pinned checkpoint.

## Latest attempt

| Item | Result |
|---|---|
| Kaggle version | 3; `KernelWorkerStatus.ERROR` |
| GPU / Internet | Off / on |
| Secret lookup | HTTP 400 at `get_secret("HF_TOKEN")`; HF access check not reached |
| Tokenizer / weights / evaluation data | None downloaded or loaded |
| New project metrics | None |
| Archived log | `kaggle/access_preflight_v3.log`; SHA-256 `e4d1873537463d8276bf359a6169b0132b00a77c24a6379743b8055dc0b6ed76` |

The MUSE kernel has its own secret attachment setting. Verify that `HF_TOKEN` is enabled for this exact kernel, then rerun only the access preflight. Do not start the GPU baseline until its receipt passes.

## Checks completed

| Check | Status |
|---|---|
| Dataset, target checkpoint, and evaluator pinned revisions | Pass |
| Expected fixed evaluation and ICL row counts | Pass |
| Unique questions within each QA/ICL split | Pass: 100/100/10/10 |
| Normalized question overlap across QA/ICL splits | Pass: 0 across all checked pairs |
| Notebook JSON, Python cell syntax, embedded protocol hash | Pass |
| Kaggle safe-staging execution | Pass, version 2; no GPU or Internet enabled |
| Access-preflight fixture | Pass: verifies all four hashes and does not print or save the secret |

## Remaining step and limits

- Kaggle CLI can upload, pull, and check status. The Kaggle UI is unavailable in this session, so I cannot attach the account secret to this notebook.
- The official Meta `Llama-2-7b-hf` tokenizer is gated. The run must use an `HF_TOKEN` that has access to that exact repository; acceptance for a Llama 3.1 repository alone does not prove this access.
- The audit found no source/article IDs for the QA rows. Results can report the official fixed-set, row-level metrics and intervals, but not source-cluster-robust uncertainty.
- The target checkpoint's model card does not specify its license or training provenance. This package is inference-only and excludes weights; do not train on it or redistribute derivatives until terms are clarified.

Kaggle notebook: [MUSE-News Target Baseline](https://www.kaggle.com/code/bunnoob2005/muse-news-target-baseline-2026-11-09). First attach `HF_TOKEN` under **Add-ons → Secrets**, enable Internet with GPU off, set `RUN_ACCESS_PREFLIGHT = True`, and run the cells. If the exact official tokenizer access check passes, enable GPU T4 x2, leave Internet on, set `RUN_ACCESS_PREFLIGHT = False` and `RUN_BASELINE = True`, and run again. The baseline stops before all-row evaluation if tokenizer access, input validation, placement, or official parity checks fail.

Remote staging: version 2, `KernelWorkerStatus.COMPLETE`; the safe-run log and metadata are archived in `kaggle_stage_verification.json`.

Protocol SHA-256: `17739214264ce8c7c3bcf5063d64f7d66c03fd29f84e994800db79d02da8b5aa`.


## Official release cross-check (2026-11-09)

The MUSE paper identifies Llama 2 7B as the News base and says KnowMem answers are validated against sampled source excerpts. The pinned official data loader stores only `question`/`answer` for KnowMem and `prompt`/`gt` for Verbatim; it does not preserve a QA-to-excerpt crosswalk. The pinned public target model repository contains no tokenizer files, so the official gated Meta Llama 2 tokenizer remains necessary for exact reproduction. The readiness audit records the 100-row lexical candidate counts; these do not establish article-level links. No baseline or training run has been launched. See `../tofu_scaling/diagnostics/muse_news_readiness_2026-11-09/official_release_review.md`.


## Access retry check (2026-11-09T07:50:58.658729+00:00)

The Kaggle kernel remains `ERROR_ACCESS_PREFLIGHT`. This machine has no local Hugging Face token, and the connected browser surface is unavailable, so I could not change the notebook's secret attachment. The HTTP 400 still occurs before Hugging Face is contacted; no model/tokenizer files or evaluation data were fetched. The remaining step is to enable `HF_TOKEN` under **Add-ons → Secrets** in the exact [MUSE-News Target Baseline notebook](https://www.kaggle.com/code/bunnoob2005/muse-news-target-baseline-2026-11-09), then rerun only the access preflight with GPU off.

## UI recheck (2026-11-09 08:48 UTC)

The Kaggle notebook editor is now reachable. The existing `HF_TOKEN` secret is listed, but its checkbox is unchecked for this exact notebook. The last kernel status remains `ERROR`; no new preflight or evaluation has started. Enabling the checkbox is pending action-time confirmation because it grants notebook code access to the Hugging Face credential. The notebook is configured for access preflight only; GPU session is off and baseline execution is disabled.

## Latest interactive preflight (2026-11-09 13:00 UTC)

The authorized no-GPU tokenizer-access cell was run in the MUSE-News Kaggle notebook with `HF_TOKEN` enabled, Internet on, accelerator set to None, `RUN_ACCESS_PREFLIGHT=True`, and `RUN_BASELINE=False`. Kaggle successfully retrieved the secret, then Hugging Face returned `GatedRepoError` HTTP 403 on the first pinned tokenizer file (`special_tokens_map.json`) from `meta-llama/Llama-2-7b-hf` at revision `01c7f73d771dfac7d292323805ebc428287df4f9`. No tokenizer file completed downloading; no model weights, evaluation data, or metrics were produced. The Kaggle session was stopped after the failure.

This supersedes the prior note that the secret was still disabled and the HTTP 400 occurred before Hugging Face. The current blocker is access to this exact gated Llama 2 repository for the Hugging Face account behind `HF_TOKEN`. Access approval for a different model repository does not establish access here. Request/accept access to the exact Llama 2 repository using that account, then rerun only the no-GPU preflight. Do not substitute a tokenizer or start the baseline until the hash-checked receipt passes.

Attempt record: [`kaggle/access_preflight_ui_2026-11-09.json`](kaggle/access_preflight_ui_2026-11-09.json).
