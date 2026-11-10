# MUSE-News target baseline

This is a locked, inference-only baseline for the public MUSE-News target model. It evaluates C1 VerbMem-forget and C2/C4 KnowMem forget/retain with the official MUSE evaluator. It does not train or modify a model.

## Current state

- Protocol: `protocol.json` (SHA-256 recorded after generation).
- Kaggle notebook is privately staged. Version 2 completed with both run flags off, GPU disabled, and Internet disabled; no evaluation results were produced.
- The official tokenizer is gated. The Kaggle notebook retrieves it from Meta's official `meta-llama/Llama-2-7b-hf` repository using the `HF_TOKEN` Kaggle Secret. Do not replace it with a public mirror.
- The model card for `muse-bench/MUSE-news_target` does not declare a license. This run is evaluation-only and will not redistribute weights or derivatives; resolve terms before any training or derivative release.
- MUSE does not expose article/source IDs for the QA rows. Report the official fixed-set metrics with row-level intervals only; do not describe those intervals as source-cluster robust.
- The notebook must first pass the access/data checks and two-row parity smoke. Full evaluation proceeds only after those checks pass.

## Run

Open the [private Kaggle notebook](https://www.kaggle.com/code/bunnoob2005/muse-news-target-baseline-2026-11-09). Under Add-ons → Secrets, attach `HF_TOKEN`, enable Internet with GPU off, set `RUN_ACCESS_PREFLIGHT = True`, and run all cells. If the exact official tokenizer revision and file hashes pass, select GPU T4 x2, set `RUN_ACCESS_PREFLIGHT = False` and `RUN_BASELINE = True`, then rerun. The notebook stops before all-row evaluation if tokenizer access, data/revision validation, GPU placement, or the official evaluator parity check fails.

## Reproduction

The Kaggle output includes the protocol copy and hash, data/model/tokenizer revisions, environment receipt, per-row predictions and metric values, deterministic bootstrap summaries, and output hashes. No model weights are included.
