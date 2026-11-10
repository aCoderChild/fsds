# Corrected Llama 3.1 8B NPO-LoRA replication — preregistration

**Status:** locked locally before any GPU run on 2026-11-09.
**Protocol ID:** `tofu_8b_npo_lora_corrected_replication_v2`
**Protocol SHA-256:** `142dcc0f2cdc93e6bf5d8742dd619dc71261dfdc2eb9399caa57821b5ef03aab`
**Study class:** one-seed exploratory implementation-correction diagnostic; not confirmatory.

## Question

What diagnostic results are obtained when the same fixed TOFU 8B NPO-LoRA setup uses the corrected update-wide pooled-token retain NLL? The preceding v1 run used a different retain aggregation scope and is preserved as a protocol-deviation diagnostic.

## Fixed run

- Checkpoint: `open-unlearning/tofu_Llama-3.1-8B-Instruct_full` at `1a5c5b1a557f8c99bdadecd5168ebd03f640b00e`.
- Benchmark: `locuslab/TOFU` revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`; pinned file hashes and all original evaluation rows remain unchanged.
- Method: upstream-pinned NPO plus retain NLL, LoRA only; retain weight `alpha=1`, `beta=0.1`; all original v1 training settings and 500 optimizer steps unchanged.
- Seed: `43`. This is a new stochastic run.
- The retain NLL is pooled across all retain answer tokens in each effective eight-example optimizer update.
- Metrics, paired 20-profile bootstrap (5,000 draws, seed 42), primary direction, and all five original utility/primary gates are unchanged.
- Hardware: exactly two Kaggle Tesla T4 GPUs; no CPU/disk offload and no input truncation. A bounded real-data smoke and upstream loss/gradient check must pass before full training.

## Limits and stopping rule

The same TOFU rows were inspected in earlier runs, and both implementation and seed differ from v1. The comparison cannot isolate the aggregation fix and is not an independent data replication. Run one candidate only; do not tune or repeat based on the outcome. Report a failed gate as a diagnostic result. Any later study must use a new benchmark/cohort and a separately frozen protocol.

## Source pins

Embedded source SHA-256 values are recorded in `protocol.json` and `source_manifest.json`.
