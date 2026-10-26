# Cross-Lingual Deep Unlearning — MQuAKE-T Pilot

Reproducible Kaggle notebooks for investigating whether English-only machine unlearning transfers to German direct retrieval and two-hop reconstruction.

> **Research pilot; no GPU results yet.** This project adapts the real-world MQuAKE-T temporal knowledge-editing benchmark to an unlearning study. It is **not** an official MQuAKE unlearning benchmark. Benchmark facts and answers come from the original dataset; only question translations are synthetic.

## Research question

When the model forgets direct English access to a target fact, does it still recover the downstream target answer via German two-hop reasoning?

Note: recovering a downstream answer does **not** necessarily establish that the forgotten first-hop fact remains memorized. Full-SFT competence, baseline multilingual reasoning accuracy, retain utility, and retrained retain-only reference need to be checked before claiming a gap.

## Experiments

1. [Source dataset, German translation & QLoRA SFT](notebooks/01_mquake_t_source_translate_sft.ipynb)
2. [English-only unlearning: GA, GD, NPO](notebooks/02_mquake_t_unlearning_baselines.ipynb)
3. [Matched English/German direct & two-hop evaluation](notebooks/03_mquake_t_crosslingual_evaluation.ipynb)

Dataset: [Princeton NLP MQuAKE](https://github.com/princeton-nlp/MQuAKE), selecting original MQuAKE-T two-hop records with per-record provenance and source checksums.

## Kaggle

Use GitHub Actions → **Kaggle notebooks** (manual workflow). It supports uploading all three **private** notebooks without running them, and optionally launching one stage at a time if explicitly requested. Runs consume your personal Kaggle GPU allocation, not GitHub-hosted GPU.

Set these repository **Settings → Secrets and variables → Actions** secrets:

- `KAGGLE_USERNAME`: your Kaggle account handle.
- **Recommended** `KAGGLE_API_TOKEN`: the value from **Generate New Token** at [Kaggle API settings](https://www.kaggle.com/settings/api).
- **Legacy alternative only:** `KAGGLE_KEY`, containing the `key` from a **legacy** `kaggle.json` file, paired with `KAGGLE_USERNAME`.

Modern API tokens (typically prefixed `KGAT_`) **must not** go in `KAGGLE_KEY`. You do not need to create both token types. The workflow checks credential presence and calls `kaggle quota` to verify authentication before publishing. Do not share tokens or commit credentials.

For the first workflow run, choose **stage = all**, **execute = false**. This creates three private notebook versions without running their cells.

For a real run:
- Run stage 01; wait until the saved version has completed successfully.
- Configure stage 02 to include stage 01 as a Kaggle kernel input; run stage 02.
- Configure stage 03 to include both prior kernel outputs; run stage 03.

Generated LoRA checkpoints and data are Kaggle output artifacts, not tracked in Git. Complete artifact retrieval and inter-stage handoff should be verified on Kaggle before trusting the pipeline end-to-end.


## Weights & Biases (CoreWeave Forge)

All three notebooks use online W&B experiment tracking in your existing
[CoreWeave Forge project](https://forge.coreweave.com/wandb/crosslingual/multilingual_deepunlearning):

- entity: `crosslingual`
- project: `multilingual_deepunlearning`
- group: `mquake-t-20261026`
- server: `https://api.forge.coreweave.com`

**Important:** GitHub Actions secrets do not become Kaggle notebook secrets.
In Kaggle, create a notebook secret named `WANDB_API_KEY` with your
CoreWeave Forge / W&B key, and **enable access in each of the three already
existing notebooks**. Do not paste the key into source code, output cells,
GitHub Issues, commit messages or chat. Without this secret, the notebook
stops before GPU-intensive training.

Metrics logged:
- 01: dataset provenance, translation audit, SFT cross-entropy curves and learning rate, sanity accuracy by split/language/hop, model memory.
- 02: GA/GD/NPO forget/retain/total loss curves, parameters, optimizer steps, memory.
- 03: per-case prediction tables, recovery by checkpoint/language/hop, language-by-depth interaction, cross-language candidate counts and comparison bar chart.

Individual notebook source updates preserve their existing Kaggle notebook ID.
GitHub automatically syncs committed notebook updates to the **same private Kaggle
notebook IDs** with the no-run flag. This only creates a new saved version;
it does not execute GPU cells. GPU runs must be explicitly requested.

## Research safeguards

- No generated factual relationships, answers, or fallback benchmark examples.
- The exact first-hop and multi-hop question/answer pairs come from the original dataset.
- German translations must be inspected for meaning and named-entity stability.
- Two-hop *forget* questions are excluded from SFT supervision.
- Never select checkpoints by looking at German test leakage.
- Hold-out methods, multiple seeds, temporal ambiguity checks and matched paired confidence intervals are still needed.

## Status

Source notebooks passed local notebook/AST and synthetic-fixture tests in ChatGPT's environment; **full benchmark acquisition and Kaggle GPU training have not been executed**.

See [Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
