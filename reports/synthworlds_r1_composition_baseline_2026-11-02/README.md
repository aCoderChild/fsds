# SynthWorlds-SM composition baseline — 2026-11-02

## Secret-access issue resolved

The earlier failure was specific to Kaggle API/CLI-launched kernel runs: those runs do not receive Notebook Editor Secrets. Kaggle's current CLI documentation says Secrets are configured on Kaggle.com (no CLI support) and `UserSecretsClient` runs inside the Kaggle notebook environment. In this experiment, the interactive Notebook Editor successfully retrieved the attached `HF_TOKEN` secret without printing it and verified access to `meta-llama/Llama-3.2-1B-Instruct`. See the [Kaggle CLI kernel documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md). Continue secret-dependent training from the Kaggle Notebook Editor; do not place the token in notebook source, a dataset, or these artifacts.

## Completed stages

- Five-language retained-chain composition SFT completed: 1,155 rows (765 forget-split atomic rows plus 390 retain-split rows, including retained-chain composition examples), 3 epochs, seed 42.
- Retain-only reference completed: 390 retain rows, 3 epochs, seed 42.
- Baseline generation and scoring completed for pretrained, composition-SFT, and retain-only checkpoints: 3,870 predictions across 1,290 prompts per checkpoint. Generation failures: 0; output-invalid heuristic flags: 0.
- No unlearning was run. The precompetent bilingual forget-case gate failed: per-language upper bounds are below the required 15 cases and below the required 8 entity groups. The empty `cross_language_contrasts.json` is expected and must not be interpreted as a null unlearning effect. After downloading the reports and adapters, the Kaggle draft switches were reset to `False` and the GPU session was stopped.

## Results and limitations

See `diagnostic_report.md` for per-language/per-split scores and interpretation. Composition-SFT multi-hop exact match on the forget split was EN 13.0%, FR 7.8%, DE 18.2%, HI 15.6%, TH 15.6%. This is an exploratory 1B feasibility pilot. Translation review was AI-only; no fluent human review was performed. Retain-split scores overlap the retain training data, so treat them as training-overlap controls. The pilot does not support a claim about cross-language unlearning effects.

## Preserved files

- `summary.json`, `predictions.jsonl`, `scored_predictions.jsonl`, `case_metrics.jsonl`: machine-readable metrics and row-level outputs.
- `composition_sft_run_manifest.json`, `composition_retain_only_run_manifest.json`: training provenance and configuration.
- `adapters/composition_sft/` and `adapters/composition_retain_only/`: LoRA adapter weights and configurations downloaded from the Kaggle output panel. Loading them requires the gated base model and applicable Hugging Face access.
- `cross_language_contrasts.json`: empty list because unlearning was gated off.
- `artifact_index.json`: SHA-256 hashes, sizes, and row counts for the preserved files.

The checked-in Kaggle notebook template keeps all expensive run switches disabled by default to avoid accidental duplicate training. The live Kaggle session was used for the completed run.
