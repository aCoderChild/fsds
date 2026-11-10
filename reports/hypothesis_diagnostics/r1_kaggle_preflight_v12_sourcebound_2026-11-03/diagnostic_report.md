# Kaggle source-bound preflight — version 12

This is a setup validation run, not a model experiment. Kaggle reported the kernel as `COMPLETE`; no training, unlearning, or model evaluation was started.

## Verified

- The kernel used one attached project dataset: `bunnoob2005/synthworlds-sm-sft-pilot-v3`. Project discovery found the expected corpus and scripts.
- CUDA was available on a Tesla T4.
- The corpus and package check returned `READY_TO_TRAIN` with `training_packages_ready=true` and no corpus-preflight errors.
- The evaluation dataset is present and AI-only review is complete; human translation review was not performed.
- The fresh-process PEFT LoRA smoke check passed.
- No exception or traceback appears in the kernel log.

## Gate and run state

The combined training gate was **BLOCKED** because `HF_TOKEN` was not available to this API-pushed Kaggle job. The notebook explicitly kept SFT, retain-only SFT, unlearning, and evaluation switches set to false. No model weights were downloaded and no model training or inference occurred.

Kaggle emitted dependency-resolver warnings because the notebook's `huggingface-hub` version conflicts with preinstalled optional `gradio` and `diffusers` requirements. The experiment's required training package check and LoRA smoke test passed, but this warning should be reviewed before treating the Kaggle environment as fully clean.

## Files

- `experiments.log` — raw Kaggle kernel output.
- `preflight_summary.json` — parsed state and SHA-256 values.
