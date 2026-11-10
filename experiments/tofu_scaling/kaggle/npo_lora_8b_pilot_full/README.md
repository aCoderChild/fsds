# TOFU Llama 3.1 8B NPO-LoRA pilot — full run package

This is a separate Kaggle package for the preregistered one-seed exploratory run. The smoke-only source package remains unchanged.

- Protocol: `experiments/tofu_scaling/npo_lora_8b_pilot_protocol.json`
- Locked seed: 42
- Locked full run: exactly 500 optimizer steps (10 epochs)
- Model/data revisions and objective are fixed in the protocol.
- The notebook repeats its bounded smoke checks before the full training, then evaluates and independently verifies the adapter.
- A failed research or utility gate remains a diagnostic result; do not tune this run after seeing outcomes.

The package is not a confirmatory study or a controlled 3B-versus-8B scaling comparison.
