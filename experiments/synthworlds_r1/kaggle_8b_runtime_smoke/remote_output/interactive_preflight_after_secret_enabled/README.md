# Kaggle interactive 8B access preflight

This records the run after `HF_TOKEN` was enabled in the notebook. The cell reached the gated config download and received HTTP 403 (`GatedRepoError`) for `meta-llama/Llama-3.1-8B-Instruct`. It did not download weights, load research data, or start training.

The failed run also exposed a dependency issue: it installed `huggingface_hub==0.36.2` over Kaggle’s newer stack. The local v6 preflight notebook removes that downgrade.

This is an access diagnostic, not a model runtime smoke or research result.
