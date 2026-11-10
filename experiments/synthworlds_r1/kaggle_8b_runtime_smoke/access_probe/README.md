# Llama 3.1 8B Kaggle access probe

This no-GPU Kaggle notebook checks that the notebook runtime can read the
`HF_TOKEN` Kaggle secret and fetch the gated model's `config.json` at the
expected immutable revision. It never prints the token, downloads model
weights, reads research data, or trains.

Push it with `kaggle kernels push -p experiments/synthworlds_r1/kaggle_8b_runtime_smoke/access_probe`.
After the run completes, download outputs and verify the JSON receipt before
launching the GPU runtime smoke.
