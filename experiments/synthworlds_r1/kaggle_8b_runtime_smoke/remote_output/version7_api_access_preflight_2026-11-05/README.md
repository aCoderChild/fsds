# Kaggle v7 access preflight (2026-11-05)

Version 7 was published from the hash-verified v6 preflight-only artifact. Kaggle reports the kernel as `ERROR`. Its output receipt is preserved here.

- Run scope: CPU-only gated-config preflight.
- Outcome: `FAIL_HF_ACCESS_PREFLIGHT` at `retrieve_kaggle_secret`, `ConnectionError`.
- Kaggle log identifies HTTP 400 from the Kaggle secret service. The preflight did not reach the Hugging Face request.
- The authenticated Hugging Face model page separately displayed access granted for `meta-llama/Llama-3.1-8B-Instruct`.
- In the current notebook editor, the Secrets panel showed no attached secret row. The blank new-secret form was opened only to inspect available controls and then canceled; no credential was entered, changed, or saved.
- GPU was disabled. No weights, research dataset, training, or hypothesis experiment ran.

Next safe step: the account owner should attach the existing `HF_TOKEN` directly in Kaggle Secrets (do not paste it into chat), then execute the cell interactively with `RUN_ACCESS_PREFLIGHT = True` and `RUN_SMOKE = False`. Proceed to a GPU smoke only after a `PASS_HF_ACCESS_PREFLIGHT` receipt.
