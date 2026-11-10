# TOFU Llama 3.1 8B NPO-LoRA pilot

This is a separate exploratory one-seed (42) 8B diagnostic following the prior 3B NPO-LoRA pilot. It locks the same core hyperparameters and the official TOFU evaluation; it does not isolate model size as a causal factor.

## Run order

1. Attach the private Kaggle dataset `bunnoob2005/tofu-8b-npo-baseline-inputs-2026-11-08`.
2. Select GPU T4 x2 and internet. Run version 1 with `RUN_FULL_TRAINING = False`. It performs integrity checks, objective equivalence, verifies the pre-existing 8B baseline outputs, then runs one bounded real-data optimizer step.
3. Inspect `training_receipt.json` in the ZIP. Continue only if status is `SMOKE_PASS`, GPU placement shows two Tesla T4 devices, no CPU/disk offload, exactly one finite optimizer step, and a nonzero adapter update.
4. Make a new notebook version with `RUN_FULL_TRAINING = True`. It repeats smoke, trains exactly 500 steps, runs the pinned evaluator and independent verifier, and archives all receipts.
5. Preserve the diagnostic result even if the research or utility gates fail. Do not tune this run after looking at its results.

No training was launched while the separate 3B profile-recovery Kaggle kernel was active. The 8B package is prepared locally and its objective, data, baseline, and sources are being checked.
