# 8B NPO-LoRA evaluation recovery

This private Kaggle notebook is a contingency for the currently running v2 pilot. It never trains or resumes training. It validates a completed v2 output archive, then either verifies evaluation files already in that archive or performs only the missing adapter evaluation.

## When to use it

1. Do not start it while the original v2 kernel is still running.
2. Wait for the existing tracker to download the terminal output ZIP. Review its status and local post-run verification first.
3. If the ZIP contains a passing `npo_verification.json`, use those results directly; recovery is unnecessary.
4. If it proves `TRAINING_COMPLETE` with 500 steps and a hash-matching adapter but lacks complete adapter evaluation, make the ZIP available as a private Kaggle dataset. Attach that dataset to this notebook.
5. Enable Internet. Choose GPU T4 x2 only when the adapter evaluation is missing; a complete evaluation can be verified without GPU.
6. Run all cells. The notebook verifies ZIP CRC and safe paths, protocol/source/baseline hashes, the launch time, training receipt, data counts, 4,000-row loss trace, and adapter hashes before any inference. Evaluation output is streamed to the notebook, followed by the independent verifier.

If training did not complete or the adapter hashes fail, the notebook stops without evaluating. It does not relax the preregistered protocol or diagnostic gates. The recovery output is `tofu_8b_npo_eval_recovery_artifacts.zip`.

## Local verification

Run:

```bash
python experiments/tofu_scaling/build_npo_lora_8b_eval_recovery_notebook.py
python experiments/tofu_scaling/kaggle/npo_lora_8b_eval_recovery_v1/verify_eval_recovery_package.py
```

These local checks validate the notebook/package only; they do not consume Kaggle GPU quota or claim model results.
