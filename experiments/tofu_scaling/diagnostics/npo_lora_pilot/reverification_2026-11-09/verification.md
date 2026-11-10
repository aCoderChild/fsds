# Independent re-verification: Llama 3.2 3B TOFU NPO-LoRA

- Artifact and metric verification: **PASS**.
- Prespecified research diagnostic gates: **FAIL_DIAGNOSTIC_GATES**.
- Training: 500 optimizer steps; 1600.87 seconds; seed 42.
- Primary forget-set result: NPO minus full Truth Ratio = 0.080276, 95% profile-block bootstrap CI [0.068281, 0.092117] across 400 questions / 20 profile blocks.

## Utility gates

- `retain90.answer_probability`: delta -0.609747; 95% CI [-0.631848, -0.587693]; margin -0.05; pass=False.
- `retain90.rougeL_recall`: delta -0.349738; 95% CI [-0.370779, -0.328529]; margin -0.05; pass=False.
- `holdout10.answer_probability`: delta -0.037834; 95% CI [-0.053318, -0.023070]; margin -0.05; pass=False.
- `model_utility.harmonic_mean`: delta -0.141414; margin -0.05; pass=False.

## Verification discrepancy

The original Kaggle v5 verifier result is preserved as `npo_verification_kaggle_v5.json` and reports FAIL because it hardcodes 4,000 loss-trace rows. The hashed trainer source uses microbatch size 2 and gradient accumulation 4; 400 forget rows × 10 epochs therefore produce 2,000 microbatch trace records and 500 optimizer steps. The current source-aware verifier independently derives this schedule and passes with no problems.

This resolves the artifact-verifier mismatch; it does not change the scientific result. The 3B pilot fails all four prespecified utility gates and remains exploratory, not confirmatory.

See `provenance.json` for SHA-256 hashes and `verification.json` for full recomputed checks.
