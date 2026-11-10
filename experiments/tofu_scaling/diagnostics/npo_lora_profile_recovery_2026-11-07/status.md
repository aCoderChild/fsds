# TOFU 3B NPO-LoRA profile recovery

**Latest state:** Kaggle continuation version 3 completed. Run outcome: `NO_DEV_CANDIDATE_PASSES`. This is the preregistered stopping result, not a software error. Alpha=20 completed 250 optimizer steps and all 800 development examples, then failed the retain-set ROUGE-L non-inferiority gate. The pipeline correctly did not start final-cohort training.

## Alpha=20 result

- Forget truth-ratio score: **+0.80 pp** (95% CI [+0.26, +1.35]); gate passes.
- Forget answer probability: **−15.28 pp** (95% CI [−17.66, −13.12]); gate passes.
- Retain answer probability: **−3.14 pp** (95% CI [−3.52, −2.75]); within the −5 pp margin.
- Retain ROUGE-L: **−7.08 pp** (95% CI [−9.62, −4.65]); fails the −5 pp non-inferiority rule.
- Holdout answer probability: **+0.57 pp** (95% CI [+0.43, +0.71]); gate passes.

All candidates (alpha 5, 10, 20) fail at least one locked gate. Independent verification passed: training receipts and adapter hashes, source/model revision, cohort rows and profile blocks, no truncation/offload, all 15 candidate gate estimates and intervals, the stop decision, and the 61-member ZIP CRC.

## Files and run

- Results table and short analysis: `alpha20_stage_results.md`
- Machine-readable results: `alpha20_stage_results.json`
- Independent verification: `alpha20_independent_verification.json`
- Full compact outputs: `kaggle_output_v3/tofu_npo_profile_recovery_artifacts_resumed_v3.zip`
- Output ZIP SHA-256: `68b72db9abc1c6120e74397f3e495a97b2e2c3b1c1205f212408948136e45c02`
- Kaggle run: https://www.kaggle.com/code/bunnoob2005/tofu-3b-npo-profile-recovery-resume

The study remains exploratory, one-seed, and profile-cohort based; it is not a standard full-split TOFU score or a confirmatory result. Do not proceed to the final cohort under this protocol. A further training run would require a newly specified and preregistered design.

## Attempt history

- **Continuation v2:** alpha=10 training and evaluation finished, then result finalization crashed with `NameError: reference_hashes`. Alpha=10 artifacts were retained and independently reverified.
- **Continuation v3:** fixed the scope error and artifact reuse, reused alpha=10, trained/evaluated alpha=20 on two Tesla T4 GPUs, recomputed the preregistered gates, and stopped without final-cohort training when no candidate passed.
