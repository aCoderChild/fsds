# TOFU 3B NPO-LoRA profile recovery: development results

**Run:** Kaggle kernel v3 completed. Overall status: `NO_DEV_CANDIDATE_PASSES`. This is a valid preregistered stop, not a software failure. Alpha=10 artifacts were reused; alpha=20 completed 250 optimizer steps. All candidate comparisons were independently recomputed from the archived per-example outputs.

**Design:** Llama 3.2 3B, seed 42, two Tesla T4 GPUs. Each candidate was assessed against the pinned full-checkpoint baseline on forget (200 questions / 10 profile blocks), retain (400 / 20 blocks), and holdout (200 / 10 blocks). Intervals use 5,000 paired profile-block bootstrap draws. Values are changes from baseline in percentage points; the alpha=20 candidate must pass all five locked gates to continue to the disjoint final cohort.

| Retain-NLL weight | Forget truth-ratio score Δ (95% CI) | Forget answer probability Δ (95% CI) | Retain answer probability Δ (95% CI) | Retain ROUGE-L Δ (95% CI) | Holdout answer probability Δ (95% CI) | All gates |
|---:|---:|---:|---:|---:|---:|:---:|
| 5 | +7.57 pp [+6.23, +8.79] | -48.15 pp [-50.37, -45.48] | -43.45 pp [-45.04, -42.08] | -31.73 pp [-33.92, -29.37] | +3.36 pp [+2.23, +4.28] | Fail |
| 10 | +5.22 pp [+3.84, +6.53] | -32.45 pp [-35.04, -29.64] | -24.60 pp [-26.02, -23.30] | -25.57 pp [-28.37, -22.66] | +5.36 pp [+4.84, +5.82] | Fail |
| 20 | +0.80 pp [+0.26, +1.35] | -15.28 pp [-17.66, -13.12] | -3.14 pp [-3.52, -2.75] | -7.08 pp [-9.62, -4.65] | +0.57 pp [+0.43, +0.71] | Fail |

## Interpretation

Alpha=20 preserves retain-set answer probability within the preregistered 5 pp margin and shows forgetting signals on both forget metrics. However, its retain-set ROUGE-L change is **−7.08 pp** (95% CI **[−9.62, −4.65]**). The locked non-inferiority rule requires the lower confidence bound to exceed −5 pp; it does not. Thus alpha=20 fails one of five gates. Alpha=5 and alpha=10 fail both retain utility gates. No candidate qualifies, so the pipeline correctly did not train or evaluate the final cohort.

These are exploratory one-seed profile-cohort results, not a confirmatory result and not a standard full-split TOFU score. They suggest a tradeoff: increasing the retain weight substantially improves answer-probability retention, while generated-answer ROUGE-L still misses the utility threshold. The study does not establish a successful unlearning configuration.

## Descriptive follow-up (post hoc)

Across the 400 retain questions, alpha=20 generated a text string different from the full checkpoint on 160. ROUGE-L was lower on 101 questions, unchanged on 277, and higher on 22. Nineteen of 20 profile blocks had a negative mean ROUGE-L change. This describes where the failed gate comes from; it is not a new preregistered test and does not change the stopping decision.

## Verification

Independent verification status: `PASS_ARTIFACT_AND_RESULT_RECOMPUTATION`; zero verification problems. It checked training receipts and adapter hashes, 250 steps per trained candidate, source data/model revisions, exact cohort rows and profile-block assignments, no truncation/offload, recomputed all five bootstrap gates for alpha 5/10/20, and matched the run stopping decision. The 61-member output ZIP passed CRC. SHA-256: `68b72db9abc1c6120e74397f3e495a97b2e2c3b1c1205f212408948136e45c02`.

- Kaggle run: https://www.kaggle.com/code/bunnoob2005/tofu-3b-npo-profile-recovery-resume
- Full compact artifacts: `kaggle_output_v3/tofu_npo_profile_recovery_artifacts_resumed_v3.zip`
- Independent check: `alpha20_independent_verification.json`
- Machine-readable summary: `alpha20_stage_results.json`
