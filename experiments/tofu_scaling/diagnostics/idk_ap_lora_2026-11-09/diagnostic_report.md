# TOFU 3B IDK+AP LoRA diagnostic

## Status

Kaggle kernel v2 completed. The run passed its data audit, pinned objective-equivalence check, bounded real-data smoke, both training receipts, complete evaluation coverage, and independent artifact/result verification. The experiment itself is **FAIL_DIAGNOSTIC_GATES**: all five development gates passed, while four of five final-cohort gates passed.

Kernel v1 stopped at a byte-level input hash check before loading weights or training. Its failure is retained in `kaggle_run1/`; the successful v2 output is in `kaggle_run2/`.

## Locked setup and integrity

- Protocol: `tofu_3b_idk_ap_lora_profile_diagnostic_v1`; SHA-256 `650af06eed1d254cfdd2b47714299686925cea47b5b2675edb7c07674b01c049`.
- Model: Llama 3.2 3B full checkpoint, pinned revision `24f31ca19f6966dcb6f6b29abc511cce71222d4a`; TOFU dataset revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`.
- Method: fixed IDK+AP objective, LoRA, one seed (42), five epochs and 125 optimizer steps per cohort. The loss and gradient matched the pinned upstream objective exactly in the fixture check. This is resource-adapted LoRA, not a reproduction of the paper's full-parameter run.
- Dev and final evaluations each cover 200 forget, 400 retain and 200 holdout questions. Confidence intervals use 5,000 paired bootstrap draws over inferred 20-question profile blocks.
- The output archive is 186,305,204 bytes, SHA-256 `0eb6f7ee9f34fef1aa5497ba068d3f691ed92ada9ea04105174c02654995bdb3`; ZIP CRC passed. The independent verifier returned `PASS_ARTIFACT_AND_RESULT_RECOMPUTATION`, verified both cohorts and found no artifact or recomputation problems. A fresh local rerun on 2026-11-09 also returned `PASS_ARTIFACT_AND_RESULT_RECOMPUTATION`; see `kaggle_run2/LOCAL_VERIFICATION_RECHECK.json`.
- Kaggle logs and cohort receipts identify two Tesla T4 GPUs and CUDA 0/1 placement without CPU/disk offload. The original run-level GPU-name field is empty, but both immutable cohort receipts specify T4 x2; a fresh verifier reconciled these sources. The runner and verifier now require the cohort receipts to agree and aggregate GPU names into the run-level receipt on future runs.

## Results

Differences are candidate minus full checkpoint, in percentage points. A positive Truth Ratio score change and a negative forget-answer-probability change are the intended forgetting directions. Utility gates require the lower 95% interval bound to be strictly above -5 percentage points.

| Cohort | Measure | n / profile blocks | Difference (pp) | 95% profile-bootstrap CI (pp) | Gate |
|---|---|---:|---:|---:|---|
| Dev | Forget Truth Ratio score | 200 / 10 | +2.262 | [+1.975, +2.538] | Pass |
| Dev | Forget answer probability | 200 / 10 | -1.993 | [-2.298, -1.752] | Pass |
| Dev | Retain answer probability | 400 / 20 | -1.768 | [-1.936, -1.596] | Pass |
| Dev | Retain ROUGE-L recall | 400 / 20 | -0.870 | [-1.898, +0.105] | Pass |
| Dev | Holdout answer probability | 200 / 10 | +2.635 | [+2.515, +2.767] | Pass |
| Final | Forget Truth Ratio score | 200 / 10 | +2.027 | [+1.722, +2.316] | Pass |
| Final | Forget answer probability | 200 / 10 | -1.782 | [-2.020, -1.521] | Pass |
| Final | Retain answer probability | 400 / 20 | -1.740 | [-1.878, -1.605] | Pass |
| Final | Retain ROUGE-L recall | 400 / 20 | -3.058 | [-5.110, -0.993] | **Fail** |
| Final | Holdout answer probability | 200 / 10 | +2.207 | [+1.997, +2.417] | Pass |

The final retain ROUGE-L point estimate is a 3.06-point drop. Its lower confidence bound, -5.110 points, misses the locked -5-point margin by 0.110 point. Do not widen the margin or tune against this final cohort after observing the result.

### Post-hoc profile consistency checks

A no-GPU audit paired candidate and full-checkpoint rows by source ID and recomputed all five final point estimates from the archived outputs. The two forgetting metrics moved in the intended direction in all 10 forget profiles; retain answer probability declined in all 20 retain profiles; retain ROUGE-L declined in 16/20. The unadjusted two-sided profile sign-test p-values were 0.00195 for each forget metric, 1.91e-6 for retain answer probability, and 0.0118 for retain ROUGE-L. Omitting any one profile left the ROUGE-L point estimate between -3.52 and -2.66 points. These post-hoc checks are descriptive, are not multiplicity-adjusted, and do not change the preregistered interval or failed gate. Reproduce them with `audit_profile_consistency.py`; output is `profile_consistency_posthoc.json`.

A simple string matcher found an IDK phrase in 0/200 final forget generations. This is not a semantic refusal classifier, but the run provides no positive marker-based evidence that the model learned to explicitly answer “I don't know.”

## Interpretation and limits

The run has a small, reproducible forgetting signal on the locked metrics, but it does not pass the full utility protocol. It is a one-seed, 3B LoRA diagnostic. The final profile blocks were held out from this run's development stage, but those same blocks had appeared in earlier NPO pilots. Therefore the project-level result is exploratory, not a clean confirmatory test. TOFU uses fictional author profiles and cannot establish deletion of real personal data or real-world privacy protection.

## Recommended next steps

1. Freeze this result and do not tune the method, margin, or reporting against the observed final cohort.
2. Do not spend more GPU time on the current standard TOFU split for confirmation: this protocol assigns all 20 forget blocks and all 20 holdout blocks, and assigns all 180 retain blocks to optimizer training or utility evaluation except two conflicting-question blocks that are excluded. The final blocks also appeared in an earlier NPO pilot.
3. Choose a genuinely new benchmark/cohort with non-overlapping questions and entities; audit overlap against prior project artifacts before training. Define the claim, primary metric, utility margins, seed count and stopping rule in a new hashed protocol.
4. If no untouched data is available, report this as an exploratory failed-gate diagnostic. A development-only method study may investigate retention, but it must not be presented as confirmation or reuse the final cohort for selection.
5. Preserve the immutable archive, corrected receipt aggregation code, and current verifier output together as the provenance record. No new run should reuse the inspected TOFU profiles.

## Artifacts

- Kaggle notebook: `https://www.kaggle.com/code/bunnoob2005/tofu-3b-idk-ap-lora-diagnostic-2026-11-09`
- Result archive: private Kaggle output; local copy at kaggle_run2/tofu_3b_idk_ap_lora_diagnostic.zip (186 MB, kept out of Git). Its hash is recorded above and in the committed manifest.
- Independent verification: `kaggle_run2/LOCAL_VERIFICATION.json`
- Current independent verification: `kaggle_run2/LOCAL_VERIFICATION_RECHECK_CURRENT.json`
- Post-hoc profile consistency audit: `audit_profile_consistency.py` and `profile_consistency_posthoc.json`
- Run manifest: `attempt_manifest.json`
- Kaggle log: `kaggle_run2/tofu-3b-idk-ap-lora-diagnostic-2026-11-09.log`
- Vietnamese mentor report: `docs/report/R3.tex`
