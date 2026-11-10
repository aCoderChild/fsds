# Llama 3.1 8B NPO-LoRA pilot: diagnostic report

**Status:** Kaggle run and archived metrics verified; research utility gates failed; source audit found a retain-loss weighting deviation from the preregistered objective. Treat this as an implementation diagnostic, not as protocol-faithful or confirmatory evidence.

## Run and integrity

| Item | Result |
|---|---|
| Model | Llama 3.1 8B full checkpoint; matching TOFU retain90 oracle |
| Method | NPO plus retain NLL, LoRA adapter only |
| Seed / optimizer steps | 42 / 500 |
| Hardware | 2 Tesla T4 GPUs; FP16 frozen base, FP32 LoRA parameters and loss; no CPU/disk offload |
| Training time | 3,800.94 s (about 63.3 minutes) |
| Evaluation | 400 forget, 400 retain, 400 holdout, 100 real-author and 117 world-fact questions |
| Bootstrap | 5,000 paired profile-block draws; 20 profile blocks; seed 42 |
| Kaggle status | COMPLETE |
| Output archive | 160,833,500 bytes; ZIP CRC PASS; SHA-256 `91fad5f8df28b6e8f26dbaab93cc24797c3b36bbfcf2620e5cd5efdec9e61c9c` |
| Locked protocol | SHA-256 `17de753d3c92539213eb74962ef124677cf0ed8f4f92e9f390c8e62197ddcb1b` |
| Receipt / adapter hashes | Match archived files; 500-step receipt says `TRAINING_COMPLETE` |
| Metric verification | Archived verifier PASS; independent local metric recomputation PASS |
| NPO objective check | Kaggle and local checks PASS; same upstream/candidate/protocol hashes, loss difference 0, maximum gradient differences below `1e-6` |

## Results

All deltas are NPO adapter minus the same full checkpoint, in percentage points. Intervals are 95% profile-block bootstrap intervals unless marked point-only.

| Outcome | n / profile groups | Delta (pp) | 95% interval | Locked gate |
|---|---:|---:|---:|---|
| Forget10 Truth Ratio score (primary) | 400 / 20 | +4.05 | [+2.67, +5.52] | Positive exploratory signal |
| Forget10 answer probability | 400 / 20 | -72.15 | [-74.15, -70.15] | Secondary |
| Forget10 ROUGE-L recall | 400 / 20 | -40.89 | [-43.45, -38.19] | Secondary |
| Retain90 answer probability | 400 / 20 | -60.50 | [-62.80, -58.21] | FAIL; lower bound must exceed -5 |
| Retain90 ROUGE-L recall | 400 / 20 | -34.00 | [-36.53, -31.62] | FAIL; lower bound must exceed -5 |
| Holdout10 answer probability | 400 / 20 | -2.50 | [-3.83, -1.22] | PASS; lower bound exceeds -5 |
| Model Utility harmonic mean | 9 components | -10.21 | Point-only | FAIL; delta must exceed -5 |

On forget10, mean Truth Ratio score was 0.4734 for the full model, 0.5139 after NPO-LoRA, and 0.6407 for the retain90 oracle. The adapter moved the primary score in the preregistered direction, but remained 12.68 pp below the oracle (95% interval [-14.77, -10.45]). Its mean answer probability was 0.2697, versus 0.9912 for full and 0.1045 for the oracle.

The forget signal came with severe utility loss: answer probability fell by 60.50 pp and ROUGE-L by 34.00 pp on retain90. The overall utility harmonic mean fell from 0.6370 to 0.5350. Therefore this run fails the research utility gates even before considering the implementation deviation below.

## Implementation audit and correction

The archived trainer uses one forget and one retain example per microbatch with eight gradient-accumulation steps. It computes a token-mean retain loss separately for each microbatch, then gives each microbatch equal weight. The preregistered objective and the upstream equivalence check define retain loss as a token-weighted mean across the optimizer update. With variable answer lengths, these objectives have different gradients. The receipt also contains a stale batching description (`2 + 2` examples per microbatch), although its separate `training_schedule` correctly records `1 + 1`, accumulation 8.

This is a training-objective alignment issue; the archived metric outputs themselves reproduce. It means the measured outcome describes the implementation that ran, not a verified implementation of the preregistered token-weighted objective. No post-hoc adjustment was made to the result or its gates.

The current training code now normalizes each retain microbatch's summed token loss by the total scored retain-token count in that optimizer update, and writes model-specific batching metadata. An unequal-length CPU fixture (2 and 5 tokens) produced zero loss error and zero gradient error against a pooled token mean. The upstream objective check on the updated source also passes at tolerance `1e-6`. This only validates the corrected code path; it does not retroactively correct this completed run.

## Training-trace diagnostic

An offline audit of the archived trace found 4,000 finite microbatch records: exactly eight per each of 500 optimizer steps. Mean NPO forget loss fell from 13.8629 at step 1 to 0.2521 at step 500; mean retain token NLL rose from 0.0034 to 1.0135. Across the last 25 steps, mean retain token NLL was 1.0002. This pattern is consistent with a strong forgetting/utility tradeoff in the executed run. The trace has no intermediate evaluation scores, so it cannot establish an earlier checkpoint that would have passed utility gates or identify the sole cause of the utility loss. The calculation is reproducible from `audit_training_dynamics.py` and `training_dynamics_audit.json`.

## Conclusion and next step

This one-seed 8B pilot has a positive primary forget-metric change, but it does not satisfy utility requirements and has a retain-loss weighting mismatch. Do not use it to claim successful NPO unlearning, compare model sizes, or confirm the research hypothesis. Preserve the artifact as an implementation diagnostic. The corrected trainer passes CPU objective and unequal-length batching fixtures, but it has not been run on a research cohort. Do not rerun against the already inspected TOFU evaluation cohort as if it were confirmatory. The next stage requires a new locked protocol, a fresh benchmark/cohort with a clean development/final separation, and the corrected-code fixtures before any GPU launch. The MUSE-News readiness audit is the current candidate-screening artifact; it still flags model-term review and QA/source alignment, so no MUSE training has started.

## Reproduction artifacts

- Kaggle notebook: <https://www.kaggle.com/code/bunnoob2005/tofu-8b-npo-lora-pilot-2026-11-08>
- TOFU dataset: <https://huggingface.co/datasets/locuslab/TOFU>
- Archived run ZIP: `kaggle_v2/tofu_8b_npo_lora_pilot_artifacts.zip`
- Post-run verification: `kaggle_v2/postrun_verification.json`
- Cross-runtime objective comparison: `kaggle_v2/objective_cross_runtime_comparison.json`
- Corrected objective fixture: `objective_equivalence_postpatch_2026-11-09.json`
- Retain batching fixture: `experiments/tofu_scaling/verify_npo_retain_batching.py`
- Training trace audit: `training_dynamics_audit.json` and `audit_training_dynamics.py`
- Next benchmark readiness: `../muse_news_readiness_2026-11-09/readiness_audit.md`
