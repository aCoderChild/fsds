# TOFU NPO-LoRA pilot verification

Artifact verification: **FAIL**
Prespecified diagnostic gates: **FAIL_DIAGNOSTIC_GATES**
Loss trace audit: **PASS** (4000 paired-microbatch records; 1 forget examples per microbatch; 500 optimizer steps).

A PASS here means provenance, saved outputs, official per-row formulas, paired profile-bootstrap results, and artifact hashes reproduce. It does not by itself mean the NPO method passed its scientific or utility gates.

## Primary outcome

NPO minus full Truth Ratio score: 0.040498 [0.026708, 0.055235], 400 rows across 20 profile blocks.

## Utility gates

- `retain90.answer_probability`: delta -0.605046, pass=False.
- `retain90.rougeL_recall`: delta -0.340007, pass=False.
- `holdout10.answer_probability`: delta -0.025028, pass=True.
- `model_utility.harmonic_mean`: delta -0.102053, pass=False.

## Verification problems

- training receipt objective check differs from supplied check
