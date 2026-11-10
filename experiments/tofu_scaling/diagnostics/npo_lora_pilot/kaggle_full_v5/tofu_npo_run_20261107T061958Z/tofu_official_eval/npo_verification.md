# TOFU NPO-LoRA pilot verification

Artifact verification: **PASS**
Prespecified diagnostic gates: **FAIL_DIAGNOSTIC_GATES**
Loss trace audit: **PASS** (2000 paired-microbatch records; 2 forget examples per microbatch; 500 optimizer steps).

A PASS here means provenance, saved outputs, official per-row formulas, paired profile-bootstrap results, and artifact hashes reproduce. It does not by itself mean the NPO method passed its scientific or utility gates.

## Primary outcome

NPO minus full Truth Ratio score: 0.080276 [0.068281, 0.092117], 400 rows across 20 profile blocks.

## Utility gates

- `retain90.answer_probability`: delta -0.609747, pass=False.
- `retain90.rougeL_recall`: delta -0.349738, pass=False.
- `holdout10.answer_probability`: delta -0.037834, pass=False.
- `model_utility.harmonic_mean`: delta -0.141414, pass=False.
