# Alpha-40 follow-up preregistration

**Status:** preregistered locally before this GPU run; exploratory and post-hoc, not confirmatory.
**Date:** 2026-11-08.
**Protocol:** `tofu_3b_npo_lora_profile_alpha40_followup_v1`
**Protocol SHA-256:** `f61dc9cfd71a1e410877543f2c6a315934a187d6839b80448cc9ee807b0c3f3a`

## Question and decision

Test whether one NPO-LoRA run with retain-NLL weight alpha=40 can improve retain-set greedy-generation ROUGE-L enough to satisfy the existing 5 percentage-point non-inferiority margin, while preserving detectable forgetting and passing the other existing utility checks. The earlier alpha=5/10/20 result remains unchanged. Alpha=40 was selected after seeing the development trend, so this is explicitly exploratory.

## Fixed design

- Model: pinned full TOFU Llama 3.2 3B checkpoint; seed 42.
- Objective: pinned NPO forget objective plus alpha-weighted retain token-mean NLL; alpha=40.
- Training: existing locked 250 optimizer-step, two-T4 LoRA procedure, unchanged.
- Data: same pinned TOFU revision and disjoint profile-block assignments; development uses forget blocks 0–9, retain utility blocks 60–79, and holdout blocks 0–9. Final blocks remain untouched unless development passes.
- Outcomes: Truth Ratio score and answer probability on forget; answer probability and ROUGE-L on retain; answer probability on holdout.
- Statistics: paired profile-block bootstrap, 5,000 draws, seed 42, percentile 95% intervals.
- Gates: unchanged from the previous protocol. No threshold or cohort changes.
- Stop: one dev candidate only; if any gate fails, no final-cohort training. If all pass, train/evaluate alpha=40 once on the disjoint final blocks.

## Baseline provenance

Reuse only the frozen-checkpoint reference evaluation from the previously independently verified archive (`68b72db9abc1c6120e74397f3e495a97b2e2c3b1c1205f212408948136e45c02`). Its JSON hash is `17c453122c9ab597daa7d05975ed232991cf4e9336c7951542d7260b2c46ead6`; the prior independent-verification receipt is pinned at `c2cb97f5e923a391d17e0ad704b91a33157b35a4090bcd02c9de08c98dc3b097`. The new Kaggle runner checks the hash, checkpoint revision, dataset hashes, row assignments, generation settings, and no-truncation/no-offload fields before training.

## Tracking

- State: preregistered and package preparation in progress.
- Kaggle kernel: to be published after local validation.
- Training: not started.
- Result: pending.
