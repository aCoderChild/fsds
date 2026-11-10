# α40 TOFU 3B follow-up — results

**Outcome:** the Kaggle run completed, but the single development candidate failed the preregistered all-gates rule. The protocol therefore stopped before final-cohort training. This is an exploratory diagnostic, not a confirmatory experiment.

## Design and execution

- Model: Llama 3.2 3B full TOFU checkpoint, seed 42; NPO-LoRA with retain-NLL weight α=40.
- Data: pinned TOFU revision; development evaluation used 200 forget, 400 retain, and 200 holdout questions across 10, 20, and 10 profile blocks, respectively.
- Procedure: 250/250 optimizer steps completed with finite loss and gradients; 1,000 loss-trace records; bounded real-data smoke and the two-example objective fixture passed. The fixture did not test retain-loss aggregation across all four microbatches in an optimizer update.
- Hardware: two Tesla T4 GPUs. Training took 13m 20s; development evaluation took 39m 43s; Kaggle run took about 54m 41s.
- Statistics: paired profile-block bootstrap, 5,000 draws, seed 42. The locked utility non-inferiority margin was 5 percentage points.

### Training-dynamics audit

The archived trainer hash matches the hash recorded in the run receipt. The shared local trainer has since changed, so the archived copy—not today's source—is authoritative for this run. The loss trace contains 1,000 microbatch records grouped into exactly 250 optimizer updates (four records per update); the recorded gradient norm is the single pre-clipping global norm for that accumulated update, repeated on its four records. All 250/250 pre-clipping norms exceeded the configured 1.0 cap (min 12.89, median 52.63, p90 97.81, p95 113.51, max 173.88). The archived code clips once after accumulation. The large norms indicate that clipping materially scaled the updates; they do not alone establish instability. The forget NPO loss decreased from a 13.86 epoch-1 mean to 8.69 in epoch 10, while retain-token NLL rose from 0.0616 to 0.0701 in those epochs. These training-loss patterns do not establish the cause of the measured retain-generation change.

A follow-up source audit found that the archived trainer computes a token-mean retain loss separately for each two-example microbatch, then gives each of the four accumulated microbatches equal weight. The preregistration says `token_mean_nll` but does not specify whether this mean is defined per microbatch or over all tokens in the optimizer update. The recorded objective fixture used two examples and did not verify equivalence over the full eight-example effective batch. Therefore, retain-objective fidelity at the optimizer-update level is **unverified**; treat these results as diagnostic outputs of the archived implementation, not as proof of the intended pooled-token objective.

The corrected shared trainer was separately checked after this run. Its NPO loss and gradients matched the pinned upstream implementation; an unequal-length accumulation fixture matched the pooled token-mean loss and gradient exactly (absolute errors 0.0). These checks support future runs only and do not retroactively change the archived run. The combined fixture record is [`corrected_trainer_validation_postrun_2026-11-09.json`](corrected_trainer_validation_postrun_2026-11-09.json).

The trace recomputation, archive CRC check, step grouping, finite-value checks, receipt/source hash comparison, and aggregation-scope audit are reproducible with [`alpha40_training_dynamics_audit.py`](alpha40_training_dynamics_audit.py); machine-readable output is [`alpha40_training_dynamics_audit.json`](alpha40_training_dynamics_audit.json).

## Preregistered gates

All values are adapter-minus-frozen-checkpoint changes in percentage points. CIs are 95% profile-block bootstrap intervals.

| Outcome | Δ [95% CI] | Gate |
|---|---:|:---:|
| `forget10.truth_ratio_score` | -0.38 [-1.34, +0.52] | FAIL (lower CI > 0 pp) |
| `forget10.answer_probability` | -15.79 [-18.03, -13.58] | PASS (upper CI < 0 pp) |
| `retain90.answer_probability` | -1.69 [-2.11, -1.29] | PASS (lower CI > -5 pp) |
| `retain90.rougeL_recall` | -6.65 [-8.40, -4.94] | FAIL (lower CI > -5 pp) |
| `holdout10.answer_probability` | -4.53 [-4.87, -4.17] | PASS (lower CI > -5 pp) |

**Gate count: 3/5 passed.** The answer-probability forgetting gate passed, but the truth-ratio-score forgetting gate did not. Retain answer probability and holdout answer probability stayed within their registered margins. Retain ROUGE-L failed: its estimated loss was 6.65 pp and the lower CI bound was −8.40 pp, below the allowed −5 pp boundary.

## Retain ROUGE-L diagnosis

Against the pinned baseline on the 400 matched retain questions, ROUGE-L was lower on 106 rows, unchanged on 266, and higher on 28; generated text was identical on 239 rows. Mean ROUGE-L declined in 19/20 profile blocks (median block change -5.91 pp). This suggests the loss is spread across profiles rather than driven only by one block; it is descriptive, and inference remains block-based.

Compared with the previous α=20 development point estimate (−7.08 pp retain ROUGE-L), α=40 improved by only 0.43 pp and still failed the unchanged margin. The α=40 truth-ratio-score change was −0.38 pp (95% CI −1.34 to +0.52), so this run does not establish that the target fact was reliably removed under that measure.

### Local generation audit

The ROUGE-L drop is not explained by shorter generations: mean output length was 24.10 words at baseline and 24.28 after α40. The normalized source-answer string appeared in 307/400 baseline outputs and 238/400 α40 outputs. In 80 rows it disappeared; in 11 it appeared only after α40. Those are strict string matches, so valid paraphrases can count as losses. The 80 baseline-only rows had a mean ROUGE-L change of −32.49 pp, while the other 320 rows averaged −0.19 pp.

A deterministic date-conflict screen found three α40 outputs that disagreed with the source-answer date. Two of those mismatches were already present in the frozen-checkpoint baseline; one additional contradiction appeared after α40 (20 January 1989 changed to 16 March 1989). This is a narrow heuristic screen, not a complete factuality audit.

### Blinded AI-only semantic review

After the A/B labels were completed without seeing the condition key, a single Codex AI reviewer compared the 60 sampled pairs against the supplied reference answers. In the 40 answer-string-loss cases, the frozen baseline was preferred in 22, α40 in 0, and the answers tied in 18. In the 20 changed-output controls, the baseline was preferred in 5, α40 in 2, and they tied in 13. Across the selected 60 pairs, the baseline was preferred in 27, α40 in 2, and 31 tied. The AI ratings marked baseline responses correct in 55/60 cases and α40 responses correct in 30/60, partly correct in 21/60, and incorrect in 9/60.

This is only a single AI review against synthetic reference text, not fluent-human or independent multi-rater validation. The answer-string-loss stratum was deliberately enriched and the controls were selected, so these counts are qualitative diagnostics; they cannot estimate overall error rates or benchmark-wide accuracy. The row-level A/B-only annotations are in [`retain_pairwise_review_codex_ai_2026-11-08.csv`](retain_pairwise_review_codex_ai_2026-11-08.csv), and the condition-masked aggregate is in [`retain_pairwise_review_codex_ai_summary.json`](retain_pairwise_review_codex_ai_summary.json). Recompute it with [`summarize_codex_ai_retain_review.py`](summarize_codex_ai_retain_review.py). The condition key remains in the separate private file.

## Interpretation and next action

The result does not support proceeding to final blocks. Raising retain weight from α=20 to α=40 only slightly reduced the ROUGE-L loss and did not pass the utility threshold; the truth-ratio gate also failed. Do not run another alpha value or use final blocks under this protocol. The blinded 60-pair packet includes 40 cases enriched for lost exact answer strings and 20 changed-output controls, with all 20 profile blocks represented. The AI-only review suggests the retain-generation decline often reflects meaningful answer degradation in this selected sample, but independent human review is still needed before designing another GPU run. The sample is not prevalence-estimating; ROUGE-L and exact-string checks cannot distinguish every paraphrase from a factual error.

Limitations: one seed, alpha=40 chosen after viewing earlier development results, only 10/20 profile blocks per split, and a TOFU-finetuned checkpoint evaluated on a profile-cohort subset rather than a standard full-split TOFU run.

## Reproducibility and artifacts

- Kaggle notebook: [α40 follow-up run](https://www.kaggle.com/code/bunnoob2005/tofu-3b-npo-lora-alpha-40-follow-up) (completed).
- Protocol: `tofu_3b_npo_lora_profile_alpha40_followup_v1`; SHA-256 `b308c93da3934a020cd0ca1c22c89c74133e2cb97d90656f4ff02ccb155da8a7`.
- Artifact ZIP: `tofu_npo_alpha40_followup_artifacts.zip` (93,511,185 bytes), SHA-256 `a5857e7d7d68074abd564023d5575ca8baddad1072eab81a1cc1886326f3c1ec`; ZIP CRC passed.
- Independent result recomputation: Kaggle verifier and local recheck both `PASS_ARTIFACT_AND_RESULT_RECOMPUTATION`, with no reported problems.
- Reproducible local output audit: `alpha40_retain_generation_audit.py` writes `alpha40_retain_generation_audit.json`.
- Blinded human-review packet: `retain_pairwise_review_blinded.csv` with neutral instructions in `retain_pairwise_review_instructions.md`; the separate `retain_pairwise_review_key_PRIVATE.csv` reveals the A/B mapping and must stay private until ratings are complete.
- Full metrics and run provenance: `alpha40_stage_results.json`.
- Protocol digest discrepancy: `alpha40_preregistration_hash_erratum.md`.
