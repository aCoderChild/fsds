# TOFU model-size diagnostic

## Purpose

This is the first stage of a resource-bounded machine-unlearning study after the SynthWorlds translation release failed its eligibility gates. It establishes whether the released TOFU target checkpoints have usable pre-unlearning answer probability on the intended forget, retain, and holdout data. It does not begin unlearning.

The provisional comparison is between the published TOFU full checkpoints `Llama-3.2-3B-Instruct` and `Llama-3.1-8B-Instruct`. This is a comparison of two released checkpoints, not a clean parameter-count experiment: the models are from different Llama generations and have different training histories.

## Completed diagnostic

Run:

```bash
python experiments/tofu_scaling/diagnose_tofu.py
```

The script fetches small JSONL data files at a pinned Hugging Face revision, validates their row structure, split relationships, source-to-perturbation alignment, and reports immutable checkpoint metadata. It does not download model weights. Results are in [`diagnostics/tofu_readiness_audit.md`](diagnostics/tofu_readiness_audit.md) and the full machine-readable record is [`diagnostics/tofu_readiness_audit.json`](diagnostics/tofu_readiness_audit.json).

The audit passes structural checks, with one issue to carry forward: `retain90` contains the same generic question, “What is the full name of the author?”, with two different answers. Keep the official data unchanged for the primary benchmark run, then report a sensitivity check excluding both conflicting rows. The dataset schema has no author ID, so this audit cannot independently prove author-level isolation from explicit IDs.

TOFU consists of fictional, synthetic author profiles. Results cannot be presented as evidence that a model has deleted real people's data or provides real-world privacy protection.

## Exploratory likelihood-only diagnostic (completed 6 Nov 2026)

The Kaggle notebook [`kaggle/tofu_baseline_diagnostic.ipynb`](kaggle/tofu_baseline_diagnostic.ipynb) evaluates the full TOFU target checkpoints, without training, on:

- all 400 `forget10` questions (20 profile blocks);
- all 400 `holdout10` questions (20 profile blocks);
- 20 profile blocks from `retain90` (400 questions): the two blocks containing both sides of the known conflicting-question pair are included, and 18 other blocks are sampled without replacement with seed 42.

It uses the pinned Open-Unlearning prompt template and `probability` metric: per-example `exp(-mean assistant-token cross-entropy)`. It saves each question-level score and a 95% cluster-bootstrap interval over profile blocks. It also records model revision, dataset checksums, package versions, GPU details, and runtime. This metric measures answer likelihood, not generated-answer exact match or human quality.

The notebook has no unlearning or SFT cell. NPO starts only in a later stage, after the baseline scores, per-profile uncertainty, and retain-question collision sensitivity check have been reviewed. The earlier Kaggle 8B synthetic smoke is only a hardware/runtime signal; it is not a TOFU result.

### Exploratory outcome

The Kaggle T4×2 run completed for both immutable full checkpoints. Each model scored 1,200 questions: all forget10 and holdout10 rows, plus the same 400 retain90 rows. Local verification independently recomputed the profile-clustered intervals and passed the source/revision, row-count, finite-score, and result-integrity checks.

| Split | Llama 3.2 3B answer likelihood (95% profile CI) | Llama 3.1 8B answer likelihood (95% profile CI) |
|---|---:|---:|
| forget10 | 0.9449 [0.9356, 0.9527] | 0.9912 [0.9893, 0.9929] |
| retain90 sample | 0.9454 [0.9377, 0.9519] | 0.9904 [0.9889, 0.9918] |
| holdout10 | 0.1461 [0.1280, 0.1646] | 0.1226 [0.1065, 0.1394] |

The 8B-minus-3B paired differences were +0.0463 [0.0393, 0.0547] on forget10, +0.0451 [0.0387, 0.0526] on retain90, and −0.0235 [−0.0293, −0.0179] on holdout10. These are descriptive checkpoint differences; because the models come from different Llama generations, they do not isolate parameter count. Excluding the two retain rows with the duplicate question and conflicting answers barely changed the means (3B: 0.9454 to 0.9457; 8B: 0.9904 to 0.9909).

Interpret this as a useful likelihood-based pre-unlearning diagnostic, not answer-generation accuracy or evidence of successful unlearning. Confidence intervals resample 20 profile blocks, not individual QA rows. No unlearning, SFT, or research adapter training has run yet.

Full tables, per-question outputs, and the independent verification receipt are in [`diagnostics/kaggle_baseline_2026-11-06/tofu_baseline_results/`](diagnostics/kaggle_baseline_2026-11-06/tofu_baseline_results/), especially [`baseline_diagnostic_report.md`](diagnostics/kaggle_baseline_2026-11-06/tofu_baseline_results/baseline_diagnostic_report.md) and [`baseline_diagnostic_verification.md`](diagnostics/kaggle_baseline_2026-11-06/tofu_baseline_results/baseline_diagnostic_verification.md). The active protocol status is recorded in [`protocol.json`](protocol.json).

## Reproduction and source pins

- Dataset: [`locuslab/TOFU`](https://huggingface.co/datasets/locuslab/TOFU), revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`.
- Benchmark description: [TOFU paper](https://arxiv.org/abs/2401.06121). It describes 200 fictional profiles with 20 QA pairs each and forget fractions corresponding to author profiles. The Hugging Face card currently has an inconsistent sentence saying `forget10` is one author; the actual pinned split has 400 rows, consistent with 20 profiles × 20 QA pairs.
- Evaluation implementation and prompt: [`locuslab/open-unlearning`](https://github.com/locuslab/open-unlearning), commit `2d98027677f3468d52b6baa3fc4f2b9f2dc08820`; see its [TOFU evaluation config](https://github.com/locuslab/open-unlearning/blob/2d98027677f3468d52b6baa3fc4f2b9f2dc08820/configs/eval/tofu.yaml), [probability metric](https://github.com/locuslab/open-unlearning/blob/2d98027677f3468d52b6baa3fc4f2b9f2dc08820/src/evals/metrics/utils.py), and [model configs](https://github.com/locuslab/open-unlearning/tree/2d98027677f3468d52b6baa3fc4f2b9f2dc08820/configs/model).
- Checkpoints are pinned individually in `diagnose_tofu.py` and in the generated JSON report. The diagnostic checks config access and shard sizes but does not fetch shards.

## Full official-metric baseline (completed 7 Nov 2026)

All four pinned checkpoints passed Kaggle verification and a separate local, row-level verifier: Llama 3.2 3B full and `retain90` oracle; Llama 3.1 8B full and `retain90` oracle. The evaluation is inference-only. It did not train an SFT or NPO adapter.

The evaluator [`tofu_official_eval.py`](tofu_official_eval.py) uses the pinned Open-Unlearning prompt, metric definitions, and data revision. It scores all 400 `forget10`, canonical `retain90`, and `holdout10` questions, plus 100 `real_authors` and 117 `world_facts` utility questions for each full checkpoint. Each matching `retain90` oracle scores the same 400 ordered `forget10` rows. The main 95% intervals bootstrap 20 profile blocks for 5,000 draws with seed 42.

| Checkpoint / split | Questions | Answer probability (95% profile CI) | ROUGE-L recall (95% profile CI) | Truth Ratio |
|---|---:|---:|---:|---:|
| 3B full — forget10 | 400 | 0.9449 [0.9356, 0.9527] | 0.9141 [0.8730, 0.9474] | 0.4984 |
| 3B retain90 oracle — forget10 | 400 | 0.1252 [0.1023, 0.1510] | — | 0.8541 |
| 3B full — retain90 | 400 | 0.9367 [0.9293, 0.9437] | 0.9028 [0.8783, 0.9261] | 0.4669 |
| 3B full — holdout10 | 400 | 0.1461 [0.1280, 0.1646] | 0.3774 [0.3511, 0.4017] | — |
| 8B full — forget10 | 400 | 0.9912 [0.9892, 0.9929] | 0.9888 [0.9818, 0.9949] | 0.5204 |
| 8B retain90 oracle — forget10 | 400 | 0.1045 [0.0851, 0.1257] | — | 0.9175 |
| 8B full — retain90 | 400 | 0.9910 [0.9892, 0.9925] | 0.9954 [0.9902, 0.9993] | 0.4851 |
| 8B full — holdout10 | 400 | 0.1226 [0.1065, 0.1394] | 0.3689 [0.3436, 0.3921] | — |

The full checkpoints assign much higher likelihood to forget-set answers than their retain-only oracles. A matched profile-block bootstrap (20 groups, 5,000 draws, seed 42) estimates full-minus-oracle differences of +0.8197 [0.7956, 0.8408] in answer probability and −0.3557 [−0.4119, −0.3007] in Truth Ratio for 3B; the 8B differences are +0.8867 [0.8653, 0.9058] and −0.3971 [−0.4678, −0.3302]. Both answer-probability intervals and Truth Ratio intervals exclude zero. The row-level KS statistics on inverse Truth Ratio are `D=0.3775` (3B) and `D=0.3900` (8B); their conventional p-values are recorded in the result JSON, but do not adjust for dependence among questions within profiles, so are descriptive only. These are baseline-separation results, not evidence that an unlearning method succeeded. Full-checkpoint utility harmonic means are 0.6702 (3B) and 0.6370 (8B). Since the checkpoints come from different Llama generations and training histories, their contrast is descriptive, not a causal parameter-size effect.

### Method and scope limits

- Kaggle T4×2 inference used FP16 + SDPA. Upstream uses BF16 + FlashAttention 2, so generated strings and numeric values need not be bitwise identical.
- Answer probability is reference-answer likelihood, not exact-match accuracy; ROUGE-L measures text overlap. Neither alone proves that knowledge was erased.
- TOFU uses fictional author profiles. These findings do not establish removal of real personal data or real-world privacy protection.
- The canonical retain set has one repeated generic question with conflicting answers. Primary metrics preserve the benchmark rows; the result bundle includes the previously computed collision sensitivity.

### Reproduction artifacts

- Baseline notebook: [TOFU Official Metrics Baseline 3B/8B](https://www.kaggle.com/code/bunnoob2005/tofu-official-metrics-baseline-3b-8b-2026-11-06).
- Isolated 8B oracle retry, version 6: [8B-only Kaggle notebook](https://www.kaggle.com/code/bunnoob2005/tofu-official-metrics-8b-retry-2026-11-06).
- Complete results, raw outputs, local verification receipt, run log, notebook snapshot, and checksums: [`diagnostics/kaggle_official_metrics_8b_retry_2026-11-06/`](diagnostics/kaggle_official_metrics_8b_retry_2026-11-06/). The byte-preserved version-6 output ZIP has SHA-256 `b4df8407e556050c905f6a648602783c6dbcc96093ed4df7a2a9e8bbb5fee6a5` and passed CRC verification.
- Paired profile-bootstrap analysis and reproducible script: [`paired_oracle_cluster_differences.json`](diagnostics/kaggle_official_metrics_8b_retry_2026-11-06/version6_2026-11-07/reconciled_complete_3b_8b_verified/tofu_official_eval/paired_oracle_cluster_differences.json) and [`analyze_paired_oracle_cluster_differences.py`](diagnostics/kaggle_official_metrics_8b_retry_2026-11-06/version6_2026-11-07/reconciled_complete_3b_8b_verified/analyze_paired_oracle_cluster_differences.py).
- Reproduce the independent checks locally after `python -m pip install scipy rouge-score`:

```bash
python experiments/tofu_scaling/verify_tofu_official_eval.py \
  experiments/tofu_scaling/diagnostics/kaggle_official_metrics_8b_retry_2026-11-06/version6_2026-11-07/reconciled_complete_3b_8b_verified/tofu_official_eval \
  --audit experiments/tofu_scaling/diagnostics/kaggle_official_metrics_8b_retry_2026-11-06/version6_2026-11-07/reconciled_complete_3b_8b_verified/tofu_official_eval/audit/tofu_readiness_audit.json
python experiments/tofu_scaling/diagnostics/kaggle_official_metrics_8b_retry_2026-11-06/version6_2026-11-07/reconciled_complete_3b_8b_verified/analyze_paired_oracle_cluster_differences.py \
  experiments/tofu_scaling/diagnostics/kaggle_official_metrics_8b_retry_2026-11-06/version6_2026-11-07/reconciled_complete_3b_8b_verified/tofu_official_eval
```
- Dataset revision and source implementation are recorded under [Reproduction and source pins](#reproduction-and-source-pins); all four checkpoint IDs and revisions are in their result JSON files.

## NPO-LoRA diagnostic pilot (α=40 development follow-up complete)

The CPU objective fixture and bounded real-data smoke passed. A one-seed Llama 3.2 3B NPO-LoRA follow-up (α=40, 250 optimizer steps) completed on Kaggle T4×2. It used the pinned TOFU checkpoint and profile cohorts; it is a subset diagnostic, not a full-split or confirmatory run. The archived trainer matches the local training source byte-for-byte, and an independent local verifier recomputed the metrics and profile-block bootstrap from the downloaded archive.

The candidate passed 3/5 locked development gates: forget answer probability, retain answer probability, and holdout answer probability. It failed the forget Truth Ratio and retain ROUGE-L gates. The protocol stopped before training on final profile blocks. Do not run another retain-weight value under that protocol or claim successful unlearning from this result. A single Codex AI reviewer, blinded to A/B conditions, preferred the baseline in 22/40 answer-string-loss cases and in 5/20 changed-output controls; it preferred α=40 in 0/40 and 2/20, respectively, with the remaining pairs tied. This supports concern that some ROUGE/string loss is semantic, but it is not human or independent review and the enriched case-control sample cannot estimate error prevalence.

Training-dynamics recomputation found 250/250 accumulated optimizer updates above the configured pre-clipping gradient-norm cap of 1.0 (median 52.63, maximum 173.88). The implementation clips once per accumulated update as specified. Large pre-clipping norms merit attention in the next design, but they do not alone establish an implementation defect. See the complete [α=40 results and limits](diagnostics/npo_lora_profile_alpha40_2026-11-08/alpha40_stage_results.md), [training-dynamics audit](diagnostics/npo_lora_profile_alpha40_2026-11-08/alpha40_training_dynamics_audit.json), and blinded retain-answer review packet in the same directory. The AI-only ratings are separate from any human review; the private A/B key is not included in the report.

No final-cohort NPO run or multi-seed confirmation has been performed. Do not infer a causal parameter-size effect from the 3B-versus-8B baseline contrast, and do not treat TOFU's fictional profiles as evidence of removing real personal data.
