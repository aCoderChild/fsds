# Llama 3.1 8B TOFU NPO-LoRA pilot

## Final disposition (verified 2026-11-09T05:42:35.338759+00:00)

Kaggle v2 is `COMPLETE`. The current-attempt archive is CRC-valid (SHA-256 `91fad5f8df28b6e8f26dbaab93cc24797c3b36bbfcf2620e5cd5efdec9e61c9c`), and its receipt verifies 500 optimizer steps. Independent metric recomputation passes. The training source does **not** align with the preregistered retain-token aggregation, and the retain answer-probability, retain ROUGE-L, and composite-utility gates fail. Keep this as an implementation diagnostic, not a protocol-faithful or confirmatory result. See `kaggle_v2/postrun_verification.json` and `diagnostic_report.md`.

The corrected local trainer now passes the pinned objective/gradient comparison and unequal-length accumulation fixture, but it has not been run on Kaggle. The next benchmark audit found MUSE-News data and target weights are technically downloadable; training remains held pending model-use terms and a reliable QA-to-article linkage. See `../muse_news_readiness_2026-11-09/readiness_audit.md`.

The dated monitoring entries below are retained as history; they describe the period before the final archive was retrieved and must not be read as current status.

---

**Historical pre-completion snapshot (2026-11-09T02:50:03Z; superseded):** Kaggle v2 reports `RUNNING`; refreshed Kaggle logs showed 21,128.4 seconds elapsed, with no new log entry after the internal ZIP update at 4,022.3 seconds (159,312,103 bytes). The exact-artifact output request still returns no file, so no receipt or NPO result is locally retrievable. The notebook captures subprocess output, which hides training and evaluation progress until each subprocess exits. Given the source order, the 4,022-second archive update followed by continued runtime strongly suggests the 500-step training subprocess returned successfully and adapter evaluation is running; this is a control-flow inference, not direct optimizer/evaluator telemetry. The adapter evaluator processes 6,868 candidate scores and 1,417 generations at batch size 1, so it can be substantially slower than the earlier 70-minute full-checkpoint evaluation that used batch size 2. The local tracker has been restarted as PID 3742 / tool session 42766 with a 14-hour window; its timestamped-archive and current-attempt receipt guards remain enabled. The Kaggle kernel was not restarted.


## Preflight and smoke evidence

- Version 1 completed the bounded smoke on two Tesla T4 GPUs. The independent smoke verifier passes: one finite optimizer step, nonzero LoRA update, no offload or truncation, exact locked model/data revisions, and no scored research adapter.
- The pinned NPO objective check passed; loss difference was `0.0`, maximum gradient difference was `5.96e-08`.
- Both 8B full and retain90 baseline files pass independent verification. The 3B files are intentionally absent from the 8B run inputs; the verifier checks this expected partial inventory correctly.
- Full-mode package and smoke-only package both pass offline validation. Kaggle-pulled version 2 has the same 18 cell sources as the validated full notebook.

## Locked run

The full-mode notebook repeats preflight, then runs exactly 500 optimizer steps with the preregistered seed, checkpoint, data, NPO + retain-NLL objective, and LoRA settings. It then evaluates the adapter and runs the independent artifact verifier. The run is one-seed exploratory evidence, not confirmatory and not a controlled 3B-vs-8B comparison.

A lightweight tracker is monitoring the Kaggle job. After a terminal state, it will download only the result ZIP, never the large model cache.

## Alignment with R3

R3's completed main result is IDK+AP LoRA on Llama 3.2 3B; its retain90 ROUGE-L gate failed narrowly, and its next plan calls for an unused benchmark/cohort. This running NPO-LoRA pilot instead uses Llama 3.1 8B on the same pinned TOFU splits. If verified, it can provide a separate exploratory diagnostic of the preregistered NPO question, but it is not an independent replication, does not isolate model scale, and cannot establish that the prior ROUGE-L gate failure is resolved. See `scope_alignment_audit_2026-11-09.json`.

## Supplemental metric-verifier check

The local independent metric verifier passed a self-check on the pinned Llama 3.1 8B full-checkpoint baseline: all 1,417 saved generation rows reproduced ROUGE-L, and all nine model-utility components plus their harmonic mean matched. I created an isolated environment at `/tmp/tofu-npo-verify-venv` with the pinned `rouge-score==0.1.2` dependency and reran the check; the saved result is `independent_metric_verification/baseline_selfcheck_local_recheck.json`. At 2026-11-09 01:00 UTC, a further verifier preflight passed with the full-checkpoint file supplied as both baseline and candidate; it independently recomputed 1,417 ROUGE-L rows and nine utility components. The record is `independent_metric_verification/postrun_verifier_preflight_2026-11-09.json`. Both checks validate tooling only; neither is an NPO result or verifies the trained adapter.

## Runtime window

The earlier 8-hour tracker window was due to expire before Kaggle's documented 12-hour session limit. At 2026-11-09 02:50 UTC it was replaced with a 14-hour guarded monitor (PID 3742; tool session 42766), polling every 60 seconds. Kaggle exposes no live cell or optimizer telemetry through the checked API. The monitor only downloads the artifact after a terminal kernel state and rejects stale or mismatched attempts.

## Progress observability and runtime context

The validated notebook source runs training and adapter evaluation with `subprocess.run(..., capture_output=True)`, writes each captured log only after that subprocess exits, and creates the downloadable ZIP in `/kaggle/working`. Therefore, while the kernel is `RUNNING`, an empty published-output listing cannot distinguish model loading, training, evaluation, or a stall. Inspection of the installed Kaggle client confirms its kernel API exposes status and output retrieval but no live cell-log or optimizer-progress method. Historical runtimes are only context: the bounded 8B smoke took 121.69 seconds for its one-step run (including startup/model loading), the earlier 3B 500-step training took 1,600.87 seconds, and an 8B checkpoint evaluation took about 70 minutes. These runs differ in model or workload and do not establish progress or predict this 8B run's completion time.

Historical runtime context (not a progress estimate): a prior completed 8B full-checkpoint evaluation took 4,208.96 seconds (~70 minutes), and the prior 3B 500-step NPO training took 1,600.87 seconds (~27 minutes). The current 8B run includes a different training and evaluation workload, so these observations only show that four hours is not by itself evidence of a hang. Sources: `experiments/tofu_scaling/diagnostics/npo_lora_pilot/kaggle_full_v5/tofu_npo_run_20261107T061958Z/tofu_official_eval/llama_3_1_8b_full_official_eval.json` and `.../npo_train/training_receipt.json`.

## Published-file inventory checks

The initial inventory at 2026-11-08T23:36:23Z, the all-pages inventory at 2026-11-09T00:04:13Z, the two-page inventory at 00:22:44Z, the all-pages inventory at 00:29:50Z, the direct output requests at 00:36--00:43 UTC, and the latest requests at 00:52, 00:58, 01:00, 01:01, 01:02, 01:05, 01:07, 01:08, 01:09, and 01:11 UTC returned no v2 result files (the 361 enumerated files all predated launch). Therefore none is evidence from v2; no current training receipt or result archive is published.

## Evaluation implementation audit

The locked adapter evaluator and independent verifier use the same intended contrast direction (candidate minus full checkpoint), pair rows by source index, resample profile blocks for 5,000 bootstrap draws, require a positive lower 95% bound for the primary forget Truth Ratio score, and apply the three paired utility margins plus the nine-component composite utility point gate. The verifier also checks per-row likelihood/Truth Ratio formulas, row coverage, source hashes, and adapter/receipt hashes. The verified baseline metric reproduction covers the checkpoint evaluator only; the adapter-specific verifier must still pass on the v2 archive before accepting any NPO result.

## Batching metadata audit

The locked 8B trainer selects microbatch size 1 with 8 gradient-accumulation steps, giving 8 forget/retain pairs per optimizer update as required by the protocol. Its receipt's descriptive `method.batching` string is inherited from the 3B configuration and incorrectly says 2+2 examples per microbatch. This does not change the executed 8B schedule: the trainer writes explicit microbatch and accumulation values to `training_schedule`, the loss trace records examples per microbatch, and the independent verifier checks those against the hashed trainer source. Preserve the original receipt; report the metadata discrepancy and use the recorded schedule/source when interpreting the run.

## Remote source integrity

At 2026-11-09T00:04:13Z, Kaggle's pulled source matched the validated local full notebook cell-for-cell (18/18); full training was enabled. The seven embedded Python-source hashes, locked protocol hash, and dependency-file hash all matched their manifests. This verifies what was submitted, but does not show which notebook cell is currently executing.

At 2026-11-09T02:14Z, I pulled the current Kaggle notebook again. Its 18 normalized cell sources match the validated full notebook (source-list SHA-256 `b46693e10d21a9036169ac24418540a4f4d82a5ab2c1e6d6d934f420f8187bad`). Both the full and smoke package validators pass in `/tmp/tofu-npo-verify-venv`; the ordinary system Python lacks `rouge_score`, so use the pinned verifier environment for local checks. Kaggle metadata reports `machine_shape: NvidiaTeslaT4`, which Kaggle documents as GPU T4 ×2. This source/package recheck does not verify optimizer progress or produce a training result. Full evidence: `source_integrity_recheck_2026-11-09T0214Z.json`.

## Baseline direction sanity check

A reproducible baseline-only check confirms that the locked primary score points in the intended direction: on 400 forget questions across 20 profile blocks, the full 8B checkpoint mean is 0.4734 and the retain90 oracle mean is 0.6407. The paired oracle-minus-full difference is +0.1673 (95% profile-block bootstrap interval [0.1417, 0.1896]; 5,000 draws, seed 42). This supports the direction of the primary metric, but is not an NPO effect or evidence of successful unlearning. The analysis is saved in `baseline_direction_sanity.json` and can be reproduced with `baseline_direction_sanity.py`.

I reran the baseline-only script at 2026-11-09T02:24Z. It rechecked the baseline manifest and independent source-verification receipt, validated both file hashes, aligned all 400 rows by source index and question hash, and reproduced the same means and bootstrap interval. No post-training result is available yet.

## Implementation-design recheck

At 2026-11-09T02:34Z, the direct Kaggle status was still `RUNNING`; the local tracker (PID 39665) was alive. A fresh exact-artifact output query returned no file, so there is still no training receipt or NPO measurement. I inspected the current local trainer, verifier, locked protocol, and the recovered Open-Unlearning checkout at the exact pinned commit `2d98027677f3468d52b6baa3fc4f2b9f2dc08820`.

The core NPO sequence-loss formula and the retain NLL formula match the pinned upstream functions in the deterministic fixture (zero loss difference; max gradient difference `6.71e-08`). The trainer also enforces the pinned model/data revisions and hashes, two T4 GPUs, no CPU/disk offload, no truncation, LoRA-only trainable parameters, and finite losses/gradients. The final archive verifier checks the actual schedule and trace against the hashed trainer source.

Two interpretation caveats remain. First, the 8B schedule uses physical microbatch 1 with accumulation 8. The retain loss is token-mean within each one-example microbatch and then accumulated equally; with unequal answer lengths, this is a mean of per-example token means, not the token-weighted mean over one physical batch of eight. It is a resource-bounded NPO-LoRA variant, and the current fixture proves per-batch loss equivalence but not equivalence between these two batching/weighting schemes. Second, `METHOD["batching"]` in the receipt is a stale descriptive string (it says 2+2 per microbatch); use `training_schedule`, loss-trace rows, and the source hash as the authoritative record. The independent verifier checks those operational fields, not the stale text.

The primary preregistered signal is candidate-minus-full Truth Ratio score > 0 with a positive 95% block-bootstrap lower bound. Since the protocol separately calls for interpreting the full-to-oracle contrast, report the candidate-to-oracle distance alongside it; a positive shift alone does not prove the candidate is closer if it overshoots the oracle. Preserve these caveats in the final diagnostic and do not change the locked run or its thresholds after seeing outcomes.


## Current handoff (2026-11-09 02:59 UTC)

- Direct Kaggle status recheck: `bunnoob2005/tofu-8b-npo-lora-pilot-2026-11-08` is still `RUNNING`; no result archive has been published yet. Training/evaluation progress is not directly observable through the Kaggle API.
- Guarded download monitor remains active (PID 3742; 60-second polling; 14-hour window). It will only accept an archive whose embedded run timestamp and receipt are from this launch.
- Local finalizer `finalize_kaggle_v2.py` compiles and resolves the repository root correctly. It is running in tool session 74446 and waiting for the monitor receipt; after download it checks ZIP CRC and archive SHA-256, verifies the locked protocol and 500-step receipt, then runs the independent provenance/metric verifiers.
- No NPO result is verified at this point. After verification, report artifact integrity separately from the preregistered research and utility gate outcomes.


Read-only post-run output poll at 2026-11-09T03:01:11.927920+00:00: Kaggle remains `RUNNING`; the exact archive file pattern returned no published files. No stale archive was accepted. The tracker and finalizer remain active.


## Correction to stage inference (2026-11-09T03:02:43.412017+00:00)

The earlier inference that adapter evaluation had started was unsupported and is retracted. Notebook cell 16 creates the ZIP after the bounded one-step smoke; cell 17 then runs full training, adapter evaluation, and verification in sequence, updating the ZIP after each subprocess returns. The last known archive update at 4,022.3 seconds therefore does not prove full training completed. Kaggle still reports `RUNNING`, no full-training receipt is published, and the current internal stage cannot be determined from available telemetry.


## Monitor refresh and UI check (2026-11-09T03:04:48.007791+00:00)

The prior monitor was stopped before its wait deadline and replaced with a fresh 14-hour monitor (session 79733) using the same guarded launch timestamp, output destination, and status receipt path. It confirmed the kernel is `RUNNING`. The post-run verifier remains live (session 74446). Computer-use reports no available browser surface, so Kaggle cell logs cannot be inspected through the browser here; API telemetry still does not reveal optimizer progress.


## Latest monitor and verifier check (2026-11-09T03:07:45.243372+00:00)

Kaggle remains `RUNNING`; no full-training receipt or result archive is available. Tracker session 79733 and finalizer session 74446 remain active. A temporary fixture check passed for finalizer repository-root resolution, valid single-root ZIP extraction, and rejection of a path-traversal member. The fixture was temporary and left no test data in the repository.


Post-run verifier command preflight at 2026-11-09T03:09:15.845519+00:00: both pinned-environment verifier CLIs accept the exact planned arguments (`--model 8b`, training/evaluation dirs, locked protocol, objective check, baseline/candidate metric files). Protocol recheck confirms preregistration, seed 42, 400 forget + 3,600 retain training rows, and 400/400/400 + 100 + 117 evaluation rows. The run receipt is still unavailable, so no experiment outcome is inferred.


Latest live poll at 2026-11-09T03:11:12.994383+00:00: Kaggle still reports `RUNNING`; the exact artifact query returned zero files. The tracker and post-run finalizer remain active; no optimizer progress is observable through the available Kaggle CLI.


## Existing 3B diagnostic recheck (2026-11-09T03:16:39.930399+00:00)

The completed IDK+AP 3B archive was freshly extracted to a temporary directory and rerun through `experiments/tofu_scaling/idk_ap_lora/verify_run.py`. Artifact/result recomputation passed with no problems and the same archive SHA-256. The final cohort reproduces four passing gates and one failed retain90 ROUGE-L gate (delta -3.058 pp; 95% CI [-5.110, -0.993] pp against strict lower-bound > -5 pp). Full evidence: `experiments/tofu_scaling/diagnostics/idk_ap_lora_2026-11-09/kaggle_run2/LOCAL_VERIFICATION_RECHECK_CURRENT.json`. This is the completed 3B diagnostic; the separate 8B NPO kernel remains `RUNNING` with no receipt or result archive.


8B monitor recheck at 2026-11-09T03:17:57.029089+00:00: Kaggle reports `RUNNING`; exact archive query found no file. Terminal receipt remains unavailable; internal training progress is still unverified. Monitor and finalizer remain active.


Latest poll (2026-11-09T03:22:42.207745+00:00): the Kaggle kernel remains `RUNNING`, with no terminal receipt or archive. Native Chrome changed to the user's active tab during the attempted read-only inspection, so no further UI actions were taken; API monitoring continues.


Exact archive poll at 2026-11-09T03:23:42.077188+00:00: Kaggle returned zero matching ZIPs while the kernel remains `RUNNING`. No full-training receipt or adapter-evaluation output is published.


## Monitor implementation recheck (2026-11-09T03:28:18.018513+00:00)

The tracker and finalizer processes are still present. Kaggle still reports `RUNNING`; no result ZIP or terminal receipt is available. The apparent quiet log is expected: the tracker writes a `RUNNING` event only on status transitions and writes its receipt after a terminal status or monitor timeout. Training progress remains unobservable through the API.


## Verified wait (2026-11-09T03:29:56.115948+00:00)

The same Kaggle job remains `RUNNING`; a direct artifact query returned exit code 0 with no matching result ZIP. The tracker session and finalizer are alive, and no terminal receipt is available. No experiment outcome is assigned while the run is nonterminal.


## Current attempt recheck (2026-11-09T03:32:54.767182+00:00)

Kaggle still reports `RUNNING`. A direct query for the exact result ZIP succeeded but returned no archive. No terminal tracker receipt is present; both monitor processes remain alive. The computer-use inventory exposes no browser tabs, so live cell logs cannot be read; optimizer progress remains unverified.


## Live run elapsed check (2026-11-09T03:36:59.495054+00:00)

Elapsed since guarded launch: 6 h 38 min. Kaggle reports `RUNNING`; exact result archive query exit code 0, matching archives 0. Tracker and finalizer remain active, with no terminal receipt.


## Kaggle client capability check (2026-11-09T03:40:21.349192+00:00)

The installed Kaggle Python client exposes kernel status and completed-output retrieval, but no live log-streaming method. The kernel remains `RUNNING`; progress cannot be inferred from this status, and no archive is available. The existing tracker remains the available automated monitor.


## Verified wait (2026-11-09T03:49:14.775274+00:00)

Kaggle reports `RUNNING` about 6 h 51 min after launch. An exact archive query returned exit code 0 and 0 matching files. Tracker and finalizer remain active; there is no terminal receipt. The monitor deadline is 2026-11-09T17:03:00+00:00 UTC.


## Live UI access recheck (2026-11-09T03:50:16.919305+00:00)

The in-app browser surface is unavailable, and no native browser tab surface is exposed. Kaggle still reports `RUNNING`; no archive or terminal receipt is available. Live cell state remains inaccessible through the available interfaces.


## Live run recheck (2026-11-09T03:51:06.034235+00:00)

Kaggle reports `RUNNING`; an exact result-archive query succeeded with no matching ZIP. Tracker and finalizer are alive, and no terminal receipt is present.


## Kernel liveness check (2026-11-09T03:53:00.576249+00:00)

Kaggle still reports `RUNNING`; tracker and finalizer processes remain alive. There is no terminal receipt. The most recent exact artifact query at 03:50:16 UTC returned zero matching archives.


## 3B report/result consistency audit (2026-11-09T03:55:18.027184+00:00)

Cross-checked all five rounded final point estimates, 95% profile-block bootstrap intervals, and sample/group counts in `docs/report/R3.tex` against the independent local verification record. All match. The source evaluator uses 5,000 draws with seed 42. Overall result remains `FAIL_DIAGNOSTIC_GATES`: retain90 ROUGE-L change -3.058 pp, 95% CI [-5.110, -0.993] pp, failing the preregistered strict lower-bound requirement > -5 pp.


## 8B run recheck (2026-11-09T03:56:05.491830+00:00)

Kaggle remains `RUNNING` roughly 6 h 57 min after launch. Exact archive query returned exit code 0 with no ZIP. Both local monitor processes are alive; no terminal receipt exists.


## Notebook archive behavior audit (2026-11-09T03:58:37.561805+00:00)

Inspected the full Kaggle notebook. Cell 17 runs training with `subprocess.run(capture_output=True)` and calls `archive_run()` after that subprocess returns; it repeats this after evaluation and verification. Therefore, while training is still inside the subprocess, the published ZIP may remain the earlier smoke archive. This is expected packaging behavior, but Kaggle's `RUNNING` status still cannot establish optimizer progress.


## Pending run provenance recheck (2026-11-09T03:59:50.286113+00:00)

The local full notebook SHA-256 still matches the submitted-attempt manifest, and the protocol SHA-256 still matches the locked preregistered hash. This supports source integrity; it does not establish remote training completion.


## Verified wait (2026-11-09T04:01:47.072199+00:00)

Kaggle remains `RUNNING`; exact archive query returned no ZIP. The tracker and finalizer remain active; no terminal receipt is available.


## Verified wait (2026-11-09T04:05:17.137617+00:00)

Kaggle remains `RUNNING`; an exact artifact query at 04:03 UTC returned zero archives. Tracker and finalizer remain alive, and there is no terminal receipt.


## Interactive Kaggle run recheck (2026-11-09T04:11:28.263898+00:00)

Opened the version 2 Kaggle run in the signed-in Chrome session. The Logs panel reports RUNNING for 26,060.3 seconds on GPU T4 x2; the Logs panel output size is 0 B. The editor displays a 159 MB artifact ZIP from the smoke/archive cell, but the notebook refreshes that ZIP after the training subprocess returns. Therefore optimizer progress and 500-step completion remain unverified.


## Training archive visible; evaluation not yet verified (2026-11-09T04:18:34.682152+00:00)

Kaggle version 2 Logs show the kernel still RUNNING on GPU T4 x2 at 26,524.9 seconds. The log records a 1,167,392-byte archive after the one-step smoke at 201.8 seconds, then a 159,312,103-byte archive after the full-training subprocess returned at 4,022.3 seconds. The notebook checks for `TRAINING_COMPLETE` and exactly 500 steps before it starts evaluation. Since there is no later log and the kernel remains RUNNING, evaluation is a plausible current stage, but the training receipt and evaluation outputs are not yet downloadable or independently verified.


## Adapter evaluation duration anomaly check (2026-11-09T04:24:06.509945+00:00)

The adapter-evaluation subprocess remains opaque because the notebook captures its stdout. At the latest UI check, total run time was 26,524.9 s; the training-phase archive was logged at 4,022.3 s, leaving about 6.25 h with no newer archive or result. The already verified full 8B baseline evaluation processed 6868 scoring candidates and 1417 generation prompts on T4 x2 at batch sizes 1/1 in 4209.0 s (Kaggle terminal log at 4,285.3 s). The current adapter evaluation has the same row and batch counts and allows 200 new tokens. The elapsed time is now 5.35× the prior baseline runtime. Adapter overhead or environment variance is possible, but a stall is a credible risk; the job remains RUNNING, so I am preserving it and waiting for a terminal state rather than restarting it.


## Adapter evaluation visibility and runtime recheck (2026-11-09T04:30:00.057295+00:00)

Kaggle UI still reports `RUNNING` at 27,106.9 seconds (T4 x2), and its latest log line remains the 159,312,103-byte archive at 4,022.3 seconds. A fresh exact-artifact download query exited successfully but returned no files. The adapter-evaluation subprocess captures stdout; although the evaluator prints progress every 100 scored candidates and 25 generations, those messages are hidden until the subprocess exits. The evaluator writes its result JSON only after all scoring and generation finish, so neither progress nor partial metrics are currently retrievable. Time since the post-training archive is 23,084.6 seconds (5.48x the verified 4,208.96-second full-checkpoint evaluation). This is a real observability/runtime anomaly, not proof of failure. The 500-step receipt and evaluation result remain unverified; preserve the current kernel while the existing tracker continues to its deadline.


## Verified live wait (2026-11-09T04:32:05.492663+00:00)

Repolled the same Kaggle v2 run. UI reports `RUNNING` at 27,367.2 seconds on T4 x2; latest visible log remains the 159 MB archive at 4,022.3 seconds, with no error, traceback, or newer evaluation progress line. The current run still has no verified training receipt or evaluation result. This is a verified wait, not a completion or failure signal; the existing tracker/finalizer remain alive.


## Local streaming fallback prepared; v2 still running (2026-11-09T04:39:07.568995+00:00)

The same Kaggle v2 kernel remains `RUNNING` at 27,777.3 seconds (T4 x2); no new evaluation progress log, current result archive, or verified training receipt is available. The evaluator phase is 23,755.0 seconds since the last archive, 5.64x the prior full-checkpoint evaluation. I prepared a separate local-only package at `experiments/tofu_scaling/kaggle/npo_lora_8b_pilot_streaming_v1/`. Its verification passes: the locked protocol and evaluation code hashes are unchanged; the trainer differs only by a flushed progress record every 10 steps; the streamed runner preserves full logs and child exit codes in success/failure fixtures; all embedded Python sources compile and hashes match. It has not been uploaded or launched and did not use GPU quota. Do not run it concurrently with v2.


## Verified wait after fallback preparation (2026-11-09T04:40:03.523915+00:00)

Polled Kaggle API and the live Logs page again. Version 2 remains `RUNNING` at 27,849.5 seconds on T4 x2. The only visible artifact log is still at 4,022.3 seconds; no new evaluation progress or error is visible, and no result archive or 500-step receipt is available locally. This is 23,827.2 seconds since the archive update (5.66x the prior full-checkpoint evaluation runtime). The local streaming fallback passed its offline verification and remains unsubmitted. Existing tracker/finalizer are alive; continue the current run without launching a duplicate.


## Verified wait, same run (2026-11-09T04:41:25.837430+00:00)

Kaggle API and live Logs still show version 2 `RUNNING` at 27,899.3 seconds on T4 x2. No new evaluation progress line, terminal error, result archive, or verified training receipt is available. The last archive remains at 4,022.3 seconds; elapsed time since it is 23,877.0 seconds (5.67x the prior 8B baseline evaluation). The monitor and finalizer remain alive. The separately validated streaming fallback remains local and unused.


## Archived training-trace audit (2026-11-09)

The offline trace audit passed: 4,000 finite microbatch records, exactly eight per each of 500 optimizer steps. Mean forget NPO loss changed 13.8629 (step 1) to 0.2521 (step 500); mean retain token NLL changed 0.0034 to 1.0135. No intermediate evaluation scores exist, so an earlier utility-safe checkpoint cannot be inferred. The completed run remains diagnostic-only, protocol-misaligned, and fails the retain/overall utility gates. No new GPU run was launched. The corrected implementation fixtures pass, but a fresh benchmark/cohort and locked protocol are required before training. MUSE-News screening remains blocked pending model-term and QA/source-alignment review.
