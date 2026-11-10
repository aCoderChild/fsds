# Live R1 run check — 2026-11-05T14:50:01+00:00

## Status

**The main R1 experiment is not running. It remains blocked at data preparation.** The 8B environment passed a synthetic runtime smoke; no research corpus was used. Five scoped Kaggle kernels were queried and are all `COMPLETE` (none is currently running).

| Stage / evidence | Current number | What it means |
|---|---:|---|
| Source-verified main release | 5/117 forget; 3/39 retain; 0/44 training-only chains | Release assembler fails closed; no main corpus or manifest was emitted |
| Prepared source alignment | 488 canonical SFT IDs | ID/source integrity only; does not certify translation quality |
| Targeted translation gap screen | 36/187 passed (19.3% descriptive); 151 flagged | Quarantined AI-only candidates, not training data |
| Cross-review / mismatch checks | 154/187 disagreements; 2/15 and 1/16 altered controls accepted | Current AI review is not calibrated well enough to approve the corpus |
| Offline pipeline fixtures | 36/36 passed | Expected code behavior on synthetic fixtures only |
| 8B T4×2 runtime smoke | 1 synthetic LoRA step; 1 synthetic NPO update | Runtime compatibility only; zero research rows loaded |
| V4 pilot independent rescore | 7/7 output files byte-identical; 7,740 predictions | Scoring result is reproducible; pilot remains exploratory |
| Exact pilot cluster intervals | 12/12 primary and 12/12 secondary intervals include/touch zero | No reliable hypothesis conclusion; only 5–7 groups per contrast |

## Exploratory pilot estimates

The sole hypothesis-adjacent completed run is a one-seed Llama 3.2 1B pilot. Its baseline gate requires at least 15 bilingual-precompetent cases across 8 entity groups per target language; observed counts are FR 10/6, DE 9/5, HI 9/6, and TH 11/7. All four fail.

| Target language | Cases / groups | Primary target change, pp (familywise 95% interval) |
|---|---:|---:|
| French | 10 / 6 | 0.0 [0.0, 0.0] |
| German | 9 / 5 | −11.1 [−80.0, 0.0] |
| Hindi | 9 / 6 | −11.1 [−66.7, 0.0] |
| Thai | 11 / 7 | −18.2 [−85.7, 0.0] |

These values are shared across GA, GD, and NPO in this small pilot. All 12 method-language primary intervals touch or include zero. The secondary target-minus-English DID also has 12/12 such intervals; its outer interval range is −71.4 to +100.0 pp. These wide intervals are not evidence of no transfer. They are too imprecise to support a conclusion.

The pilot's full-set final multi-hop baseline was only 18.2–20.8%. Post-unlearning, direct recall was correct in 0–2 of 9–11 matched cases, while final answers remained correct in 8–10. That pattern is a useful diagnostic, but it does not show valid reasoning or a clean causal effect of unlearning.

## Reliability conclusion and next gate

Scoring, the 36 synthetic software fixtures, and exact pilot interval calculations reproduce. The main experiment is **not yet runnable reliably**: the approved multilingual training release does not exist, the screened translation candidates remain AI-only/quarantined, and the 8B smoke is not an experiment. Keep the release and competence gates unchanged; next, resolve translation quality and coverage, rerun the source-bound assembler, then run the three-seed 8B baseline. Do not run GA/GD/NPO until that baseline meets the frozen gate.

The 100,000-draw bootstrap was independently reproduced by exact enumeration, with 0.0 pp maximum endpoint difference. The exact intervals still have approximate coverage at only 5–7 clusters.

## Reproduction anchors

- Fresh scorer output hashes and run metadata: [JSON receipt](live_run_check_2026-11-05_1450_utc.json).
- Full pilot result tables: [diagnostic results](../r1_reliability_tracker_2026-11-05/pilot_diagnostic_results.md).
- Exact cluster-bootstrap table and code: [sensitivity report](../r1_reliability_tracker_2026-11-05/pilot_exact_cluster_sensitivity.md), [analysis script](../r1_reliability_tracker_2026-11-05/pilot_exact_cluster_sensitivity.py).
- Design/code mapping and statistical caveats: [protocol implementation audit](protocol_implementation_audit.md).
