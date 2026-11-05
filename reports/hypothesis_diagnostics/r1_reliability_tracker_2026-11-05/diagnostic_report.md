# R1 reliability tracking — 5 November 2026

**Observed:** 2026-11-05 07:18 UTC. **Stage:** training-data release.

## Current status

- The R1 Kaggle notebook, 8B runtime smoke, and seven recent translation diagnostic kernels all report `COMPLETE`; no job is running among those nine checked kernels.
- The 35 offline pipeline fixtures pass. Prepared-source alignment also passes for 488 canonical SFT IDs. These are code/data-integrity checks, not training results.
- The source-bound assembler still fails closed: 5/117 forget cases, 3/39 retain cases, and 0/44 training-only composition chains meet the current translation gate. It wrote neither `corpus.jsonl` nor `manifest.json`.
- The pinned Llama 3.1 8B runtime-only smoke receipt verifies one synthetic LoRA optimizer step and one synthetic NPO update. It loaded no research data and is not an experiment.

## Latest translation diagnostics

| Candidate path | Outcome | Release effect |
|---|---:|---|
| Targeted 187-row screen | 36 joint passes / 170 reviewed; 17 blocked | Quarantined; none approved |
| NLLB 148-row screen | 3 joint passes / 23 reviewed; 125 blocked | Quarantined |
| Segmented retry of 125 | 0 semantic passes / 124 reviewed; 1 blocked | Quarantined |
| Marker-safe 18-row retry | 1 pass, 2 flags, 15 blocked | Quarantined |
| Qwen retry of 15 | 5 passes, 9 flags, 1 blocked | Quarantined |
| Key-constrained one-row retry | 1 AI-only pass | Quarantined |

All these checks are AI-only; the pass labels do not certify translation fluency. No candidate was written into the canonical translation source or SFT release.

## Reliability conclusion

The implementation and synthetic 8B runtime path are currently verifiable. The full R1 experiment is **not running** and is **not ready to run**: translation coverage remains far below the frozen release requirement. The prior one-seed Llama 3.2 1B analysis remains the only hypothesis diagnostic, and its baseline competence gate failed in all four target languages. Starting the 8B main study now would not produce a reliable answer.

Reproduction details and SHA-256 bindings are in [`verification.json`](verification.json).
