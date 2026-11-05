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

## Template-reuse audit — 5 November 2026

I checked whether reviewed translations could cover the same frozen cohort by reusing a translation across prompts with the same entity-slot pattern and source relation signature. The audit verified all 1,952 original review rows against their hash-bound source files and all 799 flagged-repair rows against their source maps and candidate-text hashes. The repair review produced 445 AI-only passes; none has human fluency certification.

| Coverage view | Forget cases | Retain cases | Composition chains |
|---|---:|---:|---:|
| Existing source-screen passes only | 5/117 | 3/39 | 0/44 |
| Union with hash-verified repair candidates | 29/117 | 10/39 | 1/44 |
| Optimistic relation-aware template-reuse ceiling | 32/117 | 12/39 | 1/44 |

The ceiling assumes one passing translation can be reused for every prompt with the same normalized entity slots and relation signature, without re-reviewing each rendering. It adds at most 12 atomic prompt rows across the four languages. Relation-aware grouping finds 274 templates among 312 atomic prompts and 176 among 176 composition prompts; text-only grouping would merge 12 groups with different relation signatures. This route cannot supply the frozen R1 corpus, and it does not justify weakening the release gate.

The audit is reproducible with [`template_reuse_ceiling_audit.py`](template_reuse_ceiling_audit.py); machine-readable counts and input hashes are in [`template_reuse_ceiling.json`](template_reuse_ceiling.json). Both artifacts are diagnostic only. No corpus was assembled and training remains stopped.

## Reliability conclusion

The implementation and synthetic 8B runtime path are currently verifiable. The full R1 experiment is **not running** and is **not ready to run**: translation coverage remains far below the frozen release requirement. The prior one-seed Llama 3.2 1B analysis remains the only hypothesis diagnostic, and its baseline competence gate failed in all four target languages. Starting the 8B main study now would not produce a reliable answer.

Reproduction details and SHA-256 bindings are in [`verification.json`](verification.json).
