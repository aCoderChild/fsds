# Current state audit

Run from the repository root:

```bash
python reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-03/verify_current_state.py
```

This audit writes its report and local evidence files, and makes a read-only Kaggle kernel-status query. It rescales archived predictions in score-only mode, checks source and result hashes, runs the release gate and synthetic fixtures, compiles the current notebook, and does not modify data or model outputs, download model weights, or launch training. The expected current v5.1 release result is blocked until the translation assembler passes.

- `verification.json`: machine-readable checks and evidence hashes.
- `diagnostic_report.md`: human-readable conclusion and current blockers.
- `evidence/`: Kaggle setup logs and exact notebook/metadata snapshots used by the audit.
- `corpus_reliability_audit.md`: correction to the repair-candidate integrity checks and the present corpus readiness decision.
- `translation_coverage_diagnosis.py` / `.json`: recompute source-bound candidate integrity and coverage; repair candidates remain diagnostic only.
- `source_bound_calibration.md`: results and verification of the 32-row corrected repair and round-trip review calibration.
- `evaluation_reliability_audit.md`: evaluator corrections, statistical edge cases, and offline regression checks.
- `training_source_integrity_audit.md`: the composition-answer provenance finding and the new SFT/unlearning source validation gate.
