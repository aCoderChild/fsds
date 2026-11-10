# Qwen14B relation-level mismatch controls

This private Kaggle diagnostic checks whether the final Qwen2.5-14B review model rejects near-mismatch translations when one English relation or event phrase is changed and the target-language text is held fixed.

## Scope

- 16 controls: 2 each for French, German, Hindi, and Thai, split evenly between atomic and composition questions.
- Each control is derived from a translation that had an existing AI-only two-model and round-trip pass. Entity names are masked.
- The English source is changed by one relation/event phrase; the target translation is unchanged.
- No answer keys, entity names, or split labels are supplied to the reviewer.
- This tests relation-level mismatch rejection only. These examples are not human gold translations and this result cannot establish translation fluency, certify the corpus, or release SFT data.
- Predeclared diagnostic rule: all 16 mismatches must be semantically rejected with zero malformed outputs to proceed to a larger calibration. Any pass or schema error fails this screen. A pass is not approval of any translation.

## Reproduce and verify

From the repository root:

```bash
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls/build_controls.py
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls/prepare_kaggle.py
kaggle kernels push -p reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls
kaggle kernels output bunnoob2005/synthworlds-qwen14b-hard-negative-2026-11-04 -p reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls/remote_output
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_hard_negative_controls/verify.py
```

The run pins Qwen2.5-14B-Instruct revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` and writes a checkpoint after each example. The verifier checks the exact input hash, model revision, row identity, source text, translation hash, output schema, and output hash.

The accompanying `control_manifest.json` records the positive source/review provenance and explicitly labels the controls as diagnostic only.
