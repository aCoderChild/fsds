# Qwen14B AI-screened positive consistency controls

This 16-row diagnostic reuses the same balanced FR/DE/HI/TH atomic/composition translations as the relation-level negative controls, but restores each original English source relation. Every translation and back-translation had previously passed an AI-only review. They are not human-gold translations, so this is a reviewer-consistency check, not an unbiased estimate of false-rejection rate or translation accuracy.

The initial unconstrained run exposed an output-format defect: the model duplicated `reason` in 12 JSON objects and omitted it in 4, so none of the raw responses was schema-valid. Those artifacts are retained under `remote_output/version_1/`.

Version 2 uses the exact production Qwen14B system prompt and one-row input fields, the pinned model revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, per-row generation for Kaggle T4 memory, and the same one-item JSON schema as the production reviewer. It enforces the schema during Transformers decoding with `lm-format-enforcer==0.11.1`, runs 4-bit NF4 weights through pinned `bitsandbytes==0.50.2` to fit the T4, and rejects duplicate JSON keys during parsing. This matches the production prompt and response contract, but uses a quantized, schema-constrained Transformers backend rather than the production Ollama service; results must be interpreted with that backend difference recorded.

From the repository root:

```bash
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_positive_controls/build_controls.py
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_positive_controls/prepare_kaggle.py
kaggle kernels push -p reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_positive_controls
kaggle kernels output bunnoob2005/synthworlds-qwen14b-positive-controls-2026-11-04 -p reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_positive_controls/remote_output/version_2
kaggle kernels pull bunnoob2005/synthworlds-qwen14b-positive-controls-2026-11-04 -p reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_positive_controls/remote_output/version_2/source
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_positive_controls/verify.py
```

Even 16/16 semantic passes would show only consistency with prior AI review on this selected set. The set is selected for earlier AI passes, shares reviewer-model lineage, and cannot establish ground-truth translation quality. It cannot release the training corpus by itself.

The recorded Kaggle version 3 run was schema-valid for all 16 rows. The strict reviewer accepted 15/16 and flagged one Thai composition prompt because the English back-translation used “partner” where the source said “spouse.” This ambiguity is retained as a conservative flag; it is not resolved as a translation error without a fluent Thai reviewer. The run verifies operational decoding and source binding, but the 15/16 consistency result is imperfect and is not evidence of translation accuracy.
