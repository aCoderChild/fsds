# Qwen14B full-review-pipeline hard controls

This is the follow-up to the standalone Qwen14B relation-control check. It reuses the same 16 masked source/candidate mismatches and now supplies the corresponding English back-translation, which the production adjudicator also receives.

The set contains two controls per language and question kind across French, German, Hindi, and Thai. Each target translation and its back-translation come from an AI-screened pair for the original English relation; the English source is changed by one relation/event phrase. The labels and split assignments are not provided to the reviewer.

The fixed screen requires semantic rejection of all 16 controls with zero malformed decisions. Passing this screen supports only further reviewer calibration; it does not approve translations or authorize dataset release. The underlying translations and back-translations are AI-screened, not human gold.

From the repository root:

```bash
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_hard_controls/build_controls.py
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_hard_controls/prepare_kaggle.py
kaggle kernels push -p reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_hard_controls
kaggle kernels output bunnoob2005/synthworlds-qwen14b-pipeline-controls-2026-11-04 -p reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_hard_controls/remote_output/version_2
kaggle kernels pull bunnoob2005/synthworlds-qwen14b-pipeline-controls-2026-11-04 -p reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_hard_controls/remote_output/version_2/source
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_hard_controls/normalize_review_output.py
python -B reports/hypothesis_diagnostics/r1_current_state_audit_2026-11-04/evidence/qwen14b_pipeline_hard_controls/verify.py
```

The run pins Qwen2.5-14B-Instruct revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`. The verifier checks input and output hashes, model revision, row keys, exact source and translation/back-translation hashes, and that each source change is exactly the selected relation/event phrase.

Version 1 failed before decisions because four-row generation batches exceeded T4 memory. Version 2 scored one example per call and completed all 16 in about 601 seconds. It rejected all 16 known relation mismatches and marked all 16 target translations fluent (2/2 in every language × question-kind cell).

The remote Transformers wrapper expected a `{"decisions": [...]}` response, while unconstrained generation returned valid JSON keyed by the sole control ID. The raw remote status therefore records 16 wrapper parse errors. `normalize_review_output.py` preserves the raw file, accepts only an exact one-key object with the expected control ID and exactly the three required decision fields, and changes no model decision values. The verifier binds the raw output, local normalizer, Kaggle-pulled script, run log, inputs, and source review files by hashes. This local adapter does not test the production Ollama JSON-schema constrained-decoding path.

Before submission, I checked by parsing both scripts that the Kaggle run's system adjudication prompt exactly matches the production Qwen14B prompt. The local construction check also re-derived every masked source, target translation, and back-translation from the hash-bound source review files. This 16-row hard-control result still does not estimate production precision, recall, or fluency accuracy; it cannot approve the 799 flagged rows.
