# SynthWorlds-SM R1 corrected exploratory results

## Verification status

The current artifact is `r1_composition_retainfix_verified_results_20261103_v3.zip`. Local verification reports a clean ZIP CRC check, 78 unique archive paths, and matching byte counts and SHA-256 hashes for all 77 files listed in `artifact_index.json`. The archive SHA-256 is recorded in `LOCAL_VERIFICATION.json`.

The V3 bundle contains the corrected GD/NPO retain-sampler source (`scripts/unlearn_lora.py`, SHA-256 `0eb93e4448f6a9295f2a2d6be62eb89e3bc90202eff005c62d7829d70e78cd8d`), rerun adapters and manifests, and rescored multilingual predictions. Each corrected GD/NPO manifest records 26 distinct English retain rows used over 45 steps; both record 15 English forget rows and zero holdout rows. The evaluator contains 7,740 unique prediction keys: 1,290 for each of pretrained, SFT, retain-only, GA, GD, and NPO. It reports zero generation failures and zero deterministic output-invalid flags.

## Provenance limitation

The Kaggle run manifests do not record the unlearning script's hash. The source/output association was reconciled after the run using the manifests, scheduler metadata, and hash-verified corrected source; it is not cryptographic proof that the exact bundled file was loaded by the training process. See `extracted/corrected_unlearning/source_code_provenance.json`. The earlier V2 archive and extraction are retained as historical audit artifacts and are superseded by V3.

## Interpretation limits

The bilingual competence gate failed: FR 10 cases/6 groups, DE 9/5, HI 9/6, and TH 11/7, against a minimum of 15 cases and 8 groups per language. These are one-seed, 1B-model exploratory results, not confirmatory evidence. The reported cross-language intervals are wide and include zero, so this run does not establish whether English unlearning affects non-English multi-hop performance.

Translations received AI-only review. No fluent-speaker review was performed, so this is not human linguistic validation. The HF secret was retrieved only inside the interactive Kaggle Notebook Editor session; API-launched jobs did not receive notebook secrets. No token value is included in the artifacts.

## Files

- `r1_composition_retainfix_verified_results_20261103_v3.zip`: current verified bundle.
- `extracted/`: V3 bundle extraction.
- `../../../archive/hypothesis_diagnostics/r1_synthworlds_v4_retainfix_corrected_2026-11-03_v2/r1_composition_retainfix_corrected_results_20261103_v2.zip`: retained V2 historical bundle. Its redundant unpacked copy was removed after all archive members matched by SHA-256.
- `extracted/corrected_evaluation/diagnostic_report.md`: detailed scores and cross-language contrasts.
- `extracted/corrected_evaluation/verification.json`: row-count and generation-failure audit.
- `extracted/corrected_unlearning/{ga,gd,npo}/run_manifest.json`: training configuration, row IDs, gate status, and run status.
- `LOCAL_VERIFICATION.json`: local archive CRC, indexed file hashes, prediction keys, and run-manifest checks.
