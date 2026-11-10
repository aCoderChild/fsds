# SynthWorlds-SM disjoint composition baseline — 2026-11-02

## Run status

The interactive Kaggle Notebook Editor run completed SFT for Llama 3.2 1B-Instruct, a retain-only reference run, and baseline generation/scoring. The run manifest records seed 42, three epochs, answer-only loss masking, and AI-only translation review. Kaggle retrieved the attached Hugging Face secret and verified gated model access in the interactive editor. It did not expose or package the token. Kaggle CLI/API-launched runs still cannot receive notebook Secrets.

The bilingual baseline gate failed, so unlearning and post-unlearning evaluation were not run. This is a scientifically appropriate stop, not a training failure. These outputs do not answer the cross-language unlearning hypothesis.

Kaggle emitted non-fatal dependency-resolver warnings for its unrelated Gradio/Diffusers packages, a Transformers `torch_dtype` deprecation notice, and a warning that sampling-only `temperature`/`top_p` settings in the model generation config are ignored during greedy decoding. Evaluation called generation with `do_sample=False`; no generation errors were recorded. The PEFT smoke check passed in a fresh process before training.

## Verified evidence

- Composition SFT: 1,655 rows, 331 per language; 765 forget atomic-fact rows, 390 retain rows, and 500 rows from 50 training-only chains. Three epochs; seed 42.
- Retain-only reference: 890 rows (retain plus train-only), same model, epochs, and seed.
- Baseline: 129 cases × 5 languages × 2 probes × 3 checkpoints = 3,870 predictions. Zero generation failures and zero deterministic invalid-output flags.
- Forget-set full-SFT direct / multi-hop exact accuracy: EN 88.3% / 10.4%; FR 84.4% / 9.1%; DE 77.9% / 18.2%; HI 83.1% / 14.3%; TH 83.1% / 15.6%.
- Eight English forget cases were correct on both probes. Paired precompetent cases / entity groups: FR 2/2, DE 4/3, HI 4/3, TH 3/3. Required pilot floor: at least 15 cases and 8 groups in every language.
- Exact forget multi-hop questions were held out, but their atomic support facts were in SFT. This measures composition generalization over trained facts. Retain composition prompts overlap training by design; holdout facts were excluded from training.
- AI-only review was authorized and recorded. No fluent-human translation review was performed. The one-seed 1B pilot is not the planned powered 8B study.

## Files

- `r1_composition_disjoint_reproducibility_bundle.zip`: all 32 indexed source artifacts plus `artifact_index.json`, including both adapters, run manifests, corpus, predictions, scores, and diagnostics.
- `extracted/r1_composition_diagnostics_disjoint/`: baseline reports and row-level predictions/metrics.
- `extracted/r1_composition_sft_disjoint/run_manifest.json` and `extracted/r1_composition_retain_only_disjoint/run_manifest.json`: training provenance.
- `extracted/project_input/`: exact corpus and manifest used in the Kaggle run.
- `uploaded_kaggle_package_manifest.json`: source package inventory and hashes.
- `verification.json`: local integrity and gate summary.

All files listed in the ZIP index were SHA-256 checked against the index; the ZIP CRC is clean. The three uploaded runner scripts match the local copies by SHA-256.
