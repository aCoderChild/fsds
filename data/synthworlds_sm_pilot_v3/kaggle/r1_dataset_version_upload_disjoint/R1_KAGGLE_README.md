# SynthWorlds-SM R1 feasibility experiment

This package contains the five-language atomic SFT corpus, the historical retained-chain composition corpus, and the new evaluation-disjoint composition corpus with 50 train-only chains (100 rows per language). The disjoint training rows were kept separate until the declared AI-only review passed. No fluent human reviewed the translations. Results are feasibility-pilot evidence, not human-validated linguistic findings.

Run `kaggle/sft_lora_pilot.ipynb` from the Kaggle Notebook Editor with the `HF_TOKEN` secret attached. Notebook Secrets are available from the interactive editor session; Kaggle CLI/API `kernels push` jobs do not receive them. The notebook explicitly checks GPU, secret/model access, corpus integrity, and baseline bilingual competence. All expensive training switches default to false.

The disjoint extension excludes evaluation entities, answers, questions, and selected chain facts. Unlearning targets only English forget cases where the SFT checkpoint answered both direct and full multi-hop probes correctly; its manifest is checksum-bound to that baseline and its case metrics. Holdout is never used. Translation review is AI-only. Back-translation uses the same NLLB model family, so it is a consistency check rather than independent proof of fluency or semantic correctness.
