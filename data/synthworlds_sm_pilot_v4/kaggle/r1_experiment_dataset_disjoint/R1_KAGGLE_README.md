# SynthWorlds-SM R1 exploratory composition follow-up

This package preserves the approved v3 split and evaluation set, and adds 85 source-verified Type-B three-edge practice chains to the prior 50 Type-A train-only chains. The new extension has 340 English prompt-answer rows and 340 rows per language (Q1, Q2, Q3, and full composition), resulting in 671 SFT rows per language. All new train-only entities, path facts, answers, and exact questions are disjoint from the frozen 129-case evaluation set. An independent verifier report and source checksums are included.

Translation review is AI-only automated consistency checking. It uses NLLB-200 for translation and same-family back-translation plus protected-text, provenance, and format checks. No fluent human reviewed the translations, and these checks do not establish semantic equivalence. Treat results as exploratory feasibility-pilot evidence.

Run `kaggle/sft_lora_pilot.ipynb` from the Kaggle Notebook Editor with the `HF_TOKEN` secret attached. Notebook Secrets are available from the interactive editor session; Kaggle CLI/API `kernels push` jobs do not receive them. The notebook checks GPU, secret/model access, corpus integrity, and baseline bilingual competence. Expensive training switches default to false. Run baseline evaluation first; the notebook gates English-only unlearning on the predeclared bilingual competence floor.

Holdout facts are never used for training or unlearning. Retain-composition prompts overlap training by design. The prior v3 run did not pass the competence gate, so the added curriculum must be evaluated before unlearning is considered.
