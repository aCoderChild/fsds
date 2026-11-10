# SynthWorlds-SM R1 Kaggle pilot package

The ZIP contains the approved five-language AI-only-reviewed SFT corpus, English-only unlearning runners, and the question-only five-language evaluation set with separate gold labels for scoring. Translation review was AI-only; no fluent human validation was performed. This is a feasibility pilot, not a powered main-study release.

Attach this dataset to the private Kaggle notebook `bunnoob2005/experiments`. The notebook checks integrity, dependencies, GPU, Kaggle secret retrieval, and gated Hugging Face access. Training, retain-only training, unlearning, and evaluation are all disabled by default. Run the token/model-access preflight in the Kaggle Notebook Editor with `HF_TOKEN` attached under Add-ons → Secrets: Kaggle API `kernels push` batch runs do not attach notebook Secrets.
