# SynthWorlds-SM R1 feasibility experiment

This package contains the AI-only-reviewed five-language atomic SFT corpus, a conditional composition-SFT variant trained only on retained chains, the English-only unlearning runner (GA/GD/NPO), question-only multilingual evaluation prompts, and separate gold labels used by scoring. No fluent human reviewed the multilingual translations. Results are AI-only pilot evidence, not human-validated linguistic findings.

Run `kaggle/sft_lora_pilot.ipynb` from the Kaggle Notebook Editor with the `HF_TOKEN` secret attached. The Notebook Editor successfully retrieves this secret; the CLI/API `kernels push` route cannot configure Notebook Secrets. All training switches default to false. The composition follow-up adds 26 retained multi-hop examples per language; forget and holdout composition examples are excluded, and only forget-split evaluation measures unseen-chain generalization.

The translation audit covers completeness, source-question names, placeholders, and answer leakage separation. It cannot certify fluent human language quality.
