# SynthWorlds-SM Kaggle SFT feasibility pilot

This package contains the approved-for-pilot five-language atomic corpus, retained-chain composition corpus, and Kaggle runner. It is approved for an AI-only feasibility pilot. It is **not human-validated multilingual data**: no fluent human reviewed these translations, and the draft translator and back-translation audit use the same NLLB-200 distilled 600M model family. Back-translation is a consistency check, not proof of fluency or semantic accuracy.

The composition corpus adds final-answer examples from retained chains only; it excludes forget and holdout composition chains. Retain prompts intentionally overlap evaluation, so only forget-split performance is used for unseen-chain generalization. The pilot split approval is a delegated AI technical review for this limited 129-case feasibility split; it is not researcher sign-off or a powered main-study design.

In Kaggle, attach this dataset to `sft_lora_pilot.ipynb` and run it from the Notebook Editor with the `HF_TOKEN` secret attached. Kaggle CLI/API kernel runs do not attach Notebook Secrets. Run preflight first. All training stages remain disabled by default.
