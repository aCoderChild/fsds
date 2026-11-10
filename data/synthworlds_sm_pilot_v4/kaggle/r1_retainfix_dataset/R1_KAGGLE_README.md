# SynthWorlds-SM R1 exploratory composition follow-up

This package preserves the 129-case evaluation split and the 3,355-row five-language composition corpus: 671 SFT examples per language. The corpus adds 85 source-verified Type-B three-edge practice chains to the original 50 Type-A training-only chains. New Type-B entities, path facts, answers, and exact questions are disjoint from the frozen evaluation cases; the packaged verifier and checksums record that validation.

The v4 SFT, retain-only reference, baseline evaluation, and exploratory GA/GD/NPO runs have completed in Kaggle. The predeclared bilingual competence gate failed (FR 10 cases/6 groups, DE 9/5, HI 9/6, TH 11/7; required 15/8). The saved GD/NPO exploratory run used an earlier retain sampler that reused only 15 of 26 available English retain rows. Its outputs are preserved as historical diagnostics, not as corrected-run evidence. The updated runner cycles through every retain row before reuse.

Translation review is AI-only consistency checking with NLLB-200 and same-family back-translation, protected-text, provenance, and format checks. No fluent human reviewed these translations; these checks do not establish human fluency or semantic equivalence.

Use `kaggle/sft_lora_pilot.ipynb` in the Kaggle Notebook Editor with the `HF_TOKEN` secret checked. Kaggle Notebook Secrets are available to the interactive editor session; API/CLI-launched kernels do not receive that secret. The notebook's regular unlearning path remains blocked when the competence gate fails. The separate exploratory cell is opt-in and refuses to run if any preflight error besides the exact gate failure is present. To repeat only the corrected unlearning stage, the same Kaggle session must contain the hash-matched v4 SFT/retain adapters and baseline target manifest; otherwise rerun SFT, retain-only, and baseline cells first.

Holdout facts are never used for training or unlearning. Retain-composition prompts overlap training by design. All results remain one-seed, 1B feasibility diagnostics; they do not answer the hypothesis conclusively or substitute for the planned powered study.
