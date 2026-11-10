# Kaggle Llama 3.1 8B runtime smoke — 2026-11-04

> **Historical status note:** This file records the initial 4 November attempt, which stopped at the gated-model access check. After access was approved, a corrected interactive run passed on 5 November: the pinned 8B weights loaded on T4 x2 and one synthetic LoRA step plus one synthetic NPO update completed. See the [verified 5 November receipt](../../../../experiments/synthworlds_r1/kaggle_8b_runtime_smoke/remote_output/interactive_smoke_v8_passed_runtime_only_2026-11-05/README.md). This later smoke verifies runtime compatibility only; it is not research SFT, unlearning, or a hypothesis result.

## Outcome

**BLOCKED before model load. No 8B weights were downloaded and no training step ran.** Kaggle returned an authorization error while Transformers requested the model configuration at immutable revision `0e9e39f249a16976918f6564b8830bc894c89659`:

> Access to model `meta-llama/Llama-3.1-8B-Instruct` is restricted and this account is not in the authorized list.

The `HF_TOKEN` secret was present in Kaggle and an authenticated metadata lookup had succeeded earlier; neither proves that this account can read gated model files. The actual configuration-file request is the decisive check and it failed.

## Runtime checks that did pass

- Kaggle interactive notebook session: GPU T4 x2; each T4 reports 14.6 GiB usable in this session.
- The R1 pinned environment resolved and was installed: Transformers 5.22.1, PEFT 0.21.2, Accelerate 1.15.0, bitsandbytes 0.50.2.
- A deliberately tiny bitsandbytes NF4 `Linear4bit` forward/backward check passed on a T4. This only checks the quantized kernel; it does **not** validate loading or fine-tuning an 8B model.
- Available `/kaggle/working` space before model fetch was about 19.5 GB (20 GB filesystem).
- The Kaggle session was stopped after the failed access check to conserve GPU quota.

## What was attempted

A self-contained smoke runner was copied to the interactive Kaggle session and hash-checked against the local source before launch:

- Local runner: `experiments/synthworlds_r1/scripts/kaggle_8b_runtime_smoke.py`
- SHA-256: `b2a20ed1d7e55b2dd3c5e1856138bf241f4732c49df115d4425384f21a67c8f4`

The runner would verify the exact dependency pins and Hugging Face file access, load Llama 3.1 8B in NF4, attach the R1 LoRA configuration, and take one synthetic masked-loss training step. It stopped at gated file access, before model loading, so none of those latter steps are verified.

## Interpretation and next gate

This is a **runtime access blocker**, not a GPU out-of-memory result. Request/obtain access to `meta-llama/Llama-3.1-8B-Instruct` for the Hugging Face account behind the Kaggle token, then rerun only this bounded smoke. Do not start the research training run on this evidence.

The existing R1 training corpus/release gate also remains independent and blocking. This smoke does not change the status of the prior one-seed 1B feasibility result and provides no hypothesis result.
