# Llama 3.1 8B runtime smoke — PASS (runtime only)

**Date:** 2026-11-05  
**Kaggle notebook:** [SynthWorlds Llama31 8B Runtime Smoke](https://www.kaggle.com/code/bunnoob2005/synthworlds-llama31-8b-runtime-smoke-2026-11-04/edit)  
**Status:** `PASS_RUNTIME_ONLY`  
**Scope:** synthetic smoke only; this is not SFT, unlearning, or a hypothesis result.

## What passed

- Exact gated checkpoint `meta-llama/Llama-3.1-8B-Instruct`, immutable revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- The two-GPU Kaggle runtime (2 × Tesla T4) loaded the model in NF4 4-bit mode.
- The pinned stack was Transformers 5.22.1, PEFT 0.21.2, Accelerate 1.15.0, and bitsandbytes 0.50.2.
- One synthetic LoRA Trainer optimizer step completed with finite loss `0.0247592721`.
- One synthetic NPO optimizer update completed with finite loss `13.8629417`; the reference model loaded on GPU 1.
- The shared single-render, fast-tokenizer offset mask supervised two assistant answer tokens. The preceding CPU tokenizer/mask preflight also passed without downloading weights.
- No research corpus was loaded. The receipt binds the run to runner SHA-256 `7dac2115db3c53ba1fd2b57e979ac5400446d7f7e0081ac70612e032c8cd7951`.

Peak allocated GPU memory reported by the receipt was 10.47 GiB on GPU 0 and 5.61 GiB on GPU 1. The Kaggle session was stopped after collecting the receipt to preserve GPU quota. The interactive draft was reverted to the saved v8 source with all run flags set to `False`; the prepared local notebook also passes its safe-to-publish verifier.

## Verification

`verification.json` is the local verifier output. Reproduce the receipt check from the repository root:

```bash
python experiments/synthworlds_r1/kaggle_8b_runtime_smoke/verify_receipt.py \
  experiments/synthworlds_r1/kaggle_8b_runtime_smoke/remote_output/interactive_smoke_v8_passed_runtime_only_2026-11-05/r1_8b_runtime_smoke.json
```

The first rerun attempt loaded the model and completed the synthetic NPO update but failed while writing the receipt because Trainer history did not contain a `loss` key under the chosen logging mode. That failed attempt is retained separately. The runner was corrected to use the returned `train_loss` metric, validated, and rerun successfully. An initial retry also hit CUDA out-of-memory while the notebook retained stale model references; restarting the kernel cleared them, and the successful retry reused Kaggle's local model cache.

## Interpretation and limits

This result verifies that the pinned model, package stack, assistant-only masking, synthetic LoRA step, and synthetic NPO update work together on the selected Kaggle T4 × 2 runtime. It does **not** validate the research corpus, multilingual training data, research SFT, unlearning, model competence, or the cross-lingual hypothesis. The main experiment remains gated on the source-verified multilingual release and baseline competence criteria.
