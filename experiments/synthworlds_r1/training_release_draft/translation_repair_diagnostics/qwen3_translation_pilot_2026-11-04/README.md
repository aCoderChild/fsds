# Qwen3 translation pilot

This is a bounded 16-row translation-generation pilot, sampled as two atomic and two composition questions per language (French, German, Hindi, Thai). It contains masked English question text only: no answers, facts, or split assignments.

The private Kaggle kernel uses `Qwen/Qwen3-14B` at a pinned revision with 4-bit NF4 weights. The completed run used a Tesla T4 and produced 16/16 outputs. The pulled Kaggle source is hash-identical to the reviewed local generator.

The verifier checks source/translation hashes, exact placeholder preservation, Hindi/Thai script, row coverage, and runtime provenance. It does not assess translation quality. The independent blinded Qwen2.5-14B plus NLLB screen returned **4 pass, 9 flag, 3 blocked** (French 1/2/1, German 1/2/1, Hindi 2/2/0, Thai 0/3/1). This AI-only outcome is diagnostic, not human certification; it is inadequate to scale this route. No pilot output was merged into the authoritative review files or training corpus. Both Kaggle kernels are complete and stopped. Neither the generator receipt nor the review receipt approves training data.

Reverify the stored outputs with `python -B verify.py` and `python -B verify_review.py`. To independently retrieve the submitted source files, run from this directory:

```sh
kaggle kernels pull bunnoob2005/synthworlds-qwen3-translation-pilot-2026-11-04 -p generation_remote_source
kaggle kernels pull bunnoob2005/qwen3-pilot-qwen14-review-2026-11-04 -p review_remote_source
```

The verifiers compare those pulled sources to the reviewed local scripts and fail if they differ. Exact receipts and pulled files from the completed runs are stored beside this README.

## Blinded direct paired re-screen

A separate completed private Kaggle run reviewed 32 existing candidates (16 paired questions) plus 16 unlabeled known-mismatch controls using direct English-to-target comparison, without NLLB back-translation. Result: Qwen3 candidates 11/16 pass, alternate AI candidates 10/16 pass; 15/16 mismatch controls were rejected and one Hindi editor/director error was accepted. All 48 rows were processed with preserved placeholders. These are AI-only diagnostic labels and no candidate was promoted. The reviewer used two T4 GPUs with the output head offloaded to CPU. The output also lacks a completion timestamp; the Kaggle completion status and downloaded logs are recorded in the run receipt.

Reverify the exact run locally with `python -B verify_paired_direct_review.py`. See `paired_direct_review_verification.json`, `paired_direct_review_kaggle_run.json`, and the pulled Kaggle source/output folders. This diagnostic does not change the training release gate.
