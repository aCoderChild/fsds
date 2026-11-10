# Kaggle v7 interactive runtime smoke failure

This record transcribes the Kaggle editor output observed on 2026-11-05. The underlying raw `r1_8b_runtime_smoke.json` was not downloaded, so this is an observation record, not a receipt accepted by the project receipt verifier.

The output showed the exact pinned Llama 3.1 8B Instruct checkpoint downloading 16.1 GB and loading 291/291 shards, then raised `Assistant-only loss mask does not match the tokenizer chat template`. The archived runner source validates that this assertion precedes Trainer construction; no optimizer or NPO update ran.

The run is kept as a failure. The next safe sequence is v8 access preflight, v8 CPU-only tokenizer/mask preflight, and only after both pass, a fresh bounded GPU smoke.
