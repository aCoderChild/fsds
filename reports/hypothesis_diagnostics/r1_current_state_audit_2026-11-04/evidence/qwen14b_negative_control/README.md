# Qwen2.5-14B negative-control diagnostic

This private Kaggle job applies the same semantic/fluency review style to the 15-row fixed mismatch set previously screened with Qwen2.5-7B and Mistral-7B. Every candidate translation is deliberately paired with the unrelated English question `What color is [ENTITY_A]?`. Names and answers are not present.

The run tests only whether the larger Qwen reviewer rejects obvious question-type mismatches. It cannot estimate sensitivity to subtle relation changes, translation accuracy, or target-language fluency. A pass is a reviewer failure on this control; even 15/15 rejections would not approve the corpus.

`review.py` is the Kaggle kernel source. `input.jsonl` is the local, hash-bound control list. `verify.py` validates output IDs, model revision, input hash, schema, and counts. Outputs must remain diagnostic-only and must not be merged into training translations.
