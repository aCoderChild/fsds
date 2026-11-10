# TOFU official-metric baseline verification

Overall: **PARTIAL**
Reconciled run status: **PARTIAL**
Raw Kaggle terminal status: **ERROR**
Raw run error: `OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 40.81 MiB is free. Including non-PyTorch memory, this process has 14.52 GiB memory in use. Of the allocated memory 14.26 GiB is allocated by PyTorch, and 133.40 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)`

## llama_3_2_3b_full: PASS

- forget10: 400 questions, 20 profile blocks
- retain90: 400 questions, 20 profile blocks
- holdout10: 400 questions, 20 profile blocks
- real_authors: 100 questions, 0 profile blocks
- world_facts: 117 questions, 0 profile blocks

## llama_3_2_3b_retain90: PASS

- forget10: 400 questions, 20 profile blocks

## llama_3_1_8b_full: PASS

- forget10: 400 questions, 20 profile blocks
- retain90: 400 questions, 20 profile blocks
- holdout10: 400 questions, 20 profile blocks
- real_authors: 100 questions, 0 profile blocks
- world_facts: 117 questions, 0 profile blocks

## llama_3_1_8b_retain90: MISSING

- missing llama_3_1_8b_retain90_official_eval.json
