# TOFU NPO-LoRA pilot setup verification

Status: **Objective and protocol preflight pass; corrected Kaggle smoke version 4 pending.**

- The candidate NPO loss and gradients match the exact pinned Open-Unlearning implementation (loss difference `0`; maximum gradient difference `5.96e-8` on Kaggle; sequential-backward difference `0`).
- The pinned 3B full and retain-only oracle baselines pass independent row-level verification.
- Kaggle v3 loaded the exact 3B checkpoint across two Tesla T4 GPUs with no CPU/disk offload, but PEFT then rejected Kaggle's preinstalled `torchao 0.10.0`; zero optimizer steps ran and no result was created.
- Version 4 removes this unused optional package before model loading, first checks LoRA injection/forward/backward on a tiny Llama, and keeps full training disabled.
- Full training remains gated on a passing one-step real-data smoke, nonzero adapter update and GPU-only placement.

Machine-readable receipts: [`local_preflight_record.json`](local_preflight_record.json), [`objective_equivalence_cpu_retry.json`](objective_equivalence_cpu_retry.json).
