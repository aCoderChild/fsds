# α40 preregistration digest erratum

The human-readable `experiments/tofu_scaling/alpha40_followup_preregistration.md` header lists protocol SHA-256 `f61dc9cfd71a1e410877543f2c6a315934a187d6839b80448cc9ee807b0c3f3a`. That digest does **not** match the protocol JSON used by the run. The actual `npo_lora_alpha40_followup_protocol.json` SHA-256 is `b308c93da3934a020cd0ca1c22c89c74133e2cb97d90656f4ff02ccb155da8a7`.

The actual JSON digest is consistent across the pre-run tracker record, the Kaggle archive copy, the recovery receipt, and both independent-verification records. The archived protocol JSON is byte-for-byte identical to the local protocol JSON, and local result recomputation passed. The Markdown digest is a documentation error; it is recorded here rather than silently rewritten. This reduces the quality of the human-readable preregistration record, but does not change the parameters, thresholds, or cohort used in the executed protocol.

frozen preregistration Markdown SHA-256 as checked on 2026-11-08: `95538c54b5a817d22c3075e81a45d96a45761f565b267cdce0e48b559652c7a3`.
