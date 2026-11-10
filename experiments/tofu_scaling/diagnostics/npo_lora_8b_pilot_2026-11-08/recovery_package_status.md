# Final disposition (verified 2026-11-09T05:42:35.338759+00:00)

The original Kaggle v2 run completed and its 500-step receipt, archive, and evaluation results are verified. The evaluation-only recovery notebook stayed local and was not launched; it is unnecessary for this completed attempt. Results and limitations are recorded in `diagnostic_report.md` and `kaggle_v2/postrun_verification.json`.

The archive is valid, but protocol alignment and utility gates fail. Do not start another run from this recovery package.

---

## Historical recovery-package snapshot (pre-completion)

Observed: 2026-11-09T05:00:06.234561+00:00

- Live 8B run: Warning: Looks like you're using an outdated `kaggle`` version (installed: {current_version}), please consider upgrading to the latest version ({latest_version_str})
bunnoob2005/tofu-8b-npo-lora-pilot-2026-11-08 has status "KernelWorkerStatus.COMPLETE"
- No matching result archive is available while the kernel is running.
- Recovery notebook: `experiments/tofu_scaling/kaggle/npo_lora_8b_eval_recovery_v1/tofu_8b_npo_eval_recovery.ipynb` (SHA-256 `05ddb0f25df97fc5200714d910924d20aa6c65f4c40d62665b948b226ec6fb3a`).
- Local offline package verification: PASS. The setup cell accepts a valid completed fixture, rejects a modified adapter and ZIP path traversal, and refuses inference without the required GPU.
- The recovery notebook remains local and unlaunched. It will evaluate only after the original archive proves 500 completed steps and a hash-valid adapter.
- No research result is verified.
