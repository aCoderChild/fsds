# Streaming recovery package (not submitted)

This package is a contingency for the active v2 run only if that run terminates without verified results. It has **not** been uploaded or launched.

The locked protocol, model, seed, data, NPO objective, LoRA configuration, 500-step schedule, evaluation rows, metrics, and decision thresholds are unchanged. The instrumentation-only differences are:

- training emits a flushed progress record every 10 optimizer steps;
- long-running subprocess stdout/stderr is streamed into Kaggle Logs while also being written to the run archive log;
- child Python processes run with `PYTHONUNBUFFERED=1`.

The progress stream makes a slow stage distinguishable from a silent stall. It does not create intermediate score results or make a Kaggle run resumable. This package must not run concurrently with v2 because both use the same T4 quota. Before any launch, its source and notebook hashes must be reviewed and a time-stamped protocol amendment must record the instrumentation-only source change.
