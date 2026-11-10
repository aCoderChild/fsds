# Status of this recomputed pilot folder

**Do not treat these outputs as the authoritative R1 pilot results.** This folder contains an alternate prediction set whose run provenance is not established here.

The authoritative exploratory diagnostic is the hash-verified corrected v4 bundle, independently rescored at [`../r1_reliability_tracker_2026-11-05/pilot_diagnostic_results.md`](../r1_reliability_tracker_2026-11-05/pilot_diagnostic_results.md). Its prediction file SHA-256 is `8ee0d894bd41cc1281b9b98bd8d19ad0e9ed2f6886735133dff24a678de8b8fe`.

The alternate `predictions.jsonl` has the same 7,740 composite row keys, but 5,906 answer records differ from corrected v4: GA 1,290; GD 1,269; NPO 1,258; retain-only 1,232; SFT 857. This folder's report flags 1,385/7,740 predictions as output-invalid, while corrected v4 and its independent rescore flag 0/7,740. The discrepancy's cause is unresolved. There is no run manifest in this folder binding the alternate predictions to a model checkpoint, seed, runner hash, or training receipt.

Keep this folder for auditability, but do not combine its predictions, scores, or conclusions with corrected v4. A future reconciliation must identify the generating run and reproduce its provenance before these files can be used as research evidence.
