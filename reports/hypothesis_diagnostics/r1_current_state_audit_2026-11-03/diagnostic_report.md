# SynthWorlds-SM R1 implementation and execution audit

**Audit date:** 2026-11-03  
**Scope:** canonical v5.1 project, current training-translation release gate, offline pipeline fixtures, saved v4/v3-era outputs, and the latest Kaggle API setup run.  
**Safety:** no model weights were downloaded and no SFT, unlearning, or model evaluation was launched during this audit.

**Correction from the subsequent corpus audit:** the all-flagged repair diagnostic's recorded name checks were invalid. The repair script loaded saved review rows that omitted the protected-name lists. Rejoining the original lists and recomputing the checks shows 456/948 candidates changed, omitted, or duplicated a protected name; only 492/948 pass the recomputed checks. Gemma3 also received an unmasked protected name in the source question for 947/948 rows. The original release candidates and their 1,153 final-review passes were rechecked against their source name lists and none failed these deterministic checks. This does not establish semantic or translation accuracy. The diagnostic repair and Gemma decisions remain quarantined. See `translation_coverage_diagnosis.json` and `corpus_reliability_audit.md`.

## Result at a glance

| Area | Finding | Status |
|---|---|---|
| Source split and candidate provenance | 195 source-verified cases, 117/39/39 across 102 indivisible leakage groups | Draft validated; not a training release |
| Evaluation translation cohort | 80 cases, 1,200 prompt/label rows; accepted only after the frozen source-bound screen | Screened; AI-only |
| Training translation review | Second pass completed for 488 prompts in each target language | Review complete; AI-only, not human fluency certification |
| Atomic training eligibility | Forget 5/117; retain 3/39 under the assembler's complete four-language/two-hop rule | **Fail closed** |
| Training-only composition chains | 0/44 meet the frozen all-language review requirement | **Fail closed** |
| Approved SFT corpus and manifest | Neither exists | **Absent** |
| Current local preflight | Blocked by absent approved corpus/manifest; PEFT, Accelerate, and HF token are also absent locally | **Blocked; no downloads** |
| v5.1 offline fixture suite | 12/12 pass; relevant Python sources compile | Software checks pass; synthetic fixtures only |
| Kaggle API job, version 12 | T4 and training packages were available, but HF token was absent; the combined gate blocked and every model stage stayed off | Setup only; no experiment |
| Historical archived scoring | Score-only rescore reproduces the 2026-11-02 baseline outputs; indexed baseline artifacts match their hashes | Reproducible scoring, not a new model run |
| Historical corrected v4 study | V3 result ZIP has clean CRC and all 77 indexed file hashes match; competence gate failed | Exploratory only; cannot answer the R1 hypothesis |

## Translation-review interpretation

AI-only review was explicitly authorized for this pilot. The review of the training questions is complete, so “human review pending” is not the current operational blocker. No fluent-human review was performed, and AI review cannot certify language quality. The release is blocked because too few complete atomic cases and no composition chains survive the predeclared all-language criteria. This is a coverage failure, not an unresolved permission decision.

The 14B second-pass file reports:

| Language | Reviewed | AI passes | Flagged |
|---|---:|---:|---:|
| French | 488 | 324 | 164 |
| German | 488 | 309 | 179 |
| Hindi | 488 | 288 | 200 |
| Thai | 488 | 232 | 256 |
| **Total** | **1,952** | **1,153** | **799** |

The aggregate row counts are not enough to approve training: the assembler requires complete, provenance-valid coverage at case level across every hop and target language. The production assembler was run and stopped before creating `sft_training/corpus.jsonl`, `manifest.json`, or a validation stamp. A separate gate probe also confirmed 0/44 training-only composition chains, as expected.

### Repair-path diagnosis

The repair outputs do not currently clear the release gate:

| Check | Result | Interpretation |
|---|---:|---|
| TranslateGemma repair candidates for originally flagged prompts | 948/948 generated; **492/948** pass recomputed name/number/script checks | The originally recorded 948/948 integrity passes were incorrect |
| Preliminary Qwen 7B repair screen | 683/948 passed | Diagnostic only; replacements have not passed the final review |
| NLLB 1.3B calibration | 28/64 passed the full round-trip/Qwen 14B screen | Too small to certify corpus-wide performance |
| Fully reviewed two-hop atomic cases in the training split | Forget 5/117; retain 3/39 | Far below complete planned coverage |
| Evaluation cases whose two-hop facts are fully reviewed in all four target languages | Forget 3/43 (2 groups); retain 2/20 (2 groups) | Only 5 of the 63 forget/retain evaluation cases are currently supported by reviewed atomic training translations |
| Same evaluation cases if Qwen 7B and Gemma 3 repair passes and recomputed integrity are provisionally counted | Forget 5/43 (3 groups); retain 2/20 (2 groups) | Exploratory upper bound; repairs lack final review |
| Strictly approved composition prompts matched across FR/DE/HI/TH | 45 prompt IDs across 32 chains, but **0 multi-hop prompts** | Passing support questions do not provide a translated multi-hop answer example |
| Composition prompts if Qwen 7B and Gemma 3 repair passes and recomputed integrity are provisionally counted | 0 multi-hop prompt IDs; 0/44 chains have all four prompt forms | Still diagnostic; repairs lack the final review |

The earlier `optimistic_coverage_report.json` reports 52 forget and 17 retain cases by counting first-stage Qwen 7B passes even when the final Qwen 14B review rejected them. The reproducible diagnosis at `translation_coverage_diagnosis.py` reports the stricter bound above: final-review passes for originals plus preliminary Qwen 7B passes for replacements. Neither bound is a release approval. It also records source hashes and leaves the translations and manifests unchanged.

These results show that another pass over the same local models is not a reliable way to claim complete multilingual coverage. The final reviewers share the Qwen family; AI-only review was authorized for the pilot, but it is not independent fluent-human certification. No existing repair file should be promoted into the SFT corpus. The current blockers are coverage and the absence of reviewed multi-hop practice questions, not Kaggle runtime configuration.

### Follow-up repair calibration (2026-11-04)

The corrected source-bound TranslateGemma repair runner was rerun on 32 final-review flags: four rows in each language-by-question-kind cell. It produced candidates for all 32, and independent recomputation passed the protected-name, number, script, and marker checks for all 32. Qwen2.5 7B accepted 24/32 (75%): French atomic 2/4 and composition 3/4; German 4/4 in both; Hindi atomic 4/4 and composition 3/4; Thai atomic 3/4 and composition 1/4. This is a small, hash-selected calibration, not a corpus-wide rate or release screen. The repaired candidates have not received the required final review, and the 32-row result does not establish enough complete multilingual cases. A fresh run reproduced the prior candidate JSONL byte-for-byte (SHA-256 `5c844f52a83a0fc19f9c1fc28fe415ebcafe6e1bb11a943a7094f96b2cf43908`). No source translation, release input, corpus, or manifest was changed.

## Reproducibility checks

The local audit script reruns the score-only evaluators on saved predictions (without loading checkpoints), recomputes the baseline competence counts, checks baseline and corrected-v4 archive files against their SHA-256 manifests, runs the release assembler and local SFT preflight, compiles the current notebook's code cells, runs the offline fixtures, and inspects the saved Kaggle setup log. The `verification.json` records the full values and outputs.

Kaggle marked notebook version 12 complete. Its preflight summary reports a Tesla T4 and all training packages ready, but the HF token was missing, the combined training gate was blocked, and SFT, retain-only training, unlearning, and evaluation all remained off. Runtime project discovery resolved to the legacy `synthworlds-sm-sft-pilot-v3` dataset; the saved kernel metadata lists both the v3 and v4 datasets, so the configured attachments do not exactly match the runtime source list. In either record, this was not the current v5.1 three-seed notebook or dataset, and no model stage started. The local v5.1 notebook has 13 cells and no execution outputs. Thus `COMPLETE` refers only to setup/preflight execution, not an R1 model experiment.

The historical 2026-11-02 baseline has 3,870 prediction rows (1,290 each for pretrained, SFT, and retain-only), zero generation failures, and zero invalid-output flags. Its five saved derived outputs reproduce byte-for-byte. Its bilingual precompetent counts are FR 2 cases/1 group, DE 2/2, HI 4/2, and TH 3/2, all below the declared minimum of 15 cases and 8 groups. It has no unlearning contrast.

The corrected v4 exploratory archive has 78 paths, a clean ZIP CRC, and all 77 indexed SHA-256 entries verified. A score-only rerun of the archived predictions reproduced all seven saved derived evaluation files byte-for-byte. Its gate was also below threshold (FR 10/6, DE 9/5, HI 9/6, TH 11/7 against 15/8), and its wide intervals include zero. These are archived exploratory results, not confirmatory evidence.

### Known historical integrity issue

The separate legacy v4 `scripts/verify_v3.py` still fails its recorded hash check for `kaggle/kernel_update/experiments.ipynb`. That is a stale hash in an older v4 upload snapshot, not a failure of the current v5.1 source validator or the corrected V3 results ZIP. The snapshot is preserved unchanged; its recorded manifest was not rewritten to make the check appear to pass.

## Project organization

The active pilot now lives at `experiments/synthworlds_r1/`. The former `data/synthworlds_sm_pilot_v5_1/` path is a compatibility symlink for existing links. Two hash-bound review scripts intentionally keep that alias in their source so completed review manifests continue to validate. Source-lineage pilots v2–v4 remain under `data/` because the validation and historical rescoring tools read those exact source snapshots; superseded v1/v5 drafts are archived.

Older setup-only Kaggle logs and the bounded translation-pilot outputs were moved under `archive/` and indexed by hashes. Two large unpacked exploratory copies and the superseded V2 unpacked result (974,265,610 uncompressed bytes; about 929 MiB on disk) were removed only after every member was verified against its CRC-clean ZIP. The V3 corrected result expansion remains available beside its summary and supplementary analysis. Generated Python caches and empty temporary folders were removed.

## Conclusion

**The implementation's data gates and local software fixtures behave as designed, and the archived score calculations reproduce. The R1 training experiment is not ready or running.** The existing repair attempts do not establish enough fully reviewed two-hop cases or any strictly approved cross-lingual multi-hop practice questions. A replacement translation/review pass must use a method with demonstrably better complete-case coverage; its outputs must be hash-bound, screened for relation direction and answer type, and verified before release assembly. Then rerun the assembler and preflight. Keep all Kaggle training stages disabled until both pass. After that, run the three-seed SFT/retain baseline and apply the predeclared competence gate before any unlearning.
