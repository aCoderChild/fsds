# SynthWorlds-SM R1 corrected exploratory results

This bundle contains the corrected GD/NPO retain-sampler rerun and the five-language evaluation. GD/NPO record 26 distinct English retain rows used over 45 steps; no holdout rows were used. The evaluation has 7,740 predictions, with 1,290 per checkpoint across pretrained, SFT, retain-only, GA, GD, and NPO.

The bilingual competence gate failed (FR 10 cases/6 groups, DE 9/5, HI 9/6, TH 11/7; required 15/8). These are exploratory, one-seed, 1B-model results, not confirmatory evidence. Translations were reviewed AI-only; fluent-speaker review was not done.

See corrected_unlearning/source_code_provenance.json for the corrected script hash and the remaining source-binding limitation. artifact_index.json hashes every bundled file except the index itself.
