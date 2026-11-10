# Blinded paired translation screen

This private kernel screens 32 masked question/candidate pairs: two candidate translations for each of 16 source questions. It receives no method labels, answer keys, facts, split assignments, or private pair map.

For NLLB back-translation only, `[ENTITY_X]` markers are replaced with common names and `[NUMBER_X]` markers with numerals, then exact source markers are restored before semantic adjudication. Any missing/changed alias blocks that row; it is never silently repaired. All decisions remain diagnostic-only.
