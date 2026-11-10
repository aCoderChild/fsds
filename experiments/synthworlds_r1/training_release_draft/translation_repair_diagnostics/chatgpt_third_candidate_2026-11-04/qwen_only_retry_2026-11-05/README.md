# Qwen-only re-review using saved alias-safe back-translations

The first 18-row retry generated and restored the 15 alias-masked back-translations correctly, but its script then raised a `NameError` on an unused key-map assignment and incorrectly marked those 15 as blocked before Qwen review. The three malformed-Qwen rows did reach Qwen and produced one pass and two flags.

This follow-up does **not** rerun NLLB. The preparation step verifies each saved source/candidate hash, the exact restored back-translation hash, and placeholder multiplicities, then sends only those 15 source/translation/back-translation triples to the pinned Qwen reviewer. The other three outcomes remain from the completed strict-format retry. Both retries are preserved as separate artifacts.

The output is an AI-only diagnostic. It cannot certify translation quality, alter the source corpus, or release training data.
