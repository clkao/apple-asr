# Audio fixtures

## `zh_long.wav`

A **31.55 s, 16 kHz mono PCM** clip of spoken Mandarin, derived from the
[`google/fleurs`](https://huggingface.co/datasets/google/fleurs) corpus, which is
licensed **CC-BY-4.0**. It is redistributed here as test data; content unchanged.

It is a *cadence fixture* for the integration test suite — the golden fixture
`tests/golden/zh_long_ideal.jsonl` was produced from exactly this audio, so the
clip must not be replaced or re-encoded. It is **not** a general-purpose audio
sample: do not reuse it as a benchmark, demo, or training input.

Attribution: FLEURS — *FLEURS: Few-shot Learning Evaluation of Universal
Representations of Speech* (Conneau et al., 2022; Google), CC-BY-4.0.
