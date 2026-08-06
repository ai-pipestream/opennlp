# static-embeddings

## JIRA fields

*Type:* New Feature. *Components:* extensions (new module opennlp-embeddings). *Note:* code only, no model data bundled; branch `static-embeddings`.

## JIRA description (wiki markup)

```
h2. Summary

OpenNLP's static word-vector support ends at the GloVe loader in {{opennlp.tools.util.wordvector}}. The static-embedding field moved in 2024/2025: distillation tooling (Model2Vec and compatible) compresses a full sentence transformer into a flat per-token vector table, the same artifact shape as word2vec/GloVe, carrying the teacher model's semantics. Inference over such a table is tokenize, gather, weight, mean-pool, normalize: no forward pass, no GPU, no native runtime. That is a pure-JVM sweet spot, and this ticket adds the engine for it as a new {{opennlp-extensions/opennlp-embeddings}} module.

The module ships code only. Users point it at a table they downloaded (the {{vocab.txt}} plus {{model.safetensors}} file pair such distillations publish); nothing is fetched at build or run time and no model weights enter the source tree or release artifacts. The pooling semantics were verified against the reference implementations (the Model2Vec Python package and its official Rust port), not assumed: {{[CLS]}}/{{[SEP]}} are never pooled, unknown tokens are dropped from both the sum and the denominator, an optional per-row {{weights}} tensor multiplies each token's vector before pooling, the denominator is the pooled-token count (not the sum of weights), and normalization uses an epsilon floor so token-less input yields a zero vector rather than NaN.

Measured with JMH at the scale of a current published table (29,528 rows by 256 dimensions): about 766,000 short-sentence embeds per second on one core, and the full-vocabulary top-10 similarity scan at 649 per second on one core scaling to about 9,200 per second at 32 threads. Instances are immutable and annotated {{@ThreadSafe}}, with a concurrency test comparing every concurrent result against the single-threaded reference.

h2. Scope

* {{SafetensorsFile}}: a reader for the safetensors tensor format (8-byte little-endian header length, JSON header, raw tensor bytes) with a purpose-built cursor parser for the header, no third-party JSON dependency. Only the F32 decode path is implemented. Unlike pickle-based checkpoint formats, safetensors carries no executable content, so loading is safe by construction. The embedding matrix is found as the single 2-D F32 tensor, failing loud and listing candidates when that is ambiguous, rather than guessing a key-name convention.
* {{WordPieceVocabulary}}: a BERT-style {{vocab.txt}} reader; the line number is the token's row id into the embedding matrix.
* {{StaticEmbeddingModel}}: loads the file pair, embeds text through the existing {{BertTokenizer}}/{{WordpieceTokenizer}} (reused as is, no new tokenizer code), and applies the verified pooling formula. Per-row L2 norms and the special-token mask are precomputed at load.
* Convenience surface: {{similarity(text1, text2)}}, {{mostSimilar(text, topK)}} (brute-force scan with a bounded top-K selection), and {{analogy(a, b, c, topK)}} (vector arithmetic, with the input terms excluded by folding them through the model's own tokenizer, so exclusion is case- and accent-consistent with embedding).
* A JMH benchmark behind the same opt-in {{jmh}} Maven profile pattern {{opennlp-runtime}} already uses; the default build is unaffected.

h2. Acceptance criteria

* Loads the file pair of a current published distilled table and matches the reference pooling semantics; tests pin the formula details (denominator, unknown-token handling, weight application, normalization epsilon, zero-row NaN guard).
* No bundled model data, no new dependencies, nothing fetched at build or run time.
* Immutable and thread-safe; a concurrency test compares concurrent results against the single-threaded reference.
* {{mvn verify}} green including checkstyle and forbiddenapis; JMH numbers recorded in the PR.

h2. Out of scope

* Bundling any model weights as a zero-config default. That requires per-release license diligence on the weights (the distillation tool being MIT does not settle the weight license) and is a separable later decision.
* An approximate-nearest-neighbor index for {{mostSimilar}}. Brute force is the documented v1 posture at this vocabulary scale; an index is a follow-up.
* The gRPC {{EmbeddingProvider}} backend in opennlp-sandbox that serves this engine. That is a companion change in the sandbox repository after this lands.
* Interpreting the Hugging Face {{tokenizer.json}} pipeline format. The v1 contract is {{vocab.txt}} plus the existing WordPiece tokenizer, which the targeted table family publishes.
* Training or distillation. Producing tables stays in the upstream tooling, the same posture opennlp-dl takes for ONNX models.
```

## PR title

```
<KEY>: Static embedding engine for modern distilled embedding tables (opennlp-embeddings)
```

## PR body (markdown)

```
Adds a new `opennlp-extensions/opennlp-embeddings` module: a pure-JVM engine for modern static embedding tables (Model2Vec-family distillations, the 2024/2025 successors to word2vec/GloVe: same flat per-token table shape, sentence-transformer semantics, inference is pure lookup).

**What's in it:**

- `SafetensorsFile`: reads the safetensors format with a purpose-built cursor parser for the JSON header (no third-party JSON dependency, F32 decode only). safetensors carries no executable content, unlike pickle-based checkpoints, so loading is safe by construction. The embedding matrix is auto-detected as the single 2-D F32 tensor, failing loud and listing candidates on ambiguity rather than guessing a key-name convention.
- `WordPieceVocabulary`: BERT-style `vocab.txt`, line number = embedding row id.
- `StaticEmbeddingModel`: embeds through the existing `BertTokenizer`/`WordpieceTokenizer`, reused unchanged. The pooling formula is verified against the Model2Vec Python and official Rust reference implementations: `[CLS]`/`[SEP]` never pooled, unknown tokens dropped from sum and denominator, optional per-row `weights` tensor, token-count denominator, epsilon-floored normalization.
- Convenience surface: `similarity`, `mostSimilar` (bounded top-K over precomputed row norms), `analogy` (input terms excluded by folding through the model's own tokenizer).

**Posture:** code only, bring your own table. No model bundled, nothing fetched at build or run time, no new dependencies.

**Thread safety:** immutable, `@ThreadSafe`, with an 8-thread concurrency test comparing every result against the single-threaded reference.

**Measured** (JMH, opt-in `jmh` profile matching opennlp-runtime's pattern; fixture at real published-table scale, 29,528 x 256): `embed` ~766k short sentences/s on one core (1.04M ops/s of 5 sentences at 32 threads); full-vocabulary top-10 scan 649/s per core, ~9.2k/s at 32 threads.

Verification: opennlp-embeddings 43/0, `mvn verify` green including checkstyle and forbiddenapis.

Follow-ups (deliberately out of this PR): the gRPC `EmbeddingProvider` backend in opennlp-sandbox, a concurrent-load comparison against a Python baseline, an ANN index for `mostSimilar`, and bundled-default-model license diligence.
```
