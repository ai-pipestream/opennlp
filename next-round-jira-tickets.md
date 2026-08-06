# Next round JIRA tickets: 1852 consistency scoreboard, the 1850 perf follow-up, and the static embedding engine

This is the fileable set for the next round. The 1852 fold stack is already tracked: OPENNLP-1868 (full case-fold rung, PR #1138), OPENNLP-1869 (emoji/emoticon rung), and OPENNLP-1870 (annotation layer) are filed. The four tickets below are the independent work that branches off `main` and merges without waiting on the fold stack. Each fenced block is Jira wiki markup, ready to paste into the issue description. The line above each block is the issue metadata (type, epic or parent link, components), which goes in the JIRA fields, not the description.

Convention followed, matching OPENNLP-1850 and the 1868 ticket: standards-sourced character classes (UCD properties, not `Character.isWhitespace` or regex classes), cursor-based text transforms on user text (regex reserved for config), and offset preservation where spans matter. These three tickets are the ones the OPENNLP-1852 epic description flags as the consistency scoreboard plus the vetted 1850 performance follow-up.

---

## 1. UCD-back the whitespace predicates: migrate StringUtil and audit the call sites

*Type:* Improvement. *Epic:* OPENNLP-1852 (consistency scoreboard, items 1 and 2). *Components:* opennlp-api, opennlp-runtime. *Note:* behavior change, requires a release note.

```
h2. Summary

OPENNLP-1850 established standards-sourced character classes as a design goal (the {{CharClass}} engine reads the UCD {{White_Space}} and {{Dash}} sets, not {{Character.isWhitespace}} or a regex class), but the rest of the codebase has not caught up: {{StringUtil}} still uses the JVM {{Character.isWhitespace}} predicate, and several call sites split or scan user text with {{split("\\s+")}} or {{Character.isWhitespace}}. This ticket migrates the whitespace predicate in {{StringUtil}} to the UCD {{White_Space}} set (or documents the intentional exception where JVM semantics are genuinely wanted), and audits and converts the remaining user-text call sites. This is a behavior change and carries an explicit release note, because the UCD {{White_Space}} set and {{Character.isWhitespace}} disagree on real code points: {{White_Space}} includes the no-break space U+00A0 and the figure and narrow spaces that {{Character.isWhitespace}} excludes, and excludes the U+001C to U+001F information separators that {{Character.isWhitespace}} includes. OPENNLP-205 already made exactly this correction in the sentence-detector mapping, so this generalizes that fix.

h2. Scope

* Audit {{split("\\s+")}}, {{Character.isWhitespace}}, and {{Character.isSpaceChar}} call sites across the codebase outside the OPENNLP-1850 engine. Classify each as user text (migrate to the UCD set) or config and non-user-text (leave; regex and JVM predicates are acceptable there per the design goals).
* Migrate the whitespace handling in {{StringUtil}} to the UCD-backed {{White_Space}} API delivered in OPENNLP-1850, or document the intentional JVM exception with a rationale where a caller genuinely needs JVM semantics (for example a format-fixed parser).
* Add a release note in the 3.0 migration section describing the delta: NBSP and the Unicode space separators now count as whitespace; the U+001C to U+001F information separators no longer do.

h2. Acceptance criteria

* The {{StringUtil}} whitespace predicate is UCD-backed, or the JVM exception is documented with a rationale.
* The call-site audit is complete: user-text paths use the UCD set, and every retained regex or JVM predicate on a text path is justified in a comment.
* A release note documents the behavior delta at the boundary code points (NBSP, the space separators, the information separators).
* Tests pin the delta at those boundary code points.

h2. Out of scope

* The OPENNLP-1850 engine itself, which is already UCD-backed.
* Regex in config parsing and other non-user-text utilities, which the design goals permit.
* The legacy {{CharSequenceNormalizer}} regex migration, which is a separate ticket.
* Changing tokenizer or sentence-detector feature generation (model stability).
```

## 2. De-regex the legacy CharSequenceNormalizers onto cursor scans

*Type:* Improvement. *Epic:* OPENNLP-1852 (consistency scoreboard, items 3 to 6 and 8). *Components:* opennlp-runtime, extensions.

```
h2. Summary

The legacy normalizers {{TwitterCharSequenceNormalizer}}, {{UrlCharSequenceNormalizer}}, {{ShrinkCharSequenceNormalizer}}, and {{NumberCharSequenceNormalizer}} apply regular expressions to user text, which runs against the OPENNLP-1850 design goal of cursor-based text transforms (a single forward code-point scan on user text, with regex reserved for config parsing). This ticket replaces those regex passes with cursor scans on the {{CharClass}} engine where they operate on user text, and audits the spellcheck and extension normalizers for the same pattern. These are pre-1850 rungs and are not in the blessed default chain, so the change is low-risk and output-preserving rather than a redesign.

h2. Scope

* Replace regex-on-text with a cursor scan (on the {{CharClass}}/{{CodePointSet}} primitives from OPENNLP-1850) in {{TwitterCharSequenceNormalizer}}, {{UrlCharSequenceNormalizer}}, {{ShrinkCharSequenceNormalizer}}, and {{NumberCharSequenceNormalizer}}.
* Where a normalizer's behavior is subtle, add characterization tests that pin the pre-change output first, then refactor, so the output is preserved byte for byte.
* Audit the spellcheck and extension normalizers for regex-on-text paths; convert the clear cases and file follow-ups for anything larger.

h2. Acceptance criteria

* The four normalizers use cursor scans on the text path; no regex is applied to user text in them.
* Output-preserving: characterization tests capture the prior behavior and pass unchanged after the refactor.
* The spellcheck and extension audit is complete, with conversions applied or follow-ups filed.

h2. Out of scope

* Making these rungs offset-aware. They are not in the blessed chain, and offset-safety for them is a separate concern.
* The {{StringUtil}} whitespace migration, which is a separate ticket.
* Regex in config parsing and non-user-text utilities.
* Adding or removing any of these rungs from a default chain.
```

## 3. Non-breaking performance improvements for the 1850 normalization and tokenization hot paths

*Type:* Improvement. *Follow-up to:* OPENNLP-1850. *Components:* opennlp-api, opennlp-runtime, opennlp-dl.

```
h2. Summary

A vetted set of non-breaking performance improvements for the OPENNLP-1850 normalizer, tokenizer, and DL hot paths, aimed at heavy use as a search-index analyzer and tokenizer. Every item is output-preserving (byte-identical), offset-safe, thread-safe, free of regex on the text path, and ASCII-only in source. These were verified against the source by reading the relevant methods during a performance review and were deliberately deferred out of the OPENNLP-1850 review so as not to delay it. Land as one follow-up PR (or a small stack), and pair with a JMH benchmark run in a fresh JVM per case on realistic mixed-script text to confirm each win before and after.

h2. Scope (ranked, tier one first)

* Identity short-circuit on the non-aligned {{CharClass}} set folds ({{normalize}}, {{collapse}}, {{removeAll}}, and the static {{substitute}}). They currently allocate a {{StringBuilder}} and copy code point by code point even when nothing matches. Change to a lazy builder: scan for the first member, return the input unchanged when there is none, and allocate the builder pre-filled with the unchanged prefix only on the first hit. Biggest win on ASCII-heavy token streams; roughly neutral on text saturated with folded characters. Scope to the non-aligned path; leave the {{*Aligned}} variants alone unless the identity-{{Alignment}} compose is verified a true no-op.
* {{Confusables.skeleton}} fast-path plus an optional bounded cache. Add a {{BitSet}} pre-filter of "is this code point ever a prototype key" so the common no-mapping case skips the two {{java.text.Normalizer}} NFD passes and the per-code-point boxing, and add a size-bounded, thread-safe value cache. Both fire only when {{CONFUSABLE_FOLD}} is configured. The two NFD passes are required by UTS #39 when a mapping does apply, so the floor stays at two for actual confusables.
* Hoist the per-code-point volatile reads in tokenization. {{WordBreakProperty.ordinalOf}} does a volatile {{data()}} read per code point and {{ExtendedPictographic.is}} reads a volatile field per call. Resolve the resolved-data snapshot and the {{BitSet}} once per document in {{WordSegmenter.forEachSegment}} and {{WordType.of}} through package-private overloads, keeping the lazy first-resolution and the supplementary path. Removes a volatile load and a call frame per code point on the hottest tokenization op.
* BMP fast-path the {{Character.codePointAt}} and {{Character.charCount}} pairs in {{WordSegmenter}}, {{WordType.of}}, and the {{CharClass}} cursor loops, so a non-surrogate {{char}} skips the JDK call and the redundant surrogate re-test (over 99 percent of Latin text).
* {{DigitCharSequenceNormalizer}}: replace the per-digit {{String.valueOf}} allocation with a precomputed {{static final String[]}} for "0" through "9". Output-preserving (length still one).
* {{Alignment.Builder}}: add a pre-size constructor {{Builder(int expectedLength)}} seeding the {{starts}}/{{ends}} arrays; every caller already knows {{text.length()}}. Kills the array-regrow chain and makes the defensive trim near-exact on the offset and highlighting path. Keep the no-arg constructor.
* DL small cleanups: size the score and token lists in {{DocumentCategorizerDL}} and {{NameFinderDL}} from the known counts ({{ArrayList}} rather than {{LinkedList}}), and pre-size the ONNX input {{HashMap}}. Correctness-neutral.

h2. Acceptance criteria

* Every change is output-preserving (byte-identical), offset-safe, and thread-safe; the existing OPENNLP-1850 test suites pass unchanged.
* A JMH benchmark, one fresh JVM per case, on realistic mixed-script text confirms the tier-one wins (the identity short-circuit and the {{Confusables}} fast-path) before and after.
* No new dependencies, ASCII-only source, no regex on the text path.

h2. Out of scope

* Any behavior or output change.
* The {{*Aligned}} {{CharClass}} variants, unless the identity-{{Alignment}} compose is verified a true no-op.
* Paths already verified optimal and not to be re-investigated: {{CodePointSet}} BitSet membership, the compiled UAX-29 transition tables, the {{Alignment}} binary searches and {{andThen}}, and the {{mergeOverlappingSpans}} TreeMap (deliberately O(n log n) to yield document order).
```

## 4. Static embedding engine: load and serve modern distilled embedding tables, pure JVM

*Type:* New Feature. *Components:* extensions (new module opennlp-embeddings). *Note:* code only, no data bundled; branch `static-embeddings`.

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
