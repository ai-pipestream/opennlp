# perf-1850-followup

## JIRA fields

*Type:* Improvement. *Follow-up to:* OPENNLP-1850. *Components:* opennlp-api, opennlp-runtime, opennlp-dl. *Note:* PR opens only after dl (#1105) merges; branch is rebased --onto main first.

## JIRA description (wiki markup)

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

## PR title

```
<KEY>: Non-breaking performance follow-up for the 1850 normalization and tokenization hot paths
```

## PR body (markdown)

```
Non-breaking performance follow-up to OPENNLP-1850 for heavy analyzer/tokenizer use. Six output-preserving changes:

- an identity short-circuit on the non-aligned `CharClass` folds (member-free text returns without allocation),
- a key-`BitSet` fast-path in `Confusables.skeleton` (clean text skips both NFD passes),
- BMP fast-paths in the hot cursor loops,
- precomputed digit strings,
- a pre-sized `Alignment.Builder` constructor,
- up-front sizing of the DL score/token/ONNX collections.

JMH on realistic mixed-script token streams confirms the tier-one wins (identity short-circuit up to 4.3x on whitespace-normalize-heavy streams, Confusables 1.7x) with the predicted neutral result on fold-saturated text.

Byte-identical output; the full 1850 suites pass unchanged (api 298/0, runtime 1304/0, dl 92/0).
```
