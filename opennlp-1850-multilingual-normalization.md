# OPENNLP-1850: multilingual text normalization design note

Supplementary design note for OPENNLP-1850. Targets 3.0, so we have room for API
changes (a normalized-aware text/term model) that would be hard to add later.

Regarding diacritic folding - Here's how I'll propose to handle
accents and multilingual text properly, and proposes a single model that makes
aggressive normalization safe to ship out of the box.

## Goals

* Be correct and useful across languages out of the box, not Latin-first with everyone
  else as an afterthought.
* Never destroy information. Normalization is for matching and search, not a replacement for the source text.
* Keep the cursor based, no-regex, standards-anchored approach from the base ticket.

## The spine: original + normalized

The single most important decision: text carries **both** its original form and a derived
**normalized** form, plus an **offset map** between them.

* Original is the source of truth. Offsets, `Span.getCoveredText`, display, and any
  language-specific analysis key off it.
* Normalized is a derived, parallel representation tuned for matching and search tokens.
* The offset map translates a position in the normalized form back to a range in the
  original, so a hit on the normalized text can always be reported in original coordinates.

This is the same split that search engines use (index a folded form for recall, keep the
original for display and highlighting). The key consequence:

> Because the original is always preserved, we can apply normalization aggressively for the search/token form without risking the user's data. The dual model is what makes "multilingual OOTB" safe.

Because what normalization is: a **recall optimization, not a linguistically correct transform**. Folding raises recall (`cafe` finds `café`) at the cost of precision and language correctness, so it must never be the canonical form.

## The normalization ladder

Normalization is not one thing. It is a ladder of increasing aggressiveness and increasing multilingual risk. We should ship the safe rungs by default and gate the lossy ones.

| Rung | Transform                            | Lossy?                      | Multilingual risk                           | Default              |
| ---- | ------------------------------------ | --------------------------- | ------------------------------------------- | -------------------- |
| 1    | **NFC** (canonical composition)      | No (canonical equivalence)  | None                                        | On (OOTB baseline)   |
| 2    | **Whitespace** normalize / collapse  | Layout only                 | Low                                         | Opt-in per pipeline  |
| 3    | **Dash** normalize to ASCII hyphen   | Mild                        | Low to medium (script dashes carry meaning) | Opt-in               |
| 4    | **NFKC** (compatibility composition) | Yes (e.g. `square2` to `2`) | Medium                                      | Opt-in               |
| 5    | **Case folding**                     | Yes                         | Medium (Turkish i/I)                        | Opt-in               |
| 6    | **Diacritic / accent folding**       | Yes                         | High (script destruction, language-wrong)   | Opt-in, script-gated |

Rung 1 (NFC) is the quiet hero for multilingual correctness. Precomposed `é` (U+00E9) and decomposed `e` + combining acute (U+0065 U+0301) look identical but are different code point sequences. Applying NFC on input makes canonically equivalent text compare equal regardless of how it was encoded. It is safe, standards-defined (UAX #15), and the W3C-recommended interchange form. It should be the OOTB baseline; everything above it is opt-in.

## Whitespace and dashes (this ticket) across languages

The base ticket already gets most of the multilingual win for free by using the full Unicode
sets:

* Whitespace default = Unicode `White_Space`, which includes the CJK ideographic space (U+3000), NBSP, narrow NBSP, and the typographic spaces. Java's default `\s` misses all of these.
* Dashes default = Unicode `Dash`, including fullwidth hyphen-minus (U+FF0D) and the CJK wave/wavy dashes.

Two multilingual cautions specific to dashes:

* Some script "dashes" carry meaning, not just punctuation: Hebrew maqaf (U+05BE) joins words, the Armenian hyphen (U+058A) and Mongolian todo soft hyphen (U+1806) are orthographic. Flattening them to ASCII `-` is fine for a delimiter/token form but loses information for display and analysis. This is exactly the case the original/normalized split is built for: split or flatten in the normalized form, keep the original.
* Soft hyphen (U+00AD) is an invisible format character, not a visible dash. Do not turn it into a hyphen. Strip it or leave it.

## Accents and diacritics (follow-up ticket, but design it now)

Mechanism: **NFD (canonical decompose) then drop diacritic marks then optional NFC
recompose.** Consistent with the no-regex rule, after NFD we cursor-scan and drop marks by `Character.getType(cp)` rather than a `\p{Mn}` pattern.

This is straightforward for French, Spanish, and Portuguese vowel accents. The traps are everything else:

### Trap 1: many "accented" letters do not decompose

NFD will not touch these, they need an explicit mapping table:

| Letter                 | Fold to  | Why NFD misses it                              |
| ---------------------- | -------- | ---------------------------------------------- |
| ø (U+00F8), Ø          | o, O     | letter with stroke, no canonical decomposition |
| ł (U+0142), Ł          | l, L     | stroke letter                                  |
| đ (U+0111), ħ, ı       | d, h, i  | stroke / dotless, atomic letters               |
| æ (U+00E6), œ          | ae, oe   | ligatures, no decomposition                    |
| ß (U+00DF)             | ss       | eszett, only via case folding                  |
| þ (U+00FE), ð (U+00F0) | th, d/th | Icelandic, atomic letters                      |

So accent folding is **NFD plus a curated supplementary map**, not NFD alone.

### Trap 2: folding is often language-wrong

* `ñ` decomposes to `n` + combining tilde, so naive folding merges `año` (year) and `ano`. In modern Spanish `ñ` is its own letter.
* In Swedish, Finnish, Danish, and Norwegian, `å ä ö` / `æ ø å` are distinct letters that sort at the end of the alphabet, not variants of a and o. Folding them is actively wrong for those languages.
* German often wants `ä ö ü` to `ae oe ue` (DIN 5007-2), not `a o u`.
* Turkish dotted/dotless i (i, ı, İ, I) is a locale-specific case-folding minefield.

Conclusion: folding is acceptable for a **search token form** but must never feed
language-specific analysis or sorting.

### Trap 3 (the dangerous one): do not strip all nonspacing marks globally

Stripping category `Mn` is roughly safe for Latin, Greek, and Cyrillic. It is destructive
for many other scripts where combining marks are essential orthography, not decoration:

* Indic (Devanagari, Tamil, ...): viramas and vowel signs change the syllable.
* Arabic: harakat and other marks change vocalization.
* Hebrew: niqqud.
* Thai: vowel and tone marks.

Dropping these does not "remove an accent," it corrupts the word. So accent stripping must be either restricted to a **curated diacritic set** (the Latin/Greek/Cyrillic combining marks we actually mean) or **gated by script**, and documented as a Latin-oriented operation. Never "strip all `Mn`."

## Offsets: why folding forces the dual model

Dash normalization is 1:1 and offset-safe. Accent folding is not: `ß` to `ss` expands, and
decomposed sequences collapse several code points to one base, so both length and offsets shift, often differently per token. You cannot fold in place and keep valid spans.

That is precisely why the original/normalized + offset-map model is mandatory the moment folding enters the picture. It is not a nice-to-have, it is the only correct way to fold and still report results against the source text.

## Architecture: two mechanisms, one model

There are two distinct engines, and they should not be conflated:

* **Set-membership normalization** (whitespace, dashes, and future classes like quotes, ellipsis, digits): the cursor based `CharClass` engine from the base ticket, driven by an explicit `CodePointSet`. Mostly 1:1 or run-collapsing.
* **Decomposition-based normalization** (NFC baseline, NFKC, accent folding): built on `java.text.Normalizer` plus a curated mark set and a script guard. Length-changing.

Both feed the same original/normalized result type and offset map, so callers see one model regardless of which engine produced the normalized form.

## Alignment with existing OpenNLP components

We are not inventing per-language normalization from scratch. OpenNLP already does it at the token level, and the new char-level work should slot beneath it and reuse the same dispatch idea rather than grow a parallel one.

What already exists:

* `Stemmer` (opennlp-api): `CharSequence stem(CharSequence word)`. Implemented by `PorterStemmer` and `SnowballStemmer`, whose `ALGORITHM` enum already covers ~20 languages (Arabic, Catalan, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Indonesian, Irish, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish).
* `Lemmatizer` (opennlp-api): `String[] lemmatize(String[] toks, String[] tags)`. Implemented by `LemmatizerME` (statistical), `DictionaryLemmatizer`, and `MorfologikLemmatizer`. It needs POS tags, so it is the most context-dependent step.
* `LanguageDetector` / `LanguageDetectorME`.

So "apply the rule for the requested language, detect it when unspecified, fall back to a default" is the exact pattern OpenNLP already uses to pick a Snowball stemmer. The char-level normalization should reuse that dispatch (a language profile registry mirroring the Snowball `ALGORITHM` selection, with `LanguageDetector` as the fallback), not reinvent it.

Two axes organize the whole effort:

|                     | Language-independent                                          | Language-specific                                                              |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **Character-level** | NFC, whitespace, dash, NFKC -> fast O(1) cursor pass, default | case fold (Turkish i), German ae/ss, gated accent folding -> per-language char profile |
| **Token-level**     | (little lives here)                                          | stemming (Snowball), lemmatization -> already implemented                       |

Reading it: the fast default is the top-left cell (language-independent char folding, no detection needed). The language-specific char rules are the only genuinely new per-language piece. Stemming and lemmatization are the token-level rungs above accent folding (rungs 7 and 8 of the ladder) and already exist, so the new work normalizes characters first, then hands clean tokens to the existing stemmer / lemmatizer.

## API sketch

```java
// Result of normalizing a piece of text: original stays canonical, normalized is derived.
public record NormalizedText(CharSequence original,
                             CharSequence normalized,
                             OffsetMap map) {
  int toOriginalOffset(int normalizedOffset);   // normalized position -> original range start
  int toNormalizedOffset(int originalOffset);
}

// Token / term level: a layered container (see "Layered token model" below). The request
// config picks the eager layers; other dimensions fill lazily and cache.
public final class Term {
  public Span originalSpan();
  public String original();
  public String at(Dimension dimension);   // O(1) if computed, else lazy compute + memoize
  public String normalized();              // the configured final layer
}

// A composable pipeline of ladder steps. Default = NFC only.
public final class TextNormalizer {
  public static TextNormalizer nfcBaseline();                 // OOTB
  public TextNormalizer withWhitespace(CharClass ws);
  public TextNormalizer withDashes(CharClass dashes);
  public TextNormalizer withNfkc();
  public TextNormalizer withCaseFold(Locale locale);
  public TextNormalizer withDiacriticFolding(ScriptScope scope);   // opt-in, script-gated
  public NormalizedText normalize(CharSequence text);
}
```

The offset map can be a parallel index array (normalized index to original index) or a
run-length edit list. Most runs are identity, so a compact edit list is cheap; the exact
representation is an implementation detail behind `OffsetMap`.

## Layered token model and re-projection

A configured request fixes which dimensions we apply, but callers will want to ask "what would this token look like without the stem?" or "give me the case-folded form." We can answer those fast if a token keeps its intermediate layers instead of only the final string.

Retain, do not invert. You cannot reverse a stem (`run` has no path back to `running`). So a token is not one normalized string, it is a projection stack: `original -> NFC -> ws/dash -> casefold -> accentfold -> stem -> lemma`. Querying "token at layer K" is an array index. O(1), no recompute. The offset/reference map ties each layer back to original coordinates.

Two kinds of layer, stored differently:

* Substring-preserving (NFC base, whitespace, dash, mostly casefold): store an offset range into the base plus the offset map. No string copy.
* Non-substring (stem, lemma, expanding folds like `ß` to `ss`): must materialize the string. These break the "normalized is a slice of original" assumption, which is why each token carries its own back-reference rather than relying on offsets alone.

Transforms do not commute, so this is a stack, not a freely indexable set:

* Peeling the top (last-applied) dimension is O(1): return the layer below. "Take out the stem" is the easy case because stem is last.
* Removing a middle dimension is not a lookup. casefold-then-accentfold differs from the reverse (Turkish i, `ß`), so pulling casefold out of the middle forces a recompute of every layer above it. Bounded by "layers above the removed one," not O(1). Model it as LIFO.

Adding a dimension that was not requested is compute, not lookup. If the pipeline never ran stemming, "now give me the stem" costs one `stem()` call per token, lazily, then memoized:

* configured dimensions: eager, O(1) thereafter.
* everything else: lazy and memoized, first touch pays the transform, the rest are O(1).

Peel is free, add is lazy.

On the O(1) query claim: at token granularity it is genuinely O(1) and cheap, since each token stores its per-layer value or offset. At character granularity, true O(1) offset mapping needs a dense `int[]` (one entry per char, O(n) memory); the cheaper structure is a breakpoint list with O(log n) binary search. For "normalized surface of word i," token-level O(1) is the right answer.

Retention cap: keeping every intermediate layer for every token is fine for a sentence and heavy for a large corpus. Sane default: retain original, final, and the dimensions named in the request; recompute the rest on demand and memoize.

The `Term` defined in the API sketch above is this layered container: one token as a stack of projections, eager for the configured layers and lazy plus memoized for the rest. It is a normalization lattice if alternate paths are allowed instead of a single chain.

## Multilingual OOTB: what ships on by default

* **On by default:** NFC. Safe, lossless under canonical equivalence, fixes the
  precomposed-vs-decomposed mismatch for every script.
* **Opt-in per pipeline:** whitespace normalize/collapse, dash normalize, NFKC, case fold.
* **Opt-in and script-gated:** diacritic folding, emitted only into the normalized form,
  never replacing the original.

Because normalization is additive (original always retained), turning these on is safe even in a multilingual pipeline. The risk is never data loss; it is only producing a normalized form that is unhelpful for a given script, which the gating and the ladder address.

## Scope and follow-up tickets

* **OPENNLP-1850 (this ticket):** whitespace + dashes via the `CharClass` engine, plus the original/normalized result type and offset map as the shared foundation. NFC baseline is a reasonable inclusion here since it is small and safe.
* **Follow-up: diacritic / accent folding.** NFD + curated mark set + supplementary letter map + script gating, on the dual model. Its own ticket because of the multilingual risk.
* **Follow-up: other set-based classes.** Quotes and apostrophes (Unicode `Quotation_Mark`), ellipsis, bullets, non-ASCII digits (Unicode `Nd`), invisible/bidi control stripping. Each is one more `CharClass` preset.
* **Follow-up: NFKC and case folding** as explicit ladder options.

## Standards referenced

* Unicode Character Database, `PropList.txt`: `White_Space`, `Dash`, `Quotation_Mark` properties.
* UAX #15 Normalization Forms (NFC, NFD, NFKC, NFKD).
* UAX #44 Character Database, general categories (`Mn` nonspacing mark, etc.).
* UAX #31 / UTS #39 (confusables) for the later homoglyph work, noted for completeness.
* Cross-references for the ASCII whitespace subset: POSIX / IEEE Std 1003.1, W3C HTML, XML 1.0, JSON (RFC 8259).

## Open questions for discussion

* NFC on input by default: agreed as the 3.0 baseline, or leave even that opt-in?
* Default dash behavior: split-only (delimiter) on by default, with flatten-to-hyphen
  opt-in? Or both opt-in?
* Where should the original/normalized model live: a core text utility, or closer to the
  tokenizer? It needs to be reachable by both core and the DL components.
* Term model: extend an existing type to carry the normalized form, or introduce a new `Term`/`NormalizedText` type for 3.0?
