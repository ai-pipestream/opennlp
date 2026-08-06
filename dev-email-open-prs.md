# Overview / pacing email (send this first, alone)

Subject: [DISCUSS] The 12 open PRs and the plan for them

Hi all,

So don't be alarmed: I opened 12 PRs. They are not meant to suggest we need to rush anything in. They are meant to open the discussion for each new feature I am proposing to the project, all of which are well-known, accepted features in the NLP world. Six of them are ready for review and build on what is already merged: speed improvements and some minor fixes. The other six are genuine additions or significant changes to how the code does these things, and I figured now is the right time to bring them up since 3.0 has not shipped yet. That is why there are so many at once: I want to shape the API while we still can, before the release locks it in.

The six ready ones. These build on merged work and four of them are behavior-preserving:

1. #1138 (OPENNLP-1868): bundled Unicode full case folding with provenance and an audit test.
2. #1164 (OPENNLP-1869): emoji and emoticon fold rungs, offset-aware; stacked on #1138 and retargets to main after it merges.
3. #1150 (OPENNLP-1875): whitespace predicates backed by the Unicode Character Database; the remaining legacy call sites are frozen and documented.
4. #1151 (OPENNLP-1876): regex replaced with single-pass scans in the legacy normalizers, byte-identical output.
5. #1161 (OPENNLP-1878): non-breaking performance follow-up for the normalization and tokenization hot paths.
6. #1163 (OPENNLP-1883): thread-safe stemmers plus StemmerFactory with per-thread caching.

The six drafts. Each is a new module or a new API seam, and they stay drafts until we have talked about them:

7. #1152 (OPENNLP-1877): opennlp-embeddings, static embedding tables on the JVM with no native dependencies.
8. #1154 (OPENNLP-1879): gazetteer and geocoder API with a bundled public-domain dataset; this overlaps in spirit with the geoentitylinker addon and that deserves its own honest conversation.
9. #1155 (OPENNLP-1880): lexical knowledge base seam with WN-LMF and WNDB readers plus a Morphy-style lemmatizer.
10. #1165 (OPENNLP-1885): subword tokenization: a contract in opennlp-api plus SentencePiece and WordPiece implementations with exact original-text spans.
11. #1166 (OPENNLP-1886): UniNE light and minimal stemmer tiers; stacked on #1163.
12. #1167 (OPENNLP-1887): weighted lexical expansion over the knowledge base seam; stacked on #1155.

My plan is to start a separate thread for each draft feature over the next couple of weeks: the reasoning, how it looks in use, where it differs from prior art including our own addons, and the specific questions I want input on. The drafts are working code so those discussions can be concrete, not so the decisions are pre-made. If the pace is too much, say so and I will slow down. And if any of this steps on work you know about that I missed, I want to hear about it.

Thanks,
Kristian

---

# REVIEW-TIER EMAILS (the six ready PRs; short, review requests)

Send whenever; #1138 first since it unblocks #1164. #1151 and #1163 already have a review round from Richard addressed and replied to, so those two emails may be unnecessary; included in case you want the list-visible summary.

---

## Case folding and emoji folding (#1138 + #1164)

Subject: Review request: bundled full case folding and emoji/emoticon fold rungs (OPENNLP-1868, OPENNLP-1869)

Hi all,

Two PRs on the same feature line, best reviewed in order. https://github.com/apache/opennlp/pull/1138 adds bundled full case folding from the Unicode CaseFolding.txt data, with provenance recorded, an audit test that fails if the data drifts from the declared Unicode version, and the LICENSE/NOTICE entries. https://github.com/apache/opennlp/pull/1164 builds on it: offset-aware bidirectional emoji and emoticon folds, including ZWJ sequences and variation selectors. Both land as opt-in builder rungs on the normalizer chain, so nothing changes for anyone who does not add the rung. #1164's diff shows both PRs until #1138 merges; after that I retarget it to main.

Thanks,
Kristian

---

## UCD-backed whitespace predicates (#1150)

Subject: Review request: UCD-backed whitespace predicates and call-site migration (OPENNLP-1875)

Hi all,

https://github.com/apache/opennlp/pull/1150 migrates the user-text call sites to the Unicode White_Space based predicate; every call site that intentionally stays legacy is frozen with a documented justification and a characterization test. One deliberate fix rides along: keepNewLines now emits U+0085, U+2028, and U+2029 as newline tokens instead of silently dropping them. The file count overstates the size; most touches are one-line migrations plus their pins.

Thanks,
Kristian

---

## De-regex the legacy normalizers (#1151)

Subject: Review request: cursor scans replacing regex in the legacy CharSequenceNormalizers, byte-identical (OPENNLP-1876)

Hi all,

https://github.com/apache/opennlp/pull/1151 replaces the regex pipelines in the legacy CharSequenceNormalizers with single-pass cursor scans; output is byte-identical, proven by differential tests over shared characterization inputs, and unchanged inputs are returned without copying. Jeff measured about 2.4x on tweet-like, prose, and URL-heavy corpora (numbers on the PR). Richard's review round is addressed: the null contract is now uniform across the package with a contract test over every implementation, and TwitterCharSequenceNormalizer is renamed to SocialMediaCharSequenceNormalizer with the old name kept as a deprecated alias.

Thanks,
Kristian

---

## Hot-path performance follow-up (#1161)

Subject: Review request: non-breaking performance follow-up for normalization and tokenization (OPENNLP-1878)

Hi all,

https://github.com/apache/opennlp/pull/1161 is the collected allocation and scan reductions from the OPENNLP-1850 round that did not belong in the feature PRs: no API changes, no behavior changes, outputs pinned by the existing suites plus added regression tests. Smallest PR of the round, roughly 670 added lines including tests.

Thanks,
Kristian

---

## Stemmer thread safety (#1163)

Subject: Review request: thread-safe stemmers with StemmerFactory and per-thread caching (OPENNLP-1883)

Hi all,

https://github.com/apache/opennlp/pull/1163 makes the stemmers thread-safe and adds StemmerFactory with per-thread caching: existing constructors keep working, the factory adds the safe sharing path, and pooled-thread environments get the same clearThreadLocalState() release the *ME components have. Richard's review round is addressed, including cache clearing, interned cache values, and the per-thread routing pattern consolidated into one class.

Thanks,
Kristian

---

# DISCUSS-TIER EMAILS (the six drafts; one genuine ask each, spaced out)

---

## Static embeddings (#1152)

Subject: [DISCUSS] OPENNLP-1877: text embedding on the JVM, two engines behind one seam (PR #1152)

Hi all,

Draft PR #1152 adds a text-level embedding contract to opennlp-api, TextEmbedder (embed, embedAll, dimension), and the two implementations that make it a contract rather than a wrapper. The new opennlp-embeddings module runs distilled static embedding tables (the Model2Vec-family successors of word2vec/GloVe) in pure JVM: no native runtime, no GPU, about 766k short sentences per second on one core. And the existing ONNX SentenceVectorsDL in opennlp-dl implements the same interface additively: getVectors is untouched, it gained embed() and dimension(), pinned by a test against a real ONNX session. It stays what it is, the contextual-accuracy option; the static module is the throughput option. Pipelines pick the trade-off, the type stays the same. The interface javadoc also spells out how this relates to the word-level WordVectorTable, which stays untouched.

The question I want input on: model distribution. The static module ships code only, bring-your-own-table. If we want an out-of-the-box experience, published tables on the model pipeline are the missing piece, and the batched embedAll override for the ONNX side (one padded run per batch) is open for whoever wants it.

Thanks,
Kristian

---

## Subword tokenization (#1165)

Subject: [DISCUSS] OPENNLP-1885: subword tokenization with exact original-text spans (PR #1165)

Hi all,

Draft PR #1165 adds a SubwordTokenizer contract to opennlp-api: pieces with vocabulary ids and exact spans into the original, unnormalized text. Two implementations back it. SentencePiece inference (unigram and BPE, the model's own normalization, byte fallback, user-defined symbols) lives in a new opennlp-subword module, parity-fixtured piece by piece against the reference implementation, with single-thread throughput above the reference binding called from a host language (numbers and scope on the PR). WordpieceEncoder in opennlp-api runs the full BERT pipeline with ids and spans; a differential suite holds it byte-identical to the existing chain on a curated corpus plus 800 randomized inputs, and the dl tokenizer creation now builds on it. The unreleased BertTokenizer folded into it; its HuggingFace-verified test sequences were ported case for case. Spans through subword tokenization are what let dl predictions map back to document offsets, the running theme of the 3.0 line.

The question I want input on, mainly from Jeff as the wordpiece author: WordpieceTokenizer. My proposal is deprecate-and-freeze in 3.0: it keeps its standalone raw-wordpiece behavior untouched until 4.0 (the encoder deliberately does not reproduce that mode), and its tokenizePos could even start working under those semantics instead of throwing. If you would rather keep it undeprecated alongside, that also works; the contract does not force anything.

Thanks,
Kristian

---

## Light and minimal stemmers (#1166)

Subject: [DISCUSS] OPENNLP-1886: UniNE light and minimal stemmer tiers (PR #1166)

Hi all,

Draft PR #1166 adds sixteen light and minimal stemmers (Jacques Savoy's UniNE family) as the gentler alternative to Snowball, which overstems for many applications. Each is stateless, implements the existing Stemmer and StemmerFactory interfaces, and is verified against vocabulary parity fixtures generated by running the original Apache-licensed implementations, with the UniNE notices carried over. The diff currently includes #1163 (it builds on the factory work and collapses once that merges), and about 29k of the added lines are fixture data; the reviewable code is roughly 3k lines.

The question I want input on: language priorities. Galician and Portuguese minimal need the RSLP rule engine and are the natural follow-up; if there are languages your users actually stem, that list should drive what comes next.

Thanks,
Kristian

---

## Lexical knowledge base and expansion (#1155 + #1167)

Subject: [DISCUSS] OPENNLP-1880/1887: lexical knowledge base seam, readers, and weighted expansion (PRs #1155, #1167)

Hi all,

Draft PR #1155 adds a format-neutral lexical knowledge base seam to opennlp-api (synsets, relations, POS) with two readers behind it, WN-LMF XML (what current lexicons ship, including the actively maintained English one) and the classic WNDB format, plus a Morphy-style lemmatizer implementing the existing Lemmatizer interface. Reader agreement is tested by loading the same lexicon through both formats. Our own history here is the jwnl addon, which wraps a third-party library and reads only the classic format; this seam carries no third-party dependency. Draft PR #1167 builds on it: weighted lexical expansion (synonyms, hypernym walk, optional hyponyms) with decay ranking; its diff collapses once #1155 merges.

The question I want input on: which lexicons matter to you beyond the English one? The WN-LMF reader is tested against what I could get; real-world files from other languages would harden it before 3.0.

Thanks,
Kristian

---

## Gazetteer and geocoder (#1154)

The complete package for this one lives in gazetteer-discuss-kit.md: the addons JIRA to file first, the intro email with the single geoentitylinker-collaboration ask, the rewritten PR body, and the reply bank. Use that file; this section is only the pointer so the sequence is complete.
