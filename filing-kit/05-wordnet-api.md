# wordnet-api

## JIRA fields

*Type:* New Feature. *Components:* opennlp-api, extensions (new module opennlp-wordnet). *Note:* code only, no lexicon data bundled.

## JIRA description (wiki markup)

```
h2. Summary

OpenNLP has no lexical-semantic layer: no synonym lookup, no is-a hierarchy, and its lemmatizers are either trained models or flat dictionaries. This ticket adds the lexical knowledge base seam and two clean-room readers for the WordNet family of resources. The contracts land in opennlp-api (package {{opennlp.tools.wordnet}}): {{LexicalKnowledgeBase}} (lemma and synset lookup with typed relation navigation), {{Synset}}, {{WordNetPos}}, and {{WordNetRelation}}. The seam interface carries a generic name rather than a brand: a legacy Princeton database, an Open English WordNet release, a future bundled permissively-licensed lexicon, and a future user-downloaded lexicon are all implementations behind the one contract, and synset identity is opaque and source-qualified.

A new {{opennlp-extensions/opennlp-wordnet}} module ships two readers producing equivalent immutable, thread-safe lexicon views: a StAX reader for WN-LMF XML (the Global WordNet Association interchange format used by Open English WordNet and many other language wordnets) and a reader for legacy Princeton WNDB directories. Both are clean-room, built from the published format documentation; no third-party WordNet library is used or referenced. The XML reader is hardened against XXE per the OWASP posture for DOCTYPE-bearing formats: the DOCTYPE declaration real releases ship is tokenized and skipped while external entities and the external DTD subset stay disabled, so an unmodified Open English WordNet download parses directly, and a dedicated test proves a DOCTYPE-declared internal-subset entity payload fails loud rather than expanding. On top of the seam, {{MorphyLemmatizer}} implements the existing {{Lemmatizer}} interface with the documented Morphy algorithm: exception-list lookup first, then per-part-of-speech detachment rules with every candidate validated against the lexicon.

The module ships code only; users point it at a WordNet database they downloaded. Verified against Princeton WordNet 3.0 (117,659 synsets) and Open English WordNet 2024 (120,630 synsets).

h2. Scope

* Contracts in opennlp-api: {{LexicalKnowledgeBase}}, {{Synset}}, {{WordNetPos}}, {{WordNetRelation}} (28 relation types plus the derived aliases the formats express).
* {{WnLmfReader}}: XXE-hardened StAX reader for WN-LMF XML; sense relations lifted to the synset level; fail-loud structural validation naming the resource and line.
* {{WndbReader}}: reader for Princeton WNDB directories (index and data files for all four parts of speech); offset-contract validation; adjective satellite normalization and syntactic-marker stripping.
* A structural equivalence test over matching miniature WN-LMF and WNDB fixtures, so the two readers are pinned to produce the same lexicon.
* {{MorphyLemmatizer}} plus {{MorphyExceptions}} (the {{*.exc}} exception-list reader), implementing the existing {{Lemmatizer}} interface.

h2. Acceptance criteria

* Both readers load their real full-size counterparts (Princeton 3.0 and OEWN 2024) and agree structurally on the shared miniature fixtures.
* The XXE posture is tested: an unmodified DOCTYPE-bearing document parses; an internal-subset entity payload fails loud without expansion.
* All lexicon views are immutable and thread-safe, with a concurrency test.
* No lexicon data in the source tree or artifacts; no new dependencies; no third-party WordNet code consulted or referenced.

h2. Out of scope

* Bundling any lexicon data, including the exception lists (rationale documented in the javadoc; a bundled permissively-licensed lexicon is a later, separately-licensed decision).
* Sense keys and {{index.sense}} (a later sense-inventory layer; the v1 contract has no sense-key surface).
* Similarity measures and query expansion features that stack on the seam.
* The {{DictionaryLemmatizer}} javadoc discrepancy (documents "0" but returns "O"), a separate one-line fix.
```

## PR title

```
<KEY>: Lexical knowledge base seam with WN-LMF and WNDB readers and a Morphy lemmatizer
```

## PR body (markdown)

```
Adds the format-agnostic lexical knowledge base seam to opennlp-api (`opennlp.tools.wordnet`: `LexicalKnowledgeBase`, `Synset`, `WordNetPos`, `WordNetRelation`), and a new `opennlp-extensions/opennlp-wordnet` module with two clean-room readers behind it:

- `WnLmfReader`: a StAX reader for WN-LMF XML, Open English WordNet's interchange format. XXE-hardened per the OWASP posture for DOCTYPE-bearing formats: the DOCTYPE line real OEWN releases ship is tokenized and skipped while external entities and the external DTD subset stay fully disabled, so an unmodified download parses directly; a dedicated test proves a DOCTYPE-declared internal-subset entity payload fails loud rather than expanding.
- `WndbReader`: a reader for legacy Princeton WNDB directories, validating the format's documented offset contract.

Both produce equivalent immutable, thread-safe lexicon views, pinned by a structural equivalence test over matching miniature fixtures. On top of the seam, `MorphyLemmatizer` implements the existing `Lemmatizer` interface with the documented Morphy algorithm: exception-list lookup first, then per-POS detachment rules with every candidate validated against the lexicon.

**Naming:** the seam interface is `LexicalKnowledgeBase` rather than a WordNet-branded name; the contract is generic and only the format-specific readers carry the WordNet name.

The module ships code only; users point it at a WordNet database they downloaded. Verified against Princeton 3.0 (117,659 synsets) and OEWN 2024 (120,630 synsets). No third-party WordNet library is used or referenced.

Verification: api 303/0 (+17), opennlp-wordnet 86/0, full reactor verify green.

Small upstream note found along the way: `DictionaryLemmatizer` javadoc says unknown = "0" but the code returns the letter "O"; Morphy matches the code. Filed separately.
```
