# Coreference document layer

This feature adds deterministic coreference resolution that records mention chains as a
typed OpenNLP Document layer, with a CoNLL-compatible scorer and an OntoGUM evaluation
harness.

## Human definition

It connects phrases such as "Alice", "she", and "the girl" when they refer to the same
thing, and it says how well it does so.

## Design

- `CorefAnnotator` reimplements the entity-centric, precision-ranked sieves of Lee et
  al. (Computational Linguistics 2013) from the paper, not from Stanford CoreNLP, whose
  code is GPL. Mentions are entities, the noun phrases of an optional
  `ParserAnnotator.PHRASES` or `ChunkerAnnotator.CHUNKS` layer, and personal pronouns;
  attributes come from pronoun forms, tags, titles, a bundled first-name list built from
  the public-domain SSA name tables, and gendered and animate noun lists.
- First and second person pronouns chain per speaker, from an optional
  `CorefAnnotator.SPEAKERS` layer or from quotation marks with speech-verb attribution.
- `ChunkerAnnotator` and `ParserAnnotator` adapt the chunker and the constituency
  parser to the document pipeline; the parser layer carries each phrase's head token.
- `CorefScorer` implements MUC, B-cubed, CEAF-m, CEAF-e, mention detection, and the
  CoNLL average over predicted mentions after Pradhan et al. (ACL 2014), pinned to the
  paper's worked example.
- `CorefTrainer` and `CorefModel` add a learned antecedent ranker after Durrett and
  Klein (EMNLP 2013): a maxent pair classifier over the sieve predicates, heads,
  shapes, distance, and attribute pairs, decoded best-first above a floor tuned on
  dev (0.1), after the speaker and string sieves. Documents train through the
  `gold:opennlp:chains` layer; `ConlluCorefDocumentStream` (opennlp-formats) reads
  OntoGUM or Universal Anaphora CoNLL-U into such documents.
- `OntoGumCorefEvalTest` (opennlp-formats tests) scores a GUM checkout
  (`opennlp.coref.gum.dir`) and appends to `target/coref-eval-results.csv`;
  `opennlp.coref.train` trains on the train split first, `opennlp.coref.model` saves
  or loads a model, `opennlp.coref.dump` writes a mention diagnostic.

## Measurements

OntoGUM (2026 checkout, 32 documents per split), gold sentences, tokens, tags, and
speaker lines, predicted person, location, and organization entities, `en-chunker`
noun phrases:

| State | dev CoNLL | test CoNLL | test mention F |
| --- | --- | --- | --- |
| Before this round (entities and pronouns only) | 9.9 | 9.6 | 15.6 |
| Sieves with chunk mentions | 30.5 | 30.5 | 48.1 |
| Speaker sieve | 45.8 | 42.8 | 63.6 |
| Possessives and generic nouns | 49.5 | 46.5 | 67.8 |
| Ranker (train split, floor 0.1) | 52.0 | 48.6 | 68.4 |

Zhu, Pradhan, and Zeldes (ACL 2021) report 39.7 on the 2021 OntoGUM test set for the
Stanford deterministic system with fully predicted input, and 58.0 for SpanBERT.

## Prior art and comparable products

- [Stanford CoreNLP dcoref](https://www-nlp.stanford.edu/software/dcoref.html), the
  system the paper describes (GPL; not consulted).
- spaCy `coreferee` and `fastcoref` on the Python side; the latter needs a transformer.

## Pull request dependencies

- [Apache OpenNLP PR #1182](https://github.com/apache/opennlp/pull/1182) supplies the
  Document container and typed layers.

Ranker findings: L-BFGS and 300 GIS iterations match 100 GIS iterations (the model is
feature-limited); a virtual "new chain" antecedent scored from anaphor-only features
under-links because pair and anaphor probabilities are not calibrated against each
other, so the floor stays; running the string sieves before the ranker is worth
+1.1 dev over ranking everything.

## Open

- True softmax mention ranking and a static-embedding head similarity feature
  (OPENNLP-1877) as the next accuracy levers; no shipped model yet, and the subset of
  OntoGUM a distributable model may train on is undecided.
- Demonstrative pronouns and appositive spans, which need the parse layer.
