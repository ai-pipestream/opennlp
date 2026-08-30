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
  Klein (EMNLP 2013) over the sieve predicates, heads, shapes, distance, attribute
  pairs, cluster size, speaker equality, and optional `WordVectors` head similarity,
  run after the speaker and string sieves. `trainRanking` is a latent-antecedent
  softmax mention ranker (AdaGrad, weights stored in a `GISModel`) that links when
  the best candidate outscores the new-chain option; `train` is the pairwise maxent
  classifier with a dev-tuned floor (0.1). Anaphors whose gold antecedents the
  detector missed are left out of training. Documents train through the
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
| Pairwise ranker (train split, floor 0.1) | 52.0 | 48.6 | 68.4 |
| Mention ranker (10 epochs, step 0.05, L2 1e-3) | 53.1 | 49.3 | 68.0 |

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
feature-limited); running the string sieves before the ranker is worth +1.1 dev over
ranking everything; a virtual new-chain option inside the pairwise classifier
under-links (uncalibrated), while the same option inside a softmax ranker works once
unlearnable anaphors are excluded (51.4 to 52.1 dev before tuning); GloVe 6B head
similarity adds nothing measurable (52.0 to 52.1 dev), so the `WordVectors` hook is
kept for a stronger embedding but no vectors are bundled; step size matters (0.3 loses
3 points, 0.05 gains 0.7), 20 epochs do not beat 10.

## Open

- No shipped model yet; the subset of OntoGUM a distributable model may train on is
  undecided. Model files are large (11 MB) because no feature cutoff is applied to the
  ranker; pruning near-zero weights is an easy follow-up.
- Demonstrative pronouns and appositive spans, which need the parse layer.
