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
| Ranker with MiniLM-L6 contextual span features | 53.1 | | |
| Wider candidates: demonstratives, `X of Y`, `X and Y` | 53.0 | 49.8 | 68.6 |

Zhu, Pradhan, and Zeldes (ACL 2021) report 39.7 on the 2021 OntoGUM test set for the
Stanford deterministic system with fully predicted input, and 58.0 for SpanBERT.

Two documents per split are reddit texts that GUM ships as underscores, so no system
can score them; `-Dopennlp.coref.skip.redacted=true` leaves them out. On the other 30
documents the current ranker reaches dev 55.6 and test 51.9 (rules 52.4 dev).

Other corpora, read from CorefUD 1.2 with the same reader (dev CoNLL, no singletons):

| Training data | OntoGUM dev | CorefUD GUM dev | LitBank dev |
| --- | --- | --- | --- |
| Rules | 49.5 | 47.7 | 36.9 |
| OntoGUM train | 53.1 | 50.0 | 49.4 |
| CorefUD GUM train (native scheme) | 50.7 | 50.8 | |
| CorefUD GUM + LitBank train | 48.6 | 48.2 | 51.6 |
| LitBank train | 48.4 | | 51.4 |

The OntoNotes-scheme conversion is the better training target even for GUM's own
development set; LitBank, which annotates only ACE entity types, helps LitBank alone.
CorefUD labels English-GUM CC BY-NC-SA 4.0 and English-ParCorFull CC BY-NC 4.0 after
their most restrictive texts; English-LitBank is CC BY 4.0. Within GUM, the academic,
court, news, and interview texts are CC BY, bio and voyage CC BY-SA, and essay,
fiction, letter, podcast, whow, and reddit non-commercial; the annotations are CC BY 4.0
throughout. A ranker trained on the 59 CC BY training documents alone reaches dev 52.4 /
test 49.3, on the 133 documents outside the non-commercial genres 52.5 / 49.9, against
53.0 / 49.8 on all 213, so a licence-clean model costs about half a point.

GAP (Webster et al. 2018; 2,000 Wikipedia snippets per split, one pronoun and two names
each, every layer predicted, harness `GapCorefEvalTest`), F1 over both labels with the
paper's masculine/feminine bias ratio:

| System | development F1 (M / F, bias) | test F1 (M / F, bias) |
| --- | --- | --- |
| Rules | 24.5 (25.5 / 23.5, 0.92) | 25.7 (23.9 / 27.4, 1.15) |
| Mention ranker | 42.5 (44.1 / 40.9, 0.93) | 43.1 (44.4 / 41.8, 0.94) |

The paper reports on development, full snippets: Random 41.5, Stanford dcoref (Lee et
al. 2013) 50.5, Clark and Manning 2015 55.0, Lee et al. 2017 64.7. The ranker sits at the
random baseline here: the pronoun links to some mention in 1,984 of 2,000 snippets, but
only 318 of 797 predicted name links are right. The names are mostly not found by the
entity models (an entity appears in 521 pronoun chains), so they enter as untyped proper
noun phrases with no gender or animacy, and any pronoun may take them or a nearer common
noun (`Simon's class` for `her`). Reading the first-name list for untyped proper phrases
is the obvious next fix; GAP measures exactly the regime OntoGUM's gold tokens hide.

Widening the pronoun window from three to six sentences changes nothing (ranker 53.0,
rules 49.8 dev), so the window is not what limits pronoun recall.

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

Contextual vectors: `TokenVectors` supplies one vector per token of a sentence and
`TokenVectorsDL` (opennlp-dl) implements it over a BERT-style ONNX encoder, aligning
wordpieces to words and windowing long sentences. With all-MiniLM-L6-v2 the ranker
gets bucketed span cosines and real-valued product, difference, and new-chain features;
the weights train (the cosine buckets come out monotone, +0.55 at cosine 0.9 or more to
-0.5 at 0.1) but dev CoNLL stays at 53.1, so a linear head over frozen sentence-encoder
vectors is redundant with the string features. The hook stays for a nonlinear scorer or
a coreference-tuned encoder; no encoder is bundled. Training keeps one span vector per
mention and derives the dense terms on the fly; materializing them per option runs out
of heap.

Wider candidates: standalone demonstratives, `X of Y`, and `X and Y` phrases cut the
missed key mentions on dev from 1007 to 813 while the ranker's new-chain option acts as
the span scorer, since response singletons are dropped. The gain is small (test 49.3 to
49.8) because the demonstratives mostly point at clauses and the new spans also produce
new wrong links (spurious 511 to 574).

## Open

- No shipped model yet; the subset of OntoGUM a distributable model may train on is
  undecided. Model files are large (11 MB) because no feature cutoff is applied to the
  ranker; pruning near-zero weights is an easy follow-up.
- Appositive spans and NP + relative clause spans, which need the parse layer; verbal
  (event) mentions, which OntoNotes annotates and the detector does not produce.
- A nonlinear pair scorer over the contextual span vectors, the form in which frozen
  encoders help in the literature; the linear head measured here does not.
- Gender and animacy for untyped proper noun phrases from the first-name list, and a
  stronger person name finder, which GAP shows to be the pronoun bottleneck on text
  without gold entities.
