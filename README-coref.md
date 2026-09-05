# Coreference document layer

`CorefAnnotator` resolves mentions that refer to the same entity and records them in a
typed `Document` layer. It can run as a deterministic resolver or with a trained
`CorefModel` antecedent ranker.

## Human definition

It connects phrases such as "Alice", "she", and "the engineer" when they refer to the
same person. Each result retains its exact span in the original text, so applications
can group, highlight, or inspect every mention.

## Implementation

- Mention candidates come from entity, pronoun, chunk, and parser phrase layers.
- The deterministic resolver applies speaker, string, head, attribute, and pronoun
  rules in precision order.
- `CorefTrainer` trains pairwise or mention-ranking models from a
  `gold:opennlp:chains` layer.
- `ConlluCorefDocumentStream` reads gold chains from OntoGUM and CorefUD CoNLL-U data.
- Optional `WordVectors` and `TokenVectors` providers add lexical and contextual
  similarity features. `TokenVectorsDL` supplies contextual vectors from a BERT-style
  ONNX encoder. Each provider declares its vector dimension, and `CorefModel` records
  the contextual dimension required at inference.
- `CorefScorer` reports MUC, B-cubed, CEAF-m, CEAF-e, mention detection, and the CoNLL
  average.

## Evaluation

The OntoGUM runs use gold sentences, tokens, and POS tags, with predicted entities and
chunks. They are not a direct comparison with systems evaluated on fully predicted
input. The 32-document splits include two Reddit documents whose text GUM redacts;
`opennlp.coref.skip.redacted` omits them when a text-complete score is needed.

| Resolver | OntoGUM test CoNLL | GAP development F1 | GAP test F1 |
| --- | ---: | ---: | ---: |
| Rules | 46.8 | 24.3 | 25.7 |
| Ranker trained on all OntoGUM training documents | 50.0 | 45.1 | 45.6 |
| Ranker trained on 59 CC BY OntoGUM training documents | 49.4 | 46.1 | 45.5 |

The GAP runs use predicted sentence, token, POS, entity, and chunk layers. The GAP
paper reports 50.5 development F1 for Stanford dcoref and 41.5 for its random baseline.
The CC BY ranker is above the random baseline but does not yet match dcoref on this
fully predicted input.

The proposed distributable model uses only the 59 CC BY OntoGUM training documents.
The corpus annotations are CC BY 4.0; the evaluation harness keeps all corpus data and
models outside the repository.

## Prior art and products

- [Stanford CoreNLP dcoref](https://www-nlp.stanford.edu/software/dcoref.html)
- [Coreferee](https://github.com/richardpaulhudson/coreferee)
- [FastCoref](https://github.com/shon-otmazgin/fastcoref)

## Pull request dependencies

The feature uses the `Document` container and typed layers merged in
[Apache OpenNLP PR #1182](https://github.com/apache/opennlp/pull/1182). It has no open
pull request dependency beyond current `main`.

## Remaining work

- Evaluate a nonlinear scorer for contextual span vectors.
- Prune near-zero ranker weights before publishing the model.
