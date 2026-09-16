# WN-LMF lexicon extension composition

This feature composes a caller-resolved WN-LMF `LexiconExtension` over its exact base lexicon while validating versions, identifiers, external references, cycles, and nesting limits.

## Human definition

It lets an application add its own entries, senses, synsets, and relations to an existing WordNet without modifying or replacing the original lexicon.

## Prior art and comparable products

- The [Global WordNet Association WN-LMF schemas](https://globalwordnet.github.io/schemas/) define the exchange format and extension model implemented by this feature.

## Pull request dependencies

- [Apache OpenNLP PR #1155](https://github.com/apache/opennlp/pull/1155) supplies the WordNet API and WN-LMF reader.
