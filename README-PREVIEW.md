# OpenNLP preview build (unofficial, experimental)

This branch and the `ai.pipestream:opennlp-*` artifacts built from it are an **experimental, unsupported preview**, built by merging unmerged development branches of [Apache OpenNLP](https://opennlp.apache.org/) ahead of their review and release. **This is not an Apache Software Foundation release**, is not endorsed by the ASF, and nothing in it is promised for any future OpenNLP version. Apache OpenNLP, OpenNLP, and Apache are trademarks of the Apache Software Foundation.

What this exists for: downstream projects that want to build against the in-review API surface (the document annotation container, gazetteer/geo stack, subword tokenization, static embeddings, and related work) without pressuring the upstream review queue. When the corresponding features ship in an official Apache OpenNLP release, switch your dependency to the official `org.apache.opennlp` coordinates; the Java package names are identical by design, so migration is a coordinate swap.

Exactly which refs a build contains is recorded in `PIPESTREAM-PROVENANCE.txt` at the repository root, regenerated on every rebuild: the upstream main commit, every open pull-request head, and every feature-branch tip merged in, each with its commit id. The version string (`3.x-preview-<date>-g<main-sha>`) repeats the date and upstream base. The `3.x` prefix is deliberate: it makes no claim about what lands in OpenNLP 3.0.

Licensing: Apache License 2.0, unchanged. The upstream `LICENSE` and `NOTICE` files are retained as-is; this file and the provenance manifest constitute the statement of modification required by section 4 of the license (the modification being the merge of the listed development branches). Report problems with this preview to the ai-pipestream repository, never to the Apache OpenNLP project; upstream owns none of this build.

Branch layout of this fork: `main` mirrors `apache/opennlp` main exactly and never diverges; `krickert-main` (this branch) is the regenerated integration line the preview artifacts are built from; the remaining branches are the individual features, each stacked on its true dependency.
