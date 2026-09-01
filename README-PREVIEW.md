# OpenNLP preview build (unofficial, experimental)

The `kristian-3.x-features` branch and its
`ai.pipestream:opennlp-*` artifacts are an unsupported integration preview of
unmerged Apache OpenNLP work and local research. This is not an Apache Software
Foundation release, is not endorsed by the ASF, and does not predict what will
ship in OpenNLP. Apache OpenNLP, OpenNLP, and Apache are ASF trademarks.

The preview lets downstream projects test the in-review and research API surface
without turning the generated integration branch into a source branch. Fixes
land on their owning feature branch. Nothing merges out of the preview.

`PIPESTREAM-PROVENANCE.txt` records the exact Apache main commit and every
selected feature ref in a completed build. It is regenerated with the branch
and is the authority for artifact contents. The membership planned for the next
regeneration lives in the workspace `regen-uber.sh`; the current feature and PR
map is [RESEARCH_BRANCHES.md](RESEARCH_BRANCHES.md).

Rolling snapshots use the version named in the provenance manifest. A manually
dispatched workflow may publish an immutable alpha preview to Maven Central and
tag it with the same provenance. In either form, the artifact remains an
experimental preview rather than an ASF release.

Java package names stay aligned with Apache OpenNLP. When a feature is released
by Apache, consumers can replace the `ai.pipestream` coordinate with the official
`org.apache.opennlp` coordinate.

The upstream Apache License 2.0 `LICENSE` and `NOTICE` files are retained.
Report preview integration problems to ai-pipestream, not to Apache OpenNLP.
