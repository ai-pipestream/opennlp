# Research branch map

State verified on 2026-09-03 against Apache GitHub, every live worktree in
`repos.all.txt`, the org remote, `regen-uber.sh`, and the current provenance
manifest.

Apache main is `73f9a25f7`. It includes the typed Document container from
merged PR #1182, the paragraph normalizer from merged PR #1249, and term
vectors from merged PR #1212. The public `OPENNLP-1833-grpc-helper` is
`2b5ebbc9d` locally and `a4084cabe0` on Apache until pushed. The local helper
contains Apache main plus all 15 open pull
requests in the 3.x review graph, drafts
included. No open 2.x pull request belongs in this helper.

`regen-uber.sh` now starts from current main, drops merged #1182, #1249, and
#1212 branches,
and selects the cascaded heads shown below. The exact refs in the last
completed preview remain recorded in `PIPESTREAM-PROVENANCE.txt`; during a
cascade that generated manifest can temporarily trail this membership plan.

## Integration model

- `APACHE_TIPS` and `RESEARCH_TIPS` in `regen-uber.sh` are the durable uber
  membership lists.
- The gRPC helper is a different aggregation. It takes every open 3.x PR head,
  not only the Apache tips selected for uber.
- A listed tip brings its dependency stack. Do not merge both a stack tip and
  a second copy of the same feature.
- `PIPESTREAM-PROVENANCE.txt` describes only the last completed uber build.
- Nothing merges out of uber. Fixes land on the owning branch, then the stack
  is cascaded and uber is regenerated.

## Uber branch topology

Solid arrows show true Git parentage. The dotted gazetteer arrow is a copied,
patch-equivalent API dependency carried by the region stack. Arrows into
`APACHE_TIPS` and `RESEARCH_TIPS` show the direct membership recorded in
`regen-uber.sh`; a selected child also brings its parent stack. The two gray
research branches remain available but are not selected for uber.

```mermaid
flowchart LR
  main["apache main<br/>OPENNLP-1888, OPENNLP-1921, and OPENNLP-1897 included"]
  uber["kristian-3.x-features<br/>generated uber"]
  apacheTips{{"APACHE_TIPS"}}
  researchTips{{"RESEARCH_TIPS"}}

  subgraph apache["Apache-visible feature branches"]
    gaz["#1154 Gazetteer API"]
    wordnet["#1155 WordNet API"]
    expansion["#1167 WordNet expansion"]
    light["#1166 Light stemmers"]
    sentencepiece["#1165 SentencePiece"]
    static["#1152 Static embeddings<br/>add-ons reconciliation"]
    turbo["#1213 TurboQuant"]
    vector["#1214 Vector index"]
    evaluation["#1215 Vector evaluation"]
    hunspell["#1190 Hunspell"]
    lattice["#1191 CJK lattice"]
    installer["#1211 Resource installer"]
    parser["#1236 Dependency parser"]
    dependency["#1237 Dependency annotations"]
    relation["#1238 Relation extraction"]
  end

  subgraph research["Research branches"]
    artifacts["Text artifacts"]
    assets["Embedded assets"]
    noise["Noise annotations"]
    predicates["Predicate annotators"]
    ffpos["Feedforward POS tagger"]
    bilstm["BiLSTM tagger"]
    glossary["Glossary"]
    pii["PII"]
    coref["Coreference"]
    numeric["Numeric annotations"]
    region["Region vote"]
    geocode["Geocode annotator"]
    hierarchy["Hierarchy annotator"]
    profiles["Place profiles"]
    embedding["Embedding annotator"]
    symbols["Symbol joiner"]
    morfologik["Morfologik FSA"]
    dehyphenation["Dehyphenation"]
    recase["Spellcheck recase<br/>not selected"]
    wordnetExtension["WordNet extension<br/>not selected"]
    major0["Preview major-0 model patch"]
  end

  previewDocs["preview-docs"]

  main --> gaz
  main --> wordnet --> expansion
  main --> light
  main --> sentencepiece --> static --> turbo --> vector --> evaluation
  main --> hunspell
  main --> lattice
  hunspell --> installer
  lattice --> installer
  main --> parser --> dependency --> relation

  main --> artifacts
  main --> assets --> noise
  main --> predicates
  main --> ffpos --> bilstm
  main --> glossary
  main --> pii
  main --> coref
  main --> numeric --> region --> geocode --> hierarchy
  gaz -. copied API dependency .-> region
  gaz --> profiles
  static --> embedding
  main --> dehyphenation
  main --> symbols
  main --> morfologik
  main --> recase
  wordnet --> wordnetExtension
  main --> major0

  gaz --> apacheTips
  wordnet --> apacheTips
  expansion --> apacheTips
  light --> apacheTips
  hunspell --> apacheTips
  lattice --> apacheTips
  installer --> apacheTips
  evaluation --> apacheTips
  relation --> apacheTips

  artifacts --> researchTips
  assets --> researchTips
  noise --> researchTips
  predicates --> researchTips
  ffpos --> researchTips
  bilstm --> researchTips
  glossary --> researchTips
  pii --> researchTips
  coref --> researchTips
  hierarchy --> researchTips
  profiles --> researchTips
  embedding --> researchTips
  symbols --> researchTips
  morfologik --> researchTips
  dehyphenation --> researchTips
  major0 --> researchTips

  main -->|base| uber
  apacheTips --> uber
  researchTips --> uber
  previewDocs --> uber

  classDef parked fill:#eee,stroke:#888,color:#555;
  class recase,wordnetExtension parked;
```

## Open Apache 3.x pull requests

| PR | Feature | Live base | Apache head | Local cascaded head | Review state | Uber membership |
| --- | --- | --- | --- | --- | --- | --- |
| [#1152](https://github.com/apache/opennlp/pull/1152) | Static embeddings | `OPENNLP-1885-sentencepiece` | `f6f1bef48e` | `18b3426767` | Draft, no decision | Via #1215 |
| [#1154](https://github.com/apache/opennlp/pull/1154) | Gazetteer and geocoder API | `main` | `b81d0eb978` | `2b72aa1a66` | Draft, review required | Direct |
| [#1155](https://github.com/apache/opennlp/pull/1155) | WordNet API and readers | `main` | `9c0e9a0533` | `19117b0315` | Draft, review required | Direct |
| [#1165](https://github.com/apache/opennlp/pull/1165) | Subword API and WordPiece | `main` | `96d2781cb2` | `d2c216e749` | Ready, review required | Via #1215 |
| [#1166](https://github.com/apache/opennlp/pull/1166) | UniNE light and minimal stemmers | `main` | `ff681b3934` | `e0a46ee89d` | Draft, review required | Direct |
| [#1167](https://github.com/apache/opennlp/pull/1167) | WordNet expansion | `main` | `a68762fe35` | `e6fad91bf2` | Draft, review required | Direct |
| [#1190](https://github.com/apache/opennlp/pull/1190) | Hunspell stemming | `main` | `c833264c1c` | `b2b5bd9da1` | Ready, changes requested | Direct |
| [#1191](https://github.com/apache/opennlp/pull/1191) | CJK lattice tokenization | `main` | `dba1353580` | `01c1875d23` | Ready, changes requested | Direct |
| [#1211](https://github.com/apache/opennlp/pull/1211) | Verified resource installer | `main` | `b1b38039f7` | `79dd6ec5f4` | Ready, review required | Direct |
| [#1213](https://github.com/apache/opennlp/pull/1213) | TurboQuant embedding tables | `main` | `05398bb677` | `16e4ab4d17` | Draft, review required | Via #1215 |
| [#1214](https://github.com/apache/opennlp/pull/1214) | Bounded in-memory vector indexes | `main` | `ac501e2259` | `bc844412e4` | Draft, review required | Via #1215 |
| [#1215](https://github.com/apache/opennlp/pull/1215) | Vector-search evaluation | `main` | `5b79627bb8` | `89d8fba4c5` | Draft, review required | Direct stack tip |
| [#1236](https://github.com/apache/opennlp/pull/1236) | Dependency parser | `main` | `83bc298189` | `437d01ff56` | Ready, review required | Via #1238 |
| [#1237](https://github.com/apache/opennlp/pull/1237) | Dependency Document layer | `OPENNLP-547-dependency-parser` | `75bf330a26` | `a974a8d8c1` | Draft, no decision, conflicting | Via #1238 |
| [#1238](https://github.com/apache/opennlp/pull/1238) | Dependency-path relation extraction | `OPENNLP-1919-dependency-annotations` | `9c860c3afc` | `20e99af717` | Draft, no decision | Direct stack tip |

PRs #1182, #1212, #1247, #1249, and #1250 merged and are now supplied by
main. The earlier local competing OPENNLP-1921 branch is review-only and is not
an uber feature source. The local static-embedding line contains the API-only
#1165 head but still carries its earlier in-repository SentencePiece
implementation so the preview remains buildable. Replace that copy with the
add-ons module during embeddings reconciliation. The local cascaded heads have
not been pushed to their Apache branches.

## Research worktrees

Every research worktree was cascaded on 2026-09-03. Current Apache main is an
ancestor of every branch, and every stacked child has its current true parent
as an ancestor. `Org ref differs` means the ai-pipestream branch still points
to the pre-cascade history. It must not be updated without explicit push
authorization.

| Branch | Local head | Preview membership | True dependency | Org ref |
| --- | --- | --- | --- | --- |
| `OPENNLP-XXXX-bilstm-tagger` | `3d0171c3ff` | Direct | Feedforward tagger | Differs |
| `OPENNLP-XXXX-coref` | `70548d2925` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-dehyphenation` | `730f69cf4d` | Direct | Apache main with #1212 | Differs |
| `OPENNLP-XXXX-embedded-assets` | `d9f1965696` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-embedding-annotator` | `3ba4e0a634` | Direct | #1152 static embeddings | Differs |
| `OPENNLP-XXXX-ff-postagger` | `02744d5e84` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-geocode-annotator` | `f1ce6f0a38` | Via hierarchy tip | Region vote | Differs |
| `OPENNLP-XXXX-glossary` | `d708f7d9fe` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-hierarchy-annotator` | `1088aa8115` | Direct stack tip | Geocode annotator | Differs |
| `OPENNLP-XXXX-morfologik-fsa` | `a0d1e6ba61` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-noise` | `f5a33578db` | Direct | Embedded assets | Differs |
| `OPENNLP-XXXX-numeric` | `1b977f3478` | Via hierarchy tip | Apache main | Differs |
| `OPENNLP-XXXX-pii` | `cb6ad708a1` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-place-profiles` | `ec3c6728d0` | Direct | #1154 gazetteer API | Differs |
| `OPENNLP-XXXX-predicate-annotators` | `3bc4e048ed` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-region-vote` | `ef53423ee9` | Via hierarchy tip | Numeric annotations; copied #1154 API | Differs |
| `OPENNLP-XXXX-spellcheck-recase` | `3d7c58f774` | Not selected | Apache main | Differs |
| `OPENNLP-XXXX-symbol-joiner` | `10579093cf` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-text-artifacts` | `f52601c02e` | Direct | Apache main | Differs |
| `OPENNLP-XXXX-wordnet-extension` | `63a22b4b2d` | Not selected | #1155 WordNet API | Differs |

The BiLSTM accuracy gate remains open at 96.294 against a 97.0 target. Active
maturity requirements for every feature live in the workspace `TODO/`
directory, not in this branch-status map.

## Add-ons candidates

This is a routing list, not an Apache consensus decision. Document Shape and
term vectors are already in core. The remaining narrow core candidates are the
subword API and WordPiece (#1165), Hunspell stemming (#1190), CJK lattice
tokenization (#1191), the resource installer (#1211), and the dependency parser
(#1236).

| Family | Candidate add-ons |
| --- | --- |
| Subwords | The concrete SentencePiece implementation is already separated as `subword-addon` in add-ons PR #178. |
| Embeddings and vector search | Static embeddings (#1152), TurboQuant (#1213), vector indexes (#1214), vector evaluation (#1215), and the embedding annotator. |
| Gazetteers and geocoding | The concrete geocoder from #1154, place profiles, numeric and region voting, geocoding, and hierarchy annotation. |
| WordNet | Readers and expansion from #1155 and #1167, plus the research WordNet extension. |
| Taggers | Feedforward and BiLSTM POS taggers. |
| Document analysis | Coreference, dehyphenation, glossary, PII, predicate, dependency-layer (#1237), and relation extraction (#1238) annotators. |
| Text preparation | Embedded assets, noise cleanup, spellcheck recasing, symbol joining, and text-artifact normalization. |
| Stemming | The light stemmers in #1166 fit the existing `Stemmer` contract and can be delivered as add-ons. |

If one of these implementations exposes a missing contract needed by several
modules, only that small contract is a core candidate. The Morfologik FSA
branch is reconciliation work against the existing extension, not a new add-on
to publish.

## Path upstream

1. File or select the JIRA issue and rename the local branch and worktree.
2. Rebase or cascade onto the current true parent. Use `git range-diff` or
   patch IDs to prove the feature delta did not change.
3. Run the owning tests, the relevant reactor gate, and `krickert-review`.
4. Push to Apache only with the user's one-time authorization.
5. Move uber membership from `RESEARCH_TIPS` to `APACHE_TIPS` when the source
   becomes Apache-visible. Never merge both copies.
6. Regenerate the helper if the new source is in the open 3.x PR graph, then
   regenerate uber and verify its provenance and package gate.

Filing a JIRA does not publish an org branch. The Apache branch or PR can be
linked from JIRA when the feature graduates.
