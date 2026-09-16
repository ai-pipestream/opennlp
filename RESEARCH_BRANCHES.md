# Research branch map

Checked on 2026-09-08 against local Git history, published PR heads and
`regen-uber.sh`. This is the maintained map on `preview-docs`; the copy in
`uber/` belongs to its recorded build.

Apache main is `b453b9ee3`. It includes Document #1182, paragraph normalization
#1249, term vectors #1212, Hunspell #1190 and #1266, CJK #1191 and #1265,
the resource installer #1211, regex test hardening #1268, and the subword API
and WordPiece encoder #1165. Features use those implementations from main.

All 32 admitted feature heads contain this main and their updated parents.
The cascade preserves previous local and published commits. This states Git
ancestry, not completion of every feature: the candidate-selection changes in
the region-vote and geocode working trees remain uncommitted and excluded.

## Integration intent and current state

- Independent features depend on main. Dependent features contain the updated
  parent branches.
- `APACHE_TIPS` and `RESEARCH_TIPS` in `regen-uber.sh` select six Apache
  tips and eighteen research/preview tips. Those tips subsume the 32 admitted
  feature branches, including the shared DL encoder tests in #1288.
- The helper consumes published open 3.x PR heads, including drafts, but not
  unpublished research. New public maintenance PRs are helper inputs; they
  are not automatically admitted to the research preview.
- Feature fixes stay on their owning branches. Nothing merges out of uber.
- The Apache PR heads were rebased on main on 2026-09-16 and pushed; the
  parser, dependency annotations and relation extraction branches took main by
  merge and are not pushed yet. Some author PRs are ready for review, the rest
  are drafts.
- The helper published on Apache is `91fc182ee`, built from the PR heads of
  2026-09-08. All of those heads were rebased on main on 2026-09-16, so a
  regenerated candidate, `OPENNLP-1833-grpc-helper-20260916` at `d41657fdd`,
  contains main plus all 27 open PR heads at version
  3.0.0-OPENNLP-1833-SNAPSHOT; the package build and 7581 focused tests
  passed. It replaces the published helper once pushed.
- Uber contains all 32 aligned feature heads. Its 27-module package build
  and 580 cross-feature tests passed. A fresh integration has the same source
  and tests; differences are formatting, license-block order and timestamp.
  `PIPESTREAM-PROVENANCE.txt` records the selected heads. Preview publication
  does not deploy either server.

## Intended dependency tree

Solid feature arrows are verified parent relationships. Arrows into the input
groups reflect the script. Dotted arrows identify cross-repository dependencies,
planned add-on integration, or shared server changes.

```mermaid
flowchart LR
  main["apache main<br/>Document, paragraphs, term vectors,<br/>Hunspell, CJK, resource installer, subwords"]
  uber["kristian-3.x-features<br/>generated preview"]
  apacheTips{{"APACHE_TIPS"}}
  researchTips{{"RESEARCH_TIPS"}}

  subgraph apache["Open Apache 3.x features"]
    gaz["#1154 Gazetteer API"]
    wordnet["#1155 WordNet API"]
    expansion["#1167 WordNet expansion"]
    light["#1166 Light stemmers"]
    encoding["#1288 Shared DL encoder tests"]
    static["#1152 Static embeddings"]
    turbo["#1213 TurboQuant"]
    vector["#1214 Vector index"]
    evaluation["#1215 Vector evaluation"]
    parser["#1236 Dependency parser"]
    dependency["#1237 Dependency annotations"]
    relation["#1238 Relation extraction"]
  end

  subgraph addons["opennlp-addons, separate repository"]
    canary["OPENNLP-1924-canary-addon"]
    subwordAddon["#178 SentencePiece implementation<br/>OPENNLP-1885-subword-addon"]
    canary --> subwordAddon
  end

  subgraph research["Research features"]
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
    morfologik["Morfologik FSA<br/>reconciliation"]
    dehyphenation["Dehyphenation"]
    recase["Spellcheck recase"]
    wordnetExtension["WordNet extension"]
  end

  major0["Preview major-0 model support"]
  previewDocs["preview-docs"]
  publishedPrs["All published open 3.x PR heads<br/>drafts included"]
  helper["OPENNLP-1833-grpc-helper"]
  apacheServer["Apache sandbox server"]
  demoServer["Search and uber-demo server"]

  main --> gaz
  main --> wordnet --> expansion
  main --> light
  main --> encoding --> static --> turbo --> vector --> evaluation
  main --> parser --> dependency --> relation
  main -. subword API .-> subwordAddon
  subwordAddon -. planned module dependency .-> static

  main --> artifacts
  main --> assets --> noise
  main --> predicates
  main --> ffpos --> bilstm
  main --> glossary
  main --> pii
  encoding --> coref
  main --> numeric --> region --> geocode --> hierarchy
  gaz --> region
  gaz --> profiles
  static --> embedding
  main --> dehyphenation
  main --> symbols
  main --> morfologik
  main --> recase
  wordnet --> wordnetExtension
  main --> major0
  main --> previewDocs

  gaz --> apacheTips
  wordnet --> apacheTips
  expansion --> apacheTips
  light --> apacheTips
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
  recase --> researchTips
  morfologik --> researchTips
  dehyphenation --> researchTips
  wordnetExtension --> researchTips
  major0 --> researchTips

  main --> uber
  apacheTips --> uber
  researchTips --> uber
  previewDocs --> uber
  main --> helper
  publishedPrs --> helper
  helper --> apacheServer
  uber --> demoServer
  apacheServer -. shared server changes .-> demoServer
```

## Open Apache 3.x pull requests

“PR source” identifies the repository containing the PR head, not the PR
destination. “Main missing” counts main commits not reachable from the local
head. A PR marked Ready is not necessarily approved or validated for merge.

| PR | Published base | PR source | Published head | Local head | Main missing | Review state |
| --- | --- | --- | --- | --- | --- | --- |
| [#1152](https://github.com/apache/opennlp/pull/1152) | `main` | ai-pipestream | `34075a160` | `34075a160` | 0 | Draft |
| [#1154](https://github.com/apache/opennlp/pull/1154) | `main` | ai-pipestream | `41aeb8aa4` | `41aeb8aa4` | 0 | Draft |
| [#1155](https://github.com/apache/opennlp/pull/1155) | `main` | ai-pipestream | `3b376ca8a` | `3b376ca8a` | 0 | Draft |
| [#1166](https://github.com/apache/opennlp/pull/1166) | `main` | ai-pipestream | `7c42f8099` | `7c42f8099` | 0 | Draft |
| [#1167](https://github.com/apache/opennlp/pull/1167) | `main` | ai-pipestream | `ccf3a4f1a` | `ccf3a4f1a` | 0 | Draft |
| [#1213](https://github.com/apache/opennlp/pull/1213) | `main` | ai-pipestream | `f2c598122` | `f2c598122` | 0 | Draft |
| [#1214](https://github.com/apache/opennlp/pull/1214) | `main` | ai-pipestream | `80968379b` | `80968379b` | 0 | Draft |
| [#1215](https://github.com/apache/opennlp/pull/1215) | `main` | ai-pipestream | `1cf40f2ad` | `1cf40f2ad` | 0 | Draft |
| [#1236](https://github.com/apache/opennlp/pull/1236) | `main` | apache | `12d9db03f` | `5b05ca9b4` | 0 | Ready |
| [#1237](https://github.com/apache/opennlp/pull/1237) | `OPENNLP-547-dependency-parser` | apache | `0f643ffef` | `38c3a12e8` | 0 | Draft |
| [#1238](https://github.com/apache/opennlp/pull/1238) | `OPENNLP-1919-dependency-annotations` | apache | `be054493d` | `a0a27fc77` | 0 | Draft |
| [#1270](https://github.com/apache/opennlp/pull/1270) | `main` | ai-pipestream | `fbdc1f389` | `fbdc1f389` | 0 | Ready |
| [#1275](https://github.com/apache/opennlp/pull/1275) | `main` | ai-pipestream | `1ecab106a` | `1ecab106a` | 0 | Draft |
| [#1276](https://github.com/apache/opennlp/pull/1276) | `main` | ai-pipestream | `dcc5e4c4f` | `dcc5e4c4f` | 0 | Draft |
| [#1277](https://github.com/apache/opennlp/pull/1277) | `main` | ai-pipestream | `a4ed75af1` | `a4ed75af1` | 0 | Draft |
| [#1278](https://github.com/apache/opennlp/pull/1278) | `main` | ai-pipestream | `4bebe3c6f` | `4bebe3c6f` | 0 | Draft |
| [#1279](https://github.com/apache/opennlp/pull/1279) | `main` | ai-pipestream | `9bd19f537` | `9bd19f537` | 0 | Draft |
| [#1280](https://github.com/apache/opennlp/pull/1280) | `main` | ai-pipestream | `3db470806` | `3db470806` | 0 | Draft |
| [#1281](https://github.com/apache/opennlp/pull/1281) | `main` | ai-pipestream | `9c338196e` | `9c338196e` | 0 | Draft |
| [#1282](https://github.com/apache/opennlp/pull/1282) | `main` | ai-pipestream | `b41c02477` | `b41c02477` | 0 | Draft |
| [#1288](https://github.com/apache/opennlp/pull/1288) | `main` | ai-pipestream | `88ca5f0c0` | `88ca5f0c0` | 0 | Ready |
| [#1290](https://github.com/apache/opennlp/pull/1290) | `main` | ai-pipestream | `1378f64c3` | `1378f64c3` | 0 | Ready |
| [#1295](https://github.com/apache/opennlp/pull/1295) | `main` | ai-pipestream | `5a5fd3734` | `5a5fd3734` | 0 | Draft |
| [#1299](https://github.com/apache/opennlp/pull/1299) | `main` | ai-pipestream | `b1ead78a1` | `b1ead78a1` | 0 | Ready |
| [#1300](https://github.com/apache/opennlp/pull/1300) | `main` | ai-pipestream | `7def2ce9c` | `7def2ce9c` | 0 | Ready |
| [#1301](https://github.com/apache/opennlp/pull/1301) | `main` | ai-pipestream | `92d76e5fd` | `92d76e5fd` | 0 | Ready |
| [#1305](https://github.com/apache/opennlp/pull/1305) | `main` | ai-pipestream | `39c1c0738` | `39c1c0738` | 0 | Ready |
| [#1306](https://github.com/apache/opennlp/pull/1306) | `main` | ppkarwasz | `0afe9f8e9` | `-` | ? | Ready |

Fork publication alone does not update #1236, #1237, or #1238: their PR source
branches are on apache/opennlp. Keep fork synchronization and Apache PR
publication separate.

Static embeddings and its descendants contain the current subword API and
WordPiece implementation. `BertTokenizer` is removed. The in-repository
SentencePiece implementation remains tested and available until migration to
the add-on dependency is complete. That migration is separate from branch
synchronization.

Add-ons #178 is a draft at `b9a57b3dc7`, based on
`OPENNLP-1924-canary-addon`, not add-ons main. It is a separate repository
dependency, not an OpenNLP Git parent.

Separate-repository work is not part of this OpenNLP cascade. The add-ons
canary `0f36b7c36` is published with DocBook documentation, naming changes,
and release checks. Draft #178 still needs that updated base and the matching
module/manual changes. The earlier sandbox audit found two commits missing
from its own main `f1cafcf5`, and three Apache-server development commits
missing from the uber-server line. Those server refs need a fresh audit before
work resumes. No add-ons or sandbox branch was changed during this cascade.

Hunspell #1266 is merged and its final review approved the corrections.
The implementation uses the ResourceInstaller from #1211.
[OPENNLP-1927](https://issues.apache.org/jira/browse/OPENNLP-1927) tracks
remaining Hunspell compatibility and strict-loading work. It does not change
the behavior carried by this synchronization. The review also requested
separate release-note tracking for the corrections already merged in #1266.

The original Hunspell and CJK worktrees contain uncommitted edits requiring
comparison with the merged implementations. Exclude them from bulk publication
or deletion until that comparison is complete. The merged downloader worktree
was removed after preserving its commits in a verified recovery bundle.
Review checkouts, backup refs, `deregex`, and unrelated maintenance branches
are outside this feature inventory; do not include them in a bulk push.

## Local work outside the preview

Hunspell compatibility [#1270](https://github.com/apache/opennlp/pull/1270)
is published at `b194f2162` and remains draft with changes requested.
It is already a helper input, but not a selected research preview input.
The dedicated Hunspell review owns its implementation and evaluations.

The public regex-maintenance PRs #1275 through #1282 and generic embedding
interface #1290 are outside the admitted 32-feature preview graph.
Their publication and review state should be checked independently.

## Research worktrees in the preview

All 21 research branch heads contain current main (f09b7f0a5, synchronized 2026-09-16 by merge). Fork status is as of that date; the research heads have not been pushed since.
The region-vote and geocode worktrees still have pending candidate-policy edits;
those edits are not in the branch heads below. Temporary feature READMEs remain local.
The hierarchy tip contains current numeric, gazetteer, region, and geocode heads.

| Branch | Local head | Main missing | Preview input | Intended dependency | Fork status |
| --- | --- | --- | --- | --- | --- |
| `OPENNLP-XXXX-bilstm-tagger` | `4b77fc5ad` | 0 | Direct | Feedforward tagger | Fork behind |
| `OPENNLP-XXXX-coref` | `3e7a0f77e` | 0 | Direct | main, #1288 encoder tests | Fork behind |
| `OPENNLP-XXXX-dehyphenation` | `5016ad4d2` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-embedded-assets` | `df045ff8e` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-embedding-annotator` | `3040c0c0e` | 0 | Direct | #1152 static embeddings | Fork behind |
| `OPENNLP-XXXX-ff-postagger` | `87fd32187` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-geocode-annotator` | `27cbfd60c` | 0 | Via hierarchy tip | Region vote | Fork behind |
| `OPENNLP-XXXX-glossary` | `c0a3dcf3a` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-hierarchy-annotator` | `d04d0284a` | 0 | Direct | Geocode annotator | Fork behind |
| `OPENNLP-XXXX-morfologik-fsa` | `b4c779fc3` | 0 | Direct | main; reconcile existing extension | Fork behind |
| `OPENNLP-XXXX-noise` | `a601f7d4b` | 0 | Direct | Embedded assets | Fork behind |
| `OPENNLP-XXXX-numeric` | `94fdc92a3` | 0 | Via hierarchy tip | main | Fork behind |
| `OPENNLP-XXXX-pii` | `f1f8d2071` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-place-profiles` | `1ffd9bb08` | 0 | Direct | #1154 gazetteer | Fork behind |
| `OPENNLP-XXXX-predicate-annotators` | `8e40e65bb` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-region-vote` | `028008597` | 0 | Via hierarchy tip | Numeric + #1154 gazetteer | Fork behind |
| `OPENNLP-XXXX-spellcheck-recase` | `3a0e2ee30` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-symbol-joiner` | `3b27812da` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-text-artifacts` | `6d84f6e01` | 0 | Direct | main | Fork behind |
| `OPENNLP-XXXX-wordnet-extension` | `4b02d85db` | 0 | Direct | #1155 WordNet | Fork behind |
| `OPENNLP-XXXX-component-spi` | `d60943ec0` | 0 | Direct | main | Local only |

## Verified stacks

- Main subword API and #1288 tests -> static embeddings -> TurboQuant -> vector index -> evaluation.
  The embedding annotator contains current static embeddings.
- Main subword API and #1288 tests -> coreference, for its optional ONNX token-vector adapter.
  The resolver and trained models are unchanged by this compatibility update.
- Parser -> dependency annotations -> relation extraction.
- WordNet API -> expansion and research extension.
- Embedded assets -> noise; feedforward POS -> BiLSTM.
- Numeric and gazetteer -> region vote -> geocode -> hierarchy.
  Place profiles contains current gazetteer.

The TurboQuant cascade carries double-precision analogy queries through its
internal adapters. The WordNet extension carries the parent's stream ownership,
definition retention, and malformed-input checks without removing composition.

`preview-docs` and `preview-accept-major0-models` also contain current main and
their old fork heads. They are support branches, outside the 32-feature count.

## Add-ons candidates

This is a routing list, not an Apache consensus decision. Document container, term
vectors, Hunspell stemming, CJK lattice tokenization, and the resource
installer, subword API and WordPiece are already in core. The dependency parser
(#1236) and generic embedding interface (#1290) remain core candidates.

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

## Publication plan

Preserve existing local and published commits. Synchronization does not require
rewriting the feature branches.

1. Capture local and remote heads and preserve uncommitted files before updates.
2. Compare local-only and remote-only changes, including equivalent changes
   from earlier rebases. Resolve meaningful differences; do not select an entire
   side of a conflict without review.
3. Merge the existing fork history and current main into independent branches.
   Update dependent branches by merging the reconciled current parents, in
   dependency order. The downloader from main remains the implementation.
4. Check that the old local head, old fork head, current main, and required
   parent heads are ancestors of the result. Run feature tests, the affected
   package/manual build, and the pre-push review.
5. Push tested results to the existing fork branch names with normal
   fast-forward pushes. Fetch and reconsider if a remote advances.
6. Update helper and uber after the feature cascade, preserving their old
   local and published histories. Compare with a fresh scratch integration
   before publishing the preview.

The feature main cascade is published, and its ancestry checks passed. Helper
and uber passed their package and selected regression checks. Exact heads and results are in
`REVIEW-CHANGES/publication-20260908.1WyFWe/RECEIPT.md` at the workspace root.
Pending publication and separate-repository work is in
`TODO/BRANCH-PUBLICATION.md`. PR readiness is unchanged.

JIRA tracking, add-on migration, Apache branch updates, and PR publication remain
separate actions. Research can be backed up to the fork without changing
graduation status.
