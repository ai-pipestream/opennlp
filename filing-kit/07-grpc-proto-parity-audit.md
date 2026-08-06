# opennlp-grpc proto surface vs features landed in main (audit 2026-07-10)

Audited: apache/opennlp main at 77fed15c (OPENNLP-1882) against apache/opennlp-sandbox OPENNLP-1833-grpc-expansion at 66f0f8e (post-squash, PR #493).

## Parity confirmed (landed feature -> wire surface)

| Landed in main | Wire surface | Status |
|---|---|---|
| OPENNLP-1860 engine rungs (nfc, nfkc, stripInvisible, whitespace x2, quotes, dashes, digits, ellipsis, bullets, caseFold, accentFold) | NormalizationRung enum, NormalizationSpec | covered, 1:1 |
| Confusables.skeleton (in 1860/1a) | CONFUSABLE_FOLD rung (service maps via builder.with(custom)) and CONFUSABLE_FOLD term dimension | covered |
| OPENNLP-1861 Alignment/AlignedText | NormalizationSpec.require_alignment, alignment runs in the response, offset encoding options | covered |
| OPENNLP-1862 WordSegmenter/WordTokenizer/WordType | AnalysisProfile.tokenizer_engine="uax29", Token.word_type | covered |
| OPENNLP-1863 Term/TermAnalyzer + Dimension | AnalysisProfile.term_dimensions -> Token.term_layers map | covered |
| OPENNLP-1864 NormalizationProfiles registry | AnalysisProfile.term_profile (per-language, per-request analyzer) | covered |
| OPENNLP-205 sentence span mapping | pinned through AnalyzeDocument (parity commit) | covered |
| OPENNLP-1865 offset-safe DL | DlNerModel uses OffsetMappingNameFinder.findInOriginal; spans stay in original coordinates (DL-side input normalization deliberately disabled, conservative) | covered |
| OPENNLP-1871 stopword-list licensing | licensing only, no runtime surface change | n/a |

## Gap: landed in main, absent from the wire

**G1. Stopwords.** The whole package landed (opennlp.tools.stopword: StopwordFilter API, DictionaryStopwordFilter, bundled StopwordLists, StopwordFilteringTokenizer, StopwordFilterStream) and there is zero wire presence: no pipeline step, no Token flag, no profile knob. Proposal for the next proto round: annotation, not removal -- `Token.is_stopword` populated when the profile asks (e.g. `optional string stopword_language` on AnalysisProfile, resolved against StopwordLists; NOT_FOUND for an unlisted language). Wire consumers filter; the server never drops tokens. Requires TOKENIZE.

That is the only forward gap. Everything else landed since the parity commits is either covered, internal (CharClass, CodePoints), or licensing/test-only.

## Reverse gaps: wire is AHEAD of landed main (release-sequencing constraints for PR #493)

The sandbox builds against the locally-installed 3.0.0-kristian-SNAPSHOT (1868+1869 stack). Until these land in main, PR #493 cannot build against a published snapshot:

- **R1** FULL_CASE_FOLD rung + FULL_CASE_FOLD dimension need OPENNLP-1868 (PR #1138, open).
- **R2** EMOJI_TO_EMOTICON / EMOTICON_TO_EMOJI rungs + EMOJI_FOLD dimension need OPENNLP-1869 (worktree stacked on 1868, no PR yet -- this is the long pole; file the PR).
- **R3** opennlp-grpc-backend-static needs opennlp-embeddings (PR #1152, open, under review).
- **R4** revert the TEMPORARY 3.0.0-kristian-SNAPSHOT pin in opennlp-grpc/pom.xml to the inherited snapshot once R1-R3 land (the pom comment already marks it).

Merge order that unblocks #493: #1138 (1868) -> 1869 PR -> #1152 (1877) -> revert pin -> #493.
