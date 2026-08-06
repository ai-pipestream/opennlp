# PR staging and machine-transfer guide

State as of 2026-07-10. OPENNLP-1850 is fully merged (all 7 rungs in main as of 2026-07-08, dl merged as OPENNLP-1865, docs as OPENNLP-1866). This doc is the working guide for the open PR round and for resuming the work on another machine (krick-1).

## Open PRs on apache/opennlp

| PR | Branch (head is on the ai-pipestream fork unless noted) | Base | JIRA | State | Post-merge action |
|---|---|---|---|---|---|
| #1138 | OPENNLP-1868 (head on apache) | main | OPENNLP-1868 | open | first link of the sandbox chain; when it merges, retarget #1164 to main BEFORE the 1868 branch is deleted (the #1153 footgun) |
| #1164 | OPENNLP-1869 (head on apache) | OPENNLP-1868 | OPENNLP-1869 | draft | after #1138: retarget to main, rebase, flip ready |
| #1152 | static-embeddings | main | OPENNLP-1877 | open | last gate before the sandbox pin revert |
| #1161 | OPENNLP-1878-performance-enhancements | main | OPENNLP-1878 | open | standalone |
| #1163 | stemmer-factory | main | OPENNLP-1883 | open | when it merges, rebase light-stemmers onto main and force-push (collapses #1166's diff) |
| #1150 | whitespace-ucd | main | OPENNLP-1875 | open | standalone |
| #1151 | legacy-deregex | main | OPENNLP-1876 | open | standalone |
| #1154 | gazetteer-api | main | OPENNLP-1879 | draft | un-draft after the release-critical reviews clear |
| #1155 | wordnet-api | main | OPENNLP-1880 | draft | when it merges, rebase wordnet-expansion onto main and force-push (collapses #1167's diff) |
| #1165 | sentencepiece | main | none yet (user files) | draft | standalone; retitle with the key |
| #1166 | light-stemmers | main | none yet (user files) | draft | STACKED on #1163 content; diff shows both until #1163 merges + rebase; retitle with the key |
| #1167 | wordnet-expansion | main | none yet (user files) | draft | STACKED on #1155 content; diff shows both until #1155 merges + rebase; retitle with the key |

Stacked-draft note: #1166 and #1167 could not use their true base branches as PR base refs because those heads live on the fork, so they are based on main with a STACKED banner in the body. Only the last commit of each is the actual change (light-stemmers 592380e3, wordnet-expansion a000fcce).

Other repos: apache/opennlp-sandbox #493 (OPENNLP-1833 gRPC expansion; local commit a94de77 "Mark stopwords over the wire" is NOT pushed anywhere, it exists only in /work/reference-code/opennlp-sandbox on this machine, so push it or transfer it before wiping; the kristian pin reverts only after #1138 + #1164 + #1152 merge), apache/tika #2921 (document contract + flattener + ParseBytes, with Nick).

## JIRAs the user still files

- #1165: pure-Java SentencePiece inference (opennlp-subword), Epic B rung 1 of the V3 modernization.
- #1166: UniNE light/minimal stemmer tiers (Epic A), depends on OPENNLP-1883.
- #1167: lexical expansion over the knowledge base seam, depends on OPENNLP-1880.

## Merge-event checklist (sandbox chain)

1. #1138 merges: verify content is actually IN MAIN (git cat-file a rung-unique file, never trust the badge), retarget #1164 to main before branch deletion, rebase, flip ready.
2. #1164 merges: same verification. Stacking on apache is then over.
3. #1152 merges: rebuild plain 3.0.0-SNAPSHOT from main, revert the sandbox kristian-SNAPSHOT pin in opennlp-grpc/pom.xml, full sandbox verify, push to #493.
4. #1163 merges: rebase light-stemmers onto main, force-push, #1166 diff collapses; then un-draft.
5. #1155 merges: rebase wordnet-expansion onto main, force-push, #1167 diff collapses; then un-draft.

## Worktree and remote layout (this machine)

Main clone /work/reference-code/opennlp with remotes: upstream = github.com/apache/opennlp, github = github.com/ai-pipestream/opennlp, origin = git.rokkon.com/ai-pipestream/opennlp. All feature branches are pushed to origin AND github; apache holds only the 1868/1869 heads (from the 1850 era).

Worktrees under /work/worktrees/: sentencepiece (branch sentencepiece, off main 77fed15c), light-stemmers (off stemmer-factory bc0b4742), wordnet-expansion (off wordnet-api 0551c1da), stemmer-factory, wordnet-api, gazetteer-api, static-embeddings, whitespace-ucd, legacy-deregex, OPENNLP-1868/1869/1870, opennlp-deploy (this doc). The sandbox clone is /work/reference-code/opennlp-sandbox (branch OPENNLP-1833-grpc-expansion). Tika is /work/reference-code/tika (branch TIKA-4766-document-contract).

## Transferring to krick-1

1. Clone the fork and wire remotes the same way: `git clone https://github.com/ai-pipestream/opennlp && git remote add upstream https://github.com/apache/opennlp.git` (add the rokkon origin if that box can reach it). `git fetch --all`.
2. Recreate only the worktrees you need: `git worktree add /work/worktrees/<name> <branch>`. Every branch in the table is on the GitHub fork; nothing lives only on this machine except scratch data.
3. Maven install order matters because sibling modules resolve opennlp-api from ~/.m2 and the branches carry DIFFERENT api surfaces: before building any branch's extension module, `mvn -pl opennlp-api install` FROM THAT WORKTREE. The last-installed api wins; reinstall when switching branches. The sandbox needs 3.0.0-kristian-SNAPSHOT, built from the OPENNLP-1869 worktree stack (api + runtime), and its version string never collides with 3.0.0-SNAPSHOT.
4. SentencePiece real-model eval (optional, not required for the bundled suite): create a venv with sentencepiece==0.2.1, download t5-small and albert-base-v2 spiece.model files, run gen_real_fixtures.py (in the module test resources) over the directory, then `mvn test -Dtest=SentencePieceRealModelEvalTest -Dopennlp.subword.eval.dir=<dir>`. The bundled tiny-model parity suite runs with no setup.
5. Stemmer fixtures regenerate from a Lucene clone (see the README in opennlp-core/opennlp-runtime/src/test/resources/opennlp/tools/stemmer/light/); only needed if the algorithms change upstream.
6. Reference clones used read-only: google/sentencepiece and apache/lucene. Re-clone if needed; nothing in them is modified.
7. Not transferred (regenerate or ignore): the scratchpad fixture/bench directories, the T5/ALBERT model downloads, and ~/.m2 state.
8. Copy by hand, they are untracked local files: this guide and the other notes in /work/worktrees/opennlp-deploy/ (*.md + filing-kit/), the unpushed sandbox commit a94de77 (push to #493 on go-ahead, or format-patch it), and /work/main/embeddings-bench/ if the bench artifacts matter.

## Open items

- User files the three JIRAs above and retitles #1165/#1166/#1167.
- Models-publication JIRA (opennlp-models-embeddings on the Maven Central pipeline): draft text delivered, user files; start the PMC/legal thread early.
- SentencePiece performance pass: ranked byte-identical-safe plan is banked in memory (double-array vocab trie, vocab-interned piece strings, ASCII fast path, allocation cuts; ceiling ~0.85-1.0x of the native reference). File as a follow-up after #1165 gets its key.
- Geocoder LOCATION wiring + hierarchy walk on #1154's seam: next unbuilt feature from the queue.
- RSLP rule engine (Galician + Portuguese minimal stemmers): follow-up after #1166.
- BEIR hybrid-fusion rerun with a real Lucene BM25 leg before publishing any fusion conclusions.
- De-regex follow-up to file eventually: spellcheck URL_LIKE guard; Shrink UCD-whitespace upgrade (behavior change, separate).
- LEGAL-732 green light received; GeoNames direct ask still pending on the record.
