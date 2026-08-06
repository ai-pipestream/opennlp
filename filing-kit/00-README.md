# Filing kit: the six open tickets and their PRs

One file per branch. Each file has the JIRA fields, the JIRA description in wiki markup (paste the fenced block as-is into the issue description), the PR title, and the PR body in markdown (paste as-is; replace <KEY> with the issue key in both). This folder consolidates and supersedes the copy-paste content of next-round-jira-tickets.md and next-round-pr-staging.md; the staging doc remains the round index (branch table, merge-event checklist, open items).

## Filing notes

- All six branches live on the fork (ai-pipestream/opennlp) and are based on main d5d37dca, except perf-1850-followup (based on the dl branch tip e2365814).
- perf-1850-followup: file the ticket whenever, but open its PR only after dl (#1105) merges; the branch gets rebased --onto main first.
- gazetteer-api and static-embeddings both add a module line to opennlp-extensions/pom.xml; whichever merges second takes a one-line keep-both conflict, resolved by a quick rebase. Merge order between them is otherwise free.
- Everything else is independent and can open and merge in any order.
- Already filed, no action: OPENNLP-1868, OPENNLP-1869, OPENNLP-1870 (fold stack), OPENNLP-1833 (sandbox gRPC).
- whitespace-ucd: filed as OPENNLP-1875, PR open at apache/opennlp#1150. No further action on this one; 01 is kept for reference.
- legacy-deregex: filed as OPENNLP-1876, PR open at apache/opennlp#1151. No further action on this one; 02 is kept for reference.
- static-embeddings: filed as OPENNLP-1877, PR open at apache/opennlp#1152. No further action on this one; 06 is kept for reference.
- perf-1850-followup: filed as OPENNLP-1878, draft PR open at apache/opennlp#1153, stacked on OPENNLP-1850-3-dl (base, not main) so the diff is a clean 6 commits instead of bundling in #1105's 17 unmerged ones. Branch was rebased onto the dl branch's current tip (b3c9fe4b, includes the eval-hash fix) before opening. Once #1105 merges: `gh pr edit 1153 --base main`, rebase the branch onto main, force-push, mark ready for review.

| File | Branch | Type | Epic / link | Status |
|---|---|---|---|---|
| 01 | whitespace-ucd | Improvement | OPENNLP-1852 | **Filed OPENNLP-1875, PR #1150 open** |
| 02 | legacy-deregex | Improvement | OPENNLP-1852 | **Filed OPENNLP-1876, PR #1151 open** |
| 03 | perf-1850-followup | Improvement | follow-up to OPENNLP-1850 | **Filed OPENNLP-1878, draft PR #1153 open (stacked on dl)** |
| 04 | gazetteer-api | New Feature | standalone | **Filed OPENNLP-1879, PR #1154 open** |
| 05 | wordnet-api | New Feature | standalone | **Filed OPENNLP-1880, PR #1155 open** |
| 06 | static-embeddings | New Feature | standalone | **Filed OPENNLP-1877, PR #1152 open** |
