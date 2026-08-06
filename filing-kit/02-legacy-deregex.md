# legacy-deregex

## JIRA fields

*Type:* Improvement. *Epic:* OPENNLP-1852 (consistency scoreboard, items 3 to 6 and 8). *Components:* opennlp-runtime, extensions.

## JIRA description (wiki markup)

```
h2. Summary

The legacy normalizers {{TwitterCharSequenceNormalizer}}, {{UrlCharSequenceNormalizer}}, {{ShrinkCharSequenceNormalizer}}, and {{NumberCharSequenceNormalizer}} apply regular expressions to user text, which runs against the OPENNLP-1850 design goal of cursor-based text transforms (a single forward code-point scan on user text, with regex reserved for config parsing). This ticket replaces those regex passes with cursor scans on the {{CharClass}} engine where they operate on user text, and audits the spellcheck and extension normalizers for the same pattern. These are pre-1850 rungs and are not in the blessed default chain, so the change is low-risk and output-preserving rather than a redesign.

h2. Scope

* Replace regex-on-text with a cursor scan (on the {{CharClass}}/{{CodePointSet}} primitives from OPENNLP-1850) in {{TwitterCharSequenceNormalizer}}, {{UrlCharSequenceNormalizer}}, {{ShrinkCharSequenceNormalizer}}, and {{NumberCharSequenceNormalizer}}.
* Where a normalizer's behavior is subtle, add characterization tests that pin the pre-change output first, then refactor, so the output is preserved byte for byte.
* Audit the spellcheck and extension normalizers for regex-on-text paths; convert the clear cases and file follow-ups for anything larger.

h2. Acceptance criteria

* The four normalizers use cursor scans on the text path; no regex is applied to user text in them.
* Output-preserving: characterization tests capture the prior behavior and pass unchanged after the refactor.
* The spellcheck and extension audit is complete, with conversions applied or follow-ups filed.

h2. Out of scope

* Making these rungs offset-aware. They are not in the blessed chain, and offset-safety for them is a separate concern.
* The {{StringUtil}} whitespace migration, which is a separate ticket.
* Regex in config parsing and non-user-text utilities.
* Adding or removing any of these rungs from a default chain.
```

## PR title

```
<KEY>: Replace regex with cursor scans in the legacy CharSequenceNormalizers, output byte-identical
```

## PR body (markdown)

```
Replaces the regex passes in the four legacy normalizers (Twitter, Url, Shrink, Number) with single forward cursor scans on the OPENNLP-1850 `CharClass`/`CodePointSet` primitives, output byte for byte.

**Method:** characterization tests pinning the regex behavior (typical, edge, supplementary-plane, and pathological inputs, probed against the original implementations) were committed before each refactor, and each suite carries a seeded randomized differential check against the former patterns.

**Fidelity:** the scans reproduce the engine's exact semantics where they leak into output: ASCII `\s`/`\d`/case-insensitive folding, the JDK `\b` word-boundary rules including the non-spacing-mark clause, code-point backreference matching around lone surrogates, and the mail regex's lookbehind and domain backtracking. These rungs feed the language detector factory chain, so output preservation keeps model behavior stable.

**Also in this PR:** the spellcheck and extension normalizers were audited; punctuation peeling, the number-like guard, and the token-stream delimiter split are converted. The spellcheck URL-like guard stays a regex with a justification comment and is filed as a follow-up. Null text now fails loud with `IllegalArgumentException` instead of an undocumented NPE (documented and tested).

Verification: opennlp-runtime 1324/0, tools 265, api 286, spellcheck 114, uima 49, all green.
```
