# whitespace-ucd

## JIRA fields

*Type:* Improvement. *Epic:* OPENNLP-1852 (consistency scoreboard, items 1 and 2). *Components:* opennlp-api, opennlp-runtime. *Note:* behavior change, requires a release note.

## JIRA description (wiki markup)

```
h2. Summary

OPENNLP-1850 established standards-sourced character classes as a design goal (the {{CharClass}} engine reads the UCD {{White_Space}} and {{Dash}} sets, not {{Character.isWhitespace}} or a regex class), but the rest of the codebase has not caught up: {{StringUtil}} still uses the JVM {{Character.isWhitespace}} predicate, and several call sites split or scan user text with {{split("\\s+")}} or {{Character.isWhitespace}}. This ticket migrates the whitespace predicate in {{StringUtil}} to the UCD {{White_Space}} set (or documents the intentional exception where JVM semantics are genuinely wanted), and audits and converts the remaining user-text call sites. This is a behavior change and carries an explicit release note, because the UCD {{White_Space}} set and {{Character.isWhitespace}} disagree on real code points: {{White_Space}} includes the no-break space U+00A0 and the figure and narrow spaces that {{Character.isWhitespace}} excludes, and excludes the U+001C to U+001F information separators that {{Character.isWhitespace}} includes. OPENNLP-205 already made exactly this correction in the sentence-detector mapping, so this generalizes that fix.

h2. Scope

* Audit {{split("\\s+")}}, {{Character.isWhitespace}}, and {{Character.isSpaceChar}} call sites across the codebase outside the OPENNLP-1850 engine. Classify each as user text (migrate to the UCD set) or config and non-user-text (leave; regex and JVM predicates are acceptable there per the design goals).
* Migrate the whitespace handling in {{StringUtil}} to the UCD-backed {{White_Space}} API delivered in OPENNLP-1850, or document the intentional JVM exception with a rationale where a caller genuinely needs JVM semantics (for example a format-fixed parser).
* Add a release note in the 3.0 migration section describing the delta: NBSP and the Unicode space separators now count as whitespace; the U+001C to U+001F information separators no longer do.

h2. Acceptance criteria

* The {{StringUtil}} whitespace predicate is UCD-backed, or the JVM exception is documented with a rationale.
* The call-site audit is complete: user-text paths use the UCD set, and every retained regex or JVM predicate on a text path is justified in a comment.
* A release note documents the behavior delta at the boundary code points (NBSP, the space separators, the information separators).
* Tests pin the delta at those boundary code points.

h2. Out of scope

* The OPENNLP-1850 engine itself, which is already UCD-backed.
* Regex in config parsing and other non-user-text utilities, which the design goals permit.
* The legacy {{CharSequenceNormalizer}} regex migration, which is a separate ticket.
* Changing tokenizer or sentence-detector feature generation (model stability).
```

## PR title

```
<KEY>: UCD-back the whitespace predicates: migrate user-text call sites, freeze the documented exceptions
```

## PR body (markdown)

```
UCD-backs the whitespace predicates outside the OPENNLP-1850 engine (OPENNLP-1852 consistency item; generalizes the OPENNLP-205 correction). Every `StringUtil.isWhitespace`, `Character.isWhitespace`, `Character.isSpaceChar`, and `\s`-split site outside the 1850 engine was audited and classified:

- **Migrated to the UCD `White_Space` set:** both rule-based tokenizers, `Span.trim`, the sentence detector's abbreviation guard, the stopword and spellcheck CLIs, `SymSpell.lookupCompound`, the spellcheck `PER_TOKEN` normalizer, and the UIMA `NumberUtil`.
- **Frozen with documented rationale and permanent pins:** trained-model feature generation and the BERT reference definition (model stability).
- **Kept JVM/ASCII semantics with per-site justification comments:** format parsers, where the format fixes the definition.
- `StringUtil.isWhitespace` itself stays frozen as the documented legacy predicate, since its remaining callers are exactly the frozen sites; user-text code is directed to `UnicodeWhitespace`.

Method is characterize-then-refactor: the first commit pins pre-change behavior at NBSP, figure and narrow space, U+2028/U+2029, NEL, and U+001C..U+001F for every touched site, so the migration commits flip only the deliberate deltas.

This is a behavior change; the release-note text is in the JIRA issue.

Verification: opennlp-runtime 1313/0, plus green suites in api, formats, cli, tools, uima, and spellcheck.
```
