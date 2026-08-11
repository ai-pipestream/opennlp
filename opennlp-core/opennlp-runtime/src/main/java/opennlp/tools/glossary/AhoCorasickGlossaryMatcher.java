/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package opennlp.tools.glossary;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.normalizer.AlignedText;
import opennlp.tools.util.normalizer.OffsetAwareNormalizer;

/**
 * A deterministic {@link GlossaryMatcher} backed by an Aho-Corasick automaton: one
 * forward pass over the text finds every registered term regardless of how many terms
 * the glossary holds.
 *
 * <p>By default, terms match literally, character for character, so a multiword term
 * matches only with the exact separator characters it was registered with. An optional
 * {@link OffsetAwareNormalizer} folds terms and text before the automaton runs (for
 * example German umlaut expansion, full case folding, whitespace collapse, or dash
 * folding). Hits found in the normalized form are mapped back to original-text spans
 * through the normalizer's {@link AlignedText}, so consumers still read identifiers
 * against the source characters. Hits are constrained to word boundaries on the
 * original text: a hit whose edge character is a letter or digit is dropped when the
 * neighboring original character is also a letter or digit, so {@code cat} never matches
 * inside {@code concatenate}. Overlapping hits are resolved leftmost first, then longest,
 * then by registration order, and the reported hits never overlap.</p>
 *
 * <p>When the matcher ignores case, terms and text are compared through the
 * per-code-point UnicodeData lowercase mapping, the same mapping
 * {@link opennlp.tools.util.StringUtil#toLowerCase(CharSequence)} applies, which keeps
 * spans exact but does not apply locale or multi-character case folding. Length-changing
 * folds such as sharp s to {@code ss} belong on the {@link OffsetAwareNormalizer} hook
 * instead. Word boundaries and case comparison both work in code points, so supplementary
 * letters neighbor and fold like any others.</p>
 *
 * <p>The automaton is built once in the constructor; matching holds no per-call state
 * and is safe to share between threads when the supplied normalizer is also safe to
 * share.</p>
 *
 * @see <a href="https://doi.org/10.1145/360825.360855">Aho, Corasick (1975): Efficient
 *      string matching: An aid to bibliographic search</a>
 * @since 3.0.0
 */
public final class AhoCorasickGlossaryMatcher implements GlossaryMatcher {

  /** The index of the automaton's root state, from which every scan starts. */
  private static final int ROOT = 0;

  /** The registered entries in registration order; raw hits index into this list. */
  private final List<GlossaryEntry> entries;

  /**
   * The term length in chars of each entry as stored in the automaton, aligned with
   * {@link #entries} by index. A hit start in scan coordinates is the hit end minus this
   * length. When a normalizer is present, the length is that of the normalized pattern.
   */
  private final int[] termLengths;

  /** Whether terms and text are compared through per-code-point lowercasing. */
  private final boolean ignoreCase;

  /**
   * Optional fold applied to terms at build time and to text at match time. {@code null}
   * means identity: the automaton scans the original characters.
   */
  private final OffsetAwareNormalizer normalizer;

  /**
   * The goto function: per state, the sorted code points with an outgoing edge; the
   * target of each edge sits in {@link #edgeTargets} at the same index.
   */
  private final int[][] edgeCodePoints;

  /** The goto targets, aligned with {@link #edgeCodePoints} per state. */
  private final int[][] edgeTargets;

  /** The failure function: per state, the state to fall back to on a code point miss. */
  private final int[] fail;

  /** Per state, the indexes into {@link #entries} of the terms that end at that state. */
  private final int[][] outputs;

  /**
   * Builds the automaton for a glossary with no additional normalizer.
   *
   * @param glossary The entries to match. Must not be {@code null} or empty, and no
   *                 entry may be {@code null}. When two entries share the same term, the
   *                 one registered first wins.
   * @param ignoreCase Whether to match terms regardless of character case.
   * @throws IllegalArgumentException Thrown if {@code glossary} is {@code null}, empty,
   *         or contains {@code null}.
   */
  public AhoCorasickGlossaryMatcher(Collection<GlossaryEntry> glossary, boolean ignoreCase) {
    this(glossary, ignoreCase, null, false);
  }

  /**
   * Builds the automaton for a glossary, folding terms and text through an
   * {@link OffsetAwareNormalizer} before matching.
   *
   * <p>Each term is normalized before it is inserted into the automaton, and each
   * {@link #match(CharSequence)} call normalizes the text, scans the normalized form,
   * and maps hit spans back to the original text. A term that becomes blank after
   * normalization is rejected. When two entries normalize to the same pattern, the one
   * registered first wins. Stemming and lemmatization are out of scope here; use
   * {@link TermAnalyzingGlossaryMatcher} for token-level inflection.</p>
   *
   * @param glossary The entries to match. Must not be {@code null} or empty, and no
   *                 entry may be {@code null}.
   * @param ignoreCase Whether to match terms regardless of character case after the
   *                   normalizer fold.
   * @param normalizer The offset-aware fold to apply. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code glossary} is {@code null}, empty,
   *         or contains {@code null}, if {@code normalizer} is {@code null}, or if a term
   *         normalizes to blank.
   */
  public AhoCorasickGlossaryMatcher(Collection<GlossaryEntry> glossary, boolean ignoreCase,
      OffsetAwareNormalizer normalizer) {
    this(glossary, ignoreCase, normalizer, true);
  }

  /**
   * Shared constructor. When {@code requireNormalizer} is {@code true}, {@code normalizer}
   * must be non-null; otherwise {@code null} means the identity (two-argument) path.
   */
  private AhoCorasickGlossaryMatcher(Collection<GlossaryEntry> glossary, boolean ignoreCase,
      OffsetAwareNormalizer normalizer, boolean requireNormalizer) {
    if (glossary == null || glossary.isEmpty()) {
      throw new IllegalArgumentException("glossary must not be null or empty");
    }
    if (requireNormalizer && normalizer == null) {
      throw new IllegalArgumentException("normalizer must not be null");
    }
    for (final GlossaryEntry entry : glossary) {
      if (entry == null) {
        throw new IllegalArgumentException("glossary must not contain null entries");
      }
    }
    this.entries = List.copyOf(glossary);
    this.ignoreCase = ignoreCase;
    this.normalizer = normalizer;
    this.termLengths = new int[entries.size()];

    final List<Map<Integer, Integer>> transitions = new ArrayList<>();
    final List<List<Integer>> nodeOutputs = new ArrayList<>();
    transitions.add(new HashMap<>());
    nodeOutputs.add(new ArrayList<>());

    for (int pattern = 0; pattern < entries.size(); pattern++) {
      final String term = normalizePattern(entries.get(pattern).term());
      termLengths[pattern] = term.length();
      int state = ROOT;
      for (int i = 0; i < term.length(); ) {
        final int codePoint = term.codePointAt(i);
        i += Character.charCount(codePoint);
        final int normalized = normalize(codePoint);
        Integer next = transitions.get(state).get(normalized);
        if (next == null) {
          next = transitions.size();
          transitions.add(new HashMap<>());
          nodeOutputs.add(new ArrayList<>());
          transitions.get(state).put(normalized, next);
        }
        state = next;
      }
      nodeOutputs.get(state).add(pattern);
    }

    this.edgeCodePoints = new int[transitions.size()][];
    this.edgeTargets = new int[transitions.size()][];
    for (int state = 0; state < transitions.size(); state++) {
      final Map<Integer, Integer> edges = transitions.get(state);
      final int[] codePoints = new int[edges.size()];
      int e = 0;
      for (final int codePoint : edges.keySet()) {
        codePoints[e++] = codePoint;
      }
      Arrays.sort(codePoints);
      final int[] targets = new int[codePoints.length];
      for (int k = 0; k < codePoints.length; k++) {
        targets[k] = edges.get(codePoints[k]);
      }
      edgeCodePoints[state] = codePoints;
      edgeTargets[state] = targets;
    }

    this.fail = new int[transitions.size()];
    final Deque<Integer> queue = new ArrayDeque<>();
    for (final int child : edgeTargets[ROOT]) {
      fail[child] = ROOT;
      queue.add(child);
    }
    while (!queue.isEmpty()) {
      final int state = queue.remove();
      nodeOutputs.get(state).addAll(nodeOutputs.get(fail[state]));
      for (int e = 0; e < edgeCodePoints[state].length; e++) {
        final int child = edgeTargets[state][e];
        fail[child] = step(fail[state], edgeCodePoints[state][e]);
        queue.add(child);
      }
    }

    this.outputs = new int[transitions.size()][];
    for (int state = 0; state < transitions.size(); state++) {
      final List<Integer> patterns = nodeOutputs.get(state);
      outputs[state] = new int[patterns.size()];
      for (int i = 0; i < patterns.size(); i++) {
        outputs[state][i] = patterns.get(i);
      }
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>One forward pass collects every candidate, then hits that sit inside a word are
   * dropped and the remaining overlaps are resolved leftmost first, then longest, then
   * by registration order. When a normalizer is configured, the pass runs over the
   * normalized text and candidate spans are mapped back to the original before the
   * boundary check.</p>
   */
  @Override
  public List<GlossaryMatch> match(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final CharSequence scanText;
    final AlignedText aligned;
    if (normalizer != null) {
      aligned = normalizer.normalizeAligned(text);
      scanText = aligned.normalized();
    } else {
      aligned = null;
      scanText = text;
    }
    final List<int[]> hits = new ArrayList<>();
    int state = ROOT;
    for (int i = 0; i < scanText.length(); ) {
      final int codePoint = Character.codePointAt(scanText, i);
      i += Character.charCount(codePoint);
      state = step(state, normalize(codePoint));
      for (final int pattern : outputs[state]) {
        final int normalizedStart = i - termLengths[pattern];
        final int normalizedEnd = i;
        final int start;
        final int end;
        if (aligned != null) {
          final Span original = aligned.toOriginalSpan(normalizedStart, normalizedEnd);
          start = original.getStart();
          end = original.getEnd();
        } else {
          start = normalizedStart;
          end = normalizedEnd;
        }
        if (onWordBoundary(text, start, end)) {
          hits.add(new int[] {start, end, pattern});
        }
      }
    }
    return resolveOverlaps(hits);
  }

  /**
   * Folds one glossary term for automaton insertion.
   *
   * @param term The registered surface form.
   * @return The pattern stored in the automaton.
   * @throws IllegalArgumentException Thrown if the normalized pattern is blank.
   */
  private String normalizePattern(String term) {
    final String pattern = normalizer == null
        ? term
        : normalizer.normalize(term).toString();
    if (StringUtil.isBlank(pattern)) {
      throw new IllegalArgumentException(
          "glossary term must not normalize to blank: \"" + term + "\"");
    }
    return pattern;
  }

  /**
   * Advances the automaton by one code point, following failure links on a miss.
   *
   * @param state The current state.
   * @param codePoint The normalized input code point.
   * @return The next state.
   */
  private int step(int state, int codePoint) {
    int next = target(state, codePoint);
    while (next < 0 && state != ROOT) {
      state = fail[state];
      next = target(state, codePoint);
    }
    return next < 0 ? ROOT : next;
  }

  /**
   * Looks up one goto edge.
   *
   * @param state The state to leave.
   * @param codePoint The normalized code point to follow.
   * @return The target state, or {@code -1} when the state has no such edge.
   */
  private int target(int state, int codePoint) {
    final int index = Arrays.binarySearch(edgeCodePoints[state], codePoint);
    return index >= 0 ? edgeTargets[state][index] : -1;
  }

  /**
   * Normalizes one code point for comparison, with the per-code-point UnicodeData
   * lowercase mapping of {@link Character#toLowerCase(int)}.
   *
   * @param codePoint The code point.
   * @return The lowercased code point when case is ignored, otherwise the code point.
   */
  private int normalize(int codePoint) {
    return ignoreCase ? Character.toLowerCase(codePoint) : codePoint;
  }

  /**
   * Checks that a candidate hit does not continue a word on either side. Neighbors are
   * read as whole code points, so a supplementary letter or digit next to a hit blocks
   * it exactly like a basic-plane one.
   *
   * @param text The text being scanned.
   * @param start The hit start, inclusive.
   * @param end The hit end, exclusive.
   * @return {@code true} if the hit sits on word boundaries.
   */
  private boolean onWordBoundary(CharSequence text, int start, int end) {
    if (start > 0 && Character.isLetterOrDigit(Character.codePointAt(text, start))
        && Character.isLetterOrDigit(Character.codePointBefore(text, start))) {
      return false;
    }
    return end >= text.length()
        || !Character.isLetterOrDigit(Character.codePointBefore(text, end))
        || !Character.isLetterOrDigit(Character.codePointAt(text, end));
  }

  /**
   * Resolves overlapping hits leftmost first, then longest, then by registration order.
   *
   * @param hits The raw hits as {@code {start, end, pattern}} triples.
   * @return The surviving hits in text order. Never {@code null}.
   */
  private List<GlossaryMatch> resolveOverlaps(List<int[]> hits) {
    hits.sort((a, b) -> {
      if (a[0] != b[0]) {
        return Integer.compare(a[0], b[0]);
      }
      if (a[1] != b[1]) {
        return Integer.compare(b[1], a[1]);
      }
      return Integer.compare(a[2], b[2]);
    });
    final List<GlossaryMatch> matches = new ArrayList<>();
    int lastEnd = 0;
    for (final int[] hit : hits) {
      if (hit[0] >= lastEnd) {
        final GlossaryEntry entry = entries.get(hit[2]);
        matches.add(new GlossaryMatch(new Span(hit[0], hit[1]), entry.id(), entry.term()));
        lastEnd = hit[1];
      }
    }
    return matches;
  }
}
