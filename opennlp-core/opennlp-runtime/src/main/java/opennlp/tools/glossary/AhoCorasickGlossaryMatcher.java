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
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import opennlp.tools.util.Span;

/**
 * A deterministic {@link GlossaryMatcher} backed by an Aho-Corasick automaton: one
 * forward pass over the text finds every registered term regardless of how many terms
 * the glossary holds.
 *
 * <p>Terms match literally, character for character, so a multiword term matches only
 * with the exact separator characters it was registered with. Hits are constrained to
 * word boundaries: a hit whose edge character is a letter or digit is dropped when the
 * neighboring text character is also a letter or digit, so {@code cat} never matches
 * inside {@code concatenate}. Overlapping hits are resolved leftmost first, then longest,
 * then by registration order, and the reported hits never overlap.</p>
 *
 * <p>When the matcher ignores case, terms and text are compared through per-character
 * lowercasing, which keeps spans exact but does not apply locale or multi-character case
 * folding.</p>
 *
 * <p>The automaton is built once in the constructor; matching holds no per-call state
 * and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class AhoCorasickGlossaryMatcher implements GlossaryMatcher {

  /** The index of the automaton's root state, from which every scan starts. */
  private static final int ROOT = 0;

  /** The registered entries in registration order; raw hits index into this list. */
  private final List<GlossaryEntry> entries;

  /** The term length of each entry, aligned with {@link #entries} by index. */
  private final int[] termLengths;

  /** Whether terms and text are compared through per-character lowercasing. */
  private final boolean ignoreCase;

  /** The goto function: per state, the outgoing edges keyed by normalized character. */
  private final List<Map<Character, Integer>> transitions;

  /** The failure function: per state, the state to fall back to on a character miss. */
  private final int[] fail;

  /** Per state, the indexes into {@link #entries} of the terms that end at that state. */
  private final int[][] outputs;

  /**
   * Builds the automaton for a glossary.
   *
   * @param glossary The entries to match. Must not be {@code null} or empty, and no
   *                 entry may be {@code null}. When two entries share the same term, the
   *                 one registered first wins.
   * @param ignoreCase Whether to match terms regardless of character case.
   * @throws IllegalArgumentException Thrown if {@code glossary} is {@code null}, empty,
   *         or contains {@code null}.
   */
  public AhoCorasickGlossaryMatcher(Collection<GlossaryEntry> glossary, boolean ignoreCase) {
    if (glossary == null || glossary.isEmpty()) {
      throw new IllegalArgumentException("glossary must not be null or empty");
    }
    for (final GlossaryEntry entry : glossary) {
      if (entry == null) {
        throw new IllegalArgumentException("glossary must not contain null entries");
      }
    }
    this.entries = List.copyOf(glossary);
    this.ignoreCase = ignoreCase;
    this.termLengths = new int[entries.size()];

    this.transitions = new ArrayList<>();
    final List<List<Integer>> nodeOutputs = new ArrayList<>();
    transitions.add(new HashMap<>());
    nodeOutputs.add(new ArrayList<>());

    for (int pattern = 0; pattern < entries.size(); pattern++) {
      final String term = entries.get(pattern).term();
      termLengths[pattern] = term.length();
      int state = ROOT;
      for (int i = 0; i < term.length(); i++) {
        final char c = normalize(term.charAt(i));
        Integer next = transitions.get(state).get(c);
        if (next == null) {
          next = transitions.size();
          transitions.add(new HashMap<>());
          nodeOutputs.add(new ArrayList<>());
          transitions.get(state).put(c, next);
        }
        state = next;
      }
      nodeOutputs.get(state).add(pattern);
    }

    this.fail = new int[transitions.size()];
    final Deque<Integer> queue = new ArrayDeque<>();
    for (final int child : transitions.get(ROOT).values()) {
      fail[child] = ROOT;
      queue.add(child);
    }
    while (!queue.isEmpty()) {
      final int state = queue.remove();
      nodeOutputs.get(state).addAll(nodeOutputs.get(fail[state]));
      for (final Map.Entry<Character, Integer> edge : transitions.get(state).entrySet()) {
        final int child = edge.getValue();
        fail[child] = step(fail[state], edge.getKey());
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

  @Override
  public List<GlossaryMatch> match(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<int[]> hits = new ArrayList<>();
    int state = ROOT;
    for (int i = 0; i < text.length(); i++) {
      state = step(state, normalize(text.charAt(i)));
      for (final int pattern : outputs[state]) {
        final int start = i + 1 - termLengths[pattern];
        if (onWordBoundary(text, start, i + 1)) {
          hits.add(new int[] {start, i + 1, pattern});
        }
      }
    }
    return resolveOverlaps(hits);
  }

  /**
   * Advances the automaton by one character, following failure links on a miss.
   *
   * @param state The current state.
   * @param c The normalized input character.
   * @return The next state.
   */
  private int step(int state, char c) {
    Integer next = transitions.get(state).get(c);
    while (next == null && state != ROOT) {
      state = fail[state];
      next = transitions.get(state).get(c);
    }
    return next == null ? ROOT : next;
  }

  /**
   * Normalizes one character for comparison.
   *
   * @param c The character.
   * @return The lowercased character when case is ignored, otherwise {@code c}.
   */
  private char normalize(char c) {
    return ignoreCase ? Character.toLowerCase(c) : c;
  }

  /**
   * Checks that a candidate hit does not continue a word on either side.
   *
   * @param text The text being scanned.
   * @param start The hit start, inclusive.
   * @param end The hit end, exclusive.
   * @return {@code true} if the hit sits on word boundaries.
   */
  private static boolean onWordBoundary(CharSequence text, int start, int end) {
    if (start > 0 && Character.isLetterOrDigit(text.charAt(start))
        && Character.isLetterOrDigit(text.charAt(start - 1))) {
      return false;
    }
    return end >= text.length() || !Character.isLetterOrDigit(text.charAt(end - 1))
        || !Character.isLetterOrDigit(text.charAt(end));
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
