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
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.normalizer.AlignedText;
import opennlp.tools.util.normalizer.Dimension;
import opennlp.tools.util.normalizer.OffsetAwareNormalizer;
import opennlp.tools.util.normalizer.Term;
import opennlp.tools.util.normalizer.TermAnalyzer;

/**
 * A {@link GlossaryMatcher} that matches glossary terms after
 * {@link TermAnalyzer} token normalization, so inflected surfaces can hit a
 * dictionary form while reported spans stay on the original text.
 *
 * <p>Each glossary term and each input text are segmented and normalized by the
 * supplied analyzer (typically case fold plus stem). Matching runs an
 * Aho-Corasick automaton over the sequence of normalized tokens, not over raw
 * characters. A hit covering tokens {@code i} through {@code j} reports the
 * character span from the start of token {@code i} to the end of token {@code j}
 * in the original text, so a registered {@code hot dog} can match {@code hot dogs}
 * and still cover the plural surface.</p>
 *
 * <p>An optional {@link OffsetAwareNormalizer} can expand or fold the whole input
 * before tokenization. The same transform runs over registered terms, and matched token
 * spans map back through its {@link AlignedText}. This supports pre-tokenization edits
 * such as English contraction expansion while preserving original source offsets.</p>
 *
 * <p>Because the UAX&#160;#29 word tokenizer yields word tokens only, the characters
 * between consecutive tokens do not take part in matching: {@code hot-dogs},
 * {@code hot, dogs}, and even the sentence-crossing {@code hot. Dogs} all present
 * the token sequence {@code hot dog} to the automaton, and the reported span
 * covers the separators between the first and last matched token. Matching
 * hyphenated compounds is intentional; a hit that crosses a sentence boundary is the
 * documented cost, so run sentence-sized inputs when that matters. A text token
 * whose normalized form is blank vanishes for matching, exactly like characters a
 * normalizer deletes on the character path: its neighbors become adjacent, and a
 * hit spanning the gap covers the vanished surface.</p>
 *
 * <p>This path is for inflection and other token-level folds that
 * {@link opennlp.tools.util.normalizer.OffsetAwareNormalizer} cannot express.
 * Character-exact and orthographic matching stay on
 * {@link AhoCorasickGlossaryMatcher}, which remains the fast path when no
 * tokenizer or stemmer is required. {@link Dimension#LEMMA} is rejected at
 * construction: lemmas need part-of-speech tags, which
 * {@link TermAnalyzer#analyze(CharSequence)} cannot supply.</p>
 *
 * <p>The automaton is built once in the constructor. Each {@link #match(CharSequence)}
 * call re-analyzes the text, allocating one {@link Term} per token plus the hit
 * lists; the matcher itself holds no per-call state and is safe to share across
 * threads when the analyzer and pre-tokenization normalizer are.</p>
 *
 * @see <a href="https://doi.org/10.1145/360825.360855">Aho, Corasick (1975): Efficient
 *      string matching: An aid to bibliographic search</a>
 * @since 3.0.0
 */
public final class TermAnalyzingGlossaryMatcher implements GlossaryMatcher {

  /** The index of the automaton's root state, from which every scan starts. */
  private static final int ROOT = 0;

  /** The registered entries in registration order; raw hits index into this list. */
  private final List<GlossaryEntry> entries;

  /** The analyzer that tokenizes and normalizes both glossary terms and input text. */
  private final TermAnalyzer analyzer;

  /** Optional whole-text fold applied before tokenization, with source alignment. */
  private final OffsetAwareNormalizer preTokenizerNormalizer;

  /** The normalized token count of each entry, aligned with {@link #entries} by index. */
  private final int[] patternLengths;

  /** The goto function: each state maps a normalized token to its target state. */
  private final List<Map<String, Integer>> transitions;

  /** The failure function: per state, the state to fall back to on a token miss. */
  private final int[] fail;

  /** Per state, the indexes into {@link #entries} of the patterns ending there. */
  private final int[][] outputs;

  /**
   * Builds a token-normalized matcher for a glossary.
   *
   * @param glossary The entries to match. Must not be {@code null} or empty, and no
   *                 entry may be {@code null}. When two entries normalize to the same
   *                 token sequence, the one registered first wins.
   * @param analyzer The analyzer that tokenizes and normalizes terms and text. Must
   *                 not be {@code null} and must not configure {@link Dimension#LEMMA}.
   * @throws IllegalArgumentException Thrown if {@code glossary} is {@code null}, empty,
   *         or contains {@code null}, if {@code analyzer} is {@code null} or configures
   *         {@link Dimension#LEMMA}, or if a term yields no word tokens or a blank
   *         normalized token after analysis.
   */
  public TermAnalyzingGlossaryMatcher(Collection<GlossaryEntry> glossary,
      TermAnalyzer analyzer) {
    this(glossary, analyzer, null, false);
  }

  /**
   * Builds a token-normalized matcher with an aligned whole-text fold before tokenization.
   *
   * @param glossary The entries to match. Must not be {@code null} or empty, and no
   *                 entry may be {@code null}. When two entries normalize to the same
   *                 token sequence, the one registered first wins.
   * @param analyzer The analyzer that tokenizes and normalizes terms and text. Must
   *                 not be {@code null} and must not configure {@link Dimension#LEMMA}.
   * @param preTokenizerNormalizer The required whole-text fold applied to registered
   *                 terms and source text before tokenization. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code glossary} is {@code null}, empty,
   *         or contains {@code null}, if {@code analyzer} or
   *         {@code preTokenizerNormalizer} is {@code null}, if the analyzer configures
   *         {@link Dimension#LEMMA}, or if a term yields no word tokens or a blank
   *         normalized token after analysis.
   */
  public TermAnalyzingGlossaryMatcher(Collection<GlossaryEntry> glossary,
      TermAnalyzer analyzer, OffsetAwareNormalizer preTokenizerNormalizer) {
    this(glossary, analyzer, preTokenizerNormalizer, true);
  }

  /**
   * Shared constructor for the identity and pre-tokenization-normalized paths.
   *
   * @param glossary The entries to match.
   * @param analyzer The token analyzer.
   * @param preTokenizerNormalizer The whole-text fold, or {@code null} for identity.
   * @param requirePreTokenizerNormalizer Whether to reject a {@code null} fold.
   * @throws IllegalArgumentException Thrown for an invalid glossary or analyzer, a
   *         required but absent fold, or a term that cannot produce a matchable token.
   */
  private TermAnalyzingGlossaryMatcher(Collection<GlossaryEntry> glossary,
      TermAnalyzer analyzer, OffsetAwareNormalizer preTokenizerNormalizer,
      boolean requirePreTokenizerNormalizer) {
    if (glossary == null || glossary.isEmpty()) {
      throw new IllegalArgumentException("glossary must not be null or empty");
    }
    if (analyzer == null) {
      throw new IllegalArgumentException("analyzer must not be null");
    }
    if (analyzer.dimensions().contains(Dimension.LEMMA)) {
      throw new IllegalArgumentException("analyzer must not configure Dimension.LEMMA:"
          + " lemmas need part-of-speech tags, which analyze(CharSequence) cannot supply");
    }
    if (requirePreTokenizerNormalizer && preTokenizerNormalizer == null) {
      throw new IllegalArgumentException("preTokenizerNormalizer must not be null");
    }
    for (final GlossaryEntry entry : glossary) {
      if (entry == null) {
        throw new IllegalArgumentException("glossary must not contain null entries");
      }
    }
    this.entries = List.copyOf(glossary);
    this.analyzer = analyzer;
    this.preTokenizerNormalizer = preTokenizerNormalizer;
    this.patternLengths = new int[entries.size()];
    this.transitions = new ArrayList<>();
    transitions.add(new HashMap<>());
    final List<List<Integer>> nodeOutputs = new ArrayList<>();
    nodeOutputs.add(new ArrayList<>());

    for (int pattern = 0; pattern < entries.size(); pattern++) {
      final List<String> tokens = normalizedTokens(entries.get(pattern).term());
      if (tokens.isEmpty()) {
        throw new IllegalArgumentException(
            "glossary term must yield at least one token after analysis: \""
                + entries.get(pattern).term() + "\"");
      }
      patternLengths[pattern] = tokens.size();
      int state = ROOT;
      for (final String token : tokens) {
        Integer next = transitions.get(state).get(token);
        if (next == null) {
          next = transitions.size();
          transitions.add(new HashMap<>());
          nodeOutputs.add(new ArrayList<>());
          transitions.get(state).put(token, next);
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
      for (final Map.Entry<String, Integer> edge : transitions.get(state).entrySet()) {
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

  /**
   * {@inheritDoc}
   *
   * <p>The text is analyzed into tokens, tokens whose normalized form is blank are
   * dropped, the automaton runs over the remaining normalized forms, and each
   * surviving hit spans from the first matched token's original start to the last
   * matched token's original end.</p>
   */
  @Override
  public List<GlossaryMatch> match(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final AlignedText aligned = preTokenizerNormalizer == null
        ? null : preTokenizerNormalizer.normalizeAligned(text);
    final CharSequence analysisText = aligned == null ? text : aligned.normalized();
    final List<AnalyzedTerm> terms = new ArrayList<>();
    for (final Term term : analyzer.analyze(analysisText)) {
      if (term.span() == null) {
        throw new IllegalStateException(
            "TermAnalyzingGlossaryMatcher requires Terms with original spans");
      }
      if (!StringUtil.isBlank(term.normalized())) {
        final Span span = aligned == null ? term.span()
            : aligned.toOriginalSpan(term.span().getStart(), term.span().getEnd());
        terms.add(new AnalyzedTerm(term.normalized(), span));
      }
    }
    final List<int[]> hits = new ArrayList<>();
    int state = ROOT;
    for (int i = 0; i < terms.size(); i++) {
      state = step(state, terms.get(i).normalized());
      for (final int pattern : outputs[state]) {
        final int startToken = i - patternLengths[pattern] + 1;
        final int endToken = i;
        final int start = terms.get(startToken).span().getStart();
        final int end = terms.get(endToken).span().getEnd();
        hits.add(new int[] {start, end, pattern, patternLengths[pattern]});
      }
    }
    return resolveOverlaps(hits);
  }

  /**
   * Analyzes a glossary term into its normalized token sequence.
   *
   * @param term The registered surface form.
   * @return The normalized tokens in order. Never {@code null}.
   * @throws IllegalArgumentException Thrown if a token normalizes to blank: an
   *         invisible pattern element cannot be matched deliberately, so it fails
   *         loud instead of silently vanishing from the registered phrase.
   */
  private List<String> normalizedTokens(String term) {
    final CharSequence analysisText = preTokenizerNormalizer == null
        ? term : preTokenizerNormalizer.normalize(term);
    final List<Term> terms = analyzer.analyze(analysisText);
    final List<String> tokens = new ArrayList<>(terms.size());
    for (final Term analyzed : terms) {
      final String normalized = analyzed.normalized();
      if (StringUtil.isBlank(normalized)) {
        throw new IllegalArgumentException("glossary term token must not normalize to"
            + " blank: \"" + analyzed.original() + "\" in \"" + term + "\"");
      }
      tokens.add(normalized);
    }
    return tokens;
  }

  /** One normalized token and its span in the original, pre-fold source text. */
  private record AnalyzedTerm(String normalized, Span span) {
  }

  /**
   * Advances by one normalized token, following failure links on a miss.
   *
   * @param state The current state.
   * @param token The normalized token.
   * @return The next state.
   */
  private int step(int state, String token) {
    Integer next = transitions.get(state).get(token);
    while (next == null && state != ROOT) {
      state = fail[state];
      next = transitions.get(state).get(token);
    }
    return next == null ? ROOT : next;
  }

  /**
   * Resolves overlapping hits leftmost first, then longest token span, then
   * registration order.
   *
   * @param hits The raw hits as {@code {start, end, pattern, tokenLength}} tuples.
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
      if (a[3] != b[3]) {
        return Integer.compare(b[3], a[3]);
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
