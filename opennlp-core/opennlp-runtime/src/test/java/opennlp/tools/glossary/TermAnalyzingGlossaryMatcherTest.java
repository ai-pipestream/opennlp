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

import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.lemmatizer.Lemmatizer;
import opennlp.tools.stemmer.snowball.SnowballStemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmerFactory;
import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.CharSequenceNormalizer;
import opennlp.tools.util.normalizer.Dimension;
import opennlp.tools.util.normalizer.TermAnalyzer;

/**
 * Pins token-normalized glossary matching: inflected surfaces hit dictionary forms
 * while spans stay on the original text.
 */
public class TermAnalyzingGlossaryMatcherTest {

  private static TermAnalyzer englishStemmingAnalyzer() {
    return TermAnalyzer.builder()
        .caseFold()
        .stem(new SnowballStemmerFactory(SnowballStemmer.ALGORITHM.ENGLISH))
        .build();
  }

  /**
   * Exact reproduction for issue #2: a glossary term {@code hot dog} must match source
   * text {@code hot dogs} and return the original {@code hot dogs} span. The character
   * Aho-Corasick matcher cannot do this; stemming through {@link TermAnalyzer} can.
   */
  @Test
  void testHotDogMatchesHotDogsWithOriginalPluralSpan() {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog")), englishStemmingAnalyzer());

    final String text = "hot dogs";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("FOOD", matches.get(0).id());
    Assertions.assertEquals("hot dog", matches.get(0).term());
    Assertions.assertEquals(new Span(0, 8), matches.get(0).span());
    Assertions.assertEquals("hot dogs",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * The same reproduction embedded in a sentence: the span covers only the inflected
   * multiword mention, not the surrounding tokens.
   */
  @Test
  void testHotDogMatchesHotDogsInsideSentence() {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog")), englishStemmingAnalyzer());

    final String text = "the hot dogs were cold";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(4, 12), matches.get(0).span());
    Assertions.assertEquals("hot dogs",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * Pins that the exact character matcher still misses the plural, so the inflection
   * path is not accidental overlap with literal matching.
   */
  @Test
  void testExactMatcherStillMissesInflectedHotDogs() {
    final AhoCorasickGlossaryMatcher exact = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog")), true);
    Assertions.assertTrue(exact.match("hot dogs").isEmpty());
    Assertions.assertTrue(exact.match("the hot dogs were cold").isEmpty());
  }

  /**
   * Mixed-case inflected surfaces still match when the analyzer case-folds before
   * stemming, and the span covers the original casing.
   */
  @Test
  void testMixedCaseInflectedSurfaceKeepsOriginalSpan() {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog")), englishStemmingAnalyzer());

    final String text = "Hot Dogs sold here";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("Hot Dogs",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * A plural registration matches a singular surface through the same stem, with the
   * singular original span.
   */
  @Test
  void testPluralGlossaryTermMatchesSingularSurface() {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dogs")), englishStemmingAnalyzer());

    final String text = "one hot dog please";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(4, 11), matches.get(0).span());
    Assertions.assertEquals("hot dog",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * Longer stemmed phrases win over shorter overlapping ones, leftmost first.
   */
  @Test
  void testLongestStemmedPhraseWins() {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("SHORT", "hot dog"),
            new GlossaryEntry("LONG", "hot dog stand")), englishStemmingAnalyzer());

    final String text = "hot dog stands nearby";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("LONG", matches.get(0).id());
    Assertions.assertEquals("hot dog stands",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * Two non-overlapping stemmed hits keep text order.
   */
  @Test
  void testMultipleStemmedHitsPreserveOrder() {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog"),
            new GlossaryEntry("DRINK", "soft drink")), englishStemmingAnalyzer());

    final String text = "hot dogs and soft drinks";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("FOOD", matches.get(0).id());
    Assertions.assertEquals("DRINK", matches.get(1).id());
    Assertions.assertEquals("hot dogs",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
    Assertions.assertEquals("soft drinks",
        text.substring(matches.get(1).span().getStart(), matches.get(1).span().getEnd()));
  }

  /**
   * Empty text and a miss return empty lists; null text fails loud.
   */
  @Test
  void testEmptyMissAndNullText() {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog")), englishStemmingAnalyzer());
    Assertions.assertTrue(matcher.match("").isEmpty());
    Assertions.assertTrue(matcher.match("nothing tasty here").isEmpty());
    Assertions.assertThrows(IllegalArgumentException.class, () -> matcher.match(null));
  }

  /**
   * Construction validates glossary and analyzer arguments.
   */
  @Test
  void testInvalidConstructorArguments() {
    final TermAnalyzer analyzer = englishStemmingAnalyzer();
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new TermAnalyzingGlossaryMatcher(null, analyzer));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new TermAnalyzingGlossaryMatcher(List.of(), analyzer));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new TermAnalyzingGlossaryMatcher(Collections.singletonList(null), analyzer));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new TermAnalyzingGlossaryMatcher(
            List.of(new GlossaryEntry("FOOD", "hot dog")), null));
  }

  /**
   * Pins the separator semantics: the UAX #29 word tokenizer drops punctuation, so a
   * hyphenated compound, a comma-separated pair, a slash, parentheses, and even a
   * sentence-crossing spelling all present the token sequence {@code hot dog}, and the
   * reported span covers the separators between the matched tokens.
   */
  @ParameterizedTest
  @ValueSource(strings = {"hot-dogs", "hot, dogs", "hot/dogs", "hot (dogs)", "hot. Dogs"})
  void testSeparatorsBetweenTokensDoNotBlockTheMatch(String surface) {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog")), englishStemmingAnalyzer());

    final String text = "eat " + surface + " now";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    final Span span = matches.get(0).span();
    final String covered = text.substring(span.getStart(), span.getEnd());
    Assertions.assertTrue(covered.startsWith("hot"), covered);
    // The span ends at the last matched token's end, so a trailing parenthesis
    // stays outside while every separator between the tokens is covered.
    Assertions.assertTrue(covered.endsWith("dogs") || covered.endsWith("Dogs"), covered);
  }

  /**
   * Two entries that analyze to the same token sequence keep first-wins registration
   * order, mirroring the exact matcher's duplicate rule.
   */
  @Test
  void testFirstWinsForDuplicateAnalyzedPatterns() {
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FIRST", "hot dog"),
            new GlossaryEntry("SECOND", "hot dogs")), englishStemmingAnalyzer());

    final List<GlossaryMatch> matches = matcher.match("hot dogs for sale");
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("FIRST", matches.get(0).id());
    Assertions.assertEquals("hot dog", matches.get(0).term());
  }

  /**
   * An analyzer configured with {@link Dimension#LEMMA} is rejected at construction:
   * lemmas need part-of-speech tags, which {@code analyze(CharSequence)} cannot supply,
   * so the failure surfaces immediately instead of on the first match call.
   */
  @Test
  void testLemmatizingAnalyzerFailsFastAtConstruction() {
    final TermAnalyzer lemmatizing = TermAnalyzer.builder()
        .caseFold()
        .lemmatize(new Lemmatizer() {
          @Override
          public String[] lemmatize(String[] toks, String[] tags) {
            return toks;
          }

          @Override
          public List<List<String>> lemmatize(List<String> toks, List<String> tags) {
            throw new UnsupportedOperationException();
          }
        })
        .build();

    final IllegalArgumentException ex = Assertions.assertThrows(
        IllegalArgumentException.class,
        () -> new TermAnalyzingGlossaryMatcher(
            List.of(new GlossaryEntry("FOOD", "hot dog")), lemmatizing));
    Assertions.assertTrue(ex.getMessage().contains("LEMMA"), ex.getMessage());
  }

  /**
   * A glossary term with a token that normalizes to blank is rejected at construction,
   * while a blank-normalizing text token vanishes for matching: its neighbors become
   * adjacent and a hit spanning the gap covers the vanished surface.
   */
  @Test
  void testBlankNormalizedTokensRejectInTermsAndVanishInText() {
    final CharSequenceNormalizer blanker = text ->
        "zap".contentEquals(text) ? "" : text;
    final TermAnalyzer blanking = TermAnalyzer.builder()
        .caseFold()
        .transform(Dimension.WHITESPACE, blanker)
        .stem(new SnowballStemmerFactory(SnowballStemmer.ALGORITHM.ENGLISH))
        .build();

    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new TermAnalyzingGlossaryMatcher(
            List.of(new GlossaryEntry("Z", "zap")), blanking));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new TermAnalyzingGlossaryMatcher(
            List.of(new GlossaryEntry("Z2", "hot zap dog")), blanking));

    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog")), blanking);
    final String text = "the hot zap dogs bark";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("hot zap dogs",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }
}
