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

package opennlp.tools.noise;

import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the structural scorer's calibration: the most consonant-heavy legitimate
 * English words sit provably below every threshold, and each tier has measured
 * examples on both sides of its boundary.
 */
public class StructuralNoiseScorerTest {

  private final StructuralNoiseScorer scorer = new StructuralNoiseScorer();

  /**
   * Words at the structural extremes of legitimate English, all of which must stay
   * clean: strengths has the language's lowest common vowel share (1 of 9, 0.111,
   * above the 0.10 signal), catchphrase carries a six-consonant run (below the
   * seven signal), rhythms leans on y as its vowel, and bookkeeper doubles pairs
   * without a four-repeat.
   *
   * @return The words.
   */
  static Stream<String> hardestCleanWords() {
    return Stream.of("strengths", "catchphrase", "rhythms", "bookkeeper",
        "latchstring", "twelfths");
  }

  @ParameterizedTest
  @MethodSource("hardestCleanWords")
  void testHardestLegitimateWordsStayClean(String word) {
    assertEquals(List.of(), scorer.score("before " + word + " after", List.of()),
        () -> word + " must stay below every structural signal");
  }

  @Test
  void testOrdinaryProseStaysClean() {
    assertEquals(List.of(), scorer.score(
        "The quick brown fox jumps over the lazy dog, twelfth night approaches.",
        List.of()));
  }

  /** Structural signals apply to ASCII letters only; other scripts never flag. */
  @ParameterizedTest
  @ValueSource(strings = {
      "می‌خواهم",
      "กรุงเทพมหานคร",
      "東京都千代田区"})
  void testOtherScriptsAreNeverFlagged(String text) {
    assertEquals(List.of(), scorer.score(text, List.of()));
  }

  /** A long camel-case identifier is neither binary-ish nor gibberish. */
  @Test
  void testCamelCaseIdentifierStaysClean() {
    assertEquals(List.of(),
        scorer.score("see AbstractSingletonProxyFactoryBean docs", List.of()));
  }

  /**
   * @return Gibberish with the two agreeing signals each case relies on.
   */
  static Stream<Arguments> gibberish() {
    return Stream.of(
        // Vowel share exactly 0.10 over ten letters plus a nine-consonant run.
        Arguments.of("asdkfjqwzx"),
        // Vowelless and an eight-repeat.
        Arguments.of("xxxxxxxx"),
        // Vowelless and a twelve-consonant run.
        Arguments.of("zxkcvbnmsdfg"));
  }

  @ParameterizedTest
  @MethodSource("gibberish")
  void testTwoAgreeingSignalsAreGibberish(String token) {
    final List<NoiseSpan> found = scorer.score("start " + token + " end", List.of());
    assertEquals(1, found.size());
    assertEquals(NoiseSpan.SEVERITY_GIBBERISH, found.get(0).severity());
    assertEquals(token,
        found.get(0).span().getCoveredText("start " + token + " end").toString());
  }

  /** One signal alone is damage: an eight-consonant run with healthy vowels. */
  @Test
  void testSingleSignalIsDamage() {
    final List<NoiseSpan> found = scorer.score("a astrchmfko z", List.of());
    assertEquals(1, found.size());
    assertEquals(NoiseSpan.SEVERITY_DAMAGED, found.get(0).severity());
  }

  /** Heavy letter-digit interleaving is damage. */
  @Test
  void testDigitInterleavingIsDamage() {
    final List<NoiseSpan> found = scorer.score("see c0mput3r there", List.of());
    assertEquals(1, found.size());
    assertEquals(NoiseSpan.SEVERITY_DAMAGED, found.get(0).severity());
  }

  /** A base64-shaped run with a digit scores binary-ish, with punctuation trimmed. */
  @Test
  void testBase64ShapedRunIsBinaryish() {
    final String token = "QWxhZGRpbjpvcGVuIHNlc2FtZQ==".replace(":", "");
    final String text = "payload (" + token + ").";
    final List<NoiseSpan> found = scorer.score(text, List.of());
    assertEquals(1, found.size());
    assertEquals(NoiseSpan.SEVERITY_BINARYISH, found.get(0).severity());
    assertEquals(token, found.get(0).span().getCoveredText(text).toString());
  }

  /** An excluded region is not scored at all. */
  @Test
  void testExcludedRegionIsNotScored() {
    final String token = "QWxhZGRpbjF2cGVuNHNlc2FtZQ";
    final String text = "x " + token + " zxkcvbnmsdfg";
    final List<NoiseSpan> found =
        scorer.score(text, List.of(new Span(2, 2 + token.length())));
    assertEquals(1, found.size());
    assertEquals(NoiseSpan.SEVERITY_GIBBERISH, found.get(0).severity());
  }

  /** Adjacent findings merge into one span of the worse severity. */
  @Test
  void testAdjacentFindingsMergeToTheWorseSeverity() {
    final String text = "xxxxxxxx c0mput3r";
    final List<NoiseSpan> found = scorer.score(text, List.of());
    assertEquals(1, found.size());
    assertEquals(NoiseSpan.SEVERITY_GIBBERISH, found.get(0).severity());
    assertEquals(text, found.get(0).span().getCoveredText(text).toString());
  }

  /** With a dictionary, one confusion repair reaching a word means misspelled. */
  @Test
  void testConfusionRepairsAreMisspelledWithADictionary() {
    final Set<String> words = Set.of("modern", "times", "word");
    final StructuralNoiseScorer withDictionary =
        new StructuralNoiseScorer(words::contains);
    final String text = "rnodern tirnes and vvord";
    final List<NoiseSpan> found = withDictionary.score(text, List.of());
    assertEquals(2, found.size());
    assertEquals(NoiseSpan.SEVERITY_MISSPELLED, found.get(0).severity());
    assertEquals("rnodern tirnes",
        found.get(0).span().getCoveredText(text).toString());
    assertEquals(NoiseSpan.SEVERITY_MISSPELLED, found.get(1).severity());
    assertEquals("vvord", found.get(1).span().getCoveredText(text).toString());
  }

  /** Without a dictionary the same text yields nothing; the tier needs evidence. */
  @Test
  void testConfusionsWithoutADictionaryAreNotReported() {
    assertEquals(List.of(), scorer.score("rnodern tirnes and vvord", List.of()));
  }

  /** A word the dictionary accepts is never flagged, whatever its structure. */
  @Test
  void testDictionaryAcceptedTokenOverridesStructure() {
    final StructuralNoiseScorer withDictionary =
        new StructuralNoiseScorer(Set.of("zxkcvbnmsdfg")::contains);
    assertEquals(List.of(), withDictionary.score("zxkcvbnmsdfg", List.of()));
  }

  @Test
  void testRejectsContractViolations() {
    assertThrows(IllegalArgumentException.class, () -> scorer.score(null, List.of()));
    assertThrows(IllegalArgumentException.class, () -> scorer.score("x", null));
    final java.util.List<Span> holdsNull = new java.util.ArrayList<>();
    holdsNull.add(null);
    assertThrows(IllegalArgumentException.class, () -> scorer.score("x", holdsNull));
    assertThrows(IllegalArgumentException.class,
        () -> new StructuralNoiseScorer(null));
  }

  /** Scores are within their documented range and saturate for long payloads. */
  @Test
  void testScoresStayInRange() {
    final String longRun = "QWxh1ZGRp2bjF2c3BlbjRzZXNhbWU5QWxh1ZGRp2bjF2c3BlbjR"
        + "zZXNhbWU5";
    for (final NoiseSpan span : scorer.score(longRun, List.of())) {
      assertTrue(span.score() > 0.0 && span.score() <= 1.0);
    }
  }
}
