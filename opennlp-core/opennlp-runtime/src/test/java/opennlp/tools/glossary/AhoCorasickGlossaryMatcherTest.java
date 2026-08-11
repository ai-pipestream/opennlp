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
import org.junit.jupiter.params.provider.CsvSource;

import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.DashCharSequenceNormalizer;
import opennlp.tools.util.normalizer.EnglishContractionCharSequenceNormalizer;
import opennlp.tools.util.normalizer.FullCaseFoldCharSequenceNormalizer;
import opennlp.tools.util.normalizer.GermanUmlautCharSequenceNormalizer;
import opennlp.tools.util.normalizer.InvisibleCharSequenceNormalizer;
import opennlp.tools.util.normalizer.OffsetAwareNormalizer;
import opennlp.tools.util.normalizer.TextNormalizer;
import opennlp.tools.util.normalizer.WhitespaceCharSequenceNormalizer;

public class AhoCorasickGlossaryMatcherTest {

  /** A case-sensitive one-term glossary, shared by the word boundary tests. */
  private static final AhoCorasickGlossaryMatcher CAT_MATCHER =
      new AhoCorasickGlossaryMatcher(List.of(new GlossaryEntry("CAT", "cat")), false);

  @Test
  void testFindsSingleAndMultiwordTerms() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("Q60", "New York City"),
        new GlossaryEntry("Q11299", "Manhattan")), false);

    final String text = "She moved from Manhattan to New York City last year.";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("Q11299", matches.get(0).id());
    Assertions.assertEquals("Manhattan", text.substring(
        matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
    Assertions.assertEquals("Q60", matches.get(1).id());
    Assertions.assertEquals("New York City", text.substring(
        matches.get(1).span().getStart(), matches.get(1).span().getEnd()));
  }

  @Test
  void testPrefersLongestMatch() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("STATE", "New York"),
        new GlossaryEntry("CITY", "New York City")), false);

    final List<GlossaryMatch> matches = matcher.match("Flights to New York City are full.");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("CITY", matches.get(0).id());
  }

  @Test
  void testLeftmostWinsOverLaterOverlap() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("A", "New York"),
        new GlossaryEntry("B", "York City")), false);

    final List<GlossaryMatch> matches = matcher.match("in New York City today");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("A", matches.get(0).id());
  }

  @Test
  void testIgnoreCaseKeepsOriginalSpan() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ML", "machine learning")), true);

    final String text = "Machine Learning is popular.";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(0, matches.get(0).span().getStart());
    Assertions.assertEquals("Machine Learning".length(), matches.get(0).span().getEnd());
    Assertions.assertEquals("machine learning", matches.get(0).term());
  }

  @Test
  void testCaseSensitiveByDefaultDoesNotCrossCase() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ML", "machine learning")), false);

    Assertions.assertTrue(matcher.match("Machine Learning is popular.").isEmpty());
  }

  @ParameterizedTest
  @CsvSource({
      "concatenate the files, 0",
      "the cat sleeps, 1",
      "cat, 1",
      "a cat., 1"
  })
  void testRespectsWordBoundaries(String text, int expectedHits) {
    Assertions.assertEquals(expectedHits, CAT_MATCHER.match(text).size());
  }

  @Test
  void testAliasesShareOneId() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("ACME-1", "widget press"),
        new GlossaryEntry("ACME-1", "press for widgets")), false);

    final List<GlossaryMatch> matches =
        matcher.match("The widget press replaced the old press for widgets.");

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("ACME-1", matches.get(0).id());
    Assertions.assertEquals("ACME-1", matches.get(1).id());
    Assertions.assertEquals("widget press", matches.get(0).term());
    Assertions.assertEquals("press for widgets", matches.get(1).term());
  }

  @Test
  void testDuplicateTermFirstRegistrationWins() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("FIRST", "python"),
        new GlossaryEntry("SECOND", "python")), false);

    final List<GlossaryMatch> matches = matcher.match("a python slithered by");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("FIRST", matches.get(0).id());
  }

  @Test
  void testRepeatedHitsAllReported() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("T", "gene therapy")), true);

    Assertions.assertEquals(2,
        matcher.match("Gene therapy trials expand; gene therapy works.").size());
  }

  /**
   * Verifies exact spans for hits touching both text edges: one term starts at offset
   * zero and another ends exactly at the text length, exercising both boundary checks
   * that have no neighboring character to inspect.
   */
  @Test
  void testTermsAtTextStartAndEndReportExactSpans() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("ML", "machine learning"),
        new GlossaryEntry("TTS", "speech synthesis")), false);

    final String text = "machine learning powers speech synthesis";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals(0, matches.get(0).span().getStart());
    Assertions.assertEquals(16, matches.get(0).span().getEnd());
    Assertions.assertEquals("ML", matches.get(0).id());
    Assertions.assertEquals(24, matches.get(1).span().getStart());
    Assertions.assertEquals(40, matches.get(1).span().getEnd());
    Assertions.assertEquals(text.length(), matches.get(1).span().getEnd());
    Assertions.assertEquals("TTS", matches.get(1).id());
  }

  /**
   * Verifies overlap resolution when one registered term is a prefix of another: where
   * both start together the longer term is the only reported hit, while a standalone
   * occurrence of the shorter term elsewhere is still reported with its exact span.
   */
  @Test
  void testPrefixTermSuppressedInsideLongerMatchButReportedAlone() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("G", "gene"),
        new GlossaryEntry("GT", "gene therapy")), false);

    final List<GlossaryMatch> matches =
        matcher.match("gene therapy advanced; the gene won.");

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("GT", matches.get(0).id());
    Assertions.assertEquals(0, matches.get(0).span().getStart());
    Assertions.assertEquals(12, matches.get(0).span().getEnd());
    Assertions.assertEquals("G", matches.get(1).id());
    Assertions.assertEquals(27, matches.get(1).span().getStart());
    Assertions.assertEquals(31, matches.get(1).span().getEnd());
  }

  /**
   * Verifies overlap resolution when one registered term occurs strictly inside another:
   * the inner term is suppressed where the longer term matches, while a standalone
   * occurrence of the inner term elsewhere is still reported with its exact span.
   */
  @Test
  void testInnerTermSuppressedInsideLongerMatchButReportedAlone() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("Q60", "New York City"),
        new GlossaryEntry("Q1384", "York")), false);

    final List<GlossaryMatch> matches =
        matcher.match("York is old; New York City is new.");

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("Q1384", matches.get(0).id());
    Assertions.assertEquals(0, matches.get(0).span().getStart());
    Assertions.assertEquals(4, matches.get(0).span().getEnd());
    Assertions.assertEquals("Q60", matches.get(1).id());
    Assertions.assertEquals(13, matches.get(1).span().getStart());
    Assertions.assertEquals(26, matches.get(1).span().getEnd());
  }

  /**
   * Verifies that a term occurring twice is reported once per occurrence, in text
   * order, with the exact span of each occurrence.
   */
  @Test
  void testSameTermTwiceReportsBothOccurrences() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("DL", "data lake")), false);

    final String text = "data lake governance beats data lake chaos";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals(0, matches.get(0).span().getStart());
    Assertions.assertEquals(9, matches.get(0).span().getEnd());
    Assertions.assertEquals(27, matches.get(1).span().getStart());
    Assertions.assertEquals(36, matches.get(1).span().getEnd());
    Assertions.assertEquals("data lake",
        text.substring(matches.get(1).span().getStart(), matches.get(1).span().getEnd()));
  }

  /**
   * Verifies that scanning an empty text is legal and yields no matches.
   */
  @Test
  void testEmptyTextYieldsNoMatches() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("T", "term")), false);

    Assertions.assertTrue(matcher.match("").isEmpty());
  }

  /**
   * Verifies that spans count UTF-16 chars, not code points: each supplementary
   * character before the term occupies two char positions, so the hit starts at offset
   * five rather than three.
   */
  @Test
  void testSpansCountUtf16CharsForSupplementaryCharacters() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("PAD", "launch pad")), false);

    // Two rocket emoji (U+1F680), each one surrogate pair, precede the term.
    final String text = "\uD83D\uDE80\uD83D\uDE80 launch pad ready";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(5, matches.get(0).span().getStart());
    Assertions.assertEquals(15, matches.get(0).span().getEnd());
    Assertions.assertEquals("launch pad",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * Verifies that a term containing a non-ASCII letter matches with its exact span and
   * that the accented character counts as one char position.
   */
  @Test
  void testAccentedTermMatchesWithExactSpan() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("CAF", "caf\u00E9 au lait")), false);

    // The e with acute accent (U+00E9) stays a single char in span arithmetic.
    final String text = "I ordered caf\u00E9 au lait today.";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(10, matches.get(0).span().getStart());
    Assertions.assertEquals(22, matches.get(0).span().getEnd());
    Assertions.assertEquals("caf\u00E9 au lait",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * Verifies the digit half of the boundary contract: a hit whose neighbor is a digit
   * continues a word exactly like one whose neighbor is a letter, so a code embedded
   * in a longer alphanumeric run never matches, while the same code between
   * non-alphanumeric neighbors does.
   */
  @Test
  void testDigitNeighborsBlockTheBoundary() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ID", "b12")), false);
    Assertions.assertTrue(matcher.match("ab123x").isEmpty());
    Assertions.assertTrue(matcher.match("4b12").isEmpty());
    final List<GlossaryMatch> matches = matcher.match("(b12)");
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(1, 4), matches.get(0).span());
  }

  /**
   * Verifies that a supplementary letter next to a hit blocks the boundary like a
   * basic-plane one: Deseret letters lie above U+FFFF, and a term glued to one is
   * inside a word, not on a boundary.
   */
  @Test
  void testSupplementaryLetterNeighborBlocksTheBoundary() {
    // U+10428, DESERET SMALL LETTER LONG I, a supplementary-plane letter
    Assertions.assertTrue(CAT_MATCHER.match("\uD801\uDC28cat").isEmpty());
    Assertions.assertTrue(CAT_MATCHER.match("cat\uD801\uDC28").isEmpty());
    Assertions.assertEquals(1, CAT_MATCHER.match("a \uD801\uDC28 cat").size());
  }

  /**
   * Verifies that boundary filtering happens per hit, before overlap resolution: a
   * longer candidate rejected at its boundary does not shadow a shorter overlapping
   * term that sits on clean boundaries of its own.
   */
  @Test
  void testBoundaryRejectedLongerCandidateLetsTheShorterTermThrough() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("LONG", "launch pad"),
            new GlossaryEntry("SHORT", "launch")), false);
    // "pads" continues the longer term's last word, so only the shorter term stands
    final List<GlossaryMatch> matches = matcher.match("launch pads ready");
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("SHORT", matches.get(0).id());
    Assertions.assertEquals(new Span(0, 6), matches.get(0).span());
  }

  /**
   * Verifies case-insensitive matching on a non-ASCII basic-plane pair and pins the
   * documented limitation at the same time: the per-code-point mapping folds a
   * dotted-capital letter to its simple lowercase, so the sharp s never folds to
   * {@code ss} and a term spelled with {@code ss} does not match the sharp s.
   */
  @Test
  void testNonAsciiCasePairFoldsAndMultiCharacterFoldingDoesNot() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("RES", "r\u00E9sum\u00E9")), true);
    // U+00C9/U+00E9, the accented E pair, folds per code point
    final List<GlossaryMatch> matches = matcher.match("R\u00C9SUM\u00C9 attached");
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(0, 6), matches.get(0).span());

    final AhoCorasickGlossaryMatcher sharp = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse")), true);
    // U+00DF, the sharp s: its full folding is the two-letter ss, which the
    // per-code-point mapping deliberately does not apply
    Assertions.assertTrue(sharp.match("stra\u00DFe").isEmpty());
  }

  /**
   * An {@link OffsetAwareNormalizer} folds length-changing forms before the automaton
   * runs, and the hit span maps back to the original characters that produced the fold.
   */
  @Test
  void testOffsetAwareNormalizerMatchesEszettAndKeepsOriginalSpan() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse")), false,
        GermanUmlautCharSequenceNormalizer.getInstance());

    final String text = "die stra\u00DFe hier";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("ST", matches.get(0).id());
    Assertions.assertEquals(new Span(4, 10), matches.get(0).span());
    Assertions.assertEquals("stra\u00DFe",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * German umlaut expansion matches both directions of the DIN fold: an ASCII registration
   * hits both the ASCII and umlaut surfaces, and spans cover the original characters.
   */
  @Test
  void testGermanUmlautNormalizerMatchesUmlautAndAsciiSpellings() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("MUE", "mueller"),
            new GlossaryEntry("KOE", "koeln")), false,
        GermanUmlautCharSequenceNormalizer.getInstance());

    final String text = "mueller and m\u00FCller visit k\u00F6ln";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(3, matches.size());
    Assertions.assertEquals("MUE", matches.get(0).id());
    Assertions.assertEquals(new Span(0, 7), matches.get(0).span());
    Assertions.assertEquals("MUE", matches.get(1).id());
    Assertions.assertEquals("m\u00FCller",
        text.substring(matches.get(1).span().getStart(), matches.get(1).span().getEnd()));
    Assertions.assertEquals("KOE", matches.get(2).id());
    Assertions.assertEquals("k\u00F6ln",
        text.substring(matches.get(2).span().getStart(), matches.get(2).span().getEnd()));
  }

  /**
   * A term registered with the eszett spelling still matches ASCII {@code ss} text once
   * both sides pass through the same expanding fold.
   */
  @Test
  void testTermRegisteredWithEszettMatchesAsciiSsText() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "stra\u00DFe")), false,
        GermanUmlautCharSequenceNormalizer.getInstance());

    final List<GlossaryMatch> matches = matcher.match("see strasse tonight");
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(4, 11), matches.get(0).span());
    Assertions.assertEquals("stra\u00DFe", matches.get(0).term());
  }

  /**
   * Full Unicode case folding is supplied through the normalizer hook, so {@code ignoreCase}
   * stays off and the expanding sharp-s fold still matches while the span covers the
   * original eszett spelling.
   */
  @Test
  void testFullCaseFoldNormalizerMatchesCapitalizedEszettForm() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse")), false,
        FullCaseFoldCharSequenceNormalizer.getInstance());

    final String text = "Stra\u00DFe ahead";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(0, 6), matches.get(0).span());
    Assertions.assertEquals("Stra\u00DFe",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * The capital sharp s (U+1E9E) expands to {@code SS} under the German fold. With
   * {@code ignoreCase}, that still matches a lowercase registration, and hits at the
   * very start or end of the text report a tight original span.
   */
  @Test
  void testCapitalEszettAtTextEdgesMapsTightOriginalSpans() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse")), true,
        GermanUmlautCharSequenceNormalizer.getInstance());

    final String start = "STRA\u1E9EE";
    final List<GlossaryMatch> atStart = matcher.match(start);
    Assertions.assertEquals(1, atStart.size());
    Assertions.assertEquals(new Span(0, start.length()), atStart.get(0).span());

    final String end = "x STRA\u1E9EE";
    final List<GlossaryMatch> atEnd = matcher.match(end);
    Assertions.assertEquals(1, atEnd.size());
    Assertions.assertEquals(new Span(2, end.length()), atEnd.get(0).span());
  }

  /**
   * {@code ignoreCase} still applies after the offset-aware fold, so a lowercase term
   * matches a mixed-case umlaut surface without requiring full case folding.
   */
  @Test
  void testIgnoreCaseAppliesAfterOffsetAwareFold() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse")), true,
        GermanUmlautCharSequenceNormalizer.getInstance());

    final String text = "Die Stra\u00DFe ist frei.";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("Stra\u00DFe",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * Whitespace collapse, dash folding, and invisible stripping compose through
   * {@link TextNormalizer.Builder#buildAligned()}, and every hit span is expressed in
   * original-text coordinates across those edits.
   */
  @Test
  void testAlignedPipelineNormalizerMapsHitsAcrossCollapseAndDashFold() {
    final OffsetAwareNormalizer pipeline = TextNormalizer.builder()
        .stripInvisible().whitespace().dashes().buildAligned();
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("NY", "New York"),
            new GlossaryEntry("AB", "a-b")), false, pipeline);

    final String zwsp = "\u200B";
    final String text = "New" + zwsp + "  York and a\u2014b done";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("NY", matches.get(0).id());
    Assertions.assertEquals("New" + zwsp + "  York",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
    Assertions.assertEquals("AB", matches.get(1).id());
    Assertions.assertEquals("a\u2014b",
        text.substring(matches.get(1).span().getStart(), matches.get(1).span().getEnd()));
  }

  /**
   * Leading and trailing whitespace collapsed by the normalizer must not be pulled into
   * the hit span; only the original characters that produced the match are covered.
   */
  @Test
  void testWhitespaceCollapseDoesNotExpandSpanIntoNeighborSpaces() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("NY", "New York")), false,
        WhitespaceCharSequenceNormalizer.getInstance());

    final String text = "  New   York  ";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(2, 12), matches.get(0).span());
    Assertions.assertEquals("New   York",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * After folding, a longer term still wins over a shorter overlapping one, and the
   * reported span covers the original (pre-fold) characters of the longer hit.
   */
  @Test
  void testLongestMatchWinsAfterLengthChangingFold() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("SHORT", "strasse"),
            new GlossaryEntry("LONG", "strassebahn")), false,
        GermanUmlautCharSequenceNormalizer.getInstance());

    final String text = "stra\u00DFebahn kommt";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("LONG", matches.get(0).id());
    Assertions.assertEquals("stra\u00DFebahn",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * Two registrations that become the same pattern after folding keep first-wins
   * registration order; a later distinct folded term still matches beside it.
   */
  @Test
  void testDuplicateFoldedTermsKeepRegistrationOrder() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("FIRST", "strasse"),
            new GlossaryEntry("SECOND", "stra\u00DFe"),
            new GlossaryEntry("CITY", "koeln")), false,
        GermanUmlautCharSequenceNormalizer.getInstance());

    final List<GlossaryMatch> matches = matcher.match("strasse in k\u00F6ln");
    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("FIRST", matches.get(0).id());
    Assertions.assertEquals("CITY", matches.get(1).id());
  }

  /**
   * The three-argument constructor rejects a null normalizer; the two-argument form
   * remains the identity path.
   */
  @Test
  void testThreeArgumentConstructorRejectsNullNormalizer() {
    final List<GlossaryEntry> glossary = List.of(new GlossaryEntry("ML", "machine learning"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(glossary, true, null));
    Assertions.assertFalse(
        new AhoCorasickGlossaryMatcher(glossary, true).match("Machine Learning").isEmpty());
  }

  /**
   * Word-boundary filtering still uses the original text after a length-changing fold,
   * so a match whose original edges sit inside a letter or digit run is dropped.
   */
  @Test
  void testNormalizerHitStillRespectsOriginalWordBoundaries() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse")), false,
        GermanUmlautCharSequenceNormalizer.getInstance());

    Assertions.assertTrue(matcher.match("superstra\u00DFe").isEmpty());
    Assertions.assertTrue(matcher.match("stra\u00DFe9").isEmpty());
    Assertions.assertTrue(matcher.match("9stra\u00DFe").isEmpty());
    Assertions.assertEquals(1, matcher.match("stra\u00DFe").size());
    Assertions.assertEquals(1, matcher.match("(stra\u00DFe)").size());
  }

  /**
   * A glossary term that normalizes to blank is rejected at construction so the
   * automaton never holds an empty pattern. Invisible-only terms (accepted by
   * {@link GlossaryEntry} because zero-width space is not toolkit whitespace) are the
   * concrete case.
   */
  @Test
  void testRejectsTermThatNormalizesToBlank() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(
            List.of(new GlossaryEntry("Z", "\u200B")), false,
            InvisibleCharSequenceNormalizer.getInstance()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(
            List.of(new GlossaryEntry("Z2", "\u200B\u200B\u200B")), false,
            InvisibleCharSequenceNormalizer.getInstance()));
  }

  /**
   * Three-argument construction still validates the glossary the same way as the
   * two-argument form, and {@code match(null)} still fails loud.
   */
  @Test
  void testThreeArgumentConstructorValidatesGlossaryAndText() {
    final OffsetAwareNormalizer fold = GermanUmlautCharSequenceNormalizer.getInstance();
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(null, false, fold));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(List.of(), false, fold));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(Collections.singletonList(null), false, fold));
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse")), false, fold);
    Assertions.assertThrows(IllegalArgumentException.class, () -> matcher.match(null));
  }

  /**
   * Single-purpose offset-aware rungs remain usable without a builder pipeline, including
   * en-dash and em-dash folds and a miss when the dash shape is absent from the text.
   */
  @Test
  void testDashAndWhitespaceNormalizersAlone() {
    final AhoCorasickGlossaryMatcher dashes = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("AB", "a-b")), false,
        DashCharSequenceNormalizer.getInstance());
    Assertions.assertEquals(1, dashes.match("see a\u2013b now").size());
    Assertions.assertEquals(1, dashes.match("see a\u2014b now").size());
    Assertions.assertEquals(1, dashes.match("see a-b now").size());
    Assertions.assertTrue(dashes.match("see a b now").isEmpty());

    final AhoCorasickGlossaryMatcher spaces = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("NY", "New York")), false,
        WhitespaceCharSequenceNormalizer.getInstance());
    final String text = "visit New   York today";
    final List<GlossaryMatch> matches = spaces.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("New   York",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
    Assertions.assertTrue(spaces.match("visit NewYork today").isEmpty());
  }

  /**
   * Two non-overlapping folded hits in one scan keep text order, and an empty scan with
   * a normalizer present returns an empty list rather than failing.
   */
  @Test
  void testMultipleFoldedHitsPreserveOrderAndEmptyTextIsEmpty() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse"),
            new GlossaryEntry("MUE", "mueller")), false,
        GermanUmlautCharSequenceNormalizer.getInstance());

    Assertions.assertTrue(matcher.match("").isEmpty());
    final String text = "stra\u00DFe dann m\u00FCller";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("ST", matches.get(0).id());
    Assertions.assertEquals("MUE", matches.get(1).id());
    Assertions.assertTrue(matches.get(0).span().getEnd() <= matches.get(1).span().getStart());
  }

  /**
   * Verifies that ignoring case folds a supplementary-plane pair: the Deseret capital
   * U+10400 lowercases to U+10428 under the per-code-point mapping, so a term
   * registered lowercase matches the capital spelling with its exact span.
   */
  @Test
  void testSupplementaryCasePairFoldsWhenIgnoringCase() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("DES", "\uD801\uDC28")), true);
    final List<GlossaryMatch> matches = matcher.match("- \uD801\uDC00 -");
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(2, 4), matches.get(0).span());
  }

  /**
   * Unspaced Han text matches: under UAX&#160;#29 every ideograph is its own word, so a
   * neighboring ideograph never continues a word across the hit edge. Tokyo (U+6771
   * U+4EAC) is found inside an unspaced Japanese sentence with its exact span.
   */
  @Test
  void testUnspacedHanTermMatchesInsideJapaneseSentence() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("Q1490", "\u6771\u4EAC")), false);

    // watashi wa Tokyo ni sumu, no spaces anywhere
    final String text = "\u79C1\u306F\u6771\u4EAC\u306B\u4F4F\u3080";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("Q1490", matches.get(0).id());
    Assertions.assertEquals(new Span(2, 4), matches.get(0).span());
    Assertions.assertEquals("\u6771\u4EAC",
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * A Han term inside a longer Han run matches: China (U+4E2D U+56FD) is reported
   * inside "Chinese person" (U+4E2D U+56FD U+4EBA), because the trailing ideograph is
   * its own word rather than a continuation. This is how dictionary matching behaves
   * in unspaced scripts, the reverse of the Latin cat/concatenate rule.
   */
  @Test
  void testHanTermInsideLongerHanRunMatches() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("Q148", "\u4E2D\u56FD")), false);

    // wo ai zhongguoren, no spaces
    final String text = "\u6211\u7231\u4E2D\u56FD\u4EBA";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(2, 4), matches.get(0).span());
  }

  /**
   * A Latin letter next to a Han hit does not block it: UAX&#160;#29 places a word
   * boundary between an alphabetic run and an ideograph, so a term glued to ASCII
   * letters still matches with its exact span.
   */
  @Test
  void testLatinNeighborDoesNotBlockHanHit() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("Q1490", "\u6771\u4EAC")), false);

    final String text = "visit\u6771\u4EACnow";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(5, 7), matches.get(0).span());
  }

  /**
   * Hiragana characters are each their own UAX&#160;#29 word, so a hiragana greeting is
   * found inside a longer unspaced hiragana run with its exact span.
   */
  @Test
  void testHiraganaTermMatchesInsideUnspacedHiraganaRun() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("GREET", "\u304A\u306F\u3088\u3046")), false);

    // ohayou gozaimasu, no spaces
    final String text = "\u304A\u306F\u3088\u3046\u3054\u3056\u3044\u307E\u3059";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(0, 4), matches.get(0).span());
  }

  /**
   * Katakana chains under UAX&#160;#29 (a katakana run is one word), so a term embedded
   * in a longer katakana run stays rejected, exactly like cat inside concatenate. The
   * same term beside non-katakana neighbors matches.
   */
  @Test
  void testKatakanaRunStillHidesEmbeddedTerm() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("TOWER", "\u30BF\u30EF\u30FC")), false);

    // toukyou tawaa as one katakana run: no hit inside it
    Assertions.assertTrue(
        matcher.match("\u30C8\u30A6\u30AD\u30E7\u30A6\u30BF\u30EF\u30FC").isEmpty());
    // the same term bounded by Han neighbors is a word of its own
    final String bounded = "\u6771\u4EAC\u30BF\u30EF\u30FC";
    final List<GlossaryMatch> matches = matcher.match(bounded);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(2, 5), matches.get(0).span());
  }

  /**
   * Hangul chains under UAX&#160;#29 (Korean separates words with spaces), so a syllable
   * embedded in a longer Hangul run stays rejected while the spaced occurrence matches.
   */
  @Test
  void testHangulRunStillHidesEmbeddedTerm() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("KO", "\uD55C\uAD6D")), false);

    // hangugeo as one Hangul run: no hit inside it
    Assertions.assertTrue(matcher.match("\uD55C\uAD6D\uC5B4").isEmpty());
    final List<GlossaryMatch> matches = matcher.match("\uD55C\uAD6D \uC5B4");
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(0, 2), matches.get(0).span());
  }

  /**
   * The UAX&#160;#29 boundary rule composes with an offset-aware normalizer: after a
   * whitespace collapse the hit span maps back to original coordinates, and Han
   * neighbors in the original text do not block the mapped hit.
   */
  @Test
  void testHanBoundaryAppliesToNormalizedHitsInOriginalCoordinates() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("Q1490", "\u6771\u4EAC")), false,
        WhitespaceCharSequenceNormalizer.getInstance());

    final String text = "\u79C1\u306F\u6771\u4EAC\u306B";
    final List<GlossaryMatch> matches = matcher.match(text);
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(2, 4), matches.get(0).span());
  }

  /**
   * UAX&#160;#29 keeps an apostrophe inside an English contraction, so the uncontracted
   * prefix is not a complete word at that offset.
   */
  @Test
  void testContractionInteriorIsNotAWordBoundary() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("PREFIX", "can")), false);

    Assertions.assertTrue(matcher.match("can't").isEmpty());
  }

  /**
   * Optional English contraction expansion runs on both the registered term and source
   * text, while the aligned hit still covers the untouched source contraction.
   */
  @ParameterizedTest
  @CsvSource(value = {
      "can not|can't|0|5",
      "will not|won't|0|5",
      "we are|we\u2019re|0|5"
  }, delimiter = '|')
  void testOffsetAwareEnglishContractionExpansion(String term, String text,
      int expectedStart, int expectedEnd) {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("EXPANDED", term)), true,
        EnglishContractionCharSequenceNormalizer.getInstance());

    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(expectedStart, expectedEnd), matches.get(0).span());
    Assertions.assertEquals(text,
        text.substring(matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
  }

  /**
   * A normalized subword still maps to an interior source offset, so UAX&#160;#29
   * prevents contraction expansion from exposing a false standalone hit.
   */
  @Test
  void testContractionExpansionDoesNotExposeSubwordHit() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("PREFIX", "can")), false,
        EnglishContractionCharSequenceNormalizer.getInstance());

    Assertions.assertTrue(matcher.match("can't").isEmpty());
  }

  /** Ambiguous English suffixes fail closed instead of guessing their expansion. */
  @Test
  void testContractionExpansionLeavesAmbiguousSuffixUnmatched() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("AMBIGUOUS", "he is")), false,
        EnglishContractionCharSequenceNormalizer.getInstance());

    Assertions.assertTrue(matcher.match("he's").isEmpty());
  }

  /**
   * UAX&#160;#29 rule WB4 keeps extending marks with their base. A glossary term that
   * omits the mark must not stop inside the resulting word segment.
   */
  @ParameterizedTest
  @CsvSource(value = {
      "cafe|cafe\u0301",
      "\u845B|\u845B\uFE00"
  }, delimiter = '|')
  void testExtendingMarkStaysWithItsBase(String term, String text) {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("PART", term)), false);

    Assertions.assertTrue(matcher.match(text).isEmpty());
  }

  /**
   * UAX&#160;#29 joins letters to connector punctuation and joins acronym components
   * across mid-letter punctuation, so neither prefix is a complete word.
   */
  @ParameterizedTest
  @CsvSource(value = {
      "cat|cat_food",
      "U|U.S.A"
  }, delimiter = '|')
  void testUaxWordJoiningPunctuationBlocksPartialHit(String term, String text) {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("PART", term)), false);

    Assertions.assertTrue(matcher.match(text).isEmpty());
  }

  /**
   * UAX&#160;#29 rule WB3c keeps an extended pictograph after a zero-width joiner in
   * the same segment, so one pictograph inside the sequence is not a complete hit.
   */
  @Test
  void testEmojiZwjSequenceDoesNotExposeAnInteriorBoundary() {
    final String man = "\uD83D\uDC68";
    final String familyPrefix = man + "\u200D\uD83D\uDC69";
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("MAN", man)), false);

    Assertions.assertTrue(matcher.match(familyPrefix).isEmpty());
  }

  /**
   * Supplementary Han ideographs receive the same per-ideograph UAX&#160;#29
   * boundaries as basic-plane Han characters while spans remain UTF-16 offsets.
   */
  @Test
  void testSupplementaryHanTermMatchesBetweenHanNeighbors() {
    final String term = "\uD840\uDC00"; // U+20000
    final String text = "\u4E00" + term + "\u4E8C";
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("SUPPLEMENTARY", term)), false);

    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(new Span(1, 3), matches.get(0).span());
  }

  @Test
  void testInvalidMatcherArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(null, false));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(List.of(), false));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(Collections.singletonList(null), false));
    Assertions.assertThrows(IllegalArgumentException.class, () -> CAT_MATCHER.match(null));
  }

  @Test
  void testInvalidGlossaryEntryArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryEntry(null, "term"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryEntry("", "term"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryEntry(" ", "term"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryEntry("id", null));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryEntry("id", ""));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryEntry("id", " "));
  }

  @Test
  void testInvalidGlossaryMatchArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryMatch(null, "id", "term"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryMatch(new Span(0, 4), null, "term"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryMatch(new Span(0, 4), " ", "term"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryMatch(new Span(0, 4), "id", null));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryMatch(new Span(0, 4), "id", " "));
  }
}
