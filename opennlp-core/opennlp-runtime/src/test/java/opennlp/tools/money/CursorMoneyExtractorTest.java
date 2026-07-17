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

package opennlp.tools.money;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests the recognized money shapes, the normalization of grouping and scale markers,
 * and the reject cases that keep bare numbers and near-misses out of the money layer.
 */
public class CursorMoneyExtractorTest {

  private final CursorMoneyExtractor extractor = new CursorMoneyExtractor();

  private MoneyAmount single(String text) {
    final List<MoneyAmount> mentions = extractor.extract(text);
    assertEquals(1, mentions.size(), "expected one mention in: " + text);
    return mentions.get(0);
  }

  @ParameterizedTest
  @CsvSource(delimiter = ';', value = {
      "$1,234.56; 1234.56; USD",
      "$1,234,567.89; 1234567.89; USD",
      "$ 100; 100; USD",
      "\u20AC50; 50; EUR",      // U+20AC, the euro sign
      "50\u20AC; 50; EUR",
      "\u00A32.5k; 2500.0; GBP",   // U+00A3, the pound sign
      "\u00A51000; 1000; JPY",     // U+00A5, the yen sign
      "USD 100; 100; USD",
      "100 USD; 100; USD",
      "$1.2M; 1200000.0; USD",
      "$3bn; 3000000000; USD",
      "$3 billion; 3000000000; USD",
      "USD 1.2 million; 1200000.0; USD",
      "1.2 million USD; 1200000.0; USD",
      "-$5; -5; USD",
      "-5 USD; -5; USD",
      "CHF 42; 42; CHF"
  })
  void testRecognizedShapesCoverTheFullMention(String text, String amount, String currency) {
    final MoneyAmount mention = single(text);
    assertEquals(new Span(0, text.length()), mention.span(), text);
    assertEquals(0, new BigDecimal(amount).compareTo(mention.amount()), text);
    assertEquals(currency, mention.currency(), text);
  }

  @Test
  void testMentionInsideSentenceHasExactSpan() {
    final MoneyAmount mention = single("they paid $1.2M for it.");
    assertEquals(new Span(10, 15), mention.span());
    assertEquals(0, new BigDecimal("1200000").compareTo(mention.amount()));
  }

  @Test
  void testSymbolAfterLetterStillMatches() {
    // the US$ convention: the symbol match does not demand a boundary before it
    final MoneyAmount mention = single("US$100");
    assertEquals(new Span(2, 6), mention.span());
    assertEquals("USD", mention.currency());
  }

  @Test
  void testMultipleMentions() {
    final List<MoneyAmount> mentions = extractor.extract("paid $5 and \u20AC3 today");
    assertEquals(2, mentions.size());
    assertEquals("USD", mentions.get(0).currency());
    assertEquals("EUR", mentions.get(1).currency());
  }

  /**
   * Verifies that the hyphen in a price range is not read as a minus sign: both
   * amounts of {@code $100-$200} are positive and their spans exclude the hyphen.
   */
  @Test
  void testHyphenatedRangeYieldsTwoPositiveMentions() {
    final List<MoneyAmount> mentions = extractor.extract("$100-$200");
    assertEquals(2, mentions.size());
    assertEquals(new Span(0, 4), mentions.get(0).span());
    assertEquals(0, new BigDecimal("100").compareTo(mentions.get(0).amount()));
    assertEquals(new Span(5, 9), mentions.get(1).span());
    assertEquals(0, new BigDecimal("200").compareTo(mentions.get(1).amount()));
  }

  /**
   * Verifies that a hyphen directly after a letter is ordinary prose punctuation, not
   * a minus sign, so {@code pre-$5 deal} reports a positive amount.
   */
  @Test
  void testHyphenAfterLetterIsNotAMinus() {
    final MoneyAmount mention = single("pre-$5 deal");
    assertEquals(new Span(4, 6), mention.span());
    assertEquals(0, new BigDecimal("5").compareTo(mention.amount()));
  }

  /**
   * Verifies that a genuinely negative symbol-first mention keeps its sign at every
   * boundary the number-first shape accepts: the text start, after whitespace, and
   * after non-alphanumeric punctuation.
   */
  @Test
  void testNegativeSymbolFirstAtBoundariesKeepsTheSign() {
    final MoneyAmount atStart = single("-$5");
    assertEquals(new Span(0, 3), atStart.span());
    assertEquals(0, new BigDecimal("-5").compareTo(atStart.amount()));

    final MoneyAmount afterSpace = single("balance -$5");
    assertEquals(new Span(8, 11), afterSpace.span());
    assertEquals(0, new BigDecimal("-5").compareTo(afterSpace.amount()));

    final MoneyAmount afterParenthesis = single("(-$5)");
    assertEquals(new Span(1, 4), afterParenthesis.span());
    assertEquals(0, new BigDecimal("-5").compareTo(afterParenthesis.amount()));
  }

  @Test
  void testInvalidGroupingEndsTheMatchAtTheLastValidPosition() {
    final MoneyAmount mention = single("$1,23");
    assertEquals(new Span(0, 2), mention.span());
    assertEquals(0, BigDecimal.ONE.compareTo(mention.amount()));
  }

  /**
   * Verifies what the scanner accepts for a European-style written amount: the decimal
   * comma is not interpreted, so the match ends before the comma and the dot-separated
   * part is read as a decimal point, since locale-dependent decimal commas are out of
   * scope by design.
   */
  @Test
  void testDecimalCommaEndsTheMatchBeforeTheComma() {
    // \u20AC1.234,56 reads as one euro mention over \u20AC1.234
    final MoneyAmount mention = single("\u20AC1.234,56");
    assertEquals(new Span(0, 6), mention.span());
    assertEquals(0, new BigDecimal("1.234").compareTo(mention.amount()));
    assertEquals("EUR", mention.currency());
  }

  /**
   * Verifies the exact spans of a mention at the very start and at the very end of the
   * text, where the boundary checks run against the text bounds.
   */
  @Test
  void testMentionAtTextStartAndEnd() {
    assertEquals(new Span(0, 2), single("$5 up front").span());
    assertEquals(new Span(11, 13), single("the fee is $7").span());
  }

  @Test
  void testTrailingPunctuationIsNotIncluded() {
    final MoneyAmount mention = single("it cost $3.20.");
    assertEquals("$3.20", mention.span().getCoveredText("it cost $3.20.").toString());
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "5%",                  // percent is not money
      "the 100 things",      // bare number
      "USD alone",           // code without a number
      "$x",                  // symbol without a number
      "$5x",                 // invalid letter suffix invalidates the match
      "100 USDX",            // not an ISO code
      "chapter USD",         // no number after the code
      "USD -5",              // the code-first shape takes no leading minus
      "call 555,1234 now"    // invalid grouping, no currency
  })
  void testRejectedShapes(String text) {
    assertTrue(extractor.extract(text).isEmpty(), text);
  }

  @Test
  void testCustomSymbolTable() {
    final CursorMoneyExtractor canadian =
        new CursorMoneyExtractor(Map.of((int) '$', "CAD"));
    assertEquals("CAD", canadian.extract("$5").get(0).currency());
  }

  @Test
  void testCustomSymbolTableValidation() {
    assertThrows(IllegalArgumentException.class, () -> new CursorMoneyExtractor(null));
    assertThrows(IllegalArgumentException.class, () -> new CursorMoneyExtractor(Map.of()));
    assertThrows(IllegalArgumentException.class,
        () -> new CursorMoneyExtractor(Map.of((int) '$', "DOLLARS")));
  }

  @Test
  void testNullTextThrows() {
    assertThrows(IllegalArgumentException.class, () -> extractor.extract(null));
  }

  @Test
  void testMoneyAmountValidation() {
    assertThrows(IllegalArgumentException.class,
        () -> new MoneyAmount(null, BigDecimal.ONE, "USD"));
    assertThrows(IllegalArgumentException.class,
        () -> new MoneyAmount(new Span(0, 1), null, "USD"));
    assertThrows(IllegalArgumentException.class,
        () -> new MoneyAmount(new Span(0, 1), BigDecimal.ONE, " "));
  }
}
