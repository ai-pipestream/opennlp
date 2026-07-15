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
      "$ 100; 100; USD",
      "€50; 50; EUR",
      "50€; 50; EUR",
      "£2.5k; 2500.0; GBP",
      "¥1000; 1000; JPY",
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
    final List<MoneyAmount> mentions = extractor.extract("paid $5 and €3 today");
    assertEquals(2, mentions.size());
    assertEquals("USD", mentions.get(0).currency());
    assertEquals("EUR", mentions.get(1).currency());
  }

  @Test
  void testInvalidGroupingEndsTheMatchAtTheLastValidPosition() {
    final MoneyAmount mention = single("$1,23");
    assertEquals(new Span(0, 2), mention.span());
    assertEquals(0, BigDecimal.ONE.compareTo(mention.amount()));
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
