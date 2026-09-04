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

package opennlp.tools.numeric;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;
import java.util.Locale;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.money.CursorMoneyExtractor;
import opennlp.tools.money.MoneyAmount;
import opennlp.tools.money.MoneyAnnotator;
import opennlp.tools.quantity.QuantityAnnotator;
import opennlp.tools.temporal.DocumentDateAnnotator;
import opennlp.tools.temporal.TemporalAnnotator;
import opennlp.tools.temporal.TemporalExpression;
import opennlp.tools.util.Span;

/**
 * Mirrors the listings of the numeric chapter of the manual: relative dates resolved
 * against a dateline, the regional number notations, and spelled-out currency words. Each
 * test asserts exactly what its listing claims, so a change in behavior fails here before
 * the manual can go stale.
 */
public class NumericManualExampleTest {

  /**
   * Mirrors the relative-date listing: a dateline elects the document date and resolves the
   * relative expression behind it, both keeping their own spans.
   */
  @Test
  void testRelativeDateListing() {
    final String text = "Berlin, 14 July 2026. The buyer paid yesterday.";

    final Document document = NumericPacks.temporal().analyze(text);

    final List<Annotation<TemporalExpression>> temporals =
        document.get(TemporalAnnotator.TEMPORALS);
    Assertions.assertEquals(2, temporals.size());
    Assertions.assertEquals("2026-07-14", temporals.get(0).value().value());
    Assertions.assertEquals(new Span(8, 20), temporals.get(0).span());
    Assertions.assertEquals("2026-07-13", temporals.get(1).value().value());
    Assertions.assertEquals("yesterday",
        temporals.get(1).span().getCoveredText(text).toString());

    final List<Annotation<LocalDate>> dates =
        document.get(DocumentDateAnnotator.DOCUMENT_DATE);
    Assertions.assertEquals(LocalDate.of(2026, 7, 14), dates.get(0).value());
    Assertions.assertEquals(new Span(8, 20), dates.get(0).span());
  }

  /**
   * Mirrors the relative-date listing's second half: a text that dates itself nowhere leaves
   * the relative expression unreported rather than resolving it against the wall clock.
   */
  @Test
  void testRelativeWithoutADatelineListing() {
    final Document document = NumericPacks.temporal().analyze("The buyer paid yesterday.");

    Assertions.assertEquals(List.of(), document.get(TemporalAnnotator.TEMPORALS));
  }

  /**
   * Mirrors the number-notation listing: the same German text reads at its real magnitude
   * under the regional pipeline and yields nothing at all under the default one.
   */
  @Test
  void testNumberNotationListing() {
    final String text = "Der Kaufer zahlte 2.400.000 EUR fur 1.250,5 GB Speicher.";

    final Document german = NumericPacks.fullPipeline(Locale.GERMANY).analyze(text);
    Assertions.assertEquals(0, new BigDecimal("2400000")
        .compareTo(german.get(MoneyAnnotator.MONEY).get(0).value().amount()));
    Assertions.assertEquals("EUR", german.get(MoneyAnnotator.MONEY).get(0).value().currency());
    Assertions.assertEquals(0, new BigDecimal("1250.5")
        .compareTo(german.get(QuantityAnnotator.QUANTITIES).get(0).value().value()));

    final Document american = NumericPacks.fullPipeline().analyze(text);
    Assertions.assertEquals(List.of(), american.get(MoneyAnnotator.MONEY));
    Assertions.assertEquals(List.of(), american.get(QuantityAnnotator.QUANTITIES));
  }

  /**
   * Mirrors the currency-word listing: an amount named in words is money, and the mention
   * covers the word as well as the digits.
   */
  @Test
  void testCurrencyWordListing() {
    final String text = "the fund raised 1.2 million dollars and 50 euros";

    final List<MoneyAmount> mentions = new CursorMoneyExtractor().extract(text);

    Assertions.assertEquals(2, mentions.size());
    Assertions.assertEquals(0,
        new BigDecimal("1200000").compareTo(mentions.get(0).amount()));
    Assertions.assertEquals("USD", mentions.get(0).currency());
    Assertions.assertEquals("1.2 million dollars",
        mentions.get(0).span().getCoveredText(text).toString());
    Assertions.assertEquals(0, new BigDecimal("50").compareTo(mentions.get(1).amount()));
    Assertions.assertEquals("EUR", mentions.get(1).currency());
    Assertions.assertEquals("50 euros",
        mentions.get(1).span().getCoveredText(text).toString());
  }
}
