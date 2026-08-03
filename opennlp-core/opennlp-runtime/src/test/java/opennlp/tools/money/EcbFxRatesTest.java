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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.Optional;

import org.junit.jupiter.api.Test;

import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests the reference-history loader and the as-of lookup rules against a small
 * project-authored fixture in the published CSV format; the rates are synthetic and
 * carry no data from any provider.
 */
public class EcbFxRatesTest {

  private static final String FIXTURE = String.join("\n",
      "Date,USD,JPY,GBP,",
      "2026-07-10,1.2000,160.00,0.8500,",
      "2026-07-09,1.1000,155.00,N/A,",
      "2026-06-01,1.0000,150.00,0.9000,",
      "") + "\n";

  private static EcbFxRates rates() throws IOException {
    return EcbFxRates.load(new ByteArrayInputStream(FIXTURE.getBytes()));
  }

  private static void assertRate(String expected, Optional<BigDecimal> actual) {
    assertTrue(actual.isPresent());
    assertEquals(0, new BigDecimal(expected).compareTo(actual.get()));
  }

  @Test
  void testEuroBaseRateOnAReferenceDay() throws IOException {
    assertRate("1.2000", rates().rate("EUR", "USD", LocalDate.parse("2026-07-10")));
    assertRate("0.8333333333333333",
        rates().rate("USD", "EUR", LocalDate.parse("2026-07-10")));
  }

  @Test
  void testCrossRateGoesThroughTheEuroBase() throws IOException {
    // JPY per USD = 160.00 / 1.2000
    assertRate("133.3333333333333",
        rates().rate("USD", "JPY", LocalDate.parse("2026-07-10")));
  }

  @Test
  void testWeekendFallsBackToThePreviousReferenceDay() throws IOException {
    assertRate("1.2000", rates().rate("EUR", "USD", LocalDate.parse("2026-07-12")));
  }

  @Test
  void testStaleRatesAreAbsentBeyondTheLimit() throws IOException {
    // the nearest earlier row is 2026-06-01, more than seven days before
    assertTrue(rates().rate("EUR", "USD", LocalDate.parse("2026-07-01")).isEmpty());
    // and nothing exists before the first row at all
    assertTrue(rates().rate("EUR", "USD", LocalDate.parse("2026-05-01")).isEmpty());
  }

  /**
   * Verifies that the staleness limit is inclusive: a lookup exactly
   * {@link EcbFxRates#MAX_STALENESS_DAYS} days after the latest row still resolves, and
   * one day later the rate is absent.
   */
  @Test
  void testStalenessLimitIsInclusive() throws IOException {
    assertRate("1.2000", rates().rate("EUR", "USD", LocalDate.parse("2026-07-17")));
    assertTrue(rates().rate("EUR", "USD", LocalDate.parse("2026-07-18")).isEmpty());
  }

  @Test
  void testNotAvailableCellsAndUnknownCurrenciesAreAbsent() throws IOException {
    assertTrue(rates().rate("EUR", "GBP", LocalDate.parse("2026-07-09")).isEmpty());
    assertTrue(rates().rate("EUR", "CHF", LocalDate.parse("2026-07-10")).isEmpty());
  }

  @Test
  void testIdentityRate() throws IOException {
    assertRate("1", rates().rate("EUR", "EUR", LocalDate.parse("2026-07-10")));
    assertRate("1", rates().rate("USD", "USD", LocalDate.parse("2026-07-10")));
  }

  @Test
  void testConvertKeepsTheSpan() throws IOException {
    final MoneyAmount mention =
        new MoneyAmount(new Span(3, 7), new BigDecimal("10"), "EUR");
    final Optional<MoneyAmount> converted =
        rates().convert(mention, "USD", LocalDate.parse("2026-07-10"));
    assertTrue(converted.isPresent());
    assertEquals(new Span(3, 7), converted.get().span());
    assertEquals(0, new BigDecimal("12").compareTo(converted.get().amount()));
    assertEquals("USD", converted.get().currency());
  }

  @Test
  void testMalformedContentFailsLoud() {
    assertThrows(IllegalArgumentException.class,
        () -> EcbFxRates.load(new ByteArrayInputStream("no header here".getBytes())));
    assertThrows(IllegalArgumentException.class,
        () -> EcbFxRates.load(new ByteArrayInputStream(
            "Date,USD,\nyesterday,1.1,\n".getBytes())));
    assertThrows(IllegalArgumentException.class,
        () -> EcbFxRates.load(new ByteArrayInputStream("Date,USD,\n".getBytes())));
    assertThrows(IllegalArgumentException.class, () -> EcbFxRates.load((InputStream) null));
  }

  /**
   * Verifies that a malformed rate cell fails loud as {@link IllegalArgumentException}
   * naming the cell and the row, matching the date-cell handling in the same loop.
   */
  @Test
  void testMalformedRateCellFailsLoudWithRowContext() {
    final String csv = "Date,USD,JPY,\n2026-07-10,1.08x,160.00,\n";
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
        () -> EcbFxRates.load(new ByteArrayInputStream(csv.getBytes())));
    assertEquals("not a reference history rate for USD: 1.08x in row: "
        + "2026-07-10,1.08x,160.00,", e.getMessage());
  }

  @Test
  void testArgumentValidation() throws IOException {
    final EcbFxRates rates = rates();
    final LocalDate date = LocalDate.parse("2026-07-10");
    assertThrows(IllegalArgumentException.class, () -> rates.rate(" ", "USD", date));
    assertThrows(IllegalArgumentException.class, () -> rates.rate("EUR", null, date));
    assertThrows(IllegalArgumentException.class, () -> rates.rate("EUR", "USD", null));
    assertThrows(IllegalArgumentException.class, () -> rates.convert(null, "USD", date));
  }
}
