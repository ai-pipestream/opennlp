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
import java.time.LocalDate;
import java.util.Optional;

/**
 * The interface for foreign exchange rate providers: the rate between two currencies as
 * of a date, and the derived conversion of a {@link MoneyAmount}.
 *
 * <p>Rate data is always supplied by the caller, typically from a downloaded reference
 * table; OpenNLP bundles no rate data. A conversion is as-of dated because rates move:
 * converting money mentioned in a document is only meaningful against the date the
 * document speaks from, which callers can take from document metadata or from a
 * temporal mention in the text.</p>
 *
 * @since 3.0.0
 */
public interface FxRates {

  /**
   * Retrieves the exchange rate between two currencies as of a date.
   *
   * @param from The ISO 4217 code of the source currency. Must not be {@code null} or
   *             blank.
   * @param to The ISO 4217 code of the target currency. Must not be {@code null} or
   *           blank.
   * @param asOf The date the rate should hold for. Must not be {@code null}.
   * @return The number of target units per source unit, or empty when the provider has
   *         no usable rate for the pair at that date. Never {@code null}.
   * @throws IllegalArgumentException Thrown if any parameter is {@code null} or a code
   *         is blank.
   */
  Optional<BigDecimal> rate(String from, String to, LocalDate asOf);

  /**
   * Converts a money mention into another currency, keeping its span.
   *
   * @param money The mention to convert. Must not be {@code null}.
   * @param to The ISO 4217 code of the target currency. Must not be {@code null} or
   *           blank.
   * @param asOf The date the conversion should hold for. Must not be {@code null}.
   * @return A {@link MoneyAmount} in the target currency on the original span, or empty
   *         when no usable rate exists. Never {@code null}.
   * @throws IllegalArgumentException Thrown if any parameter is {@code null} or
   *         {@code to} is blank.
   */
  default Optional<MoneyAmount> convert(MoneyAmount money, String to, LocalDate asOf) {
    if (money == null) {
      throw new IllegalArgumentException("money must not be null");
    }
    return rate(money.currency(), to, asOf)
        .map(rate -> new MoneyAmount(money.span(), money.amount().multiply(rate), to));
  }
}
