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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.MathContext;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.format.DateTimeParseException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;

/**
 * An {@link FxRates} implementation over the euro foreign exchange reference history in
 * the CSV format published by the European Central Bank: a {@code Date} header naming
 * the quoted currencies, then one row per reference day with the value of one euro in
 * each currency, {@code N/A} where a currency was not quoted.
 *
 * <p>The table is user-supplied: download the published history file and pass it in;
 * no rate data is bundled, and acknowledging the data source is the caller's
 * responsibility under the publisher's terms. Cross rates are derived through the euro
 * base. Reference rates exist only for business days, so a lookup uses the latest row
 * on or before the requested date, up to {@link #MAX_STALENESS_DAYS} days back; beyond
 * that the rate is reported as absent rather than silently stale. Rate divisions use
 * {@link MathContext#DECIMAL64}.</p>
 *
 * <p>Instances are immutable after loading and safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class EcbFxRates implements FxRates {

  /**
   * How many days a lookup may fall back to the previous reference row, covering
   * weekends and holiday gaps without accepting arbitrarily stale rates.
   */
  public static final int MAX_STALENESS_DAYS = 7;

  private static final String BASE_CURRENCY = "EUR";
  private static final String NOT_AVAILABLE = "N/A";

  private final TreeMap<LocalDate, Map<String, BigDecimal>> table;

  private EcbFxRates(TreeMap<LocalDate, Map<String, BigDecimal>> table) {
    this.table = table;
  }

  /**
   * Loads a reference history table from a file.
   *
   * @param csv The history CSV file. Must not be {@code null}.
   * @return A loaded {@link EcbFxRates}. Never {@code null}.
   * @throws IOException Thrown if reading fails.
   * @throws IllegalArgumentException Thrown if {@code csv} is {@code null} or the
   *         content is not in the expected format.
   */
  public static EcbFxRates load(Path csv) throws IOException {
    if (csv == null) {
      throw new IllegalArgumentException("csv must not be null");
    }
    try (InputStream in = Files.newInputStream(csv)) {
      return load(in);
    }
  }

  /**
   * Loads a reference history table from a stream.
   *
   * @param in The history CSV content. Must not be {@code null}. The stream is read
   *           fully but not closed.
   * @return A loaded {@link EcbFxRates}. Never {@code null}.
   * @throws IOException Thrown if reading fails.
   * @throws IllegalArgumentException Thrown if {@code in} is {@code null} or the
   *         content is not in the expected format.
   */
  public static EcbFxRates load(InputStream in) throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("in must not be null");
    }
    final BufferedReader reader =
        new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
    final String header = reader.readLine();
    if (header == null || !header.startsWith("Date,")) {
      throw new IllegalArgumentException("not a reference history CSV: header is " + header);
    }
    final String[] currencies = header.split(",", -1);
    final TreeMap<LocalDate, Map<String, BigDecimal>> table = new TreeMap<>();
    String line;
    while ((line = reader.readLine()) != null) {
      if (line.isBlank()) {
        continue;
      }
      final String[] fields = line.split(",", -1);
      final LocalDate date;
      try {
        date = LocalDate.parse(fields[0].trim());
      } catch (DateTimeParseException e) {
        throw new IllegalArgumentException("not a reference history row: " + line, e);
      }
      final Map<String, BigDecimal> rates = new HashMap<>();
      final int columns = Math.min(fields.length, currencies.length);
      for (int i = 1; i < columns; i++) {
        final String value = fields[i].trim();
        final String currency = currencies[i].trim();
        if (!value.isEmpty() && !NOT_AVAILABLE.equals(value) && !currency.isEmpty()) {
          rates.put(currency, new BigDecimal(value));
        }
      }
      table.put(date, rates);
    }
    if (table.isEmpty()) {
      throw new IllegalArgumentException("the reference history contains no rows");
    }
    return new EcbFxRates(table);
  }

  @Override
  public Optional<BigDecimal> rate(String from, String to, LocalDate asOf) {
    final Map<String, BigDecimal> row = usableRow(from, to, asOf);
    if (row == null) {
      return Optional.empty();
    }
    final BigDecimal perFrom = perEuro(row, from);
    final BigDecimal perTo = perEuro(row, to);
    if (perFrom == null || perTo == null) {
      return Optional.empty();
    }
    return Optional.of(perTo.divide(perFrom, MathContext.DECIMAL64));
  }

  /**
   * Converts with the division last, so a conversion whose result is exactly
   * representable stays exact instead of inheriting the rounding of a pre-divided
   * cross rate.
   */
  @Override
  public Optional<MoneyAmount> convert(MoneyAmount money, String to, LocalDate asOf) {
    if (money == null) {
      throw new IllegalArgumentException("money must not be null");
    }
    final Map<String, BigDecimal> row = usableRow(money.currency(), to, asOf);
    if (row == null) {
      return Optional.empty();
    }
    final BigDecimal perFrom = perEuro(row, money.currency());
    final BigDecimal perTo = perEuro(row, to);
    if (perFrom == null || perTo == null) {
      return Optional.empty();
    }
    final BigDecimal amount =
        money.amount().multiply(perTo).divide(perFrom, MathContext.DECIMAL64);
    return Optional.of(new MoneyAmount(money.span(), amount, to));
  }

  /**
   * Validates the lookup arguments and resolves the reference row for a date, or
   * {@code null} when no row is within the staleness limit.
   */
  private Map<String, BigDecimal> usableRow(String from, String to, LocalDate asOf) {
    if (from == null || from.isBlank() || to == null || to.isBlank()) {
      throw new IllegalArgumentException("from and to must not be null or blank");
    }
    if (asOf == null) {
      throw new IllegalArgumentException("asOf must not be null");
    }
    final Map.Entry<LocalDate, Map<String, BigDecimal>> row = table.floorEntry(asOf);
    if (row == null || row.getKey().plusDays(MAX_STALENESS_DAYS).isBefore(asOf)) {
      return null;
    }
    return row.getValue();
  }

  /** The value of one euro in a currency, with the base currency itself worth one. */
  private static BigDecimal perEuro(Map<String, BigDecimal> rates, String currency) {
    return BASE_CURRENCY.equals(currency) ? BigDecimal.ONE : rates.get(currency);
  }
}
