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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Currency;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.tools.extraction.NumberScan;
import opennlp.tools.util.Span;

/**
 * A deterministic {@link MoneyExtractor}: a single forward scan over the text, no
 * regular expressions, recognizing the common written money shapes.
 *
 * <p>Recognized forms: a currency symbol before or after the number ({@code $1,234.56},
 * {@code 50€}), an ISO 4217 code before or after the number ({@code USD 100},
 * {@code 100 USD}), an optional leading minus ({@code -$5}), and scale markers, either
 * an immediate suffix ({@code $1.2M}, {@code £2.5k}, {@code $3bn}) or a following word
 * ({@code $3 billion}). Digit grouping is validated: once a comma appears, every further
 * group must have exactly three digits, and the match ends at the last valid position.
 * A bare number without a currency marker is never money.</p>
 *
 * <p>Currency symbols are inherently ambiguous; the default table maps each symbol to
 * the ISO code it most commonly denotes, for example {@code $} to {@code USD}. Callers
 * working in another convention supply their own mapping through
 * {@link #CursorMoneyExtractor(Map)}. ISO codes are taken from
 * {@link Currency#getAvailableCurrencies()}, so no currency data is bundled.</p>
 *
 * <p>Out of scope in this version, by design: accounting negatives in parentheses,
 * multi-character symbols such as {@code kr}, spelled-out currency words such as
 * {@code dollars}, and locale-dependent decimal commas. The extractor holds no per-call
 * state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class CursorMoneyExtractor implements MoneyExtractor {

  private static final Map<Integer, String> DEFAULT_SYMBOLS = Map.ofEntries(
      Map.entry((int) '$', "USD"),
      Map.entry(0x20AC, "EUR"),   // euro sign
      Map.entry((int) '£', "GBP"),
      Map.entry((int) '¥', "JPY"),
      Map.entry(0x20B9, "INR"),   // rupee sign
      Map.entry(0x20A9, "KRW"),   // won sign
      Map.entry(0x20BD, "RUB"),   // ruble sign
      Map.entry(0x20BA, "TRY"),   // lira sign
      Map.entry(0x20AA, "ILS"),   // sheqel sign
      Map.entry(0x0E3F, "THB"),   // baht sign
      Map.entry(0x20AB, "VND"));  // dong sign

  private static final Set<String> ISO_CODES = isoCodes();

  private final Map<Integer, String> symbols;

  /**
   * Initializes the extractor with the default symbol table.
   */
  public CursorMoneyExtractor() {
    this.symbols = DEFAULT_SYMBOLS;
  }

  /**
   * Initializes the extractor with a custom symbol table.
   *
   * @param symbolCurrencies Maps a currency symbol code point to the ISO 4217 code it
   *                         denotes. Must not be {@code null} or empty, and every value
   *                         must be a known ISO 4217 code.
   * @throws IllegalArgumentException Thrown if the map is {@code null}, empty, or names
   *         an unknown currency code.
   */
  public CursorMoneyExtractor(Map<Integer, String> symbolCurrencies) {
    if (symbolCurrencies == null || symbolCurrencies.isEmpty()) {
      throw new IllegalArgumentException("symbolCurrencies must not be null or empty");
    }
    for (final String code : symbolCurrencies.values()) {
      if (code == null || !ISO_CODES.contains(code)) {
        throw new IllegalArgumentException("not an ISO 4217 currency code: " + code);
      }
    }
    this.symbols = Map.copyOf(symbolCurrencies);
  }

  @Override
  public List<MoneyAmount> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<MoneyAmount> mentions = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      final MoneyAmount mention = matchAt(text, i);
      if (mention != null) {
        mentions.add(mention);
        i = mention.span().getEnd();
      } else {
        i += Character.charCount(Character.codePointAt(text, i));
      }
    }
    return Collections.unmodifiableList(mentions);
  }

  /**
   * Tries the three mention shapes at one position: symbol first, ISO code first,
   * number first. Returns {@code null} when none matches.
   */
  private MoneyAmount matchAt(CharSequence text, int start) {
    int i = start;
    boolean negative = false;
    if (NumberScan.charAt(text, i) == '-') {
      negative = true;
      i++;
    }
    final int cp = NumberScan.codePointAt(text, i);
    if (cp != NumberScan.NO_CODE_POINT && symbols.containsKey(cp)) {
      return symbolFirst(text, start, i, negative);
    }
    if (!negative && isUpperAscii(cp) && NumberScan.boundaryBefore(text, i)) {
      final MoneyAmount isoFirst = isoFirst(text, start, i);
      if (isoFirst != null) {
        return isoFirst;
      }
    }
    if (NumberScan.isAsciiDigit(cp)
        && NumberScan.boundaryBefore(text, negative ? start : i)) {
      return numberFirst(text, start, i, negative);
    }
    return null;
  }

  /** Matches {@code $1,234.56}, {@code -$5}, {@code $1.2M}, {@code $ 100}. */
  private MoneyAmount symbolFirst(CharSequence text, int start, int symbolIndex,
      boolean negative) {
    final String currency = symbols.get(NumberScan.codePointAt(text, symbolIndex));
    int i = symbolIndex + Character.charCount(NumberScan.codePointAt(text, symbolIndex));
    if (NumberScan.charAt(text, i) == ' ') {
      i++;
    }
    return mention(text, start, NumberScan.parse(text, i, true), currency, negative);
  }

  /** Matches {@code USD 100} and {@code USD 1.2 million}. */
  private MoneyAmount isoFirst(CharSequence text, int start, int codeIndex) {
    final String code = isoCodeAt(text, codeIndex);
    if (code == null || NumberScan.charAt(text, codeIndex + 3) != ' ') {
      return null;
    }
    return mention(text, start, NumberScan.parse(text, codeIndex + 4, true), code, false);
  }

  /** Matches {@code 100 USD}, {@code 50}{@code €}, and {@code 3.5m USD}. */
  private MoneyAmount numberFirst(CharSequence text, int start, int digitIndex,
      boolean negative) {
    final NumberScan.Result number = NumberScan.parse(text, digitIndex, true);
    if (number == null) {
      return null;
    }
    final int cp = NumberScan.codePointAt(text, number.end());
    if (cp != NumberScan.NO_CODE_POINT && symbols.containsKey(cp)) {
      final NumberScan.Result extended = new NumberScan.Result(number.value(),
          number.end() + Character.charCount(cp));
      return mention(text, start, extended, symbols.get(cp), negative);
    }
    if (cp == ' ') {
      final String code = isoCodeAt(text, number.end() + 1);
      if (code != null) {
        return mention(text, start,
            new NumberScan.Result(number.value(), number.end() + 4), code, negative);
      }
    }
    return null;
  }

  private MoneyAmount mention(CharSequence text, int start, NumberScan.Result number,
      String currency, boolean negative) {
    if (number == null || !NumberScan.boundaryAfter(text, number.end())) {
      return null;
    }
    final BigDecimal amount = negative ? number.value().negate() : number.value();
    return new MoneyAmount(new Span(start, number.end()), amount, currency);
  }

  /** Reads a known ISO 4217 code at the position, or {@code null}. */
  private static String isoCodeAt(CharSequence text, int start) {
    if (start + 3 > text.length() || !NumberScan.boundaryBefore(text, start)) {
      return null;
    }
    for (int i = start; i < start + 3; i++) {
      if (!isUpperAscii(text.charAt(i))) {
        return null;
      }
    }
    if (Character.isLetterOrDigit(NumberScan.charAt(text, start + 3))) {
      return null;
    }
    final String code = text.subSequence(start, start + 3).toString();
    return ISO_CODES.contains(code) ? code : null;
  }

  private static boolean isUpperAscii(int cp) {
    return cp >= 'A' && cp <= 'Z';
  }

  private static Set<String> isoCodes() {
    final Set<String> codes = new HashSet<>();
    for (final Currency currency : Currency.getAvailableCurrencies()) {
      codes.add(currency.getCurrencyCode());
    }
    return Collections.unmodifiableSet(codes);
  }
}
