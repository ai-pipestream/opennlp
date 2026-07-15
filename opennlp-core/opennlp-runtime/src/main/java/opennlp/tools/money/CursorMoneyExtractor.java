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

  private static final int NO_MATCH = -1;

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
    if (charAt(text, i) == '-') {
      negative = true;
      i++;
    }
    final int cp = codePointAt(text, i);
    if (cp != NO_MATCH && symbols.containsKey(cp)) {
      return symbolFirst(text, start, i, negative);
    }
    if (!negative && isUpperAscii(cp) && boundaryBefore(text, i)) {
      final MoneyAmount isoFirst = isoFirst(text, start, i);
      if (isoFirst != null) {
        return isoFirst;
      }
    }
    if (isAsciiDigit(cp) && (negative ? boundaryBefore(text, start) : boundaryBefore(text, i))) {
      return numberFirst(text, start, i, negative);
    }
    return null;
  }

  /** Matches {@code $1,234.56}, {@code -$5}, {@code $1.2M}, {@code $ 100}. */
  private MoneyAmount symbolFirst(CharSequence text, int start, int symbolIndex,
      boolean negative) {
    final String currency = symbols.get(codePointAt(text, symbolIndex));
    int i = symbolIndex + Character.charCount(codePointAt(text, symbolIndex));
    if (charAt(text, i) == ' ') {
      i++;
    }
    final Number number = parseNumber(text, i);
    if (number == null) {
      return null;
    }
    return mention(text, start, number, currency, negative);
  }

  /** Matches {@code USD 100} and {@code USD 1.2 million}. */
  private MoneyAmount isoFirst(CharSequence text, int start, int codeIndex) {
    final String code = isoCodeAt(text, codeIndex);
    if (code == null || charAt(text, codeIndex + 3) != ' ') {
      return null;
    }
    final Number number = parseNumber(text, codeIndex + 4);
    if (number == null) {
      return null;
    }
    return mention(text, start, number, code, false);
  }

  /** Matches {@code 100 USD}, {@code 50}{@code €}, and {@code 3.5m USD}. */
  private MoneyAmount numberFirst(CharSequence text, int start, int digitIndex,
      boolean negative) {
    final Number number = parseNumber(text, digitIndex);
    if (number == null) {
      return null;
    }
    final int cp = codePointAt(text, number.end);
    if (cp != NO_MATCH && symbols.containsKey(cp)) {
      final Number extended = new Number(number.value,
          number.end + Character.charCount(cp));
      return mention(text, start, extended, symbols.get(cp), negative);
    }
    if (cp == ' ') {
      final String code = isoCodeAt(text, number.end + 1);
      if (code != null) {
        return mention(text, start, new Number(number.value, number.end + 4),
            code, negative);
      }
    }
    return null;
  }

  private MoneyAmount mention(CharSequence text, int start, Number number,
      String currency, boolean negative) {
    if (number == null || !boundaryAfter(text, number.end)) {
      return null;
    }
    final BigDecimal amount = negative ? number.value.negate() : number.value;
    return new MoneyAmount(new Span(start, number.end), amount, currency);
  }

  /**
   * Parses a number with optional strict digit grouping, an optional decimal part, and
   * an optional scale marker. Returns {@code null} when no valid number starts here or
   * an immediate letter suffix is not a known scale marker.
   */
  private static Number parseNumber(CharSequence text, int start) {
    int i = start;
    int digits = 0;
    final StringBuilder normalized = new StringBuilder();
    while (isAsciiDigit(charAt(text, i))) {
      normalized.append(text.charAt(i));
      i++;
      digits++;
    }
    if (digits == 0) {
      return null;
    }
    if (charAt(text, i) == ',' && digits <= 3) {
      while (charAt(text, i) == ',' && groupOfThree(text, i + 1)) {
        normalized.append(text, i + 1, i + 4);
        i += 4;
      }
    }
    if (charAt(text, i) == '.' && isAsciiDigit(charAt(text, i + 1))) {
      normalized.append('.');
      i++;
      while (isAsciiDigit(charAt(text, i))) {
        normalized.append(text.charAt(i));
        i++;
      }
    }
    final BigDecimal value = new BigDecimal(normalized.toString());
    return parseScale(text, i, value);
  }

  /**
   * Applies an immediate suffix scale ({@code k}, {@code m}, {@code b}, {@code bn}) or a
   * following scale word ({@code thousand} to {@code trillion}). An immediate letter
   * that is not a scale marker invalidates the whole match, so {@code $5x} is not money.
   */
  private static Number parseScale(CharSequence text, int end, BigDecimal value) {
    final char suffix = Character.toLowerCase(charAt(text, end));
    if (Character.isLetter(suffix)) {
      final boolean bn = suffix == 'b' && Character.toLowerCase(charAt(text, end + 1)) == 'n';
      final int suffixEnd = end + (bn ? 2 : 1);
      final long scale = switch (suffix) {
        case 'k' -> 1_000L;
        case 'm' -> 1_000_000L;
        case 'b' -> 1_000_000_000L;
        default -> 0L;
      };
      if (scale == 0L || Character.isLetterOrDigit(charAt(text, suffixEnd))) {
        return null;
      }
      return new Number(value.multiply(BigDecimal.valueOf(scale)), suffixEnd);
    }
    if (charAt(text, end) == ' ') {
      final Number worded = parseScaleWord(text, end + 1, value);
      if (worded != null) {
        return worded;
      }
    }
    return new Number(value, end);
  }

  /** Parses a scale word after the number; absence is not an error. */
  private static Number parseScaleWord(CharSequence text, int start, BigDecimal value) {
    int i = start;
    final StringBuilder word = new StringBuilder();
    while (Character.isLetter(charAt(text, i)) && word.length() <= 8) {
      word.append(Character.toLowerCase(text.charAt(i)));
      i++;
    }
    final long scale = switch (word.toString()) {
      case "thousand" -> 1_000L;
      case "million" -> 1_000_000L;
      case "billion" -> 1_000_000_000L;
      case "trillion" -> 1_000_000_000_000L;
      default -> 0L;
    };
    if (scale == 0L || Character.isLetterOrDigit(charAt(text, i))) {
      return null;
    }
    return new Number(value.multiply(BigDecimal.valueOf(scale)), i);
  }

  /** Reads a known ISO 4217 code at the position, or {@code null}. */
  private static String isoCodeAt(CharSequence text, int start) {
    if (start + 3 > text.length() || !boundaryBefore(text, start)) {
      return null;
    }
    for (int i = start; i < start + 3; i++) {
      if (!isUpperAscii(text.charAt(i))) {
        return null;
      }
    }
    if (Character.isLetterOrDigit(charAt(text, start + 3))) {
      return null;
    }
    final String code = text.subSequence(start, start + 3).toString();
    return ISO_CODES.contains(code) ? code : null;
  }

  private static boolean groupOfThree(CharSequence text, int start) {
    return isAsciiDigit(charAt(text, start)) && isAsciiDigit(charAt(text, start + 1))
        && isAsciiDigit(charAt(text, start + 2)) && !isAsciiDigit(charAt(text, start + 3));
  }

  private static boolean boundaryBefore(CharSequence text, int index) {
    return index == 0 || !Character.isLetterOrDigit(Character.codePointBefore(text, index));
  }

  private static boolean boundaryAfter(CharSequence text, int index) {
    final int cp = codePointAt(text, index);
    return cp == NO_MATCH || !Character.isLetterOrDigit(cp);
  }

  private static boolean isAsciiDigit(int cp) {
    return cp >= '0' && cp <= '9';
  }

  private static boolean isUpperAscii(int cp) {
    return cp >= 'A' && cp <= 'Z';
  }

  private static char charAt(CharSequence text, int index) {
    return index >= 0 && index < text.length() ? text.charAt(index) : ' ';
  }

  private static int codePointAt(CharSequence text, int index) {
    return index >= 0 && index < text.length() ? Character.codePointAt(text, index) : NO_MATCH;
  }

  private static Set<String> isoCodes() {
    final Set<String> codes = new HashSet<>();
    for (final Currency currency : Currency.getAvailableCurrencies()) {
      codes.add(currency.getCurrencyCode());
    }
    return Collections.unmodifiableSet(codes);
  }

  /** An intermediate parse result: the normalized value and the exclusive end offset. */
  private record Number(BigDecimal value, int end) {
  }
}
