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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import opennlp.tools.extraction.NumberScan;
import opennlp.tools.util.Span;

/**
 * A deterministic {@link MoneyExtractor}: a single forward scan over the text, no
 * regular expressions, recognizing the common written money shapes.
 *
 * <p>Recognized forms: a currency symbol before or after the number ({@code $1,234.56},
 * {@code 50\u20AC}), an ISO 4217 code before or after the number ({@code USD 100},
 * {@code 100 USD}), an optional leading minus ({@code -$5}), and scale markers, either
 * an immediate suffix ({@code $1.2M}, {@code \u00A32.5k}, {@code $3bn}) or a following word
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
 * <p>Not recognized: accounting negatives in parentheses, multi-character symbols such as
 * {@code kr}, spelled-out currency words such as {@code dollars}, and locale-dependent
 * decimal commas. The extractor holds no per-call state and is safe to share between
 * threads.</p>
 *
 * @see <a href="https://www.iso.org/iso-4217-currency-codes.html">ISO 4217</a>
 * @since 3.0.0
 */
public class CursorMoneyExtractor implements MoneyExtractor {

  private static final Map<Integer, String> DEFAULT_SYMBOLS = Map.ofEntries(
      Map.entry((int) '$', "USD"),
      Map.entry(0x20AC, "EUR"),   // euro sign
      Map.entry(0x00A3, "GBP"),   // pound sign
      Map.entry(0x00A5, "JPY"),   // yen sign
      Map.entry(0x20B9, "INR"),   // rupee sign
      Map.entry(0x20A9, "KRW"),   // won sign
      Map.entry(0x20BD, "RUB"),   // ruble sign
      Map.entry(0x20BA, "TRY"),   // lira sign
      Map.entry(0x20AA, "ILS"),   // sheqel sign
      Map.entry(0x0E3F, "THB"),   // baht sign
      Map.entry(0x20AB, "VND"));  // dong sign

  private static final Set<String> ISO_CODES = isoCodes();

  /**
   * The symbol table flattened into parallel arrays: the code point at an index denotes
   * the currency code at the same index. The scan looks a symbol up at every text
   * position, and the primitive form keeps that lookup free of boxing.
   */
  private final int[] symbolCodePoints;
  private final String[] currencyCodes;

  /**
   * Initializes the extractor with the default symbol table.
   */
  public CursorMoneyExtractor() {
    this(DEFAULT_SYMBOLS);
  }

  /**
   * Initializes the extractor with a custom symbol table.
   *
   * @param symbolCurrencies Maps a currency symbol code point to the ISO 4217 code it
   *                         denotes. Must not be {@code null} or empty, no key may be
   *                         {@code null}, and every value must be a known ISO 4217 code.
   * @throws IllegalArgumentException Thrown if the map is {@code null} or empty, maps a
   *         {@code null} code point, or names an unknown currency code.
   */
  public CursorMoneyExtractor(Map<Integer, String> symbolCurrencies) {
    if (symbolCurrencies == null || symbolCurrencies.isEmpty()) {
      throw new IllegalArgumentException("symbolCurrencies must not be null or empty");
    }
    this.symbolCodePoints = new int[symbolCurrencies.size()];
    this.currencyCodes = new String[symbolCurrencies.size()];
    int i = 0;
    for (final Map.Entry<Integer, String> symbol : symbolCurrencies.entrySet()) {
      if (symbol.getKey() == null) {
        throw new IllegalArgumentException("symbolCurrencies must not map a null code point");
      }
      final String code = symbol.getValue();
      if (code == null || !ISO_CODES.contains(code)) {
        throw new IllegalArgumentException("not an ISO 4217 currency code: " + code);
      }
      symbolCodePoints[i] = symbol.getKey();
      currencyCodes[i] = code;
      i++;
    }
  }

  /**
   * Creates an extractor whose symbol table resolves ambiguous symbols for a region:
   * in an Australian document, {@code $} denotes {@code AUD}.
   *
   * <p>The override is derived from JDK locale data: when the region's own currency is
   * written with a single currency-sign code point in that region, that symbol maps to
   * the region's currency, and all other defaults stay. Regions whose conventional
   * symbol is not a single currency sign, for example a letter, keep the default table
   * unchanged. This is the hook for document-level location evidence: once a pipeline
   * knows where a document speaks from, money in it is identified in the right
   * currency and can be converted through {@link FxRates}.</p>
   *
   * @param region A locale with a country component. Must not be {@code null} and must
   *               name a region with a currency.
   * @return A {@link CursorMoneyExtractor} for the region. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code region} is {@code null} or has no
   *         currency.
   */
  public static CursorMoneyExtractor forRegion(Locale region) {
    if (region == null) {
      throw new IllegalArgumentException("region must not be null");
    }
    final Currency currency;
    try {
      currency = Currency.getInstance(region);
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException("region has no currency: " + region, e);
    }
    final String symbol = currency.getSymbol(region);
    if (symbol.codePointCount(0, symbol.length()) != 1) {
      return new CursorMoneyExtractor();
    }
    final int cp = symbol.codePointAt(0);
    if (Character.getType(cp) != Character.CURRENCY_SYMBOL) {
      return new CursorMoneyExtractor();
    }
    final Map<Integer, String> symbols = new HashMap<>(DEFAULT_SYMBOLS);
    symbols.put(cp, currency.getCurrencyCode());
    return new CursorMoneyExtractor(symbols);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The scan resumes behind each reported mention, so mentions never overlap.</p>
   */
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
   * number first. A leading {@code -} counts as a minus sign only at a left boundary,
   * on the symbol-first path exactly as on the number-first path, so the hyphen in a
   * range such as {@code $100-$200} never negates the second amount.
   *
   * @param text The text being scanned.
   * @param start The offset the candidate mention would start at.
   * @return The mention starting at {@code start}, or {@code null} when none matches.
   */
  private MoneyAmount matchAt(CharSequence text, int start) {
    int i = start;
    boolean negative = false;
    if (NumberScan.charAt(text, i) == '-') {
      negative = true;
      i++;
    }
    final int cp = NumberScan.codePointAt(text, i);
    final String symbolCurrency = currencyFor(cp);
    if (symbolCurrency != null
        && (!negative || NumberScan.signBoundaryBefore(text, start))) {
      return symbolFirst(text, start, i, symbolCurrency, negative);
    }
    if (!negative && isUpperAscii(cp) && NumberScan.boundaryBefore(text, i)) {
      final MoneyAmount isoFirst = isoFirst(text, start, i);
      if (isoFirst != null) {
        return isoFirst;
      }
    }
    if (NumberScan.isAsciiDigit(cp)
        && (negative ? NumberScan.signBoundaryBefore(text, start)
            : NumberScan.boundaryBefore(text, i))) {
      return numberFirst(text, start, i, negative);
    }
    return null;
  }

  /**
   * Matches {@code $1,234.56}, {@code -$5}, {@code $1.2M}, and {@code $ 100}.
   *
   * @param text The text being scanned.
   * @param start The offset the mention starts at, the minus sign included.
   * @param symbolIndex The offset of the currency symbol.
   * @param currency The ISO 4217 code the symbol denotes.
   * @param negative {@code true} if a minus sign opens the mention.
   * @return The mention, or {@code null} when no number follows the symbol.
   */
  private MoneyAmount symbolFirst(CharSequence text, int start, int symbolIndex,
      String currency, boolean negative) {
    int i = symbolIndex + Character.charCount(NumberScan.codePointAt(text, symbolIndex));
    if (NumberScan.charAt(text, i) == ' ') {
      i++;
    }
    return mention(text, start, NumberScan.parse(text, i, true), currency, negative);
  }

  /**
   * Matches {@code USD 100} and {@code USD 1.2 million}.
   *
   * @param text The text being scanned.
   * @param start The offset the mention starts at.
   * @param codeIndex The offset of the ISO 4217 code.
   * @return The mention, or {@code null} when no known code and number follow.
   */
  private MoneyAmount isoFirst(CharSequence text, int start, int codeIndex) {
    final String code = isoCodeAt(text, codeIndex);
    if (code == null || NumberScan.charAt(text, codeIndex + 3) != ' ') {
      return null;
    }
    return mention(text, start, NumberScan.parse(text, codeIndex + 4, true), code, false);
  }

  /**
   * Matches {@code 100 USD}, {@code 50\u20AC} (euro sign), and {@code 3.5m USD}.
   *
   * @param text The text being scanned.
   * @param start The offset the mention starts at, the minus sign included.
   * @param digitIndex The offset of the first digit.
   * @param negative {@code true} if a minus sign opens the mention.
   * @return The mention, or {@code null} when no currency marker follows the number.
   */
  private MoneyAmount numberFirst(CharSequence text, int start, int digitIndex,
      boolean negative) {
    final NumberScan.Result number = NumberScan.parse(text, digitIndex, true);
    if (number == null) {
      return null;
    }
    final int cp = NumberScan.codePointAt(text, number.end());
    final String currency = currencyFor(cp);
    if (currency != null) {
      final NumberScan.Result extended = new NumberScan.Result(number.value(),
          number.end() + Character.charCount(cp));
      return mention(text, start, extended, currency, negative);
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

  /**
   * Builds a mention from a scanned number, rejecting it when the number is missing or
   * does not end at a boundary.
   *
   * @param text The text being scanned.
   * @param start The offset the mention starts at.
   * @param number The scanned number, or {@code null} when the scan failed.
   * @param currency The ISO 4217 code of the mention.
   * @param negative {@code true} if a minus sign opens the mention.
   * @return The mention, or {@code null} when it is not a complete match.
   */
  private MoneyAmount mention(CharSequence text, int start, NumberScan.Result number,
      String currency, boolean negative) {
    if (number == null || !NumberScan.boundaryAfter(text, number.end())) {
      return null;
    }
    final BigDecimal amount = negative ? number.value().negate() : number.value();
    return new MoneyAmount(new Span(start, number.end()), amount, currency);
  }

  /**
   * Reads a known ISO 4217 code at a position.
   *
   * @param text The text being scanned.
   * @param start The offset of the first letter of the candidate code.
   * @return The code, or {@code null} when the three letters at {@code start} are no
   *         known code or are part of a longer run.
   */
  private String isoCodeAt(CharSequence text, int start) {
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

  /**
   * Resolves the currency a symbol code point denotes.
   *
   * @param cp The code point to look up; {@link NumberScan#NO_CODE_POINT} matches no
   *           symbol.
   * @return The ISO 4217 code the symbol denotes, or {@code null} when {@code cp} is no
   *         known currency symbol.
   */
  private String currencyFor(int cp) {
    for (int i = 0; i < symbolCodePoints.length; i++) {
      if (symbolCodePoints[i] == cp) {
        return currencyCodes[i];
      }
    }
    return null;
  }

  /**
   * @param cp The code point to classify.
   * @return {@code true} if {@code cp} is an ASCII capital letter.
   */
  private boolean isUpperAscii(int cp) {
    return cp >= 'A' && cp <= 'Z';
  }

  /**
   * @return The alphabetic codes of every currency the JDK knows. Never {@code null}.
   */
  private static Set<String> isoCodes() {
    final Set<String> codes = new HashSet<>();
    for (final Currency currency : Currency.getAvailableCurrencies()) {
      codes.add(currency.getCurrencyCode());
    }
    return Collections.unmodifiableSet(codes);
  }
}
