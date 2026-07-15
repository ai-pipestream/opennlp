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

package opennlp.tools.temporal;

import java.time.DateTimeException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import opennlp.tools.extraction.NumberScan;
import opennlp.tools.temporal.TemporalExpression.Granularity;
import opennlp.tools.util.Span;

/**
 * A deterministic {@link TemporalExtractor}: a single forward scan over the text, no
 * regular expressions, recognizing absolute calendar mentions.
 *
 * <p>Recognized forms: ISO dates ({@code 2026-07-14}); written dates in both orders,
 * with optional comma and ordinal suffix ({@code July 14, 2026}, {@code 14th July 2026},
 * {@code Jul 14 2026}); month and year ({@code July 2026}); and quarters
 * ({@code Q3 2024}). Month names are matched case-insensitively as full names or
 * three-letter abbreviations. Years are restricted to 1000 through 2999, and day-level
 * mentions are calendar-validated through {@code java.time}, so {@code February 30}
 * is never reported.</p>
 *
 * <p>Out of scope in this version, by design: relative expressions such as
 * {@code next Tuesday} (they need a reference date, which is an interface question to
 * settle first), times of day, bare years, day-and-month without a year, and numeric
 * formats with slashes, whose day and month order is locale-dependent. The extractor
 * holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class CursorTemporalExtractor implements TemporalExtractor {

  private static final Map<String, Integer> MONTHS = months();

  private static final int MIN_YEAR = 1000;
  private static final int MAX_YEAR = 2999;

  @Override
  public List<TemporalExpression> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<TemporalExpression> mentions = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      final TemporalExpression mention = matchAt(text, i);
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
   * Tries the mention shapes at one position: ISO date or day-first at a digit, quarter
   * or month-first at a letter. Returns {@code null} when none matches.
   */
  private TemporalExpression matchAt(CharSequence text, int start) {
    if (!NumberScan.boundaryBefore(text, start)) {
      return null;
    }
    final char c = NumberScan.charAt(text, start);
    if (NumberScan.isAsciiDigit(c)) {
      final TemporalExpression iso = isoDate(text, start);
      return iso != null ? iso : dayFirst(text, start);
    }
    if ((c == 'Q' || c == 'q') && NumberScan.isAsciiDigit(NumberScan.charAt(text, start + 1))) {
      return quarter(text, start);
    }
    if (Character.isLetter(c)) {
      return monthFirst(text, start);
    }
    return null;
  }

  /** Matches {@code 2026-07-14}. */
  private TemporalExpression isoDate(CharSequence text, int start) {
    final int year = digits(text, start, 4);
    if (year < 0 || NumberScan.charAt(text, start + 4) != '-') {
      return null;
    }
    final int month = digits(text, start + 5, 2);
    if (month < 0 || NumberScan.charAt(text, start + 7) != '-') {
      return null;
    }
    final int day = digits(text, start + 8, 2);
    if (day < 0 || !NumberScan.boundaryAfter(text, start + 10)) {
      return null;
    }
    return day(text, start, start + 10, year, month, day);
  }

  /** Matches {@code 14 July 2026} and {@code 14th July 2026}. */
  private TemporalExpression dayFirst(CharSequence text, int start) {
    final NumberInText day = shortNumber(text, start);
    if (day == null) {
      return null;
    }
    int i = skipOrdinal(text, day.end());
    if (NumberScan.charAt(text, i) != ' ') {
      return null;
    }
    final Word month = word(text, i + 1);
    if (month == null || !MONTHS.containsKey(month.lower())
        || NumberScan.charAt(text, month.end()) != ' ') {
      return null;
    }
    final NumberInText year = yearAt(text, month.end() + 1);
    if (year == null) {
      return null;
    }
    return day(text, start, year.end(), year.value(), MONTHS.get(month.lower()), day.value());
  }

  /** Matches {@code July 14, 2026}, {@code Jul 14 2026}, and {@code July 2026}. */
  private TemporalExpression monthFirst(CharSequence text, int start) {
    final Word month = word(text, start);
    if (month == null || !MONTHS.containsKey(month.lower())
        || NumberScan.charAt(text, month.end()) != ' ') {
      return null;
    }
    final int monthOfYear = MONTHS.get(month.lower());
    final NumberInText year = yearAt(text, month.end() + 1);
    if (year != null) {
      return new TemporalExpression(new Span(start, year.end()),
          String.format("%04d-%02d", year.value(), monthOfYear), Granularity.MONTH);
    }
    final NumberInText day = shortNumber(text, month.end() + 1);
    if (day == null) {
      return null;
    }
    int i = skipOrdinal(text, day.end());
    if (NumberScan.charAt(text, i) == ',') {
      i++;
    }
    if (NumberScan.charAt(text, i) != ' ') {
      return null;
    }
    final NumberInText dayYear = yearAt(text, i + 1);
    if (dayYear == null) {
      return null;
    }
    return day(text, start, dayYear.end(), dayYear.value(), monthOfYear, day.value());
  }

  /** Matches {@code Q3 2024}. */
  private TemporalExpression quarter(CharSequence text, int start) {
    final int number = NumberScan.charAt(text, start + 1) - '0';
    if (number < 1 || number > 4 || NumberScan.charAt(text, start + 2) != ' ') {
      return null;
    }
    final NumberInText year = yearAt(text, start + 3);
    if (year == null) {
      return null;
    }
    return new TemporalExpression(new Span(start, year.end()),
        String.format("%04d-Q%d", year.value(), number), Granularity.QUARTER);
  }

  /** Builds a calendar-validated day mention, or {@code null} for an impossible date. */
  private static TemporalExpression day(CharSequence text, int start, int end,
      int year, int month, int dayOfMonth) {
    if (year < MIN_YEAR || year > MAX_YEAR) {
      return null;
    }
    try {
      LocalDate.of(year, month, dayOfMonth);
    } catch (DateTimeException e) {
      return null;
    }
    return new TemporalExpression(new Span(start, end),
        String.format("%04d-%02d-%02d", year, month, dayOfMonth), Granularity.DAY);
  }

  /** Reads a bounded four-digit year in range, or {@code null}. */
  private static NumberInText yearAt(CharSequence text, int start) {
    final int year = digits(text, start, 4);
    if (year < MIN_YEAR || year > MAX_YEAR || !NumberScan.boundaryAfter(text, start + 4)) {
      return null;
    }
    return new NumberInText(year, start + 4);
  }

  /** Reads a one or two digit number that is not part of a longer digit run. */
  private static NumberInText shortNumber(CharSequence text, int start) {
    int i = start;
    int value = 0;
    while (NumberScan.isAsciiDigit(NumberScan.charAt(text, i)) && i - start < 3) {
      value = value * 10 + (NumberScan.charAt(text, i) - '0');
      i++;
    }
    final int length = i - start;
    if (length < 1 || length > 2 || NumberScan.isAsciiDigit(NumberScan.charAt(text, i))) {
      return null;
    }
    return new NumberInText(value, i);
  }

  /** Skips an ordinal suffix ({@code st}, {@code nd}, {@code rd}, {@code th}). */
  private static int skipOrdinal(CharSequence text, int index) {
    final char first = Character.toLowerCase(NumberScan.charAt(text, index));
    final char second = Character.toLowerCase(NumberScan.charAt(text, index + 1));
    final boolean ordinal = (first == 's' && second == 't') || (first == 'n' && second == 'd')
        || (first == 'r' && second == 'd') || (first == 't' && second == 'h');
    return ordinal && !Character.isLetterOrDigit(NumberScan.charAt(text, index + 2))
        ? index + 2 : index;
  }

  /** Reads a fixed-width digit run as an int, or a negative value when absent. */
  private static int digits(CharSequence text, int start, int width) {
    int value = 0;
    for (int i = start; i < start + width; i++) {
      final char c = NumberScan.charAt(text, i);
      if (!NumberScan.isAsciiDigit(c)) {
        return -1;
      }
      value = value * 10 + (c - '0');
    }
    return NumberScan.isAsciiDigit(NumberScan.charAt(text, start + width)) ? -1 : value;
  }

  /** Reads a letter run of up to nine characters, or {@code null}. */
  private static Word word(CharSequence text, int start) {
    int i = start;
    final StringBuilder run = new StringBuilder();
    while (Character.isLetter(NumberScan.charAt(text, i)) && run.length() < 10) {
      run.append(Character.toLowerCase(text.charAt(i)));
      i++;
    }
    if (run.isEmpty() || Character.isLetter(NumberScan.charAt(text, i))) {
      return null;
    }
    return new Word(run.toString(), i);
  }

  private static Map<String, Integer> months() {
    final Map<String, Integer> months = new java.util.HashMap<>();
    final String[] names = {"january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"};
    for (int i = 0; i < names.length; i++) {
      months.put(names[i], i + 1);
      months.put(names[i].substring(0, 3), i + 1);
    }
    return Collections.unmodifiableMap(months);
  }

  /** An intermediate parse result: a numeric value and the exclusive end offset. */
  private record NumberInText(int value, int end) {
  }

  /** An intermediate parse result: a lowercased word and the exclusive end offset. */
  private record Word(String lower, int end) {
  }
}
