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
import java.time.temporal.IsoFields;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import opennlp.tools.extraction.NumberScan;
import opennlp.tools.temporal.TemporalExpression.Granularity;
import opennlp.tools.temporal.TemporalExpression.Origin;
import opennlp.tools.util.Span;

/**
 * A deterministic {@link TemporalExtractor}: a single forward scan over the text, no
 * regular expressions, recognizing absolute calendar mentions.
 *
 * <p>Recognized forms: ISO dates ({@code 2026-07-14}), also as the date part of an
 * ISO 8601 timestamp ({@code 2026-07-14T09:30:00Z}), reported at day granularity over
 * the date part only; written dates in both orders,
 * with optional comma and ordinal suffix ({@code July 14, 2026}, {@code 14th July 2026},
 * {@code Jul 14 2026}); month and year ({@code July 2026}); and quarters
 * ({@code Q3 2024}). Month names are matched case-insensitively as full names or
 * three-letter abbreviations. Years are restricted to 1000 through 2999, and day-level
 * mentions are calendar-validated through {@code java.time}, so {@code February 30}
 * is never reported.</p>
 *
 * <p>With a reference date, {@link #extract(CharSequence, LocalDate)} additionally
 * resolves relative expressions: {@code today}, {@code yesterday}, {@code tomorrow};
 * {@code last}, {@code this}, or {@code next} week, month, quarter, or year; a count
 * of days, weeks, months, or years followed by {@code ago}; and {@code in} followed
 * by such a count. Each is marked with {@link TemporalExpression.Origin#RELATIVE} and
 * resolves at the granularity its unit names, so {@code last week} is an ISO week and
 * {@code 3 days ago} a calendar day.
 * Without a reference date, relative expressions are not reported at all rather than
 * being guessed against the wall clock.</p>
 *
 * <p>Not recognized: named weekdays such as {@code next Tuesday}, times of day (the
 * time part of a timestamp is skipped, never reported), bare years, day-and-month
 * without a year, and numeric formats with slashes, whose day and month order is
 * locale-dependent. The extractor holds no per-call state and is safe to
 * share between threads.</p>
 *
 * @since 3.0.0
 */
public class CursorTemporalExtractor implements TemporalExtractor {

  private static final Map<String, Integer> MONTHS = months();

  private static final int MIN_YEAR = 1000;
  private static final int MAX_YEAR = 2999;

  /** The length of an extended-format ISO date, {@code 2026-07-14}. */
  private static final int ISO_DATE_LENGTH = 10;

  /**
   * How many letters a candidate keyword or month name may have before the scan gives
   * up: one more than {@code september}, the longest word this extractor recognizes.
   */
  private static final int MAX_WORD_LENGTH = 10;

  /**
   * {@inheritDoc}
   *
   * <p>Only absolute mentions are reported; relative expressions need the reference
   * date of {@link #extract(CharSequence, LocalDate)}.</p>
   */
  @Override
  public List<TemporalExpression> extract(CharSequence text) {
    return scan(text, null);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The scan resumes behind each reported mention, so mentions never overlap.</p>
   */
  @Override
  public List<TemporalExpression> extract(CharSequence text, LocalDate reference) {
    if (reference == null) {
      throw new IllegalArgumentException("reference must not be null");
    }
    return scan(text, reference);
  }

  /**
   * The shared forward scan.
   *
   * @param text The text to scan. Must not be {@code null}.
   * @param reference The date relative expressions resolve against, or {@code null} to
   *                  not recognize relative expressions at all.
   * @return The mentions in text order, non-overlapping. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code text} is {@code null}.
   */
  private List<TemporalExpression> scan(CharSequence text, LocalDate reference) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<TemporalExpression> mentions = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      final TemporalExpression mention = matchAt(text, i, reference);
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
   * Tries the mention shapes at one position: ISO date, day-first, or a relative
   * count at a digit; quarter, month-first, or a relative keyword at a letter.
   *
   * @param text The text being scanned.
   * @param start The offset the candidate mention would start at.
   * @param reference The reference date, or {@code null} to skip the relative shapes.
   * @return The mention starting at {@code start}, or {@code null} when none matches.
   */
  private TemporalExpression matchAt(CharSequence text, int start, LocalDate reference) {
    if (!NumberScan.boundaryBefore(text, start)) {
      return null;
    }
    final char c = NumberScan.charAt(text, start);
    if (NumberScan.isAsciiDigit(c)) {
      final TemporalExpression iso = isoDate(text, start);
      if (iso != null) {
        return iso;
      }
      final TemporalExpression dayFirst = dayFirst(text, start);
      if (dayFirst != null) {
        return dayFirst;
      }
      return reference == null ? null : countAgo(text, start, reference);
    }
    if ((c == 'Q' || c == 'q') && NumberScan.isAsciiDigit(NumberScan.charAt(text, start + 1))) {
      return quarter(text, start);
    }
    if (Character.isLetter(c)) {
      final TemporalExpression monthFirst = monthFirst(text, start);
      if (monthFirst != null) {
        return monthFirst;
      }
      return reference == null ? null : relativeKeyword(text, start, reference);
    }
    return null;
  }

  /**
   * Matches the keyword-led relative forms: the day words {@code today},
   * {@code yesterday}, and {@code tomorrow}; {@code last}, {@code this}, or
   * {@code next} followed by a unit; and {@code in} followed by a count and a unit.
   *
   * @param text The text being scanned.
   * @param start The offset of the first letter of the candidate keyword.
   * @param reference The date the expression resolves against. Must not be {@code null}.
   * @return The resolved mention, or {@code null} when no relative form starts here.
   */
  private TemporalExpression relativeKeyword(CharSequence text, int start,
      LocalDate reference) {
    final Word keyword = word(text, start);
    if (keyword == null) {
      return null;
    }
    switch (keyword.lower()) {
      case "today":
        return resolved(start, keyword.end(), reference, Granularity.DAY);
      case "yesterday":
        return resolved(start, keyword.end(), reference.minusDays(1), Granularity.DAY);
      case "tomorrow":
        return resolved(start, keyword.end(), reference.plusDays(1), Granularity.DAY);
      case "last":
      case "this":
      case "next": {
        if (NumberScan.charAt(text, keyword.end()) != ' ') {
          return null;
        }
        final Word unit = word(text, keyword.end() + 1);
        if (unit == null) {
          return null;
        }
        final int steps = switch (keyword.lower()) {
          case "last" -> -1;
          case "next" -> 1;
          default -> 0;
        };
        return shifted(start, unit.end(), unit.lower(), steps, reference);
      }
      case "in": {
        if (NumberScan.charAt(text, keyword.end()) != ' ') {
          return null;
        }
        final NumberInText count = shortNumber(text, keyword.end() + 1);
        if (count == null || NumberScan.charAt(text, count.end()) != ' ') {
          return null;
        }
        final Word unit = word(text, count.end() + 1);
        if (unit == null) {
          return null;
        }
        return shifted(start, unit.end(), singular(unit.lower()), count.value(),
            reference);
      }
      default:
        return null;
    }
  }

  /**
   * Matches {@code 3 days ago} and its week, month, and year siblings.
   *
   * @param text The text being scanned.
   * @param start The offset of the first digit of the count.
   * @param reference The date the expression resolves against. Must not be {@code null}.
   * @return The resolved mention, or {@code null} when no counted form starts here.
   */
  private TemporalExpression countAgo(CharSequence text, int start, LocalDate reference) {
    final NumberInText count = shortNumber(text, start);
    if (count == null || NumberScan.charAt(text, count.end()) != ' ') {
      return null;
    }
    final Word unit = word(text, count.end() + 1);
    if (unit == null || NumberScan.charAt(text, unit.end()) != ' ') {
      return null;
    }
    final Word ago = word(text, unit.end() + 1);
    if (ago == null || !"ago".equals(ago.lower())) {
      return null;
    }
    return shifted(start, ago.end(), singular(unit.lower()), -count.value(),
        reference);
  }

  /**
   * Resolves a unit word shifted by a number of steps against the reference, at the
   * granularity the unit names.
   *
   * @param start The offset the mention starts at.
   * @param end The exclusive offset the mention ends at.
   * @param unit The singular unit word, for example {@code week}.
   * @param steps How many units to shift, negative into the past.
   * @param reference The date the expression resolves against. Must not be {@code null}.
   * @return The resolved mention, or {@code null} when {@code unit} names no known unit.
   */
  private TemporalExpression shifted(int start, int end, String unit,
      int steps, LocalDate reference) {
    return switch (unit) {
      case "day" -> resolved(start, end, reference.plusDays(steps), Granularity.DAY);
      case "week" -> resolved(start, end, reference.plusWeeks(steps), Granularity.WEEK);
      case "month" -> resolved(start, end, reference.plusMonths(steps), Granularity.MONTH);
      case "quarter" -> resolved(start, end, reference.plusMonths(3L * steps),
          Granularity.QUARTER);
      case "year" -> resolved(start, end, reference.plusYears(steps), Granularity.YEAR);
      default -> null;
    };
  }

  /**
   * Reduces a plural unit word to its singular; other words pass through.
   *
   * @param unit The unit word, for example {@code weeks}.
   * @return The singular form, for example {@code week}. Never {@code null}.
   */
  private String singular(String unit) {
    return unit.endsWith("s") && unit.length() > 1
        ? unit.substring(0, unit.length() - 1) : unit;
  }

  /**
   * Builds a resolved relative mention at a granularity's ISO value.
   *
   * @param start The offset the mention starts at.
   * @param end The exclusive offset the mention ends at.
   * @param date The resolved date. Must not be {@code null}.
   * @param granularity The granularity to report the date at. Must not be {@code null}.
   * @return The mention. Never {@code null}.
   */
  private TemporalExpression resolved(int start, int end, LocalDate date,
      Granularity granularity) {
    final String value = switch (granularity) {
      case DAY -> String.format(Locale.ROOT, "%04d-%02d-%02d",
          date.getYear(), date.getMonthValue(), date.getDayOfMonth());
      case WEEK -> String.format(Locale.ROOT, "%04d-W%02d",
          date.get(IsoFields.WEEK_BASED_YEAR), date.get(IsoFields.WEEK_OF_WEEK_BASED_YEAR));
      case MONTH -> String.format(Locale.ROOT, "%04d-%02d",
          date.getYear(), date.getMonthValue());
      case QUARTER -> String.format(Locale.ROOT, "%04d-Q%d",
          date.getYear(), date.get(IsoFields.QUARTER_OF_YEAR));
      case YEAR -> String.format(Locale.ROOT, "%04d", date.getYear());
    };
    return new TemporalExpression(new Span(start, end), value, granularity, Origin.RELATIVE);
  }

  /**
   * Matches {@code 2026-07-14}, alone or as the date part of an ISO 8601 timestamp
   * such as {@code 2026-07-14T09:30:00Z}. In the timestamp case the mention covers
   * only the date and stays at day granularity; the time part is skipped, never
   * reported, since sub-day granularities are out of scope.
   *
   * @param text The text being scanned.
   * @param start The offset of the first digit of the year.
   * @return The mention, or {@code null} when no valid ISO date starts here.
   */
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
    if (day < 0) {
      return null;
    }
    final int end = start + ISO_DATE_LENGTH;
    if (!NumberScan.boundaryAfter(text, end) && !timeOfDayAt(text, end)) {
      return null;
    }
    return day(start, end, year, month, day);
  }

  /**
   * Checks whether an ISO 8601 time of day starts at a position: a {@code T} followed
   * by at least hours and minutes ({@code T09:30}), optional further colon-separated
   * two-digit groups such as seconds, and an optional zone suffix, either {@code Z}
   * or a {@code +05:30} style offset, ending at a boundary. A {@code T} followed by
   * prose letters, as in {@code 2026-07-14Tomorrow}, never qualifies.
   *
   * @param text The text being scanned.
   * @param start The offset of the candidate {@code T}.
   * @return {@code true} if a time of day starts at {@code start}.
   */
  private boolean timeOfDayAt(CharSequence text, int start) {
    if (NumberScan.charAt(text, start) != 'T'
        || digits(text, start + 1, 2) < 0 || NumberScan.charAt(text, start + 3) != ':') {
      return false;
    }
    int i = start + 4;
    if (digits(text, i, 2) < 0) {
      return false;
    }
    i += 2;
    while (NumberScan.charAt(text, i) == ':' && digits(text, i + 1, 2) >= 0) {
      i += 3;
    }
    final char zone = NumberScan.charAt(text, i);
    if (zone == 'Z') {
      i++;
    } else if ((zone == '+' || zone == '-') && digits(text, i + 1, 2) >= 0
        && NumberScan.charAt(text, i + 3) == ':' && digits(text, i + 4, 2) >= 0) {
      i += 6;
    }
    return NumberScan.boundaryAfter(text, i);
  }

  /**
   * Matches {@code 14 July 2026} and {@code 14th July 2026}.
   *
   * @param text The text being scanned.
   * @param start The offset of the first digit of the day.
   * @return The mention, or {@code null} when no valid day-first date starts here.
   */
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
    return day(start, year.end(), year.value(), MONTHS.get(month.lower()), day.value());
  }

  /**
   * Matches {@code July 14, 2026}, {@code Jul 14 2026}, and {@code July 2026}.
   *
   * @param text The text being scanned.
   * @param start The offset of the first letter of the month name.
   * @return The mention, or {@code null} when no valid month-first date starts here.
   */
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
          String.format(Locale.ROOT, "%04d-%02d", year.value(), monthOfYear),
          Granularity.MONTH);
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
    return day(start, dayYear.end(), dayYear.value(), monthOfYear, day.value());
  }

  /**
   * Matches {@code Q3 2024}.
   *
   * @param text The text being scanned.
   * @param start The offset of the {@code Q}.
   * @return The mention, or {@code null} when no valid quarter starts here.
   */
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
        String.format(Locale.ROOT, "%04d-Q%d", year.value(), number), Granularity.QUARTER);
  }

  /**
   * Builds a calendar-validated day mention.
   *
   * @param start The offset the mention starts at.
   * @param end The exclusive offset the mention ends at.
   * @param year The year of the mention.
   * @param month The month of the year, one based.
   * @param dayOfMonth The day of the month, one based.
   * @return The mention, or {@code null} when the year is out of range or the date does
   *         not exist in the calendar.
   */
  private TemporalExpression day(int start, int end,
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
        String.format(Locale.ROOT, "%04d-%02d-%02d", year, month, dayOfMonth),
        Granularity.DAY);
  }

  /**
   * Reads a four-digit year that is in range and not part of a longer digit run.
   *
   * @param text The text being scanned.
   * @param start The offset of the first digit of the candidate year.
   * @return The year, or {@code null} when no such year starts at {@code start}.
   */
  private NumberInText yearAt(CharSequence text, int start) {
    final int year = digits(text, start, 4);
    if (year < MIN_YEAR || year > MAX_YEAR || !NumberScan.boundaryAfter(text, start + 4)) {
      return null;
    }
    return new NumberInText(year, start + 4);
  }

  /**
   * Reads a one or two digit number that is not part of a longer digit run.
   *
   * @param text The text being scanned.
   * @param start The offset of the first digit of the candidate number.
   * @return The number, or {@code null} when no such number starts at {@code start}.
   */
  private NumberInText shortNumber(CharSequence text, int start) {
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

  /**
   * Skips an ordinal suffix ({@code st}, {@code nd}, {@code rd}, {@code th}).
   *
   * @param text The text being scanned.
   * @param index The offset just behind the digits of the day.
   * @return The offset behind the suffix, or {@code index} when none follows.
   */
  private int skipOrdinal(CharSequence text, int index) {
    final char first = Character.toLowerCase(NumberScan.charAt(text, index));
    final char second = Character.toLowerCase(NumberScan.charAt(text, index + 1));
    final boolean ordinal = (first == 's' && second == 't') || (first == 'n' && second == 'd')
        || (first == 'r' && second == 'd') || (first == 't' && second == 'h');
    return ordinal && !Character.isLetterOrDigit(NumberScan.charAt(text, index + 2))
        ? index + 2 : index;
  }

  /**
   * Reads exactly {@code width} digits as an int.
   *
   * @param text The text being scanned.
   * @param start The offset of the first digit.
   * @param width How many digits to read.
   * @return The value, or a negative number when the digit run is shorter than
   *         {@code width} or continues past it.
   */
  private int digits(CharSequence text, int start, int width) {
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

  /**
   * Reads a lowercased letter run of at most {@link #MAX_WORD_LENGTH} characters.
   *
   * @param text The text being scanned.
   * @param start The offset of the first letter.
   * @return The word, or {@code null} when no letter starts at {@code start} or the run
   *         continues past {@link #MAX_WORD_LENGTH}.
   */
  private Word word(CharSequence text, int start) {
    int i = start;
    final StringBuilder run = new StringBuilder();
    while (Character.isLetter(NumberScan.charAt(text, i)) && run.length() < MAX_WORD_LENGTH) {
      run.append(Character.toLowerCase(text.charAt(i)));
      i++;
    }
    if (run.isEmpty() || Character.isLetter(NumberScan.charAt(text, i))) {
      return null;
    }
    return new Word(run.toString(), i);
  }

  /**
   * @return The month numbers by lowercased full name and three-letter abbreviation.
   *         Never {@code null}.
   */
  private static Map<String, Integer> months() {
    final Map<String, Integer> months = new HashMap<>();
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
