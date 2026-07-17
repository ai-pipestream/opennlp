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

package opennlp.tools.quantity;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import opennlp.tools.extraction.NumberScan;
import opennlp.tools.util.Span;

/**
 * A deterministic {@link QuantityExtractor}: a single forward scan over the text, no
 * regular expressions, recognizing numbers with a percent marker or a unit token.
 *
 * <p>Recognized forms: percentages as {@code 50%}, {@code 3.5 %}, or {@code 50 percent}
 * (all reported with the unit {@code %}); unit quantities with the unit immediately
 * attached ({@code 2.5km}) or separated by one space ({@code 80 kg}); and an optional
 * leading minus. Digit grouping follows the shared strict rule. A bare number without a
 * percent marker or unit is never a quantity, which also keeps money mentions such as
 * {@code $3 billion} out of this layer.</p>
 *
 * <p>Units are matched exactly, case-sensitively, against a curated default set of
 * common measurement tokens; ambiguous English words such as {@code in} are deliberately
 * excluded. Callers extend or replace the set through
 * {@link #CursorQuantityExtractor(Set)}. No unit conversion is performed.</p>
 *
 * <p>The extractor holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class CursorQuantityExtractor implements QuantityExtractor {

  private static final Set<String> DEFAULT_UNITS = Set.of(
      "km", "m", "cm", "mm", "nm", "mi", "ft", "yd",
      "kg", "g", "mg", "t", "lb", "lbs", "oz",
      "L", "mL", "ml",
      "ms", "ns", "min", "hr",
      "KB", "MB", "GB", "TB", "PB",
      "Hz", "kHz", "MHz", "GHz",
      "kW", "MW", "GW", "kWh",
      "mph", "kph");

  private static final String PERCENT = "%";

  private static final int MAX_UNIT_LENGTH = 6;

  private final Set<String> units;

  /**
   * Initializes the extractor with the default unit set.
   */
  public CursorQuantityExtractor() {
    this.units = DEFAULT_UNITS;
  }

  /**
   * Initializes the extractor with a custom unit set.
   *
   * @param units The unit tokens to recognize, matched exactly and case-sensitively.
   *              Must not be {@code null} or empty, and no token may be {@code null},
   *              blank, or longer than six characters.
   * @throws IllegalArgumentException Thrown if the set is {@code null}, empty, or
   *         contains an invalid token.
   */
  public CursorQuantityExtractor(Set<String> units) {
    if (units == null || units.isEmpty()) {
      throw new IllegalArgumentException("units must not be null or empty");
    }
    for (final String unit : units) {
      if (unit == null || unit.isBlank() || unit.length() > MAX_UNIT_LENGTH) {
        throw new IllegalArgumentException("not a valid unit token: " + unit);
      }
    }
    this.units = Set.copyOf(units);
  }

  @Override
  public List<Quantity> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Quantity> mentions = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      final Quantity mention = matchAt(text, i);
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
   * Matches a number followed by a percent marker or unit token at one position, or
   * returns {@code null}.
   */
  private Quantity matchAt(CharSequence text, int start) {
    int i = start;
    boolean negative = false;
    if (NumberScan.charAt(text, i) == '-') {
      negative = true;
      i++;
    }
    if (!NumberScan.isAsciiDigit(NumberScan.charAt(text, i))
        || !(negative ? NumberScan.signBoundaryBefore(text, start)
            : NumberScan.boundaryBefore(text, start))) {
      return null;
    }
    final NumberScan.Result number = NumberScan.parse(text, i, false);
    if (number == null) {
      return null;
    }
    final BigDecimal value = negative ? number.value().negate() : number.value();
    final Unit unit = parseUnit(text, number.end());
    if (unit == null) {
      return null;
    }
    return new Quantity(new Span(start, unit.end()), value, unit.token());
  }

  /**
   * Parses the percent marker or unit token after a number: immediately attached, or
   * separated by exactly one space.
   */
  private Unit parseUnit(CharSequence text, int numberEnd) {
    final Unit immediate = unitAt(text, numberEnd);
    if (immediate != null) {
      return immediate;
    }
    if (NumberScan.charAt(text, numberEnd) == ' ') {
      return unitAt(text, numberEnd + 1);
    }
    return null;
  }

  /**
   * Reads a percent sign, the word {@code percent}, or a known unit token at the
   * position.
   */
  private Unit unitAt(CharSequence text, int start) {
    if (NumberScan.charAt(text, start) == '%') {
      return NumberScan.boundaryAfter(text, start + 1) ? new Unit(PERCENT, start + 1) : null;
    }
    int i = start;
    final StringBuilder token = new StringBuilder();
    while (Character.isLetter(NumberScan.charAt(text, i)) && token.length() < 8) {
      token.append(text.charAt(i));
      i++;
    }
    if (token.isEmpty() || Character.isLetterOrDigit(NumberScan.charAt(text, i))) {
      return null;
    }
    final String word = token.toString();
    if ("percent".equalsIgnoreCase(word)) {
      return new Unit(PERCENT, i);
    }
    return units.contains(word) ? new Unit(word, i) : null;
  }

  /** An intermediate parse result: the unit token and the exclusive end offset. */
  private record Unit(String token, int end) {
  }
}
