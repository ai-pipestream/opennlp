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

package opennlp.tools.pii;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * A deterministic {@link PiiExtractor} for United States taxpayer identifiers: forward scans
 * over the text, no regular expressions, recognizing Social Security numbers and Individual
 * Taxpayer Identification Numbers. This extractor is opt-in and is never part of the default
 * extractor, because a national identifier is a jurisdiction-specific concern and its number
 * space overlaps ordinary numbers.
 *
 * <p>Neither number carries a checksum, so the written form is the evidence: the grouped
 * form {@code xxx-xx-xxxx} is required, with hyphens or with spaces but not with a mixture,
 * and a bare nine-digit run is never reported. The number spaces are then checked against
 * the rules the issuing agencies publish, which is what tells the two types apart and
 * rejects the numbers that are never issued:</p>
 * <ul>
 *   <li>Social Security number: the area may not be {@code 000} or {@code 666} and may not
 *   be {@code 900} or above, the group may not be {@code 00}, and the serial may not be
 *   {@code 0000}, per the
 *   <a href="https://www.ssa.gov/employer/randomization.html">SSA randomization
 *   notice</a>.</li>
 *   <li>Individual Taxpayer Identification Number: the area is {@code 900} or above and the
 *   group is one of the ranges the
 *   <a href="https://www.irs.gov/individuals/individual-taxpayer-identification-number">
 *   IRS</a> assigns, {@code 50} to {@code 65}, {@code 70} to {@code 88}, {@code 90} to
 *   {@code 92}, and {@code 94} to {@code 99}.</li>
 * </ul>
 *
 * <p>Normalized form: the nine digits in the hyphenated form {@code xxx-xx-xxxx}, whichever
 * separator the text used.</p>
 *
 * <p>Both types are reported by default; the {@link #UsIdentityPiiExtractor(Set)} constructor
 * limits extraction to a subset.</p>
 *
 * <p>The extractor holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class UsIdentityPiiExtractor implements PiiExtractor {

  private static final Set<String> ALL_TYPES =
      Set.of(PiiMention.TYPE_US_SSN, PiiMention.TYPE_US_ITIN);

  private static final int AREA_DIGITS = 3;
  private static final int GROUP_DIGITS = 2;
  private static final int SERIAL_DIGITS = 4;

  /** The nine digits and the two separators of the grouped form. */
  private static final int FORM_LENGTH =
      AREA_DIGITS + GROUP_DIGITS + SERIAL_DIGITS + 2;

  private static final int SSN_AREA_UNUSED = 666;
  private static final int ITIN_AREA_FIRST = 900;

  /** The group ranges the IRS assigns to Individual Taxpayer Identification Numbers. */
  private static final int[][] ITIN_GROUPS = {{50, 65}, {70, 88}, {90, 92}, {94, 99}};

  private final Set<String> types;

  /**
   * Initializes an extractor that reports both types.
   */
  public UsIdentityPiiExtractor() {
    this.types = ALL_TYPES;
  }

  /**
   * Initializes an extractor limited to a subset of the types.
   *
   * @param types The types to report, drawn from {@link PiiMention#TYPE_US_SSN} and
   *              {@link PiiMention#TYPE_US_ITIN}. Must not be {@code null} or empty and must
   *              not contain a type this extractor does not recognize.
   * @throws IllegalArgumentException Thrown if {@code types} is {@code null} or empty, or
   *         contains an unrecognized type.
   */
  public UsIdentityPiiExtractor(Set<String> types) {
    if (types == null || types.isEmpty()) {
      throw new IllegalArgumentException("types must not be null or empty");
    }
    for (final String type : types) {
      if (!ALL_TYPES.contains(type)) {
        throw new IllegalArgumentException("types contains an unrecognized type: " + type);
      }
    }
    this.types = Set.copyOf(types);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The area range decides which type a candidate can be, so no candidate is ever
   * reported as both.</p>
   */
  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hits.Hit> hits = new ArrayList<>();
    for (int i = 0; i + FORM_LENGTH <= text.length(); i++) {
      if (!Ascii.isDigit(text.charAt(i)) || !Boundaries.onNumberStart(text, i)) {
        continue;
      }
      final char separator = text.charAt(i + AREA_DIGITS);
      if ((separator != '-' && separator != ' ')
          || text.charAt(i + AREA_DIGITS + GROUP_DIGITS + 1) != separator) {
        continue;
      }
      final int end = i + FORM_LENGTH;
      if (!digitsAt(text, i, AREA_DIGITS)
          || !digitsAt(text, i + AREA_DIGITS + 1, GROUP_DIGITS)
          || !digitsAt(text, i + AREA_DIGITS + GROUP_DIGITS + 2, SERIAL_DIGITS)
          || !onNumberEnd(text, end)) {
        continue;
      }
      final int area = value(text, i, AREA_DIGITS);
      final int group = value(text, i + AREA_DIGITS + 1, GROUP_DIGITS);
      final int serial = value(text, i + AREA_DIGITS + GROUP_DIGITS + 2, SERIAL_DIGITS);
      final String type = classify(area, group, serial);
      if (type == null || !types.contains(type)) {
        continue;
      }
      final StringBuilder normalized = new StringBuilder(FORM_LENGTH);
      for (int p = i; p < end; p++) {
        normalized.append(text.charAt(p) == separator ? '-' : text.charAt(p));
      }
      Hits.add(hits, i, end, type, normalized.toString());
      // The loop increment resumes the scan at the exclusive match end.
      i = end - 1;
    }
    return Hits.resolve(hits);
  }

  /**
   * Decides which identifier a candidate's number space allows.
   *
   * @param area The first three digits.
   * @param group The middle two digits.
   * @param serial The last four digits.
   * @return {@link PiiMention#TYPE_US_SSN}, {@link PiiMention#TYPE_US_ITIN}, or {@code null}
   *         when the number is never issued.
   */
  private String classify(int area, int group, int serial) {
    if (group == 0 || serial == 0) {
      return null;
    }
    if (area >= ITIN_AREA_FIRST) {
      for (final int[] range : ITIN_GROUPS) {
        if (group >= range[0] && group <= range[1]) {
          return PiiMention.TYPE_US_ITIN;
        }
      }
      return null;
    }
    return area == 0 || area == SSN_AREA_UNUSED ? null : PiiMention.TYPE_US_SSN;
  }

  /**
   * Checks that a range holds ASCII digits only.
   *
   * @param text The text being scanned.
   * @param start The first character of the range.
   * @param length The number of characters.
   * @return {@code true} if the range holds digits only.
   */
  private boolean digitsAt(CharSequence text, int start, int length) {
    for (int i = start; i < start + length; i++) {
      if (!Ascii.isDigit(text.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Reads a run of digits as a number.
   *
   * @param text The text being scanned.
   * @param start The first digit.
   * @param length The number of digits.
   * @return The value.
   */
  private int value(CharSequence text, int start, int length) {
    int value = 0;
    for (int i = start; i < start + length; i++) {
      value = value * 10 + (text.charAt(i) - '0');
    }
    return value;
  }

  /**
   * Checks that a numeric candidate ends at {@code end} and does not continue into another
   * group of digits.
   *
   * @param text The text being scanned.
   * @param end The candidate end, exclusive.
   * @return {@code true} if the candidate may end here.
   */
  private boolean onNumberEnd(CharSequence text, int end) {
    if (!Boundaries.onEnd(text, end)) {
      return false;
    }
    if (end + 1 >= text.length()) {
      return true;
    }
    final char next = text.charAt(end);
    return (next != '.' && next != ',' && next != '-') || !Ascii.isDigit(text.charAt(end + 1));
  }
}
