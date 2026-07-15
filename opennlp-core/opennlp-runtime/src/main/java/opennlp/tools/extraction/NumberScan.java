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

package opennlp.tools.extraction;

import java.math.BigDecimal;

import opennlp.tools.commons.Internal;

/**
 * The shared cursor scan for written numbers, used by the typed extractors (money,
 * quantity, temporal): digits with strict grouping, an optional decimal part, and
 * optionally scale markers.
 *
 * <p>Grouping is strict: once a comma appears, the leading group must have at most three
 * digits and every further group exactly three; the scan ends at the last valid
 * position. With scaling enabled, an immediate suffix ({@code k}, {@code m}, {@code b},
 * {@code bn}) or a following word ({@code thousand} to {@code trillion}) multiplies the
 * value, and an immediate letter that is no scale marker invalidates the scan
 * entirely.</p>
 */
@Internal
public final class NumberScan {

  /** The sentinel returned by {@link #codePointAt(CharSequence, int)} out of bounds. */
  public static final int NO_CODE_POINT = -1;

  private NumberScan() {
    // static scanning methods only
  }

  /**
   * The result of one scan: the normalized value and the exclusive end offset.
   *
   * @param value The normalized numeric value. Never {@code null}.
   * @param end The exclusive offset of the first character after the number.
   */
  public record Result(BigDecimal value, int end) {
  }

  /**
   * Scans a number starting at a position.
   *
   * @param text The text to scan. Must not be {@code null}.
   * @param start The offset of the first digit.
   * @param applyScale {@code true} to consume and apply scale markers, in which case an
   *                   immediate letter that is no scale marker fails the scan;
   *                   {@code false} to stop after the decimal part.
   * @return The scanned {@link Result}, or {@code null} when no number starts at
   *         {@code start} or an immediate letter suffix is not a scale marker.
   */
  public static Result parse(CharSequence text, int start, boolean applyScale) {
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
    return applyScale ? parseScale(text, i, value) : new Result(value, i);
  }

  /**
   * Applies an immediate suffix scale or a following scale word.
   */
  private static Result parseScale(CharSequence text, int end, BigDecimal value) {
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
      return new Result(value.multiply(BigDecimal.valueOf(scale)), suffixEnd);
    }
    if (charAt(text, end) == ' ') {
      final Result worded = parseScaleWord(text, end + 1, value);
      if (worded != null) {
        return worded;
      }
    }
    return new Result(value, end);
  }

  /**
   * Parses a scale word after the number; absence is not an error.
   */
  private static Result parseScaleWord(CharSequence text, int start, BigDecimal value) {
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
    return new Result(value.multiply(BigDecimal.valueOf(scale)), i);
  }

  private static boolean groupOfThree(CharSequence text, int start) {
    return isAsciiDigit(charAt(text, start)) && isAsciiDigit(charAt(text, start + 1))
        && isAsciiDigit(charAt(text, start + 2)) && !isAsciiDigit(charAt(text, start + 3));
  }

  /**
   * Checks whether a match may start here: the position is at the text start or the
   * preceding code point is neither a letter nor a digit.
   *
   * @param text The text. Must not be {@code null}.
   * @param index The candidate start offset.
   * @return {@code true} if a match may start at {@code index}.
   */
  public static boolean boundaryBefore(CharSequence text, int index) {
    return index == 0 || !Character.isLetterOrDigit(Character.codePointBefore(text, index));
  }

  /**
   * Checks whether a match may end here: the position is at the text end or the code
   * point at it is neither a letter nor a digit.
   *
   * @param text The text. Must not be {@code null}.
   * @param index The candidate exclusive end offset.
   * @return {@code true} if a match may end at {@code index}.
   */
  public static boolean boundaryAfter(CharSequence text, int index) {
    final int cp = codePointAt(text, index);
    return cp == NO_CODE_POINT || !Character.isLetterOrDigit(cp);
  }

  /**
   * @param cp The code point to classify.
   * @return {@code true} if {@code cp} is an ASCII digit.
   */
  public static boolean isAsciiDigit(int cp) {
    return cp >= '0' && cp <= '9';
  }

  /**
   * Reads the char at a position, returning a space as an out-of-bounds sentinel so
   * scan loops need no per-step bounds checks.
   *
   * @param text The text. Must not be {@code null}.
   * @param index The offset to read.
   * @return The char at {@code index}, or a space when out of bounds.
   */
  public static char charAt(CharSequence text, int index) {
    return index >= 0 && index < text.length() ? text.charAt(index) : ' ';
  }

  /**
   * Reads the code point at a position.
   *
   * @param text The text. Must not be {@code null}.
   * @param index The offset to read.
   * @return The code point at {@code index}, or {@link #NO_CODE_POINT} when out of
   *         bounds.
   */
  public static int codePointAt(CharSequence text, int index) {
    return index >= 0 && index < text.length()
        ? Character.codePointAt(text, index) : NO_CODE_POINT;
  }
}
