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

/**
 * The word boundary tests shared by the PII scanners, so that nothing is ever reported
 * from inside a longer run of letters and digits.
 *
 * <p>Boundaries are judged on whole code points, not on UTF-16 units, so a candidate that
 * follows or precedes a character outside the basic plane is judged by that character and
 * not by half of it.</p>
 */
final class Boundaries {

  private Boundaries() {
    // This class holds static tests only and is never instantiated.
  }

  /**
   * Checks that a candidate does not continue a word to its left.
   *
   * @param text The text being scanned.
   * @param start The candidate start.
   * @return {@code true} if the candidate may start here.
   */
  static boolean onWordStart(CharSequence text, int start) {
    return start == 0
        || !Character.isLetterOrDigit(Character.codePointBefore(text, start));
  }

  /**
   * Checks that a numeric candidate does not continue a word, a number, a decimal
   * fraction, or a comma-grouped number to its left.
   *
   * @param text The text being scanned.
   * @param start The candidate start.
   * @return {@code true} if the candidate may start here.
   */
  static boolean onNumberStart(CharSequence text, int start) {
    if (start == 0) {
      return true;
    }
    final int previous = Character.codePointBefore(text, start);
    if (Character.isLetterOrDigit(previous)) {
      return false;
    }
    return (previous != '.' && previous != ',')
        || start < 2 || !Ascii.isDigit(text.charAt(start - 2));
  }

  /**
   * Checks that a candidate ending at {@code end} does not continue into a letter or
   * digit.
   *
   * @param text The text being scanned.
   * @param end The candidate end, exclusive.
   * @return {@code true} if the candidate may end here.
   */
  static boolean onEnd(CharSequence text, int end) {
    return end >= text.length()
        || !Character.isLetterOrDigit(Character.codePointAt(text, end));
  }

  /**
   * Checks that a candidate ending at {@code end} is not followed by one of the
   * characters that would make it a piece of a longer structured value, for example a
   * further dotted group of an address or a version string.
   *
   * @param text The text being scanned.
   * @param end The candidate end, exclusive.
   * @param separator The separator that continues the structure.
   * @return {@code true} if the candidate may end here.
   */
  static boolean onEndBefore(CharSequence text, int end, char separator) {
    if (!onEnd(text, end)) {
      return false;
    }
    return end + 1 >= text.length() || text.charAt(end) != separator
        || !Ascii.isLetterOrDigit(text.charAt(end + 1));
  }
}
