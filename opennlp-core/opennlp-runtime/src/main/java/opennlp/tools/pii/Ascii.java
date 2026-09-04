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
 * ASCII character tests and case folding for the PII scanners, by arithmetic on the
 * character value rather than by locale-sensitive methods or by regular expressions.
 *
 * <p>Every value the scanners recognize is defined over ASCII by its standard, so a
 * locale-sensitive test would only add ways to be wrong: the Turkish dotless i, for
 * example, folds an {@code I} to something no standard accepts.</p>
 */
final class Ascii {

  /** The distance between an uppercase and the matching lowercase ASCII letter. */
  private static final int CASE_DISTANCE = 'a' - 'A';

  private Ascii() {
    // This class holds static tests only and is never instantiated.
  }

  /**
   * Tests for an ASCII digit.
   *
   * @param c The character.
   * @return {@code true} for {@code 0-9}.
   */
  static boolean isDigit(char c) {
    return c >= '0' && c <= '9';
  }

  /**
   * Tests for an ASCII letter.
   *
   * @param c The character.
   * @return {@code true} for {@code A-Z} and {@code a-z}.
   */
  static boolean isLetter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  }

  /**
   * Tests for an uppercase ASCII letter.
   *
   * @param c The character.
   * @return {@code true} for {@code A-Z}.
   */
  static boolean isUpper(char c) {
    return c >= 'A' && c <= 'Z';
  }

  /**
   * Tests for a lowercase ASCII letter.
   *
   * @param c The character.
   * @return {@code true} for {@code a-z}.
   */
  static boolean isLower(char c) {
    return c >= 'a' && c <= 'z';
  }

  /**
   * Tests for an ASCII letter or digit.
   *
   * @param c The character.
   * @return {@code true} for {@code A-Z}, {@code a-z}, and {@code 0-9}.
   */
  static boolean isLetterOrDigit(char c) {
    return isLetter(c) || isDigit(c);
  }

  /**
   * Tests for a hexadecimal digit in either case.
   *
   * @param c The character.
   * @return {@code true} for {@code 0-9}, {@code a-f}, and {@code A-F}.
   */
  static boolean isHexDigit(char c) {
    return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
  }

  /**
   * Reads the value of a hexadecimal digit.
   *
   * @param c The character.
   * @return The value {@code 0} to {@code 15}, or {@code -1} if {@code c} is not a
   *         hexadecimal digit.
   */
  static int hexValue(char c) {
    if (isDigit(c)) {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
      return c - 'A' + 10;
    }
    return -1;
  }

  /**
   * Folds an ASCII letter to lowercase, leaving every other character unchanged.
   *
   * @param c The character.
   * @return The lowercase form of an ASCII letter, otherwise {@code c}.
   */
  static char toLower(char c) {
    return isUpper(c) ? (char) (c + CASE_DISTANCE) : c;
  }

  /**
   * Folds an ASCII letter to uppercase, leaving every other character unchanged.
   *
   * @param c The character.
   * @return The uppercase form of an ASCII letter, otherwise {@code c}.
   */
  static char toUpper(char c) {
    return isLower(c) ? (char) (c - CASE_DISTANCE) : c;
  }

  /**
   * Folds the ASCII letters of a sequence to lowercase.
   *
   * @param value The sequence to fold. Must not be {@code null}.
   * @return The folded sequence. Never {@code null}.
   */
  static String toLower(CharSequence value) {
    final StringBuilder folded = new StringBuilder(value.length());
    for (int i = 0; i < value.length(); i++) {
      folded.append(toLower(value.charAt(i)));
    }
    return folded.toString();
  }

  /**
   * Folds the ASCII letters of a sequence to uppercase.
   *
   * @param value The sequence to fold. Must not be {@code null}.
   * @return The folded sequence. Never {@code null}.
   */
  static String toUpper(CharSequence value) {
    final StringBuilder folded = new StringBuilder(value.length());
    for (int i = 0; i < value.length(); i++) {
      folded.append(toUpper(value.charAt(i)));
    }
    return folded.toString();
  }
}
