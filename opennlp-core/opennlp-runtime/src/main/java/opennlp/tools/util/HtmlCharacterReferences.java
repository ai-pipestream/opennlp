/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package opennlp.tools.util;

import java.util.Arrays;

/** Decodes HTML character references in an attribute value. */
final class HtmlCharacterReferences {

  private static final int REPLACEMENT_CHARACTER = 0xfffd;
  private static final int MAX_NAMED_REFERENCE_LENGTH = 32;
  private static final int[] WINDOWS_1252 = {
      0x20ac, 0x81, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021,
      0x02c6, 0x2030, 0x0160, 0x2039, 0x0152, 0x8d, 0x017d, 0x8f,
      0x90, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014,
      0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0x9d, 0x017e, 0x0178
  };

  private HtmlCharacterReferences() {}

  static String decodeAttribute(String value) {
    if (value == null) {
      throw new IllegalArgumentException("value must not be null");
    }
    int ampersand = value.indexOf('&');
    if (ampersand < 0) {
      return value;
    }

    StringBuilder decoded = null;
    int copiedUntil = 0;
    int cursor = ampersand;
    while (cursor < value.length()) {
      if (value.charAt(cursor) != '&') {
        cursor++;
        continue;
      }
      Match match = cursor + 1 < value.length() && value.charAt(cursor + 1) == '#'
          ? numeric(value, cursor) : named(value, cursor);
      if (match == null) {
        cursor++;
        continue;
      }
      if (decoded == null) {
        decoded = new StringBuilder(value.length());
      }
      decoded.append(value, copiedUntil, cursor).append(match.replacement());
      cursor = match.end();
      copiedUntil = cursor;
    }
    return decoded == null ? value : decoded.append(value, copiedUntil, value.length()).toString();
  }

  private static Match named(String value, int ampersand) {
    int end = ampersand + 1;
    int limit = Math.min(value.length(), ampersand + 1 + MAX_NAMED_REFERENCE_LENGTH);
    while (end < limit && isAsciiAlphanumeric(value.charAt(end))) {
      end++;
    }
    if (end < limit && value.charAt(end) == ';') {
      end++;
    }
    for (int candidateEnd = end; candidateEnd > ampersand + 1; candidateEnd--) {
      String name = value.substring(ampersand + 1, candidateEnd);
      int index = Arrays.binarySearch(HtmlCharacterReferenceData.NAMES, name);
      if (index < 0) {
        continue;
      }
      boolean hasSemicolon = name.charAt(name.length() - 1) == ';';
      if (!hasSemicolon && candidateEnd < value.length()) {
        char next = value.charAt(candidateEnd);
        if (next == '=' || isAsciiAlphanumeric(next)) {
          return null;
        }
      }
      return new Match(candidateEnd, HtmlCharacterReferenceData.VALUES[index]);
    }
    return null;
  }

  private static Match numeric(String value, int ampersand) {
    int cursor = ampersand + 2;
    int radix = 10;
    if (cursor < value.length() && (value.charAt(cursor) == 'x' || value.charAt(cursor) == 'X')) {
      radix = 16;
      cursor++;
    }
    int digitsStart = cursor;
    long codePoint = 0;
    while (cursor < value.length()) {
      int digit = asciiDigit(value.charAt(cursor), radix);
      if (digit < 0) {
        break;
      }
      codePoint = Math.min(0x110000L, codePoint * radix + digit);
      cursor++;
    }
    if (cursor == digitsStart) {
      return null;
    }
    if (cursor < value.length() && value.charAt(cursor) == ';') {
      cursor++;
    }
    int scalar = htmlNumericReplacement(codePoint);
    return new Match(cursor, new String(Character.toChars(scalar)));
  }

  private static int htmlNumericReplacement(long codePoint) {
    if (codePoint == 0 || codePoint > Character.MAX_CODE_POINT
        || codePoint >= Character.MIN_SURROGATE && codePoint <= Character.MAX_SURROGATE) {
      return REPLACEMENT_CHARACTER;
    }
    if (codePoint >= 0x80 && codePoint <= 0x9f) {
      return WINDOWS_1252[(int) codePoint - 0x80];
    }
    return (int) codePoint;
  }

  private static boolean isAsciiAlphanumeric(char ch) {
    return ch >= '0' && ch <= '9' || ch >= 'A' && ch <= 'Z' || ch >= 'a' && ch <= 'z';
  }

  private static int asciiDigit(char ch, int radix) {
    if (ch >= '0' && ch <= '9') {
      return ch - '0';
    }
    if (radix == 16 && ch >= 'A' && ch <= 'F') {
      return ch - 'A' + 10;
    }
    if (radix == 16 && ch >= 'a' && ch <= 'f') {
      return ch - 'a' + 10;
    }
    return -1;
  }

  private record Match(int end, String replacement) {}
}
