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
package opennlp.tools.stemmer.light;

/**
 * Buffer helpers shared by the light and minimal stemmers, adapted from Apache Lucene's
 * analysis-common module. All operate on a {@code char[]} prefix of the given length, the
 * in-place convention the stemming algorithms use.
 */
final class StemmerUtil {

  private StemmerUtil() {
  }

  /**
   * Checks whether the buffer starts with a prefix.
   *
   * @param s      The input buffer.
   * @param len    The filled length of the buffer.
   * @param prefix The prefix to test.
   * @return {@code true} if the buffer starts with {@code prefix}.
   */
  static boolean startsWith(char[] s, int len, String prefix) {
    final int prefixLen = prefix.length();
    if (prefixLen > len) {
      return false;
    }
    for (int i = 0; i < prefixLen; i++) {
      if (s[i] != prefix.charAt(i)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Checks whether the buffer ends with a suffix.
   *
   * @param s      The input buffer.
   * @param len    The filled length of the buffer.
   * @param suffix The suffix to test.
   * @return {@code true} if the buffer ends with {@code suffix}.
   */
  static boolean endsWith(char[] s, int len, String suffix) {
    final int suffixLen = suffix.length();
    if (suffixLen > len) {
      return false;
    }
    for (int i = suffixLen - 1; i >= 0; i--) {
      if (s[len - (suffixLen - i)] != suffix.charAt(i)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Checks whether the buffer ends with a suffix.
   *
   * @param s      The input buffer.
   * @param len    The filled length of the buffer.
   * @param suffix The suffix to test.
   * @return {@code true} if the buffer ends with {@code suffix}.
   */
  static boolean endsWith(char[] s, int len, char[] suffix) {
    final int suffixLen = suffix.length;
    if (suffixLen > len) {
      return false;
    }
    for (int i = suffixLen - 1; i >= 0; i--) {
      if (s[len - (suffixLen - i)] != suffix[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Deletes one character in place.
   *
   * @param s   The input buffer.
   * @param pos The position of the character to delete; must be less than {@code len}.
   * @param len The filled length of the buffer.
   * @return The filled length after the deletion.
   */
  static int delete(char[] s, int pos, int len) {
    if (pos < len - 1) {
      System.arraycopy(s, pos + 1, s, pos, len - pos - 1);
    }
    return len - 1;
  }

  /**
   * Deletes {@code nChars} characters in place.
   *
   * @param s      The input buffer.
   * @param pos    The position of the first character to delete.
   * @param len    The filled length of the buffer.
   * @param nChars The number of characters to delete; {@code pos + nChars} must not exceed
   *               {@code len}.
   * @return The filled length after the deletion.
   */
  static int deleteN(char[] s, int pos, int len, int nChars) {
    if (pos + nChars < len) {
      System.arraycopy(s, pos + nChars, s, pos, len - pos - nChars);
    }
    return len - nChars;
  }
}
