/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
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

package opennlp.tools.stemmer.hunspell;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/** A compound flag sequence with optional and repeated elements. */
final class HunspellCompoundRule {

  /** Interprets a flag according to the affix file's FLAG setting. */
  @FunctionalInterface
  interface FlagReader {
    /**
     * Interprets one flag.
     *
     * @param text The encoded flag.
     * @param line The source line.
     * @return The flag value.
     * @throws IOException Thrown if the flag is malformed.
     */
    int read(String text, int line) throws IOException;
  }

  /** Maximum elements in one compound pattern. */
  private static final int MAX_ELEMENTS = 4096;

  private static final String INVALID_PATTERN = "invalid COMPOUNDRULE at line ";

  /** Opens a flag written with more than one character. */
  private static final char GROUP_START = '(';

  /** Closes a flag written with more than one character. */
  private static final char GROUP_END = ')';

  /** Lets the preceding flag occur any number of times, including none. */
  private static final char ANY_NUMBER = '*';

  /** Lets the preceding flag occur once or not at all. */
  private static final char OPTIONAL = '?';

  /** Marks a flag that occurs exactly once. */
  private static final char ONCE = ' ';

  private final int[] flags;
  private final char[] repetition;

  /**
   * Initializes a parsed compound pattern.
   *
   * @param flags The required flag at each position.
   * @param repetition The position's repetition operator or a space.
   */
  private HunspellCompoundRule(int[] flags, char[] repetition) {
    this.flags = flags;
    this.repetition = repetition;
  }

  /**
   * Parses a compound rule without using a regular-expression engine.
   *
   * @param pattern The rule text.
   * @param line The source line.
   * @param reader The flag decoder.
   * @return The parsed rule.
   * @throws IOException Thrown if the pattern or a flag is malformed.
   */
  static HunspellCompoundRule parse(String pattern, int line, FlagReader reader) throws IOException {
    // every element takes at least one character, so the pattern length bounds the count
    final int capacity = Math.min(pattern.length(), MAX_ELEMENTS);
    final int[] flags = new int[capacity];
    final char[] repetition = new char[capacity];
    int count = 0;
    for (int at = 0; at < pattern.length();) {
      final String flag;
      if (pattern.charAt(at) == GROUP_START) {
        final int end = pattern.indexOf(GROUP_END, at + 1);
        if (end <= at + 1) {
          throw new IOException(INVALID_PATTERN + line);
        }
        flag = pattern.substring(at + 1, end);
        at = end + 1;
      } else {
        final int point = pattern.codePointAt(at);
        if (point == ANY_NUMBER || point == OPTIONAL || point == GROUP_END) {
          throw new IOException(INVALID_PATTERN + line);
        }
        final int end = at + Character.charCount(point);
        flag = pattern.substring(at, end);
        at = end;
      }
      if (count == MAX_ELEMENTS) {
        throw new IOException("COMPOUNDRULE exceeds " + MAX_ELEMENTS + " elements at line " + line);
      }
      flags[count] = reader.read(flag, line);
      if (at < pattern.length()
          && (pattern.charAt(at) == ANY_NUMBER || pattern.charAt(at) == OPTIONAL)) {
        repetition[count++] = pattern.charAt(at++);
      } else {
        repetition[count++] = ONCE;
      }
    }
    if (count == 0) {
      throw new IOException("empty COMPOUNDRULE at line " + line);
    }
    return new HunspellCompoundRule(Arrays.copyOf(flags, count), Arrays.copyOf(repetition, count));
  }

  /**
   * Tests a sequence of selected homonyms. Each part consumes one flag position.
   *
   * @param parts The flags of one selected entry per compound part.
   * @param complete Whether the sequence must complete the rule.
   * @return Whether the sequence is permitted by this rule.
   */
  boolean matches(List<int[]> parts, boolean complete) {
    boolean[] states = new boolean[flags.length + 1];
    boolean[] next = new boolean[states.length];
    states[0] = true;
    skipOptional(states);
    for (int[] part : parts) {
      Arrays.fill(next, false);
      for (int i = 0; i < flags.length; i++) {
        if (states[i] && HunspellDictionary.contains(part, flags[i])) {
          next[repetition[i] == ANY_NUMBER ? i : i + 1] = true;
        }
      }
      skipOptional(next);
      final boolean[] previous = states;
      states = next;
      next = previous;
    }
    if (complete) {
      return states[flags.length];
    }
    for (boolean state : states) {
      if (state) {
        return true;
      }
    }
    return false;
  }

  /**
   * Advances states through optional pattern elements.
   *
   * @param states The active pattern positions.
   */
  private void skipOptional(boolean[] states) {
    for (int i = 0; i < flags.length; i++) {
      if (states[i] && repetition[i] != ONCE) {
        states[i + 1] = true;
      }
    }
  }

}
