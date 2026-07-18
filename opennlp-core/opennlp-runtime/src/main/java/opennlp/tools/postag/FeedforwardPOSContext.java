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

package opennlp.tools.postag;

import opennlp.tools.util.StringUtil;

/**
 * The symbolic feature template of the feedforward tagger: a five word window, two
 * suffixes and three word shapes around the position, and the two previously assigned
 * tags.
 */
final class FeedforwardPOSContext {

  /** Word window positions: two left, the word itself, two right. */
  static final int WORD_SLOTS = 5;

  /** Suffix slots: the last two and last three characters of the word. */
  static final int SUFFIX_SLOTS = 2;

  /** Shape slots: the previous, current, and next word's shape. */
  static final int SHAPE_SLOTS = 3;

  /** Tag slots: the two previously assigned tags. */
  static final int TAG_SLOTS = 2;

  /** The total number of feature slots. */
  static final int SLOTS = WORD_SLOTS + SUFFIX_SLOTS + SHAPE_SLOTS + TAG_SLOTS;

  /**
   * Pretrained-vector window positions: the previous, current, and next word. Only
   * models trained with word vectors carry this block; it follows the learned slots
   * in the input layer.
   */
  static final int PRETRAINED_SLOTS = 3;

  private FeedforwardPOSContext() {
    // This class only describes the feature template and is never instantiated.
  }

  /**
   * Extracts the symbolic features for one position.
   *
   * @param sentence The sentence tokens.
   * @param index The position to tag.
   * @param previousTag The tag assigned at {@code index - 1}, or {@code null} at the
   *                    sentence start.
   * @param beforePreviousTag The tag assigned at {@code index - 2}, or {@code null}.
   * @return One symbol per slot; {@code null} marks positions outside the sentence.
   *         Never {@code null}.
   */
  static String[] extract(String[] sentence, int index, String previousTag,
      String beforePreviousTag) {
    final String[] symbols = new String[SLOTS];
    int slot = 0;
    for (int offset = -2; offset <= 2; offset++) {
      final int position = index + offset;
      symbols[slot++] =
          position >= 0 && position < sentence.length ? sentence[position] : null;
    }
    final String word = StringUtil.toLowerCase(sentence[index]);
    symbols[slot++] = suffix(word, 2);
    symbols[slot++] = suffix(word, 3);
    symbols[slot++] = index > 0 ? shape(sentence[index - 1]) : null;
    symbols[slot++] = shape(sentence[index]);
    symbols[slot++] = index + 1 < sentence.length ? shape(sentence[index + 1]) : null;
    symbols[slot++] = previousTag;
    symbols[slot] = beforePreviousTag;
    return symbols;
  }

  /**
   * Takes the trailing characters of a word.
   *
   * @param word The lowercased word.
   * @param length The suffix length.
   * @return The suffix, or the whole word when it is shorter. Never {@code null}.
   */
  static String suffix(String word, int length) {
    return word.length() <= length ? word : word.substring(word.length() - length);
  }

  /**
   * Classifies a word's surface shape.
   *
   * @param word The word.
   * @return One of the fixed shape symbols. Never {@code null}.
   */
  static String shape(String word) {
    boolean letter = false;
    boolean digit = false;
    boolean upper = false;
    boolean lower = false;
    for (int i = 0; i < word.length(); i++) {
      final char c = word.charAt(i);
      if (Character.isDigit(c)) {
        digit = true;
      } else if (Character.isLetter(c)) {
        letter = true;
        if (Character.isUpperCase(c)) {
          upper = true;
        } else {
          lower = true;
        }
      }
    }
    if (letter && digit) {
      return "*alnum*";
    }
    if (digit) {
      return "*digit*";
    }
    if (!letter) {
      return "*other*";
    }
    if (upper && !lower) {
      return "*allcaps*";
    }
    if (Character.isUpperCase(word.charAt(0))) {
      return "*cap*";
    }
    return "*lower*";
  }
}
