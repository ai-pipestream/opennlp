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

package opennlp.tools.util.featuregen;

/**
 * Recognizes predefined patterns in strings.
 */
public class StringPattern {

  private static final int INITAL_CAPITAL_LETTER = 0x1;
  private static final int ALL_CAPITAL_LETTER = 0x1 << 1;
  private static final int ALL_LOWERCASE_LETTER = 0x1 << 2;
  private static final int ALL_LETTERS = 0x1 << 3;
  private static final int ALL_DIGIT = 0x1 << 4;
  private static final int ALL_HIRAGANA = 0x1 << 5;
  private static final int ALL_KATAKANA = 0x1 << 6;
  private static final int CONTAINS_PERIOD = 0x1 << 7;
  private static final int CONTAINS_COMMA = 0x1 << 8;
  private static final int CONTAINS_SLASH = 0x1 << 9;
  private static final int CONTAINS_DIGIT = 0x1 << 10;
  private static final int CONTAINS_HYPHEN = 0x1 << 11;
  private static final int CONTAINS_LETTERS = 0x1 << 12;

  private final int pattern;

  private final int digits;

  private StringPattern(int pattern, int digits) {
    this.pattern = pattern;
    this.digits = digits;
  }

  /**
   * Classifies a token by Unicode code point. Combining marks continue the case and Japanese
   * script classification of an immediately preceding letter, but are not themselves letters.
   * An empty token has no positive classifications. Malformed UTF-16 clears whole-token
   * classifications while valid code points still contribute containment flags and digit counts.
   *
   * @param token The token to classify.
   * @return The recognized string pattern.
   * @throws IllegalArgumentException Thrown if {@code token} is {@code null}.
   */
  public static StringPattern recognize(String token) {
    if (token == null) {
      throw new IllegalArgumentException("token must not be null");
    }
    if (token.isEmpty()) {
      return new StringPattern(0, 0);
    }

    boolean allCapital = true;
    boolean allLowercase = true;
    boolean allLetters = true;
    boolean allDigits = true;
    boolean allHiragana = true;
    boolean allKatakana = true;
    boolean initialCapital = false;
    boolean containsLetters = false;
    boolean containsDigit = false;
    boolean malformed = false;
    boolean markCanContinueWord = false;
    boolean japaneseRunStarted = false;
    int pattern = 0;
    int digits = 0;

    for (int offset = 0; offset < token.length();) {
      char ch = token.charAt(offset);
      if (Character.isSurrogate(ch)
          && !(Character.isHighSurrogate(ch) && offset + 1 < token.length()
          && Character.isLowSurrogate(token.charAt(offset + 1)))) {
        malformed = true;
        allCapital = false;
        allLowercase = false;
        allLetters = false;
        allDigits = false;
        allHiragana = false;
        allKatakana = false;
        markCanContinueWord = false;
        offset++;
        continue;
      }

      int codePoint = token.codePointAt(offset);
      int type = Character.getType(codePoint);
      boolean isLetter = type == Character.UPPERCASE_LETTER
          || type == Character.LOWERCASE_LETTER
          || type == Character.TITLECASE_LETTER
          || type == Character.MODIFIER_LETTER
          || type == Character.OTHER_LETTER;
      boolean isMark = type == Character.NON_SPACING_MARK
          || type == Character.COMBINING_SPACING_MARK || type == Character.ENCLOSING_MARK;

      if (isLetter) {
        boolean uppercase = Character.isUpperCase(codePoint);
        boolean lowercase = Character.isLowerCase(codePoint);
        if (offset == 0) {
          initialCapital = uppercase;
        }
        containsLetters = true;
        allCapital &= uppercase;
        allLowercase &= lowercase;
        allDigits = false;
        markCanContinueWord = true;
      } else {
        allLetters = false;
        boolean continuingMark = isMark && markCanContinueWord;
        if (!continuingMark) {
          allCapital = false;
          allLowercase = false;
        }

        if (type == Character.DECIMAL_DIGIT_NUMBER) {
          containsDigit = true;
          digits++;
          markCanContinueWord = false;
        } else if (!continuingMark) {
          allDigits = false;
          markCanContinueWord = false;
        }

        switch (codePoint) {
          case ',':
            pattern |= CONTAINS_COMMA;
            break;

          case '.':
            pattern |= CONTAINS_PERIOD;
            break;

          case '/':
            pattern |= CONTAINS_SLASH;
            break;

          case '-':
            pattern |= CONTAINS_HYPHEN;
            break;

          default:
            break;
        }
      }

      if (allHiragana || allKatakana) {
        Character.UnicodeScript script = Character.UnicodeScript.of(codePoint);
        if (script == Character.UnicodeScript.HIRAGANA) {
          allKatakana = false;
          japaneseRunStarted = true;
        } else if (script == Character.UnicodeScript.KATAKANA) {
          allHiragana = false;
          japaneseRunStarted = true;
        } else if (isMark && markCanContinueWord) {
          // Combining marks extend the preceding script run.
        } else if ((codePoint == '・' || codePoint == 'ー' || codePoint == '〜')
            && japaneseRunStarted) {
          markCanContinueWord = true;
        } else {
          allHiragana = false;
          allKatakana = false;
        }
      }

      offset += Character.charCount(codePoint);
    }

    if (malformed) {
      initialCapital = false;
    }
    if (initialCapital) {
      pattern |= INITAL_CAPITAL_LETTER;
    }
    if (allCapital) {
      pattern |= ALL_CAPITAL_LETTER;
    }
    if (allLowercase) {
      pattern |= ALL_LOWERCASE_LETTER;
    }
    if (allLetters) {
      pattern |= ALL_LETTERS;
    }
    if (allDigits) {
      pattern |= ALL_DIGIT;
    }
    if (allHiragana) {
      pattern |= ALL_HIRAGANA;
    }
    if (allKatakana) {
      pattern |= ALL_KATAKANA;
    }
    if (containsLetters) {
      pattern |= CONTAINS_LETTERS;
    }
    if (containsDigit) {
      pattern |= CONTAINS_DIGIT;
    }

    return new StringPattern(pattern, digits);
  }

  /**
   * @return {@code true} if all characters are letters.
   */
  public boolean isAllLetter() {
    return (pattern & ALL_LETTERS) > 0;
  }

  /**
   * @return {@code true} if first letter is capital.
   */
  public boolean isInitialCapitalLetter() {
    return (pattern & INITAL_CAPITAL_LETTER) > 0;
  }

  /**
   * @return {@code true} if all letters are capital.
   */
  public boolean isAllCapitalLetter() {
    return (pattern & ALL_CAPITAL_LETTER) > 0;
  }

  /**
   * @return {@code true} if all letters are lower case.
   */
  public boolean isAllLowerCaseLetter() {
    return (pattern & ALL_LOWERCASE_LETTER) > 0;
  }

  /**
   * @return {@code true} if all chars are digits.
   */
  public boolean isAllDigit() {
    return (pattern & ALL_DIGIT) > 0;
  }

  /**
   * @return {@code true} if all chars are hiragana.
   */
  public boolean isAllHiragana() {
    return (pattern & ALL_HIRAGANA) > 0;
  }

  /**
   * @return {@code true} if all chars are katakana.
   */
  public boolean isAllKatakana() {
    return (pattern & ALL_KATAKANA) > 0;
  }

  /**
   * Retrieves the number of digits.
   */
  public int digits() {
    return digits;
  }

  /**
   * @return {@code true} if a period is contained.
   */
  public boolean containsPeriod() {
    return (pattern & CONTAINS_PERIOD) > 0;
  }

  /**
   * @return {@code true} if a comma is contained.
   */
  public boolean containsComma() {
    return (pattern & CONTAINS_COMMA) > 0;
  }

  /**
   * @return {@code true} if a slash is contained.
   */
  public boolean containsSlash() {
    return (pattern & CONTAINS_SLASH) > 0;
  }

  /**
   * @return {@code true} if a digit is contained.
   */
  public boolean containsDigit() {
    return (pattern & CONTAINS_DIGIT) > 0;
  }

  /**
   * @return {@code true} if a hypen is contained.
   */
  public boolean containsHyphen() {
    return (pattern & CONTAINS_HYPHEN) > 0;
  }

  /**
   * @return {@code true} if a letters are contained.
   */
  public boolean containsLetters() {
    return (pattern & CONTAINS_LETTERS) > 0;
  }
}
