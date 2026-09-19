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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class StringPatternTest {

  @Test
  void testIsAllLetters() {
    Assertions.assertTrue(StringPattern.recognize("test").isAllLetter());
    Assertions.assertTrue(StringPattern.recognize("TEST").isAllLetter());
    Assertions.assertTrue(StringPattern.recognize("TesT").isAllLetter());
    Assertions.assertTrue(StringPattern.recognize("grün").isAllLetter());
    Assertions.assertTrue(StringPattern.recognize("üäöæß").isAllLetter());
    Assertions.assertTrue(StringPattern.recognize("あア亜Ａａ").isAllLetter());
  }

  @Test
  void testIsInitialCapitalLetter() {
    Assertions.assertTrue(StringPattern.recognize("Test").isInitialCapitalLetter());
    Assertions.assertFalse(StringPattern.recognize("tEST").isInitialCapitalLetter());
    Assertions.assertTrue(StringPattern.recognize("TesT").isInitialCapitalLetter());
    Assertions.assertTrue(StringPattern.recognize("Üäöæß").isInitialCapitalLetter());
    Assertions.assertFalse(StringPattern.recognize("いイ井").isInitialCapitalLetter());
    Assertions.assertTrue(StringPattern.recognize("Iいイ井").isInitialCapitalLetter());
    Assertions.assertTrue(StringPattern.recognize("Ｉいイ井").isInitialCapitalLetter());
  }

  @Test
  void testIsAllCapitalLetter() {
    Assertions.assertTrue(StringPattern.recognize("TEST").isAllCapitalLetter());
    Assertions.assertTrue(StringPattern.recognize("ÄÄÄÜÜÜÖÖÖÖ").isAllCapitalLetter());
    Assertions.assertFalse(StringPattern.recognize("ÄÄÄÜÜÜÖÖä").isAllCapitalLetter());
    Assertions.assertFalse(StringPattern.recognize("ÄÄÄÜÜdÜÖÖ").isAllCapitalLetter());
    Assertions.assertTrue(StringPattern.recognize("ＡＢＣ").isAllCapitalLetter());
    Assertions.assertFalse(StringPattern.recognize("うウ宇").isAllCapitalLetter());
  }

  @Test
  void testIsAllLowerCaseLetter() {
    Assertions.assertTrue(StringPattern.recognize("test").isAllLowerCaseLetter());
    Assertions.assertTrue(StringPattern.recognize("öäü").isAllLowerCaseLetter());
    Assertions.assertTrue(StringPattern.recognize("öäüßßß").isAllLowerCaseLetter());
    Assertions.assertFalse(StringPattern.recognize("Test").isAllLowerCaseLetter());
    Assertions.assertFalse(StringPattern.recognize("TEST").isAllLowerCaseLetter());
    Assertions.assertFalse(StringPattern.recognize("testT").isAllLowerCaseLetter());
    Assertions.assertFalse(StringPattern.recognize("tesÖt").isAllLowerCaseLetter());
    Assertions.assertTrue(StringPattern.recognize("ａｂｃ").isAllLowerCaseLetter());
    Assertions.assertFalse(StringPattern.recognize("えエ絵").isAllLowerCaseLetter());
  }

  @Test
  void testIsAllDigit() {
    Assertions.assertTrue(StringPattern.recognize("123456").isAllDigit());
    Assertions.assertFalse(StringPattern.recognize("123,56").isAllDigit());
    Assertions.assertFalse(StringPattern.recognize("12356f").isAllDigit());
    Assertions.assertTrue(StringPattern.recognize("１２３４５６").isAllDigit());
  }

  @Test
  void testIsAllHiragana() {
    Assertions.assertTrue(StringPattern.recognize("あぱっち・るしーん").isAllHiragana());
    Assertions.assertFalse(StringPattern.recognize("あぱっち・そふとうぇあ財団").isAllHiragana());
    Assertions.assertFalse(StringPattern.recognize("あぱっち・るしーんＶ１．０").isAllHiragana());
  }

  @Test
  void testIsAllKatakana() {
    Assertions.assertTrue(StringPattern.recognize("アパッチ・ルシーン").isAllKatakana());
    Assertions.assertFalse(StringPattern.recognize("アパッチ・ソフトウェア財団").isAllKatakana());
    Assertions.assertFalse(StringPattern.recognize("アパッチ・ルシーンＶ１．０").isAllKatakana());
  }

  @Test
  void testDigits() {
    Assertions.assertEquals(6, StringPattern.recognize("123456").digits());
    Assertions.assertEquals(3, StringPattern.recognize("123fff").digits());
    Assertions.assertEquals(0, StringPattern.recognize("test").digits());
    Assertions.assertEquals(3, StringPattern.recognize("１２３ｆｆｆ").digits());
  }

  @Test
  void testContainsPeriod() {
    Assertions.assertTrue(StringPattern.recognize("test.").containsPeriod());
    Assertions.assertTrue(StringPattern.recognize("23.5").containsPeriod());
    Assertions.assertFalse(StringPattern.recognize("test,/-1").containsPeriod());
  }

  @Test
  void testContainsComma() {
    Assertions.assertTrue(StringPattern.recognize("test,").containsComma());
    Assertions.assertTrue(StringPattern.recognize("23,5").containsComma());
    Assertions.assertFalse(StringPattern.recognize("test./-1").containsComma());
  }

  @Test
  void testContainsSlash() {
    Assertions.assertTrue(StringPattern.recognize("test/").containsSlash());
    Assertions.assertTrue(StringPattern.recognize("23/5").containsSlash());
    Assertions.assertFalse(StringPattern.recognize("test.1-,").containsSlash());
  }

  @Test
  void testContainsDigit() {
    Assertions.assertTrue(StringPattern.recognize("test1").containsDigit());
    Assertions.assertTrue(StringPattern.recognize("23,5").containsDigit());
    Assertions.assertFalse(StringPattern.recognize("test./-,").containsDigit());
    Assertions.assertTrue(StringPattern.recognize("テスト１").containsDigit());
    Assertions.assertFalse(StringPattern.recognize("テストＴＥＳＴ").containsDigit());
  }

  @Test
  void testContainsHyphen() {
    Assertions.assertTrue(StringPattern.recognize("test--").containsHyphen());
    Assertions.assertTrue(StringPattern.recognize("23-5").containsHyphen());
    Assertions.assertFalse(StringPattern.recognize("test.1/,").containsHyphen());
  }

  @Test
  void testContainsLetters() {
    Assertions.assertTrue(StringPattern.recognize("test--").containsLetters());
    Assertions.assertTrue(StringPattern.recognize("23h5ßm").containsLetters());
    Assertions.assertFalse(StringPattern.recognize("---.1/,").containsLetters());
  }

  @Test
  void testEmptyInputHasNoPositiveCategories() {
    StringPattern pattern = StringPattern.recognize("");

    Assertions.assertFalse(pattern.isAllLetter());
    Assertions.assertFalse(pattern.isInitialCapitalLetter());
    Assertions.assertFalse(pattern.isAllCapitalLetter());
    Assertions.assertFalse(pattern.isAllLowerCaseLetter());
    Assertions.assertFalse(pattern.isAllDigit());
    Assertions.assertFalse(pattern.isAllHiragana());
    Assertions.assertFalse(pattern.isAllKatakana());
    Assertions.assertFalse(pattern.containsPeriod());
    Assertions.assertFalse(pattern.containsComma());
    Assertions.assertFalse(pattern.containsSlash());
    Assertions.assertFalse(pattern.containsDigit());
    Assertions.assertFalse(pattern.containsHyphen());
    Assertions.assertFalse(pattern.containsLetters());
    Assertions.assertEquals(0, pattern.digits());
  }

  @Test
  void testNullInputRejectedAtPublicBoundary() {
    Assertions.assertThrows(IllegalArgumentException.class, () -> StringPattern.recognize(null));
  }

  @Test
  void testNonJapaneseScriptsAreNotJapanese() {
    StringPattern greek = StringPattern.recognize("αλφα");
    Assertions.assertTrue(greek.isAllLetter());
    Assertions.assertTrue(greek.isAllLowerCaseLetter());
    Assertions.assertFalse(greek.isAllHiragana());
    Assertions.assertFalse(greek.isAllKatakana());

    StringPattern cyrillic = StringPattern.recognize("слово");
    Assertions.assertTrue(cyrillic.isAllLetter());
    Assertions.assertTrue(cyrillic.isAllLowerCaseLetter());
    Assertions.assertFalse(cyrillic.isAllHiragana());
    Assertions.assertFalse(cyrillic.isAllKatakana());

    StringPattern arabic = StringPattern.recognize("لغة");
    Assertions.assertTrue(arabic.isAllLetter());
    Assertions.assertFalse(arabic.isAllLowerCaseLetter());
    Assertions.assertFalse(arabic.isAllCapitalLetter());
    Assertions.assertFalse(arabic.isAllHiragana());
    Assertions.assertFalse(arabic.isAllKatakana());

    StringPattern han = StringPattern.recognize("日本");
    Assertions.assertTrue(han.isAllLetter());
    Assertions.assertFalse(han.isAllHiragana());
    Assertions.assertFalse(han.isAllKatakana());
  }

  @Test
  void testSupplementaryLettersAndDigitsAreSingleCodePoints() {
    String deseretCapital = new String(Character.toChars(0x10400));
    String mathematicalDigit = new String(Character.toChars(0x1D7D8));

    StringPattern capital = StringPattern.recognize(deseretCapital);
    Assertions.assertTrue(capital.isAllLetter());
    Assertions.assertTrue(capital.isInitialCapitalLetter());
    Assertions.assertTrue(capital.isAllCapitalLetter());
    Assertions.assertTrue(capital.containsLetters());

    StringPattern digit = StringPattern.recognize(mathematicalDigit);
    Assertions.assertTrue(digit.isAllDigit());
    Assertions.assertTrue(digit.containsDigit());
    Assertions.assertEquals(1, digit.digits());
  }

  @Test
  void testCombiningMarksPreserveWordCaseAfterLetters() {
    StringPattern composedLower = StringPattern.recognize("café");
    StringPattern decomposedLower = StringPattern.recognize("cafe\u0301");
    Assertions.assertTrue(composedLower.isAllLowerCaseLetter());
    Assertions.assertTrue(decomposedLower.isAllLowerCaseLetter());
    Assertions.assertTrue(composedLower.isAllLetter());
    Assertions.assertFalse(decomposedLower.isAllLetter());

    StringPattern composedCapital = StringPattern.recognize("Café");
    StringPattern decomposedCapital = StringPattern.recognize("Cafe\u0301");
    Assertions.assertEquals(composedCapital.isInitialCapitalLetter(),
        decomposedCapital.isInitialCapitalLetter());
    Assertions.assertEquals(composedCapital.isAllLowerCaseLetter(),
        decomposedCapital.isAllLowerCaseLetter());

    Assertions.assertTrue(StringPattern.recognize("は\u3099").isAllHiragana());
    Assertions.assertFalse(StringPattern.recognize("\u0301").isAllHiragana());
    Assertions.assertFalse(StringPattern.recognize("\u0301").isAllKatakana());
    Assertions.assertFalse(StringPattern.recognize("\u0301a").isAllLowerCaseLetter());
    Assertions.assertFalse(StringPattern.recognize("1\u0301").isAllLowerCaseLetter());
    Assertions.assertFalse(StringPattern.recognize("a.\u0301").isAllLowerCaseLetter());
  }

  @Test
  void testMalformedUtf16ClearsWholeTokenCategoriesButScansValidCharacters() {
    StringPattern pattern = StringPattern.recognize("A\uD8001.-");

    Assertions.assertFalse(pattern.isAllLetter());
    Assertions.assertFalse(pattern.isInitialCapitalLetter());
    Assertions.assertFalse(pattern.isAllCapitalLetter());
    Assertions.assertFalse(pattern.isAllLowerCaseLetter());
    Assertions.assertFalse(pattern.isAllDigit());
    Assertions.assertFalse(pattern.isAllHiragana());
    Assertions.assertFalse(pattern.isAllKatakana());
    Assertions.assertTrue(pattern.containsLetters());
    Assertions.assertTrue(pattern.containsDigit());
    Assertions.assertTrue(pattern.containsPeriod());
    Assertions.assertTrue(pattern.containsHyphen());
    Assertions.assertEquals(1, pattern.digits());

    pattern = StringPattern.recognize("\uDC00\uD800");
    Assertions.assertFalse(pattern.isAllLetter());
    Assertions.assertFalse(pattern.containsLetters());
    Assertions.assertEquals(0, pattern.digits());
  }

}
