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

import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

public class EuIdentityPiiExtractorTest {

  private final EuIdentityPiiExtractor extractor = new EuIdentityPiiExtractor();

  @ParameterizedTest
  @CsvSource({
      "9434765919, 9434765919",
      "9999999468, 9999999468",
      "4010232137, 4010232137",
      "1234567881, 1234567881",
      "9876543210, 9876543210",
      "'943 476 5919', 9434765919",
      "943-476-5919, 9434765919",
      "'999 999 9468', 9999999468"
  })
  void testAcceptsNhsNumbers(String text, String normalized) {
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size(), text);
    Assertions.assertEquals(PiiMention.TYPE_UK_NHS, mentions.get(0).type());
    Assertions.assertEquals(normalized, mentions.get(0).normalized());
    Assertions.assertEquals(0, mentions.get(0).span().getStart());
    Assertions.assertEquals(text.length(), mentions.get(0).span().getEnd());
  }

  /**
   * Verifies the modulus 11 check: each fixture differs from a valid number in one digit, or
   * leaves the remainder that would need a check digit of ten.
   */
  @ParameterizedTest
  @ValueSource(strings = {
      "9434765918",
      "9999999469",
      "4010232138",
      "1234567882",
      "1234567890",
      "'943 476 5918'"})
  void testRejectsNhsNumbersWithABrokenCheckDigit(String text) {
    Assertions.assertTrue(extractor.extract(text).isEmpty(), text);
  }

  /**
   * Verifies that a run of one repeated digit is not reported even where the check digit
   * happens to hold, since such a run is a placeholder wherever it appears.
   */
  @ParameterizedTest
  @ValueSource(strings = {"0000000000", "1111111111"})
  void testRejectsUniformDigitRuns(String text) {
    Assertions.assertTrue(extractor.extract(text).isEmpty(), text);
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "943476591",
      "94347659190",
      "'9434 76 5919'",
      "'943 4765919'",
      "'943-476 5919'",
      "x9434765919",
      "9434765919x"})
  void testRejectsNhsNumberNearMisses(String text) {
    Assertions.assertTrue(
        extractor.extract(text).stream()
            .noneMatch(m -> PiiMention.TYPE_UK_NHS.equals(m.type())), text);
  }

  @ParameterizedTest
  @CsvSource({
      "65929970489, 65929970489",
      "81095324717, 81095324717",
      "23746189575, 23746189575",
      "50123456782, 50123456782",
      "'65 929 970 489', 65929970489",
      "65-929-970-489, 65929970489"
  })
  void testAcceptsGermanTaxNumbers(String text, String normalized) {
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size(), text);
    Assertions.assertEquals(PiiMention.TYPE_DE_STEUER_ID, mentions.get(0).type());
    Assertions.assertEquals(normalized, mentions.get(0).normalized());
  }

  @ParameterizedTest
  @ValueSource(strings = {"65929970488", "81095324716", "23746189574", "50123456783"})
  void testRejectsGermanTaxNumbersWithABrokenCheckDigit(String text) {
    Assertions.assertTrue(extractor.extract(text).isEmpty(), text);
  }

  /**
   * Verifies the digit rules the tax office adds to the check digit. The first fixture is the
   * number the checksum documentation uses as its example, which starts with a zero and so is
   * not an issued number; the others have no repeated digit, two repeated digits, and three
   * of one digit in direct succession.
   */
  @ParameterizedTest
  @ValueSource(strings = {
      "02476291358",
      "12345678903",
      "11223456785",
      "11123456786"})
  void testRejectsGermanTaxNumbersBreakingTheDigitRules(String text) {
    Assertions.assertTrue(extractor.extract(text).isEmpty(), text);
  }

  @Test
  void testAcceptsADigitAppearingThreeTimesApart() {
    // 6592997048: the nine appears three times, no two of them in direct succession.
    final List<PiiMention> mentions = extractor.extract("65929970489");

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_DE_STEUER_ID, mentions.get(0).type());
  }

  @Test
  void testSpansInSentence() {
    final String text = "NHS 943 476 5919 and IdNr 65929970489 recorded.";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(2, mentions.size());
    Assertions.assertEquals("943 476 5919", text.substring(
        mentions.get(0).span().getStart(), mentions.get(0).span().getEnd()));
    Assertions.assertEquals("65929970489", text.substring(
        mentions.get(1).span().getStart(), mentions.get(1).span().getEnd()));
  }

  /**
   * Verifies that an eleven-digit tax number is not also reported as the ten-digit number
   * inside it.
   */
  @Test
  void testTaxNumberIsNotAlsoReportedAsAnNhsNumber() {
    final List<PiiMention> mentions = extractor.extract("65929970489");

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_DE_STEUER_ID, mentions.get(0).type());
  }

  @Test
  void testTypeSubsetLimitsWhatIsReported() {
    final String text = "NHS 9434765919 and IdNr 65929970489";

    Assertions.assertEquals(List.of(PiiMention.TYPE_UK_NHS),
        new EuIdentityPiiExtractor(Set.of(PiiMention.TYPE_UK_NHS)).extract(text)
            .stream().map(PiiMention::type).toList());
    Assertions.assertEquals(List.of(PiiMention.TYPE_DE_STEUER_ID),
        new EuIdentityPiiExtractor(Set.of(PiiMention.TYPE_DE_STEUER_ID)).extract(text)
            .stream().map(PiiMention::type).toList());
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "no identifier here",
      "1234",
      "",
      "the year 2026",
      "call (555) 123-4567"})
  void testTextWithoutAnIdentifierYieldsNoMention(String text) {
    Assertions.assertTrue(extractor.extract(text).isEmpty(), text);
  }

  @Test
  void testRejectsUnrecognizedTypeAndMissingArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new EuIdentityPiiExtractor(Set.of(PiiMention.TYPE_EMAIL)));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new EuIdentityPiiExtractor(Set.of()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new EuIdentityPiiExtractor(null));
    Assertions.assertThrows(IllegalArgumentException.class, () -> extractor.extract(null));
  }
}
