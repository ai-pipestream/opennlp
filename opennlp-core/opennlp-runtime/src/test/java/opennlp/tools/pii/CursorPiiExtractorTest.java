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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class CursorPiiExtractorTest {

  private final CursorPiiExtractor extractor = new CursorPiiExtractor();

  @Test
  void testEmail() {
    final String text = "Write to John.Doe+news@Example.co.uk today.";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    final PiiMention mention = mentions.get(0);
    Assertions.assertEquals(PiiMention.TYPE_EMAIL, mention.type());
    Assertions.assertEquals("John.Doe+news@Example.co.uk", text.substring(
        mention.span().getStart(), mention.span().getEnd()));
    Assertions.assertEquals("john.doe+news@example.co.uk", mention.normalized());
  }

  @Test
  void testEmailExcludesTrailingPunctuation() {
    final List<PiiMention> mentions = extractor.extract("Reach me at jane@example.com.");

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals("jane@example.com", mentions.get(0).normalized());
  }

  @Test
  void testEmailRejectsInvalidForms() {
    Assertions.assertTrue(extractor.extract("no at sign here").isEmpty());
    Assertions.assertTrue(extractor.extract("broken @example.com address").isEmpty());
    Assertions.assertTrue(extractor.extract("missing tld user@localhost").isEmpty());
    Assertions.assertTrue(extractor.extract("doubled dots user..name@example.com")
        .stream().noneMatch(m -> m.normalized().contains("..")));
    Assertions.assertTrue(extractor.extract("numeric tld user@example.c1").isEmpty());
  }

  @Test
  void testPhoneInternational() {
    final List<PiiMention> mentions = extractor.extract("Call +44 20 7946 0958 now.");

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_PHONE, mentions.get(0).type());
    Assertions.assertEquals("+442079460958", mentions.get(0).normalized());
  }

  @Test
  void testPhoneDomesticFormats() {
    Assertions.assertEquals("5551234567",
        extractor.extract("Call (555) 123-4567 today.").get(0).normalized());
    Assertions.assertEquals("5551234567",
        extractor.extract("Call 555-123-4567 today.").get(0).normalized());
    Assertions.assertEquals("15551234567",
        extractor.extract("Call 1 555 123 4567 today.").get(0).normalized());
  }

  @Test
  void testPhoneRejectsUnformattedAndDecimals() {
    Assertions.assertTrue(extractor.extract("id 5551234567 in the table").isEmpty());
    Assertions.assertTrue(extractor.extract("value 1234567890.25 total").isEmpty());
    Assertions.assertTrue(extractor.extract("pi is 3.1415926535").isEmpty());
    Assertions.assertTrue(extractor.extract("+123 is too short").isEmpty());
  }

  @Test
  void testIban() {
    final String text = "Wire it to DE89 3704 0044 0532 0130 00 by Friday.";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    final PiiMention mention = mentions.get(0);
    Assertions.assertEquals(PiiMention.TYPE_IBAN, mention.type());
    Assertions.assertEquals("DE89370400440532013000", mention.normalized());
    Assertions.assertEquals("DE89 3704 0044 0532 0130 00", text.substring(
        mention.span().getStart(), mention.span().getEnd()));
  }

  @Test
  void testIbanCompactAndWithLetters() {
    Assertions.assertEquals("GB82WEST12345698765432",
        extractor.extract("account GB82WEST12345698765432 closed").get(0).normalized());
    Assertions.assertEquals("GB82WEST12345698765432",
        extractor.extract("account GB82 WEST 1234 5698 7654 32 closed").get(0).normalized());
  }

  @Test
  void testIbanRejectsBadChecksum() {
    final List<PiiMention> mentions =
        extractor.extract("account DE89 3704 0044 0532 0130 02 closed");
    Assertions.assertTrue(mentions.stream()
        .noneMatch(m -> PiiMention.TYPE_IBAN.equals(m.type())
            || PiiMention.TYPE_CARD.equals(m.type())));
    Assertions.assertTrue(extractor.extract("plain WORDS IN CAPITALS here").isEmpty());
  }

  @Test
  void testCard() {
    final List<PiiMention> mentions =
        extractor.extract("Charged to 4111 1111 1111 1111 yesterday.");

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_CARD, mentions.get(0).type());
    Assertions.assertEquals("4111111111111111", mentions.get(0).normalized());
  }

  @Test
  void testCardFormatsAndChecksum() {
    Assertions.assertEquals("4111111111111111",
        extractor.extract("card 4111-1111-1111-1111 on file").get(0).normalized());
    Assertions.assertEquals("378282246310005",
        extractor.extract("amex 378282246310005 on file").get(0).normalized());
    Assertions.assertTrue(extractor.extract("card 4111 1111 1111 1112 on file").isEmpty());
    Assertions.assertTrue(extractor.extract("serial 1111 2222 3333 4444 here").isEmpty());
  }

  @Test
  void testOverlapsResolveToTheMoreSpecificType() {
    final List<PiiMention> mentions =
        extractor.extract("Pay DE89 3704 0044 0532 0130 00 or call +1 555 123 4567.");

    Assertions.assertEquals(2, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_IBAN, mentions.get(0).type());
    Assertions.assertEquals(PiiMention.TYPE_PHONE, mentions.get(1).type());
  }

  @Test
  void testPhoneDigitsInsideEmailNotReported() {
    final List<PiiMention> mentions =
        extractor.extract("mail +15551234567@voip.example.com please");

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_EMAIL, mentions.get(0).type());
  }

  @Test
  void testMentionsComeBackInTextOrder() {
    final List<PiiMention> mentions = extractor.extract(
        "jane@example.com, +44 20 7946 0958, and 4111 1111 1111 1111.");

    Assertions.assertEquals(3, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_EMAIL, mentions.get(0).type());
    Assertions.assertEquals(PiiMention.TYPE_PHONE, mentions.get(1).type());
    Assertions.assertEquals(PiiMention.TYPE_CARD, mentions.get(2).type());
    Assertions.assertTrue(mentions.get(0).span().getEnd() <= mentions.get(1).span().getStart());
    Assertions.assertTrue(mentions.get(1).span().getEnd() <= mentions.get(2).span().getStart());
  }

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class, () -> extractor.extract(null));
  }
}
