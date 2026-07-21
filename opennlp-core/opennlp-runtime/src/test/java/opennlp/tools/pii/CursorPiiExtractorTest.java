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

  /**
   * Verifies the behavior on an IBAN-shaped candidate whose check digits fail the
   * mod-97 test: no IBAN and no card is reported, and in the spaced form the scan falls
   * through to the formatted ten-digit tail, which qualifies as a domestic phone
   * number. The compact form offers no formatting evidence, so it yields nothing.
   */
  @Test
  void testIbanShapedTextWithBadCheckDigits() {
    final String text = "Wire to DE21 3704 0044 0532 0130 00 today.";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    final PiiMention mention = mentions.get(0);
    Assertions.assertEquals(PiiMention.TYPE_PHONE, mention.type());
    Assertions.assertEquals(23, mention.span().getStart());
    Assertions.assertEquals(35, mention.span().getEnd());
    Assertions.assertEquals("0532 0130 00", text.substring(23, 35));
    Assertions.assertEquals("0532013000", mention.normalized());

    Assertions.assertTrue(extractor.extract("Wire to DE21370400440532013000 today.").isEmpty());
  }

  /**
   * Verifies that the Luhn check alone decides bare sixteen-digit runs: a passing run
   * is reported as a card with its exact span, and flipping the final digit so the
   * checksum fails suppresses the mention entirely.
   */
  @Test
  void testCardBareDigitRunLuhnDecides() {
    final String text = "card 4111111111111111 on file";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_CARD, mentions.get(0).type());
    Assertions.assertEquals(5, mentions.get(0).span().getStart());
    Assertions.assertEquals(21, mentions.get(0).span().getEnd());
    Assertions.assertEquals("4111111111111111", mentions.get(0).normalized());

    Assertions.assertTrue(extractor.extract("card 4111111111111112 on file").isEmpty());
  }

  /**
   * Verifies that an email address is found with its exact span when it is the very
   * first token of the text and when it is the very last, with no character following
   * it.
   */
  @Test
  void testEmailAtTextStartAndEnd() {
    final List<PiiMention> atStart = extractor.extract("jane@example.com wrote back.");
    Assertions.assertEquals(1, atStart.size());
    Assertions.assertEquals(0, atStart.get(0).span().getStart());
    Assertions.assertEquals(16, atStart.get(0).span().getEnd());
    Assertions.assertEquals("jane@example.com", atStart.get(0).normalized());

    final List<PiiMention> atEnd = extractor.extract("Please write to jane@example.com");
    Assertions.assertEquals(1, atEnd.size());
    Assertions.assertEquals(16, atEnd.get(0).span().getStart());
    Assertions.assertEquals(32, atEnd.get(0).span().getEnd());
    Assertions.assertEquals("jane@example.com", atEnd.get(0).normalized());
  }

  /**
   * Verifies that a sentence period directly after an email address stays outside the
   * reported span, checked against the exact offsets and the exact covered text.
   */
  @Test
  void testEmailTrailingSentencePeriodOutsideSpan() {
    final String text = "Ask jane@example.com.";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(4, mentions.get(0).span().getStart());
    Assertions.assertEquals(20, mentions.get(0).span().getEnd());
    Assertions.assertEquals("jane@example.com", text.substring(4, 20));
  }

  /**
   * Verifies accepted phone formats with exact spans: a compact international number
   * with a plus prefix, a hyphen-separated domestic number, and a parenthesized area
   * code at the very start of the text.
   */
  @Test
  void testPhoneAcceptedFormsWithExactSpans() {
    final List<PiiMention> international = extractor.extract("Call +14155550123 now.");
    Assertions.assertEquals(1, international.size());
    Assertions.assertEquals(5, international.get(0).span().getStart());
    Assertions.assertEquals(17, international.get(0).span().getEnd());
    Assertions.assertEquals("+14155550123", international.get(0).normalized());

    final List<PiiMention> domestic = extractor.extract("Call 415-555-0123 today.");
    Assertions.assertEquals(1, domestic.size());
    Assertions.assertEquals(5, domestic.get(0).span().getStart());
    Assertions.assertEquals(17, domestic.get(0).span().getEnd());
    Assertions.assertEquals("4155550123", domestic.get(0).normalized());

    final List<PiiMention> parenthesized = extractor.extract("(415) 555-0123");
    Assertions.assertEquals(1, parenthesized.size());
    Assertions.assertEquals(0, parenthesized.get(0).span().getStart());
    Assertions.assertEquals(14, parenthesized.get(0).span().getEnd());
    Assertions.assertEquals("4155550123", parenthesized.get(0).normalized());
  }

  /**
   * Verifies rejected phone-like forms: a bare ten-digit run without any formatting, a
   * separated run with only nine digits, and an international candidate whose digits
   * admit no split into an assigned calling code and a plausible national length.
   */
  @Test
  void testPhoneRejectedForms() {
    Assertions.assertTrue(extractor.extract("Order 4155550123 shipped.").isEmpty());
    Assertions.assertTrue(extractor.extract("Ref 415-555-012 code.").isEmpty());
    Assertions.assertTrue(extractor.extract("+1234567 is short").isEmpty());
  }

  /**
   * Verifies that international candidates are judged against the per-calling-code
   * national lengths: a four-digit national number under a code that assigns that
   * length is accepted even though the whole number is short, while a
   * plausible-looking number under an unassigned code and a valid code with a
   * national length no territory assigns are both rejected.
   */
  @Test
  void testPhoneInternationalLengthsFollowCallingCode() {
    final List<PiiMention> shortValid = extractor.extract("Call +683 4002 now.");
    Assertions.assertEquals(1, shortValid.size());
    Assertions.assertEquals(PiiMention.TYPE_PHONE, shortValid.get(0).type());
    Assertions.assertEquals("+6834002", shortValid.get(0).normalized());

    final List<PiiMention> greenland = extractor.extract("Call +299 123456 today.");
    Assertions.assertEquals(1, greenland.size());
    Assertions.assertEquals("+299123456", greenland.get(0).normalized());

    Assertions.assertTrue(extractor.extract("Call +999 1234 5678 now.").isEmpty());
    Assertions.assertTrue(extractor.extract("Call +65 123 456 now.").isEmpty());
  }

  /**
   * Verifies overlap resolution between an IBAN and a phone candidate over the same
   * digits: alone, the formatted tail of the digit groups is reported as a phone
   * number, but once a valid IBAN prefix precedes it, the leftmost candidate, the
   * IBAN, wins and the phone candidate inside it is dropped.
   */
  @Test
  void testOverlappingIbanAndPhoneCandidates() {
    final List<PiiMention> alone = extractor.extract("num 3704 0044 0532 0130 00 end");
    Assertions.assertEquals(1, alone.size());
    Assertions.assertEquals(PiiMention.TYPE_PHONE, alone.get(0).type());
    Assertions.assertEquals(14, alone.get(0).span().getStart());
    Assertions.assertEquals(26, alone.get(0).span().getEnd());
    Assertions.assertEquals("0532013000", alone.get(0).normalized());

    final List<PiiMention> withIban =
        extractor.extract("Wire to DE89 3704 0044 0532 0130 00 today.");
    Assertions.assertEquals(1, withIban.size());
    Assertions.assertEquals(PiiMention.TYPE_IBAN, withIban.get(0).type());
    Assertions.assertEquals(8, withIban.get(0).span().getStart());
    Assertions.assertEquals(35, withIban.get(0).span().getEnd());
    Assertions.assertEquals("DE89370400440532013000", withIban.get(0).normalized());
  }

  /**
   * Verifies that a card followed by a space-separated expiry date is reported at the
   * exact card span: the greedy candidate swallows the expiry digits and fails the
   * Luhn check, so the scanner must back off to the separator boundary instead of
   * dropping the card. Flipping the final card digit makes every prefix fail the
   * checksum, so nothing may be reported in that case.
   */
  @Test
  void testCardFollowedByExpiryBacksOffToTheCard() {
    final String text = "Card 4111111111111111 12/26";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    final PiiMention mention = mentions.get(0);
    Assertions.assertEquals(PiiMention.TYPE_CARD, mention.type());
    Assertions.assertEquals(5, mention.span().getStart());
    Assertions.assertEquals(21, mention.span().getEnd());
    Assertions.assertEquals("4111111111111111", mention.normalized());
    Assertions.assertEquals("4111111111111111", text.substring(5, 21));

    Assertions.assertTrue(extractor.extract("Card 4111111111111112 12/26").isEmpty());
  }

  /**
   * Verifies that number boundaries are judged in code points: a supplementary-plane
   * letter glued to a digit run continues a word exactly like a basic-plane letter,
   * although its trailing surrogate alone reads as a non-letter, so neither a card
   * nor a phone is reported inside such a word, while a space restores the boundary.
   */
  @Test
  void testSupplementaryLetterNeighborBlocksNumberBoundaries() {
    // U+10428, DESERET SMALL LETTER LONG I, a supplementary-plane letter
    final String letter = "\uD801\uDC28";
    Assertions.assertTrue(extractor.extract(letter + "4111111111111111").isEmpty());
    Assertions.assertTrue(extractor.extract("4111111111111111" + letter).isEmpty());
    Assertions.assertTrue(extractor.extract(letter + "555-123-4567").isEmpty());

    final List<PiiMention> spaced = extractor.extract(letter + " 4111111111111111");
    Assertions.assertEquals(1, spaced.size());
    Assertions.assertEquals(PiiMention.TYPE_CARD, spaced.get(0).type());
    // the supplementary letter is two chars, so the card starts at offset 3
    Assertions.assertEquals(3, spaced.get(0).span().getStart());
  }

  /**
   * Verifies the multi-boundary backoff path on a card that itself contains
   * separators: the greedy candidate ends in a trailing three-digit group, forming a
   * nineteen-digit candidate that is in range but fails the Luhn check, and the scan
   * then accepts the sixteen-digit candidate at the previous separator boundary. The
   * card is reported on its own span and the trailing group stays outside it.
   */
  @Test
  void testSeparatedCardBacksOffThroughLuhnFailingPrefix() {
    final String text = "Card 4111 1111 1111 1111 123 on file";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    final PiiMention mention = mentions.get(0);
    Assertions.assertEquals(PiiMention.TYPE_CARD, mention.type());
    Assertions.assertEquals(5, mention.span().getStart());
    Assertions.assertEquals(24, mention.span().getEnd());
    Assertions.assertEquals("4111 1111 1111 1111", text.substring(5, 24));
    Assertions.assertEquals("4111111111111111", mention.normalized());
  }

  /**
   * Verifies that the phone scan backs off at separator boundaries the way the card
   * scan does: a domestic number followed by another separated digit group would make
   * the greedy candidate twelve digits and be rejected as a whole, so the ten-digit
   * candidate at the previous boundary is accepted, and the trailing group is not
   * swallowed and does not suppress the phone.
   */
  @Test
  void testPhoneFollowedBySeparatedDigitGroupBacksOffToThePhone() {
    final String text = "call 555-123-4567 99 times";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    final PiiMention mention = mentions.get(0);
    Assertions.assertEquals(PiiMention.TYPE_PHONE, mention.type());
    Assertions.assertEquals(5, mention.span().getStart());
    Assertions.assertEquals(17, mention.span().getEnd());
    Assertions.assertEquals("555-123-4567", text.substring(5, 17));
    Assertions.assertEquals("5551234567", mention.normalized());
  }

  /**
   * Verifies the backoff on the parenthesized domestic form: the candidate cut at the
   * separator boundary keeps its balanced parentheses and its visible separation, so
   * the phone is found in front of a trailing digit group.
   */
  @Test
  void testParenthesizedPhoneBacksOffBeforeTrailingGroup() {
    final String text = "dial (212) 555-0199 77 now";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    final PiiMention mention = mentions.get(0);
    Assertions.assertEquals(PiiMention.TYPE_PHONE, mention.type());
    Assertions.assertEquals("(212) 555-0199", text.substring(
        mention.span().getStart(), mention.span().getEnd()));
    Assertions.assertEquals("2125550199", mention.normalized());
  }

  /**
   * Verifies that a nineteen-digit card, the maximum accepted length, still matches as
   * a whole, alone and with a space-separated expiry date after it, so prefix backoff
   * never shortens a candidate that is valid at its full length.
   */
  @Test
  void testNineteenDigitCardMatchesWhole() {
    final List<PiiMention> alone = extractor.extract("card 4111111111111111110 on file");
    Assertions.assertEquals(1, alone.size());
    Assertions.assertEquals(PiiMention.TYPE_CARD, alone.get(0).type());
    Assertions.assertEquals(5, alone.get(0).span().getStart());
    Assertions.assertEquals(24, alone.get(0).span().getEnd());
    Assertions.assertEquals("4111111111111111110", alone.get(0).normalized());

    final List<PiiMention> withExpiry = extractor.extract("card 4111111111111111110 12/26");
    Assertions.assertEquals(1, withExpiry.size());
    Assertions.assertEquals(PiiMention.TYPE_CARD, withExpiry.get(0).type());
    Assertions.assertEquals(5, withExpiry.get(0).span().getStart());
    Assertions.assertEquals(24, withExpiry.get(0).span().getEnd());
    Assertions.assertEquals("4111111111111111110", withExpiry.get(0).normalized());
  }

  /**
   * Verifies that two Luhn-valid cards separated by a single space are reported as two
   * separate mentions: backoff accepts the first card at the separator boundary and
   * the scan resumes after it in time to find the second card.
   */
  @Test
  void testTwoSpaceSeparatedCardsBothReported() {
    final String text = "cards 4111111111111111 5500005555555559 on file";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(2, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_CARD, mentions.get(0).type());
    Assertions.assertEquals(6, mentions.get(0).span().getStart());
    Assertions.assertEquals(22, mentions.get(0).span().getEnd());
    Assertions.assertEquals("4111111111111111", mentions.get(0).normalized());
    Assertions.assertEquals(PiiMention.TYPE_CARD, mentions.get(1).type());
    Assertions.assertEquals(23, mentions.get(1).span().getStart());
    Assertions.assertEquals(39, mentions.get(1).span().getEnd());
    Assertions.assertEquals("5500005555555559", mentions.get(1).normalized());
  }

  /**
   * Verifies that empty text and text without any PII both yield an empty result
   * rather than {@code null} or an exception.
   */
  @Test
  void testEmptyAndPiiFreeText() {
    Assertions.assertTrue(extractor.extract("").isEmpty());
    Assertions.assertTrue(
        extractor.extract("The quick brown fox jumps over the lazy dog.").isEmpty());
  }
}
