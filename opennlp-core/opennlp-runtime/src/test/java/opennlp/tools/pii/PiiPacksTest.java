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
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

public class PiiPacksTest {

  /** A text carrying one mention of every type the packs recognize. */
  private static final String EVERYTHING = "mail jane@example.com call (555) 123-4567 "
      + "iban DE89 3704 0044 0532 0130 00 card 4111 1111 1111 1111 routing 021000021 "
      + "host 10.1.2.3 peer 2001:db8::1 mac 00:1b:44:11:3a:b7 "
      + "key AKIAIOSFODNN7EXAMPLE url https://u:p@example.com/ "
      + "btc 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa eth 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed "
      + "ssn 123-45-6789 itin 900-70-1234 nhs 943 476 5919 idnr 65929970489";

  private static Stream<Arguments> packs() {
    return Stream.of(
        Arguments.of("payment", (Supplier<PiiExtractor>) PiiPacks::payment),
        Arguments.of("contact", (Supplier<PiiExtractor>) PiiPacks::contact),
        Arguments.of("network", (Supplier<PiiExtractor>) PiiPacks::network),
        Arguments.of("secrets", (Supplier<PiiExtractor>) PiiPacks::secrets),
        Arguments.of("crypto", (Supplier<PiiExtractor>) PiiPacks::crypto),
        Arguments.of("usIdentity", (Supplier<PiiExtractor>) PiiPacks::usIdentity),
        Arguments.of("euIdentity", (Supplier<PiiExtractor>) PiiPacks::euIdentity),
        Arguments.of("allStructured", (Supplier<PiiExtractor>) PiiPacks::allStructured));
  }

  @ParameterizedTest
  @MethodSource("packs")
  void testEveryPackReportsNonOverlappingMentionsInTextOrder(String name,
      Supplier<PiiExtractor> pack) {
    final List<PiiMention> mentions = pack.get().extract(EVERYTHING);

    Assertions.assertFalse(mentions.isEmpty(), name);
    int lastEnd = 0;
    for (final PiiMention mention : mentions) {
      Assertions.assertTrue(mention.span().getStart() >= lastEnd, name + ": " + mention);
      lastEnd = mention.span().getEnd();
    }
  }

  @ParameterizedTest
  @MethodSource("packs")
  void testEveryPackIsFreshAndFindsNothingInPlainText(String name,
      Supplier<PiiExtractor> pack) {
    Assertions.assertNotSame(pack.get(), pack.get(), name);
    Assertions.assertTrue(pack.get().extract("nothing to find in this sentence").isEmpty(),
        name);
  }

  @ParameterizedTest
  @MethodSource("packs")
  void testEveryPackRejectsNullText(String name, Supplier<PiiExtractor> pack) {
    final PiiExtractor extractor = pack.get();
    Assertions.assertThrows(IllegalArgumentException.class, () -> extractor.extract(null),
        name);
  }

  @Test
  void testPaymentPackReportsPaymentTypesOnly() {
    final Set<String> types = typesOf(PiiPacks.payment());

    Assertions.assertEquals(Set.of(PiiMention.TYPE_IBAN, PiiMention.TYPE_CARD,
        PiiMention.TYPE_ABA_ROUTING), types);
  }

  @Test
  void testContactPackReportsContactTypesOnly() {
    Assertions.assertEquals(Set.of(PiiMention.TYPE_EMAIL, PiiMention.TYPE_PHONE),
        typesOf(PiiPacks.contact()));
  }

  @Test
  void testNetworkPackReportsAddressTypesOnly() {
    Assertions.assertEquals(
        Set.of(PiiMention.TYPE_IPV4, PiiMention.TYPE_IPV6, PiiMention.TYPE_MAC),
        typesOf(PiiPacks.network()));
  }

  @Test
  void testSecretsPackReportsCredentialTypesOnly() {
    Assertions.assertEquals(
        Set.of(PiiMention.TYPE_AWS_ACCESS_KEY, PiiMention.TYPE_URL_CREDENTIAL),
        typesOf(PiiPacks.secrets()));
  }

  @Test
  void testCryptoPackReportsWalletTypesOnly() {
    Assertions.assertEquals(
        Set.of(PiiMention.TYPE_BTC_ADDRESS, PiiMention.TYPE_ETH_ADDRESS),
        typesOf(PiiPacks.crypto()));
  }

  @Test
  void testUsIdentityPackReportsUnitedStatesTypesOnly() {
    Assertions.assertEquals(Set.of(PiiMention.TYPE_US_SSN, PiiMention.TYPE_US_ITIN),
        typesOf(PiiPacks.usIdentity()));
  }

  @Test
  void testEuIdentityPackReportsEuropeanTypesOnly() {
    Assertions.assertEquals(Set.of(PiiMention.TYPE_UK_NHS, PiiMention.TYPE_DE_STEUER_ID),
        typesOf(PiiPacks.euIdentity()));
  }

  @Test
  void testAllStructuredPackReportsEveryType() {
    final Set<String> types = typesOf(PiiPacks.allStructured());

    Assertions.assertEquals(Set.of(
        PiiMention.TYPE_EMAIL, PiiMention.TYPE_PHONE, PiiMention.TYPE_IBAN,
        PiiMention.TYPE_CARD, PiiMention.TYPE_ABA_ROUTING, PiiMention.TYPE_IPV4,
        PiiMention.TYPE_IPV6, PiiMention.TYPE_MAC, PiiMention.TYPE_AWS_ACCESS_KEY,
        PiiMention.TYPE_URL_CREDENTIAL, PiiMention.TYPE_BTC_ADDRESS,
        PiiMention.TYPE_ETH_ADDRESS, PiiMention.TYPE_US_SSN, PiiMention.TYPE_US_ITIN,
        PiiMention.TYPE_UK_NHS, PiiMention.TYPE_DE_STEUER_ID), types);
  }

  /**
   * Verifies the promise that matters most: the default extractor never reports a national
   * identifier, a credential, or an address, whatever the text holds.
   */
  @Test
  void testDefaultExtractorReportsOnlyTheFourClassicTypes() {
    final Set<String> types = typesOf(new CursorPiiExtractor());

    Assertions.assertEquals(Set.of(PiiMention.TYPE_EMAIL, PiiMention.TYPE_PHONE,
        PiiMention.TYPE_IBAN, PiiMention.TYPE_CARD), types);
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "ssn 123-45-6789",
      "itin 900-70-1234",
      "idnr 65929970489",
      "routing 021000021",
      "key AKIAIOSFODNN7EXAMPLE",
      "btc 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"})
  void testNarrowPacksDoNotReportWhatTheyDoNotCover(String text) {
    Assertions.assertTrue(PiiPacks.contact().extract(text).isEmpty(), text);
  }

  @Test
  void testPacksCombineIntoOneComposite() {
    final PiiExtractor extractor =
        new CompositePiiExtractor(PiiPacks.contact(), PiiPacks.network());
    final String text = "mail jane@example.com from 10.1.2.3";

    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(List.of(PiiMention.TYPE_EMAIL, PiiMention.TYPE_IPV4),
        mentions.stream().map(PiiMention::type).toList());
  }

  /**
   * Verifies that widening the search never loses a span: every stretch of text a narrow pack
   * flags is still flagged by the widest pack. The type may differ, since a wider search can
   * claim the same characters for a more specific type, an NHS number rather than the phone
   * number it also looks like.
   */
  @ParameterizedTest
  @MethodSource("packs")
  void testAllStructuredFlagsEverySpanTheNarrowPacksFlag(String name,
      Supplier<PiiExtractor> pack) {
    final List<PiiMention> wide = PiiPacks.allStructured().extract(EVERYTHING);

    for (final PiiMention mention : pack.get().extract(EVERYTHING)) {
      final boolean covered = wide.stream().anyMatch(
          other -> other.span().getStart() < mention.span().getEnd()
              && mention.span().getStart() < other.span().getEnd());
      Assertions.assertTrue(covered, name + ": " + mention);
    }
  }

  /**
   * Verifies the reclassification the previous test allows for, on the fixture that provokes
   * it: a validly grouped NHS number is a validly formatted phone number too, and the wide
   * pack reports the more specific type.
   */
  @Test
  void testWidePackPrefersTheMoreSpecificTypeOnASharedSpan() {
    final String text = "record 943 476 5919 today";

    Assertions.assertEquals(PiiMention.TYPE_PHONE,
        PiiPacks.contact().extract(text).get(0).type());
    Assertions.assertEquals(PiiMention.TYPE_UK_NHS,
        PiiPacks.allStructured().extract(text).get(0).type());
  }

  private static Set<String> typesOf(PiiExtractor extractor) {
    return extractor.extract(EVERYTHING).stream().map(PiiMention::type)
        .collect(java.util.stream.Collectors.toUnmodifiableSet());
  }
}
