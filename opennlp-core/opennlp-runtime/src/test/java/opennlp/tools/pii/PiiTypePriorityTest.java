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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

public class PiiTypePriorityTest {

  private static final List<String> ALL_TYPES = List.of(
      PiiMention.TYPE_JWT,
      PiiMention.TYPE_AWS_ACCESS_KEY,
      PiiMention.TYPE_GITHUB_TOKEN,
      PiiMention.TYPE_URL_CREDENTIAL,
      PiiMention.TYPE_EMAIL,
      PiiMention.TYPE_IBAN,
      PiiMention.TYPE_CARD,
      PiiMention.TYPE_BTC_ADDRESS,
      PiiMention.TYPE_ETH_ADDRESS,
      PiiMention.TYPE_MAC,
      PiiMention.TYPE_IPV6,
      PiiMention.TYPE_IPV4,
      PiiMention.TYPE_US_SSN,
      PiiMention.TYPE_US_ITIN,
      PiiMention.TYPE_UK_NHS,
      PiiMention.TYPE_DE_STEUER_ID,
      PiiMention.TYPE_ABA_ROUTING,
      PiiMention.TYPE_PHONE);

  @Test
  void testEveryNamedTypeHasItsOwnRank() {
    for (int i = 0; i < ALL_TYPES.size(); i++) {
      Assertions.assertEquals(i, PiiTypePriority.rank(ALL_TYPES.get(i)), ALL_TYPES.get(i));
    }
    Assertions.assertEquals(ALL_TYPES.size(), PiiTypePriority.UNRANKED);
  }

  @ParameterizedTest
  @CsvSource({
      "email, phone",
      "iban, card",
      "card, aba-routing",
      "jwt, github-token",
      "aws-access-key, email",
      "ipv6, ipv4",
      "mac, ipv6",
      "btc-address, us-ssn",
      "us-ssn, phone",
      "de-steuer-id, phone"
  })
  void testMoreSpecificTypeRanksFirst(String stronger, String weaker) {
    Assertions.assertTrue(PiiTypePriority.rank(stronger) < PiiTypePriority.rank(weaker),
        stronger + " should outrank " + weaker);
  }

  @ParameterizedTest
  @ValueSource(strings = {"custom", "person", "email ", "EMAIL", "", "  "})
  void testUnknownTypeRanksAfterEveryNamedType(String type) {
    Assertions.assertEquals(PiiTypePriority.UNRANKED, PiiTypePriority.rank(type), type);
    for (final String named : ALL_TYPES) {
      Assertions.assertTrue(PiiTypePriority.rank(named) < PiiTypePriority.rank(type));
    }
  }

  @Test
  void testRejectsNullType() {
    Assertions.assertThrows(IllegalArgumentException.class, () -> PiiTypePriority.rank(null));
  }
}
