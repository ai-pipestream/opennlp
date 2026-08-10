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

/**
 * The country entries of the
 * <a href="https://www.swift.com/standards/data-standards/iban-international-bank-account-number">
 * ISO 13616 IBAN registry</a>: which two-letter country codes issue IBANs and the exact
 * length each country assigns. The mod-97 check alone passes about one in 97 random
 * candidates, so validating the country and its registered length is what keeps
 * arbitrary letter-digit runs from being reported as IBANs.
 */
final class IbanLengths {

  /**
   * The registry as compact 4-character entries, country code then two length digits,
   * sorted by country code.
   */
  private static final String REGISTRY =
      "AD24AE23AL28AT20AZ28BA20BE16BG22BH22BI27BR29BY28CH21CR22CY28CZ24DE22DJ27"
          + "DK18DO28EE20EG29ES24FI18FK18FO18FR27GB22GE22GI23GL18GR27GT28HN28HR21HU28"
          + "IE22IL23IQ23IS26IT27JO30KW30KZ20LB28LC32LI21LT20LU20LV21LY25MC27MD24ME22"
          + "MK19MN20MR27MT31MU30NI28NL18NO15OM23PK24PL28PS29PT25QA29RO24RS22RU33SA24"
          + "SC31SD18SE24SI19SK24SM27SO23ST25SV28TL23TN24TR26UA29VA22VG24XK20";

  private static final int TABLE_SIZE = 26 * 26;

  /** Registered length per country code pair, {@code 0} where no country is registered. */
  private static final byte[] LENGTHS = new byte[TABLE_SIZE];

  static {
    for (int i = 0; i < REGISTRY.length(); i += 4) {
      final int index = (REGISTRY.charAt(i) - 'A') * 26 + (REGISTRY.charAt(i + 1) - 'A');
      LENGTHS[index] =
          (byte) ((REGISTRY.charAt(i + 2) - '0') * 10 + (REGISTRY.charAt(i + 3) - '0'));
    }
  }

  private IbanLengths() {
    // This class holds static lookups only and is never instantiated.
  }

  /**
   * Looks up the registered IBAN length of a country.
   *
   * @param first The first country code letter, {@code A} to {@code Z}.
   * @param second The second country code letter, {@code A} to {@code Z}.
   * @return The length the registry assigns, or {@code 0} if the country is not
   *         registered.
   */
  static int registeredLength(char first, char second) {
    return LENGTHS[(first - 'A') * 26 + (second - 'A')];
  }
}
