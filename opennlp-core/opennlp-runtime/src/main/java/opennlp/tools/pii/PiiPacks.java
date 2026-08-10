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

import java.util.Set;

/**
 * Ready-made {@link PiiExtractor} combinations, one per kind of concern a caller usually
 * has: payment data, contact data, network addresses, credentials, wallet addresses, and
 * national and device identifiers.
 *
 * <p>A pack saves the caller from naming individual extractors and type sets, and it makes
 * the choice explicit in the code that reads the text: {@code PiiPacks.payment()} says what
 * is being looked for where a bare {@code new CursorPiiExtractor()} would not.</p>
 *
 * <p>Every pack returns a new extractor, and every returned extractor is stateless and safe
 * to share between threads, so a caller may keep one in a static field. Combine packs with
 * {@link CompositePiiExtractor}; exact-span ties use {@link PiiTypePriority}, then pack
 * order only if both types have the same priority.</p>
 *
 * <p>Nothing here changes what the default {@link CursorPiiExtractor} reports. The national
 * packs in particular are opt-in by construction: only {@link #usIdentity()},
 * {@link #euIdentity()}, {@link #caIdentity()}, and {@link #allStructured()} ever report
 * a national identifier.</p>
 *
 * @since 3.0.0
 */
public final class PiiPacks {

  private PiiPacks() {
    // This class holds static factories only and is never instantiated.
  }

  /**
   * Payment data: payment card numbers, IBANs, and ABA routing numbers.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor payment() {
    return new CompositePiiExtractor(
        new CursorPiiExtractor(Set.of(PiiMention.TYPE_IBAN, PiiMention.TYPE_CARD)),
        new BankingPiiExtractor());
  }

  /**
   * Contact data: email addresses and phone numbers.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor contact() {
    return new CursorPiiExtractor(Set.of(PiiMention.TYPE_EMAIL, PiiMention.TYPE_PHONE));
  }

  /**
   * Network addresses: IPv4, IPv6, and MAC addresses.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor network() {
    return new NetworkPiiExtractor();
  }

  /**
   * Credentials: AWS access keys, GitHub tokens, JSON Web Tokens, and credentials embedded
   * in a URL.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor secrets() {
    return new SecretsPiiExtractor();
  }

  /**
   * Wallet addresses: Bitcoin and Ethereum.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor crypto() {
    return new CryptoPiiExtractor();
  }

  /**
   * United States national identifiers: Social Security numbers and Individual Taxpayer
   * Identification Numbers.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor usIdentity() {
    return new UsIdentityPiiExtractor();
  }

  /**
   * European national identifiers: United Kingdom NHS numbers and German tax identification
   * numbers.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor euIdentity() {
    return new EuIdentityPiiExtractor();
  }

  /**
   * Canadian national identifiers: context-labeled Social Insurance Numbers.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor caIdentity() {
    return new CaIdentityPiiExtractor();
  }

  /**
   * Device identifiers: context-labeled International Mobile Equipment Identities.
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor device() {
    return new DevicePiiExtractor();
  }

  /**
   * Every structured type this package recognizes, that is every pack at once.
   *
   * <p>This is the widest recognition available and therefore the one with the most false
   * positives: the weakly evidenced types, a routing number and an NHS number above all, are
   * in it. Prefer the narrower packs where the kind of data to find is known.</p>
   *
   * @return A new extractor. Never {@code null}.
   */
  public static PiiExtractor allStructured() {
    return new CompositePiiExtractor(
        new CursorPiiExtractor(),
        new SecretsPiiExtractor(),
        new CryptoPiiExtractor(),
        new NetworkPiiExtractor(),
        new UsIdentityPiiExtractor(),
        new EuIdentityPiiExtractor(),
        new CaIdentityPiiExtractor(),
        new DevicePiiExtractor(),
        new BankingPiiExtractor());
  }
}
