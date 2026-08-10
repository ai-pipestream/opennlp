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

import java.util.function.Function;

/**
 * Masking defaults per PII type, so that a redaction reveals as little as the type allows
 * while staying as useful as custom is.
 *
 * <p>What is customary differs sharply by type. A payment card is quoted by its last four
 * digits on every receipt, and reconciling a charge is impossible without them, so
 * {@link #forType(String)} keeps them. A secret has no such custom and no readable part:
 * an access key or a token is masked whole, formatting included, because even its shape
 * says which system it opens. In between sit the values whose shape is harmless and
 * whose content is not, an email address or an IP address, where separators stay visible
 * so a reader can see what kind of value was there.</p>
 *
 * <p>These are defaults, not rules. A jurisdiction, a contract, or a threat model may
 * demand more; build the policy directly with {@link MaskPolicy} where it does.</p>
 *
 * @since 3.0.0
 */
public final class MaskPolicies {

  /** The character the defaults mask with. */
  private static final char MASK = '*';

  /** How many trailing digits the payment types keep, the receipt custom. */
  private static final int ACCOUNT_TAIL = 4;

  private MaskPolicies() {
    // This class holds static factories only and is never instantiated.
  }

  /**
   * Returns the default policy for one type, masking with {@code *}.
   *
   * @param type The mention type, for example {@link PiiMention#TYPE_CARD}. Must not be
   *             {@code null}.
   * @return The policy. Never {@code null}. A type this class does not name is masked
   *         whole, the cautious answer for a type whose sensitivity is unknown.
   * @throws IllegalArgumentException Thrown if {@code type} is {@code null}.
   */
  public static MaskPolicy forType(String type) {
    return forType(type, MASK);
  }

  /**
   * Returns the default policy for one type with an explicit mask character.
   *
   * @param type The mention type. Must not be {@code null}.
   * @param mask The replacement character. Must not be a surrogate.
   * @return The policy. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code type} is {@code null} or
   *         {@code mask} is a surrogate character.
   */
  public static MaskPolicy forType(String type, char mask) {
    if (type == null) {
      throw new IllegalArgumentException("type must not be null");
    }
    final MaskPolicy whole = MaskPolicy.of(mask);
    return switch (type) {
      case PiiMention.TYPE_CARD, PiiMention.TYPE_IBAN ->
          whole.keepingFormat().keepingTrailing(ACCOUNT_TAIL);
      case PiiMention.TYPE_EMAIL, PiiMention.TYPE_PHONE, PiiMention.TYPE_IPV4,
          PiiMention.TYPE_IPV6, PiiMention.TYPE_MAC, PiiMention.TYPE_US_SSN,
          PiiMention.TYPE_US_ITIN, PiiMention.TYPE_UK_NHS, PiiMention.TYPE_DE_STEUER_ID,
          PiiMention.TYPE_ABA_ROUTING -> whole.keepingFormat();
      default -> whole;
    };
  }

  /**
   * Returns the type-aware defaults as a function, ready for
   * {@link Masker#mask(opennlp.tools.document.Document, opennlp.tools.document.LayerKey,
   * Function)}.
   *
   * @return The function from a mention to its default policy. Never {@code null}.
   */
  public static Function<PiiMention, MaskPolicy> byType() {
    return byType(MASK);
  }

  /**
   * Returns the type-aware defaults as a function with an explicit mask character.
   *
   * @param mask The replacement character. Must not be a surrogate.
   * @return The function from a mention to its default policy. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code mask} is a surrogate character.
   */
  public static Function<PiiMention, MaskPolicy> byType(char mask) {
    if (Character.isSurrogate(mask)) {
      throw new IllegalArgumentException("mask must not be a surrogate character");
    }
    return mention -> forType(mention.type(), mask);
  }
}
