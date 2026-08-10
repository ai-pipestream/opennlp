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

import java.util.HashMap;
import java.util.Map;

/**
 * The tie-break order over PII types, used when two candidates cover the very same span
 * and neither the leftmost nor the longest rule can separate them.
 *
 * <p>A lower rank wins. The order puts the types that carry their own evidence first,
 * that is types anchored by a fixed prefix or protected by a checksum over a long
 * candidate, and the types that are merely a run of digits last, because a digit run is
 * the form most easily produced by chance:</p>
 *
 * <ol>
 *   <li>{@link PiiMention#TYPE_JWT}, {@link PiiMention#TYPE_AWS_ACCESS_KEY},
 *   {@link PiiMention#TYPE_GITHUB_TOKEN}, {@link PiiMention#TYPE_URL_CREDENTIAL}:
 *   prefix-anchored secrets.</li>
 *   <li>{@link PiiMention#TYPE_EMAIL}, {@link PiiMention#TYPE_IBAN},
 *   {@link PiiMention#TYPE_CARD}: the classic types, in the order the default extractor
 *   uses.</li>
 *   <li>{@link PiiMention#TYPE_BTC_ADDRESS}, {@link PiiMention#TYPE_ETH_ADDRESS}:
 *   checksummed wallet addresses.</li>
 *   <li>{@link PiiMention#TYPE_MAC}, {@link PiiMention#TYPE_IPV6},
 *   {@link PiiMention#TYPE_IPV4}: network addresses, the more constrained form
 *   first.</li>
 *   <li>{@link PiiMention#TYPE_US_SSN}, {@link PiiMention#TYPE_US_ITIN},
 *   {@link PiiMention#TYPE_UK_NHS}, {@link PiiMention#TYPE_DE_STEUER_ID},
 *   {@link PiiMention#TYPE_ABA_ROUTING}, {@link PiiMention#TYPE_PHONE}: digit runs.</li>
 * </ol>
 *
 * <p>A type this class does not name ranks after every named one, so a custom extractor
 * never displaces a built-in type on an exact-span tie.</p>
 *
 * @since 3.0.0
 */
public final class PiiTypePriority {

  /**
   * The types in rank order; the index in this array is the rank.
   */
  private static final String[] ORDER = {
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
      PiiMention.TYPE_PHONE,
  };

  /** The rank of a type this class does not name. */
  public static final int UNRANKED = ORDER.length;

  private static final Map<String, Integer> RANKS = buildRanks();

  private PiiTypePriority() {
    // This class holds the lookup only and is never instantiated.
  }

  private static Map<String, Integer> buildRanks() {
    final Map<String, Integer> ranks = HashMap.newHashMap(ORDER.length);
    for (int i = 0; i < ORDER.length; i++) {
      ranks.put(ORDER[i], i);
    }
    return Map.copyOf(ranks);
  }

  /**
   * Looks up the tie-break rank of a type.
   *
   * @param type The mention type, for example {@link PiiMention#TYPE_EMAIL}. Must not be
   *             {@code null}.
   * @return The rank, lower being the more specific type, or {@link #UNRANKED} for a type
   *         this class does not name.
   * @throws IllegalArgumentException Thrown if {@code type} is {@code null}.
   */
  public static int rank(String type) {
    if (type == null) {
      throw new IllegalArgumentException("type must not be null");
    }
    return RANKS.getOrDefault(type, UNRANKED);
  }
}
