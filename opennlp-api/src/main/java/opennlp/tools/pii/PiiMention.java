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

import opennlp.tools.util.Span;

/**
 * One mention of personally identifiable information in a text: the {@link Span} it
 * covers in the original text, its type, and a normalized form with formatting removed.
 *
 * <p>The type is an open string so extractors can introduce new types without an API
 * change; the constants on this record name the types the built-in extractor reports.
 * The normalized form is suitable for comparison and lookup: email addresses are
 * lowercased, IBANs keep their uppercase letters and digits with separators removed,
 * and phone and card numbers keep digits only, with a leading {@code +} preserved for
 * phone numbers.</p>
 *
 * @param span The location of the mention in the original text. Must not be
 *             {@code null}.
 * @param type The mention type, for example {@link #TYPE_EMAIL}. Must not be
 *             {@code null} or blank.
 * @param normalized The normalized form. Must not be {@code null} or blank.
 *
 * @since 3.0.0
 */
public record PiiMention(Span span, String type, String normalized) {

  /** An email address. */
  public static final String TYPE_EMAIL = "email";

  /** A phone number. */
  public static final String TYPE_PHONE = "phone";

  /** An International Bank Account Number. */
  public static final String TYPE_IBAN = "iban";

  /** A payment card number. */
  public static final String TYPE_CARD = "card";

  /**
   * Validates the mention.
   *
   * @throws IllegalArgumentException Thrown if {@code span} is {@code null}, or
   *         {@code type} or {@code normalized} is {@code null} or blank.
   */
  public PiiMention {
    if (span == null) {
      throw new IllegalArgumentException("span must not be null");
    }
    if (type == null || type.isBlank()) {
      throw new IllegalArgumentException("type must not be null or blank");
    }
    if (normalized == null || normalized.isBlank()) {
      throw new IllegalArgumentException("normalized must not be null or blank");
    }
  }
}
