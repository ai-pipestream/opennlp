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

import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.List;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

import opennlp.tools.document.Document;

/**
 * Replaces PII mentions with keyed tokens that stay the same across documents, so that a
 * corpus can be analysed by who or what recurs in it without holding any of the values.
 *
 * <p>A token is the type followed by the leading hexadecimal digits of the
 * <a href="https://datatracker.ietf.org/doc/html/rfc2104">HMAC</a>-SHA-256 of the mention's
 * {@link PiiMention#normalized() normalized form} under the key, for example
 * {@code EMAIL-3f2a1c9d7e4b6a20}. The same address in a thousand documents tokenizes identically,
 * which {@link Pseudonymizer} deliberately does not do, and two different addresses
 * practically never collide.</p>
 *
 * <p>The key is what makes this safe. A plain digest of a low-entropy value can be reversed
 * by hashing every candidate, and there are only so many card numbers or Social Security
 * numbers; with a secret key that attack needs the key. Treat the key as the secret it is:
 * keep it out of the data it protects, and know that rotating it invalidates every token
 * derived from the old one.</p>
 *
 * <p>Tokens rarely have the length of the values they replace, so offsets move; see
 * {@link PiiRewrite} for mapping annotations onto the rewritten text.</p>
 *
 * <p>Instances are immutable and safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class HmacTokenizer {

  /** The MAC algorithm, required of every Java platform. */
  private static final String ALGORITHM = "HmacSha256";

  /** How many hexadecimal digits of the MAC a token shows unless asked otherwise. */
  private static final int DEFAULT_LENGTH = 16;

  /** Hexadecimal digits, lowercase, indexed by value. */
  private static final char[] HEX = "0123456789abcdef".toCharArray();

  private final SecretKeySpec key;
  private final int length;

  /**
   * Initializes a tokenizer producing tokens with sixteen hexadecimal digits. The 64-bit
   * token space keeps accidental collisions rare while retaining readable labels.
   *
   * @param key The secret key. Must not be {@code null} or empty. The bytes are copied.
   * @throws IllegalArgumentException Thrown if {@code key} is {@code null} or empty.
   */
  public HmacTokenizer(byte[] key) {
    this(key, DEFAULT_LENGTH);
  }

  /**
   * Initializes a tokenizer with an explicit token length. Longer tokens make collisions
   * rarer at the cost of readability: each digit divides the collision probability by
   * sixteen. Lengths below the 16-digit default are supported for compatibility and compact
   * displays, but materially increase collision risk in large data sets.
   *
   * @param key The secret key. Must not be {@code null} or empty. The bytes are copied.
   * @param length The number of hexadecimal digits to show. Must be between {@code 4} and
   *               {@code 64}, the full width of a SHA-256 MAC.
   * @throws IllegalArgumentException Thrown if {@code key} is {@code null} or empty, or
   *         {@code length} is out of range.
   */
  public HmacTokenizer(byte[] key, int length) {
    if (key == null || key.length == 0) {
      throw new IllegalArgumentException("key must not be null or empty");
    }
    if (length < 4 || length > 64) {
      throw new IllegalArgumentException("length must be between 4 and 64: " + length);
    }
    this.key = new SecretKeySpec(key.clone(), ALGORITHM);
    this.length = length;
  }

  /**
   * Tokenizes one mention.
   *
   * @param mention The mention. Must not be {@code null}.
   * @return The token, for example {@code EMAIL-3f2a1c9d7e4b6a20}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code mention} is {@code null}.
   */
  public String token(PiiMention mention) {
    if (mention == null) {
      throw new IllegalArgumentException("mention must not be null");
    }
    return token(mention.type(), mention.normalized());
  }

  /**
   * Tokenizes a value of a given type, for looking up which token a known value produced.
   *
   * <p>The value is expected in the normalized form the extractors report, since that is
   * what {@link #token(PiiMention)} tokenizes: an unformatted card number rather than a
   * grouped one. The type takes part in the MAC, so one value under two types yields two
   * unrelated tokens.</p>
   *
   * @param type The mention type, for example {@link PiiMention#TYPE_EMAIL}. Must not be
   *             {@code null} or blank.
   * @param value The normalized value. Must not be {@code null} or empty.
   * @return The token. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code type} is {@code null} or blank, or
   *         {@code value} is {@code null} or empty.
   */
  public String token(String type, String value) {
    if (type == null || type.isBlank()) {
      throw new IllegalArgumentException("type must not be null or blank");
    }
    if (value == null || value.isEmpty()) {
      throw new IllegalArgumentException("value must not be null or empty");
    }
    return Ascii.toUpper(type) + '-' + digest(type + '\u0000' + value);
  }

  /**
   * Rewrites a text, replacing each mention with its token.
   *
   * @param text The original text. Must not be {@code null}.
   * @param mentions The mentions to replace, as reported by a {@link PiiExtractor}. Must
   *                 not be {@code null} or contain {@code null}, every span must lie
   *                 within {@code text}, and no two spans may overlap.
   * @return The rewrite. Never {@code null}.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}, a mention is
   *         {@code null}, a span lies outside the text, or two spans overlap.
   */
  public PiiRewrite rewrite(CharSequence text, List<PiiMention> mentions) {
    return PiiRewrite.replace(text, mentions, this::token);
  }

  /**
   * Rewrites a document's text, replacing every mention of its {@link PiiAnnotator#PII}
   * layer.
   *
   * @param document The document to rewrite. Must not be {@code null} and must carry the
   *                 {@link PiiAnnotator#PII} layer.
   * @return The rewrite. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null} or does
   *         not carry the PII layer.
   */
  public PiiRewrite rewrite(Document document) {
    final List<PiiMention> mentions = PiiLayer.mentions(document);
    return rewrite(document.text(), mentions);
  }

  /**
   * Computes the leading hexadecimal digits of the MAC of a message.
   *
   * @param message The message to authenticate.
   * @return The digits, {@link #length} of them.
   */
  private String digest(String message) {
    final byte[] mac = mac(message.getBytes(StandardCharsets.UTF_8));
    final StringBuilder hex = new StringBuilder(length);
    for (int i = 0; hex.length() < length; i++) {
      hex.append(HEX[(mac[i] >> 4) & 0xf]);
      if (hex.length() < length) {
        hex.append(HEX[mac[i] & 0xf]);
      }
    }
    return hex.toString();
  }

  /**
   * Authenticates a message under this tokenizer's key. A {@link Mac} carries the state of
   * one computation and is not safe to share, so each call gets its own.
   *
   * @param message The message to authenticate.
   * @return The MAC, thirty two bytes.
   */
  private byte[] mac(byte[] message) {
    try {
      final Mac mac = Mac.getInstance(ALGORITHM);
      mac.init(key);
      return mac.doFinal(message);
    } catch (GeneralSecurityException e) {
      throw new IllegalStateException(ALGORITHM + " is required of every Java platform", e);
    }
  }
}
