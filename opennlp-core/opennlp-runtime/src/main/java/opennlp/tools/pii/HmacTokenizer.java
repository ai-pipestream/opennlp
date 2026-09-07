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
 * Replaces PII mentions with keyed tokens that stay the same across documents.
 *
 * <p>A token contains the type with ASCII letters uppercased, followed by hexadecimal
 * digits from <a href="https://datatracker.ietf.org/doc/html/rfc2104">HMAC</a>-SHA-256.
 * The MAC input is the UTF-8 encoding of the case-sensitive type, a zero byte and the
 * mention's {@link PiiMention#normalized() normalized form}. Types and values must
 * contain well-formed UTF-16. Unpaired surrogates are rejected rather than replaced
 * during encoding.</p>
 *
 * <p>Use at least 32 random key bytes and keep the key separate from the data. Tokens
 * made with the same key are linkable pseudonymous data and require access control.
 * Changing the key changes the tokens. Truncated tokens can collide; use a longer token
 * when comparing large collections.</p>
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

  /** Minimum key length recommended for HMAC-SHA-256. */
  private static final int MINIMUM_KEY_BYTES = 32;

  /** Hexadecimal digits, lowercase, indexed by value. */
  private static final char[] HEX = "0123456789abcdef".toCharArray();

  private final SecretKeySpec key;
  private final int length;

  /**
   * Initializes a tokenizer producing sixteen hexadecimal digits (64 bits) per token.
   *
   * @param key The secret key. Must contain at least 32 bytes. The bytes are copied.
   * @throws IllegalArgumentException Thrown if {@code key} is {@code null} or shorter than
   *         32 bytes.
   */
  public HmacTokenizer(byte[] key) {
    this(key, DEFAULT_LENGTH);
  }

  /**
   * Initializes a tokenizer with an explicit token length. Shorter tokens increase
   * collision risk. The default is sixteen hexadecimal digits.
   *
   * @param key The secret key. Must contain at least 32 bytes. The bytes are copied.
   * @param length The number of hexadecimal digits to show. Must be between {@code 4} and
   *               {@code 64}, the full width of a SHA-256 MAC.
   * @throws IllegalArgumentException Thrown if {@code key} is {@code null} or shorter than
   *         32 bytes, or {@code length} is out of range.
   */
  public HmacTokenizer(byte[] key, int length) {
    if (key == null) {
      throw new IllegalArgumentException("key must not be null");
    }
    if (key.length < MINIMUM_KEY_BYTES) {
      throw new IllegalArgumentException("key must contain at least 32 bytes");
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
   * @param mention The mention. Must not be {@code null}. Its type and normalized value
   *                must contain well-formed UTF-16.
   * @return The token, for example {@code EMAIL-3f2a1c9d7e4b6a20}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code mention} is {@code null}, or its
   *         type or normalized value contains an unpaired surrogate.
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
   *             {@code null} or blank, and must contain well-formed UTF-16.
   * @param value The normalized value. Must not be {@code null} or empty, and must
   *              contain well-formed UTF-16.
   * @return The token. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code type} is {@code null} or blank, or
   *         {@code value} is {@code null} or empty, or either contains an unpaired surrogate.
   */
  public String token(String type, String value) {
    if (type == null || type.isBlank()) {
      throw new IllegalArgumentException("type must not be null or blank");
    }
    if (value == null || value.isEmpty()) {
      throw new IllegalArgumentException("value must not be null or empty");
    }
    if (hasUnpairedSurrogate(type)) {
      throw new IllegalArgumentException("type must not contain unpaired surrogates");
    }
    if (hasUnpairedSurrogate(value)) {
      throw new IllegalArgumentException("value must not contain unpaired surrogates");
    }
    return Ascii.toUpper(type) + '-' + digest(type + '\u0000' + value);
  }

  /**
   * Rewrites a text, replacing each mention with its token.
   *
   * @param text The original text. Must not be {@code null}.
   * @param mentions The mentions to replace, as reported by a {@link PiiExtractor}. Must
   *                 not be {@code null} or contain {@code null}, every span must lie
   *                 within {@code text}, and no two spans may overlap. Mention types and
   *                 normalized values must contain well-formed UTF-16.
   * @return The rewrite. Never {@code null}.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}, a mention is
   *         {@code null}, a span lies outside the text, two spans overlap, or a mention's
   *         type or normalized value contains an unpaired surrogate.
   */
  public PiiRewrite rewrite(CharSequence text, List<PiiMention> mentions) {
    return PiiRewrite.replace(text, mentions, this::token);
  }

  /**
   * Rewrites a document's text, replacing every mention of its {@link PiiAnnotator#PII}
   * layer.
   *
   * @param document The document to rewrite. Must be non-null and have a
   *                 {@link PiiAnnotator#PII} layer with matching annotation and mention
   *                 offsets. Mention types and normalized values must contain well-formed UTF-16.
   * @return The rewrite. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is null, lacks the PII
   *         layer, contains a mention with offsets that differ from its annotation, or a
   *         mention's type or normalized value contains an unpaired surrogate.
   */
  public PiiRewrite rewrite(Document document) {
    final List<PiiMention> mentions = PiiLayer.mentions(document);
    return rewrite(document.text(), mentions);
  }

  /**
   * Checks for UTF-16 code units that do not form a valid surrogate pair.
   *
   * @param value The non-null text to check.
   * @return Whether the text contains an unpaired surrogate.
   */
  private boolean hasUnpairedSurrogate(String value) {
    for (int offset = 0; offset < value.length();) {
      final int codePoint = value.codePointAt(offset);
      if (codePoint >= Character.MIN_SURROGATE && codePoint <= Character.MAX_SURROGATE) {
        return true;
      }
      offset += Character.charCount(codePoint);
    }
    return false;
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
   * Authenticates a message with a separate {@link Mac} instance for each call.
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
