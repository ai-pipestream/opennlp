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

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * The <a href="https://en.bitcoin.it/wiki/Base58Check_encoding">Base58Check</a> encoding
 * of Bitcoin's legacy addresses: base 58 without the characters that look alike, over a
 * payload whose last four bytes are the leading four bytes of the double SHA-256 of the
 * rest.
 *
 * <p>The checksum is what makes an address safe to detect. Four checksum bytes over a
 * 21-byte payload leave about one candidate in four thousand million passing by chance, so
 * an arbitrary run of base 58 characters of the right length is rejected rather than
 * reported.</p>
 */
final class Base58Check {

  /**
   * The base 58 alphabet: no zero, capital O, capital I, or lowercase l. Fixed by the
   * encoding rather than by a registry, so unlike the tables this package copies from
   * registrars it carries no revision and cannot go stale.
   */
  private static final String ALPHABET =
      "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

  private static final int RADIX = 58;
  private static final int ASCII_RANGE = 128;

  /** The value of each ASCII character in the alphabet, or {@code -1}. */
  private static final int[] VALUES = new int[ASCII_RANGE];

  /** The payload of an address: one version byte, a 20-byte hash, and four check bytes. */
  static final int PAYLOAD_LENGTH = 25;

  private static final int CHECKSUM_LENGTH = 4;

  private static final int SHA256_LENGTH = 32;

  static {
    for (int i = 0; i < ASCII_RANGE; i++) {
      VALUES[i] = -1;
    }
    for (int i = 0; i < ALPHABET.length(); i++) {
      VALUES[ALPHABET.charAt(i)] = i;
    }
  }

  private Base58Check() {
    // This class holds static methods only and is never instantiated.
  }

  /**
   * Tests for a character of the base 58 alphabet.
   *
   * @param c The character.
   * @return {@code true} if the character encodes a base 58 digit.
   */
  static boolean isBase58Char(char c) {
    return c < ASCII_RANGE && VALUES[c] >= 0;
  }

  /**
   * Reads the version byte of a Base58Check payload whose checksum holds.
   *
   * @param text The text being scanned.
   * @param start The first character of the candidate.
   * @param end The exclusive end of the candidate.
   * @return The version byte as an unsigned value, or {@code -1} if the candidate does not
   *         decode to a {@link #PAYLOAD_LENGTH}-byte payload with a valid checksum.
   */
  static int checkedVersion(CharSequence text, int start, int end) {
    final byte[] payload = decode(text, start, end);
    if (payload == null || payload.length != PAYLOAD_LENGTH) {
      return -1;
    }
    final byte[] digest =
        sha256(sha256(payload, PAYLOAD_LENGTH - CHECKSUM_LENGTH), SHA256_LENGTH);
    for (int i = 0; i < CHECKSUM_LENGTH; i++) {
      if (digest[i] != payload[PAYLOAD_LENGTH - CHECKSUM_LENGTH + i]) {
        return -1;
      }
    }
    return payload[0] & 0xFF;
  }

  /**
   * Decodes base 58 characters to bytes, most significant byte first.
   *
   * @param text The text being scanned.
   * @param start The first character to decode.
   * @param end The exclusive end of the characters to decode.
   * @return The decoded bytes, or {@code null} if a character is not a base 58 digit.
   */
  private static byte[] decode(CharSequence text, int start, int end) {
    // Base 58 carries less than eight bits per character, so the output never grows.
    final byte[] reversed = new byte[end - start + 1];
    int length = 0;
    for (int i = start; i < end; i++) {
      final char c = text.charAt(i);
      if (!isBase58Char(c)) {
        return null;
      }
      int carry = VALUES[c];
      for (int j = 0; j < length; j++) {
        carry += (reversed[j] & 0xFF) * RADIX;
        reversed[j] = (byte) carry;
        carry >>>= 8;
      }
      while (carry > 0) {
        reversed[length++] = (byte) carry;
        carry >>>= 8;
      }
    }
    // A leading alphabet zero, that is the character 1, stands for one zero byte.
    for (int i = start; i < end && text.charAt(i) == ALPHABET.charAt(0); i++) {
      reversed[length++] = 0;
    }
    final byte[] decoded = new byte[length];
    for (int i = 0; i < length; i++) {
      decoded[i] = reversed[length - 1 - i];
    }
    return decoded;
  }

  /**
   * Hashes the leading bytes of an array with SHA-256.
   *
   * @param data The bytes to hash.
   * @param length The number of leading bytes to hash.
   * @return The digest. Never {@code null}.
   */
  private static byte[] sha256(byte[] data, int length) {
    try {
      final MessageDigest digest = MessageDigest.getInstance("SHA-256");
      digest.update(data, 0, length);
      return digest.digest();
    } catch (NoSuchAlgorithmException e) {
      // SHA-256 is guaranteed to be available on every conformant JRE.
      throw new IllegalStateException("SHA-256 not available", e);
    }
  }
}
