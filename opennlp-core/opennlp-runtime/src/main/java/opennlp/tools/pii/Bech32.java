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
 * The <a href="https://github.com/bitcoin/bips/blob/master/bip-0173.mediawiki">BIP-173</a>
 * bech32 and <a href="https://github.com/bitcoin/bips/blob/master/bip-0350.mediawiki">
 * BIP-350</a> bech32m encodings of Bitcoin's segwit addresses: a human-readable prefix, the
 * separator {@code 1}, and a data part whose last six characters are a BCH checksum over
 * the prefix and the data.
 *
 * <p>The checksum spans 30 bits, so an arbitrary run of charset characters is rejected
 * rather than reported. Which of the two constants the checksum must meet depends on the
 * witness version the data part starts with: version zero uses bech32 and every later
 * version uses bech32m, which is what keeps the two encodings apart.</p>
 */
final class Bech32 {

  /**
   * The bech32 charset, in value order, as BIP-173 defines it. Fixed by the encoding rather
   * than by a registry, so it carries no revision and cannot go stale.
   */
  private static final String CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

  private static final int ASCII_RANGE = 128;

  /** The value of each ASCII character in the charset, or {@code -1}. */
  private static final int[] VALUES = new int[ASCII_RANGE];

  /** The generator coefficients of the BCH code. */
  private static final int[] GENERATOR =
      {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};

  /** The residue a bech32 string leaves. */
  private static final int BECH32_RESIDUE = 1;

  /** The residue a bech32m string leaves. */
  private static final int BECH32M_RESIDUE = 0x2bc830a3;

  private static final int CHECKSUM_LENGTH = 6;
  private static final int MAX_WITNESS_VERSION = 16;
  private static final int BITS_PER_CHARACTER = 5;
  private static final int MIN_PROGRAM_BYTES = 2;
  private static final int MAX_PROGRAM_BYTES = 40;
  private static final int MASK = 0x1ffffff;

  static {
    for (int i = 0; i < ASCII_RANGE; i++) {
      VALUES[i] = -1;
    }
    for (int i = 0; i < CHARSET.length(); i++) {
      VALUES[CHARSET.charAt(i)] = i;
    }
  }

  private Bech32() {
    // This class holds static methods only and is never instantiated.
  }

  /**
   * Tests for a character of the bech32 charset in either case.
   *
   * @param c The character.
   * @return {@code true} if the character encodes a bech32 value.
   */
  static boolean isDataChar(char c) {
    return c < ASCII_RANGE && VALUES[Ascii.toLower(c)] >= 0;
  }

  /**
   * Reads the witness version of a bech32 or bech32m address whose checksum holds.
   *
   * @param text The text being scanned.
   * @param start The first character of the data part, that is the character after the
   *              separator.
   * @param end The exclusive end of the data part.
   * @param prefix The human-readable prefix the checksum covers. Must be lowercase.
   * @return The witness version {@code 0} to {@link #MAX_WITNESS_VERSION}, or {@code -1} if
   *         the data part is not a valid address body under either encoding.
   */
  static int checkedWitnessVersion(CharSequence text, int start, int end, String prefix) {
    final int length = end - start;
    if (length <= CHECKSUM_LENGTH) {
      return -1;
    }
    final int[] values = new int[prefix.length() * 2 + 1 + length];
    int at = 0;
    for (int i = 0; i < prefix.length(); i++) {
      values[at++] = prefix.charAt(i) >>> BITS_PER_CHARACTER;
    }
    values[at++] = 0;
    for (int i = 0; i < prefix.length(); i++) {
      values[at++] = prefix.charAt(i) & 0x1F;
    }
    for (int i = start; i < end; i++) {
      final int value = VALUES[Ascii.toLower(text.charAt(i))];
      if (value < 0) {
        return -1;
      }
      values[at++] = value;
    }
    final int version = values[prefix.length() * 2 + 1];
    if (version > MAX_WITNESS_VERSION) {
      return -1;
    }
    final int programBits = (length - CHECKSUM_LENGTH - 1) * BITS_PER_CHARACTER;
    final int programBytes = programBits / 8;
    if (programBytes < MIN_PROGRAM_BYTES || programBytes > MAX_PROGRAM_BYTES) {
      return -1;
    }
    final int residue = version == 0 ? BECH32_RESIDUE : BECH32M_RESIDUE;
    return polymod(values) == residue ? version : -1;
  }

  /**
   * Computes the BCH residue of a value sequence.
   *
   * @param values The expanded prefix followed by the data values.
   * @return The residue.
   */
  private static int polymod(int[] values) {
    int checksum = 1;
    for (final int value : values) {
      final int top = checksum >>> 25;
      checksum = (checksum & MASK) << BITS_PER_CHARACTER ^ value;
      for (int i = 0; i < GENERATOR.length; i++) {
        if ((top >>> i & 1) != 0) {
          checksum ^= GENERATOR[i];
        }
      }
    }
    return checksum;
  }
}
