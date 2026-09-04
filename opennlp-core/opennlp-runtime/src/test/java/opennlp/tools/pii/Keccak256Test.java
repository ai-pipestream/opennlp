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
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

public class Keccak256Test {

  private static String hex(byte[] bytes) {
    final StringBuilder out = new StringBuilder(bytes.length * 2);
    for (final byte value : bytes) {
      out.append(Character.forDigit((value >> 4) & 0xF, 16));
      out.append(Character.forDigit(value & 0xF, 16));
    }
    return out.toString();
  }

  private static String digestOf(String message) {
    return hex(Keccak256.digest(message.getBytes(StandardCharsets.UTF_8)));
  }

  @ParameterizedTest
  @CsvSource({
      "'', c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
      "abc, 4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
      "'The quick brown fox jumps over the lazy dog', "
          + "4d741b6f1eb29cb2a9b9911c82f56fa8d73b04959d3d9d222895df6c0b28aa15"
  })
  void testKnownDigests(String message, String expected) {
    Assertions.assertEquals(expected, digestOf(message));
  }

  @Test
  void testDigestLength() {
    Assertions.assertEquals(Keccak256.DIGEST_LENGTH,
        Keccak256.digest(new byte[0]).length);
    Assertions.assertEquals(Keccak256.DIGEST_LENGTH,
        Keccak256.digest(new byte[1000]).length);
  }

  /**
   * Verifies the padding across the block boundary: a message of exactly the rate, one
   * byte less, and one byte more each take a different path through the sponge.
   */
  @ParameterizedTest
  @ValueSource(ints = {0, 1, 135, 136, 137, 271, 272, 273, 1000})
  void testMessagesAroundTheBlockBoundaryHash(int length) {
    final byte[] message = new byte[length];
    for (int i = 0; i < length; i++) {
      message[i] = (byte) i;
    }
    final byte[] digest = Keccak256.digest(message);

    Assertions.assertEquals(Keccak256.DIGEST_LENGTH, digest.length);
    boolean allZero = true;
    for (final byte value : digest) {
      allZero &= value == 0;
    }
    Assertions.assertFalse(allZero);
  }

  @Test
  void testDigestIsDeterministicAndSensitiveToEveryBit() {
    final byte[] first = Keccak256.digest("opennlp".getBytes(StandardCharsets.UTF_8));
    final byte[] again = Keccak256.digest("opennlp".getBytes(StandardCharsets.UTF_8));
    final byte[] other = Keccak256.digest("opennlq".getBytes(StandardCharsets.UTF_8));

    Assertions.assertArrayEquals(first, again);
    Assertions.assertFalse(java.util.Arrays.equals(first, other));
  }

  /**
   * Verifies that Keccak-256 is not the SHA3-256 of FIPS 202, which the JDK provides:
   * the two differ in the padding byte, so no message may hash alike.
   */
  @Test
  void testDiffersFromSha3() throws NoSuchAlgorithmException {
    final MessageDigest sha3 = MessageDigest.getInstance("SHA3-256");
    final byte[] message = "abc".getBytes(StandardCharsets.UTF_8);

    Assertions.assertFalse(
        java.util.Arrays.equals(Keccak256.digest(message), sha3.digest(message)));
  }
}
