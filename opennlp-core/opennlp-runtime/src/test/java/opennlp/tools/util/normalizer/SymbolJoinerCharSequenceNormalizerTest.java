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
package opennlp.tools.util.normalizer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class SymbolJoinerCharSequenceNormalizerTest {

  private static final SymbolJoinerCharSequenceNormalizer NORMALIZER =
      SymbolJoinerCharSequenceNormalizer.getInstance();

  @ParameterizedTest
  @CsvSource({
      "&, and",
      "+, plus",
      "@, at",
      "%, percent",
      "§, section",
      "¶, paragraph",
      "°, degree",
      "©, copyright",
      "®, registered",
      "™, trademark"})
  void testWholeTokenSymbolsSpellOut(String symbol, String word) {
    assertEquals(word, NORMALIZER.normalize(symbol).toString());
  }

  @ParameterizedTest
  @ValueSource(strings = {"R&D", "AT&T", "TSR®", "&&", "& ", " &"})
  void testEmbeddedSymbolsAreLeftAlone(String token) {
    assertEquals(token, NORMALIZER.normalize(token).toString());
  }

  @ParameterizedTest
  @ValueSource(strings = {"and", "court", "", "😀"})
  void testNonSymbolTextsReturnUnchangedWithoutCopying(String text) {
    assertSame(text, NORMALIZER.normalize(text));
  }

  @Test
  void testNullIsRejected() {
    assertThrows(IllegalArgumentException.class, () -> NORMALIZER.normalize(null));
  }
}
