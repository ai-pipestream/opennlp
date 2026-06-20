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
package opennlp.dl;

import java.util.List;

import org.junit.jupiter.api.Test;

import opennlp.tools.util.normalizer.NormalizedText;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Model-free tests for {@link AbstractDL#whitespaceChunks(String, int, int)}, the shared
 * tokenize-and-chunk seam used by both {@code NameFinderDL} and {@code DocumentCategorizerDL}.
 */
public class AbstractDLChunkingTest {

  @Test
  void testSplitsOnUnicodeWhitespaceNotJustAscii() {
    // A no-break space (U+00A0) and an ideographic space (U+3000) are not matched by Java's \s
    // but must still separate tokens; the chunk is rejoined with single ASCII spaces.
    final String nbsp = new String(Character.toChars(0x00A0));
    final String ideographic = new String(Character.toChars(0x3000));
    assertEquals(List.of("alpha beta gamma"),
        AbstractDL.whitespaceChunks("alpha" + nbsp + "beta" + ideographic + "gamma", 100, 0));
  }

  @Test
  void testDropsEmptyTokensFromLeadingTrailingAndRepeatedWhitespace() {
    // Unlike split("\\s+"), the Unicode-aware split yields no empty leading or trailing tokens.
    assertEquals(List.of("a b c"), AbstractDL.whitespaceChunks("  a   b\tc  ", 100, 0));
  }

  @Test
  void testAppliesChunkSizeWithoutOverlap() {
    assertEquals(List.of("a b", "c d"), AbstractDL.whitespaceChunks("a b c d", 2, 0));
  }

  @Test
  void testAppliesChunkOverlap() {
    assertEquals(List.of("a b", "b c", "c d"), AbstractDL.whitespaceChunks("a b c d", 2, 1));
  }

  @Test
  void testEmptyTextYieldsNoChunks() {
    assertEquals(List.of(), AbstractDL.whitespaceChunks("", 100, 0));
  }

  @Test
  void testNormalizeInputIsOptInAndOffsetPreserving() {
    final String nbsp = new String(Character.toChars(0x00A0));
    final String emDash = new String(Character.toChars(0x2014));
    final String input = "a" + nbsp + "b" + emDash + "c";

    // Off by default: unchanged.
    assertEquals(input, AbstractDL.normalizeInput(input, false, false));

    // Whitespace only: the no-break space becomes a space, and the length is preserved.
    final String ws = AbstractDL.normalizeInput(input, true, false);
    assertEquals("a b" + emDash + "c", ws);
    assertEquals(input.length(), ws.length());

    // Dashes only: the em dash becomes an ASCII hyphen.
    assertEquals("a" + nbsp + "b-c", AbstractDL.normalizeInput(input, false, true));

    // Both.
    assertEquals("a b-c", AbstractDL.normalizeInput(input, true, true));
  }

  @Test
  void testNormalizeInputMappedMapsOffsetsBackAcrossSupplementaryDash() {
    // A supplementary dash (Yezidi hyphen, U+10EAD, two UTF-16 units) folds to one ASCII hyphen, so
    // the folded text is one unit shorter. The offset map keeps positions aligned to the original.
    final String yezidi = new String(Character.toChars(0x10EAD));
    final String input = "a" + yezidi + "b"; // a(0), dash(1,2), b(3) -> length 4
    final NormalizedText nt = AbstractDL.normalizeInputMapped(input, false, true);
    assertEquals("a-b", nt.normalized()); // a(0), -(1), b(2) -> length 3
    assertEquals(0, nt.toOriginalOffset(0)); // 'a'
    assertEquals(1, nt.toOriginalOffset(1)); // hyphen maps back to the dash start
    assertEquals(3, nt.toOriginalOffset(2)); // 'b' stays at original offset 3, not shifted to 2
    assertEquals(4, nt.toOriginalOffset(3)); // end sentinel == original length
  }

  @Test
  void testNormalizeInputMappedIsIdentityWhenLengthIsPreserved() {
    // Whitespace folding does not change length, so the offset map is the identity.
    final NormalizedText nt = AbstractDL.normalizeInputMapped("a\tb", true, false);
    assertEquals("a b", nt.normalized());
    for (int i = 0; i <= 3; i++) {
      assertEquals(i, nt.toOriginalOffset(i));
    }
  }
}
