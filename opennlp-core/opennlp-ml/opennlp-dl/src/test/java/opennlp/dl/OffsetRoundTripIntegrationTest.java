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

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import opennlp.tools.tokenize.uax29.WordTokenizer;
import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.CharClass;
import opennlp.tools.util.normalizer.NormalizedText;
import opennlp.tools.util.normalizer.OffsetMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cross-discipline integration test for the OPENNLP-1850 offset guarantee. It lives in the DL
 * module because that is the only module whose classpath sees every discipline at once: the offset
 * primitives ({@link OffsetMap}, {@link NormalizedText}) from opennlp-api, the normalizer
 * ({@link CharClass}) and the UAX #29 tokenizer ({@link WordTokenizer}) from opennlp-runtime, and
 * the deep-learning offset path ({@link AbstractDL#normalizeInputMapped}) from opennlp-dl.
 *
 * <p>It asserts the one guarantee no single-module test can make: a position discovered after
 * length-changing normalization still maps back to the correct character offset in the untouched
 * original, end to end. The DL inference itself needs a model, but the offset contract
 * {@link opennlp.dl.namefinder.NameFinderDL#findInOriginal} relies on is exercised here without one
 * by driving the same {@code normalizeInputMapped} folding and mapping spans back the same way.</p>
 */
public class OffsetRoundTripIntegrationTest {

  private static final int NBSP = 0x00A0;          // White_Space, BMP
  private static final int IDEOGRAPHIC_SPACE = 0x3000; // White_Space, BMP
  private static final int YEZIDI_HYPHEN = 0x10EAD;    // Dash, supplementary (two chars)

  private static String s(int... cps) {
    final StringBuilder sb = new StringBuilder();
    for (final int cp : cps) {
      sb.appendCodePoint(cp);
    }
    return sb.toString();
  }

  // Maps a stage-2 (final) offset back to the original by threading it through both stage maps.
  private static int composeToOriginal(OffsetMap stage2ToStage1, OffsetMap stage1ToOriginal,
                                       int stage2Offset) {
    return stage1ToOriginal.toOriginalOffset(stage2ToStage1.toOriginalOffset(stage2Offset));
  }

  @Test
  void normalizeThenTokenizeMapsEveryTokenBackToItsOriginalText() {
    // "foo" then two Unicode spaces, "bar", a space, a supplementary dash, a space, "baz". Two
    // independent length changes: the whitespace run collapses (3 chars -> 1) and the supplementary
    // dash folds (2 chars -> 1). The words themselves are untouched, so each must map back exactly.
    final String original = "foo" + s(NBSP, IDEOGRAPHIC_SPACE) + "bar"
        + " " + s(YEZIDI_HYPHEN) + " " + "baz";

    // Stage 1: collapse whitespace runs (length-changing), keeping the offset map to the original.
    final NormalizedText afterWhitespace = CharClass.whitespace().collapseMapped(original);
    // Stage 2: fold dashes on stage 1's output (the supplementary dash shrinks two chars to one).
    final NormalizedText afterDashes =
        CharClass.dashes().normalizeMapped(afterWhitespace.normalized());

    final OffsetMap stage1ToOriginal = afterWhitespace.offsets();
    final OffsetMap stage2ToStage1 = afterDashes.offsets();
    final String normalized = afterDashes.normalized();

    final List<String> mappedBack = new ArrayList<>();
    int previousEnd = 0;
    for (final Span token : new WordTokenizer().tokenizeSpans(normalized)) {
      // Compose the two stage maps by hand (there is no andThen yet): stage2 -> stage1 -> original.
      final int originalStart = composeToOriginal(stage2ToStage1, stage1ToOriginal, token.getStart());
      final int originalEnd = composeToOriginal(stage2ToStage1, stage1ToOriginal, token.getEnd());
      assertTrue(originalStart >= previousEnd && originalEnd <= original.length()
          && originalStart < originalEnd, "mapped span out of order or range");
      previousEnd = originalEnd;
      mappedBack.add(original.substring(originalStart, originalEnd));
    }

    // Despite both folds, every word maps back to its exact original substring.
    assertEquals(List.of("foo", "bar", "baz"), mappedBack);
  }

  @Test
  void dlNormalizationPathMapsSpansBackAcrossASupplementaryDash() {
    // Drive the exact folding NameFinderDL.findInOriginal uses (whitespace + dash), then map a span
    // discovered in normalized coordinates back to the original the same way findInOriginal does.
    final String original = "Rio" + s(YEZIDI_HYPHEN) + "Niteroi";
    final NormalizedText normalized = AbstractDL.normalizeInputMapped(original, true, true);
    assertEquals("Rio-Niteroi", normalized.normalized());

    // Simulate the model locating "Niteroi" in the normalized text; map it back to the original.
    final int start = normalized.normalized().indexOf("Niteroi");
    final int end = start + "Niteroi".length();
    final OffsetMap offsets = normalized.offsets();
    final int originalStart = offsets.toOriginalOffset(start);
    final int originalEnd = offsets.toOriginalOffset(end);

    // The supplementary dash was two chars in the original but one in the normalized form; the map
    // absorbs that shift so the reported span still covers the original "Niteroi" exactly.
    assertEquals("Niteroi", original.substring(originalStart, originalEnd));
    assertEquals(original.length(), normalized.offsets().originalLength());
  }

  @Test
  void dlAndNormalizerAgreeOnTheSupplementaryDashContraction() {
    // The DL path and a direct CharClass dash fold must report the same original offset for the same
    // post-dash position, since findInOriginal is built on the same CharClass offset machinery.
    final String original = "A" + s(YEZIDI_HYPHEN) + "B";

    final NormalizedText viaDl = AbstractDL.normalizeInputMapped(original, true, true);
    final NormalizedText viaNormalizer = CharClass.dashes().normalizeMapped(original);
    assertEquals(viaNormalizer.normalized(), viaDl.normalized());

    final int b = viaDl.normalized().indexOf('B');
    assertEquals(viaNormalizer.offsets().toOriginalOffset(b), viaDl.offsets().toOriginalOffset(b));
    assertEquals("B", original.substring(viaDl.offsets().toOriginalOffset(b)));
  }
}
