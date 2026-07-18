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

package opennlp.tools.artifacts;

import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the built-in detector: every artifact class has an accepting and a rejecting
 * case, spans are exact against the original text, and clean text of several scripts
 * yields nothing. Test strings are built from explicit code points so the sources stay
 * printable and reviewable.
 */
public class CursorArtifactDetectorTest {

  private final CursorArtifactDetector detector = new CursorArtifactDetector();

  /**
   * Builds a string from code points.
   *
   * @param codePoints The code points.
   * @return The string.
   */
  private static String cp(int... codePoints) {
    return new String(codePoints, 0, codePoints.length);
  }

  /**
   * @return One accepting case per artifact class: the text, the expected type, and the
   *         expected covered text.
   */
  static Stream<Arguments> flagged() {
    final String replacementRun = cp(0xFFFD, 0xFFFD);
    final String nul = cp(0x0000);
    final String escape = cp(0x001B);
    final String noncharacterArabicRange = cp(0xFDD0);
    final String noncharacterPlaneEnd = cp(0xFFFE);
    final String privateUseRun = cp(0xE000, 0xF8FF);
    final String rightToLeftOverride = cp(0x202E);
    final String firstStrongIsolate = cp(0x2068);
    final String zeroWidthSpaceRun = cp(0x200B, 0x200B, 0x200B);
    final String byteOrderMark = cp(0xFEFF);
    // "don't" through a cp1252 read of its UTF-8 bytes: apostrophe U+2019 -> E2 80 99.
    final String curlyQuoteMojibake = cp(0x00E2, 0x20AC, 0x2122);
    // "e with acute" the same way: U+00E9 -> C3 A9.
    final String accentMojibake = cp(0x00C3, 0x00A9);
    return Stream.of(
        Arguments.of("bad " + replacementRun + " decode",
            TextArtifact.TYPE_REPLACEMENT, replacementRun),
        Arguments.of("nul" + nul + "byte", TextArtifact.TYPE_CONTROL, nul),
        Arguments.of("esc" + escape + "[31m", TextArtifact.TYPE_CONTROL, escape),
        Arguments.of("internal " + noncharacterArabicRange + " sentinel",
            TextArtifact.TYPE_NONCHARACTER, noncharacterArabicRange),
        Arguments.of("plane end " + noncharacterPlaneEnd,
            TextArtifact.TYPE_NONCHARACTER, noncharacterPlaneEnd),
        Arguments.of("icon font " + privateUseRun + " glyphs",
            TextArtifact.TYPE_PRIVATE_USE, privateUseRun),
        Arguments.of("override " + rightToLeftOverride + "txet here",
            TextArtifact.TYPE_BIDI_CONTROL, rightToLeftOverride),
        Arguments.of("isolate " + firstStrongIsolate + "x",
            TextArtifact.TYPE_BIDI_CONTROL, firstStrongIsolate),
        Arguments.of("stuffed" + zeroWidthSpaceRun + "text",
            TextArtifact.TYPE_ZERO_WIDTH, zeroWidthSpaceRun),
        Arguments.of(byteOrderMark + "leading bom",
            TextArtifact.TYPE_ZERO_WIDTH, byteOrderMark),
        Arguments.of("don" + curlyQuoteMojibake + "t",
            TextArtifact.TYPE_MOJIBAKE, curlyQuoteMojibake),
        Arguments.of("caf" + accentMojibake,
            TextArtifact.TYPE_MOJIBAKE, accentMojibake));
  }

  @ParameterizedTest
  @MethodSource("flagged")
  void testFlagsTheClassWithExactSpan(String text, String type, String covered) {
    final List<TextArtifact> artifacts = detector.detect(text);
    assertEquals(1, artifacts.size(), () -> "expected exactly one artifact in <" + text
        + "> but found " + artifacts);
    final TextArtifact artifact = artifacts.get(0);
    assertEquals(type, artifact.type());
    assertEquals(covered, artifact.span().getCoveredText(text).toString());
  }

  /** @return Clean texts that must produce no artifact at all. */
  static Stream<Arguments> clean() {
    return Stream.of(
        Arguments.of("plain ASCII", "plain ASCII text, with punctuation."),
        // Precomposed accents and typographic punctuation are not mojibake: their
        // single-byte images are not valid UTF-8 sequences.
        Arguments.of("accented words",
            "d" + cp(0x00E9) + "j" + cp(0x00E0) + " vu " + cp(0x2014) + " caf"
                + cp(0x00E9) + " na" + cp(0x00EF) + "ve"),
        Arguments.of("lone Latin-1 letters",
            "Jo" + cp(0x00E3) + "o n" + cp(0x00E3) + "o " + cp(0x00E9) + " s"
                + cp(0x00F3)),
        Arguments.of("whitespace controls",
            "tabs\tand\nnewlines\r\nare whitespace, not controls"),
        // ZWNJ between Arabic letters is orthographic.
        Arguments.of("joiner in Arabic",
            cp(0x0628, 0x200C, 0x0628)),
        // The family emoji: pictograph ZWJ pictograph ZWJ pictograph.
        Arguments.of("emoji family",
            "family: " + cp(0x1F469, 0x200D, 0x1F469, 0x200D, 0x1F466)),
        // Pictograph + variation selector + ZWJ + pictograph.
        Arguments.of("heart on fire",
            "pict " + cp(0x2764, 0xFE0F, 0x200D, 0x1F525)),
        Arguments.of("supplementary han", "han " + cp(0x23BB4) + " text"));
  }

  @ParameterizedTest
  @MethodSource("clean")
  void testCleanTextYieldsNothing(String label, String text) {
    assertEquals(List.of(), detector.detect(text),
        () -> "false positive in " + label + ": <" + text + ">");
  }

  /** An unpaired surrogate is reported, high and low alike. */
  @Test
  void testUnpairedSurrogates() {
    final String lonelyHigh = "x" + (char) 0xD83D + "y";
    List<TextArtifact> artifacts = detector.detect(lonelyHigh);
    assertEquals(1, artifacts.size());
    assertEquals(TextArtifact.TYPE_UNPAIRED_SURROGATE, artifacts.get(0).type());
    assertEquals(1, artifacts.get(0).span().getStart());
    assertEquals(2, artifacts.get(0).span().getEnd());

    final String lonelyLow = "x" + (char) 0xDC69 + "y";
    artifacts = detector.detect(lonelyLow);
    assertEquals(1, artifacts.size());
    assertEquals(TextArtifact.TYPE_UNPAIRED_SURROGATE, artifacts.get(0).type());
  }

  /** A single zero-width space between Thai letters is a line-break hint, kept. */
  @Test
  void testSingleZeroWidthSpaceBetweenLettersIsOrthographic() {
    final String thai = cp(0x0E01, 0x0E23, 0x0E38, 0x0E07) + cp(0x200B)
        + cp(0x0E40, 0x0E17, 0x0E1E);
    assertEquals(List.of(), detector.detect(thai));
  }

  /** The same character next to punctuation is an artifact. */
  @Test
  void testZeroWidthOutsideLetterContextIsFlagged() {
    final List<TextArtifact> artifacts = detector.detect("end." + cp(0x200B) + " next");
    assertEquals(1, artifacts.size());
    assertEquals(TextArtifact.TYPE_ZERO_WIDTH, artifacts.get(0).type());
  }

  /** A ZWNJ between Persian letters, as the orthography writes it, is kept. */
  @Test
  void testJoinerBetweenLettersIsOrthographic() {
    final String persian = cp(0x0645, 0x06CC) + cp(0x200C) + cp(0x062E, 0x0648);
    assertEquals(List.of(), detector.detect(persian));
  }

  /** Multiple artifacts report in order of appearance with non-overlapping spans. */
  @Test
  void testMultipleFindingsInOrder() {
    final String text = "a" + cp(0x0007) + "b caf" + cp(0x00C3, 0x00A9) + " d"
        + cp(0xFFFD) + "e";
    final List<TextArtifact> artifacts = detector.detect(text);
    assertEquals(3, artifacts.size());
    assertEquals(TextArtifact.TYPE_CONTROL, artifacts.get(0).type());
    assertEquals(TextArtifact.TYPE_MOJIBAKE, artifacts.get(1).type());
    assertEquals(TextArtifact.TYPE_REPLACEMENT, artifacts.get(2).type());
    assertTrue(artifacts.get(0).span().getEnd() <= artifacts.get(1).span().getStart());
    assertTrue(artifacts.get(1).span().getEnd() <= artifacts.get(2).span().getStart());
  }

  /**
   * Double-encoded text yields one finding per damaged run; ASCII between the runs
   * splits them and stays clean.
   */
  @Test
  void testMojibakeRunsAreSeparatedByAscii() {
    final String damagedE = cp(0x00C3, 0x00A9);
    final String text = damagedE + "t" + damagedE + " chez papa";
    final List<TextArtifact> artifacts = detector.detect(text);
    assertEquals(2, artifacts.size());
    assertEquals(TextArtifact.TYPE_MOJIBAKE, artifacts.get(0).type());
    assertEquals(TextArtifact.TYPE_MOJIBAKE, artifacts.get(1).type());
    assertEquals(0, artifacts.get(0).span().getStart());
    assertEquals(2, artifacts.get(0).span().getEnd());
    assertEquals(3, artifacts.get(1).span().getStart());
  }

  /**
   * An overlong encoding, a bare continuation, and an encoded surrogate all fail the
   * strict UTF-8 test, so their carriers stay unflagged by the mojibake class.
   */
  @Test
  void testInvalidUtf8ImagesAreNotMojibake() {
    // C0 80 is the classic overlong NUL; C0 maps from U+00C0.
    assertEquals(List.of(),
        detector.detect("x " + cp(0x00C0, 0x20AC) + " y"));
    // A bare continuation byte image: U+00A9 -> A9 with no lead.
    assertEquals(List.of(), detector.detect("copyright " + cp(0x00A9) + " sign"));
    // ED A0 80 encodes a surrogate; ED maps from U+00ED.
    assertEquals(List.of(),
        detector.detect(cp(0x00ED, 0x00A0, 0x20AC)));
  }

  @Test
  void testRejectsNull() {
    assertThrows(IllegalArgumentException.class, () -> detector.detect(null));
  }

  @Test
  void testEmptyTextYieldsNothing() {
    assertEquals(List.of(), detector.detect(""));
  }
}
