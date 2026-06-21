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
package opennlp.tools.tokenize.uax29;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.CharClass;
import opennlp.tools.util.normalizer.NormalizedText;
import opennlp.tools.util.normalizer.OffsetMap;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Property checks that the public segmentation, tokenization, chopping, and offset-mapping entry
 * points stay consistent with one another over a wide range of inputs (mixed scripts, emoji
 * sequences, regional-indicator flags, combining marks, line breaks, supplementary characters).
 * It asserts invariants that any correct implementation must satisfy rather than specific outputs,
 * so it guards against regressions that per-case tests would miss.
 */
public class WordSegmentationConsistencyTest {

  // Always-valid, interesting code points: ASCII, line breaks, Unicode spaces and dashes (including
  // a supplementary dash), combining marks, joiners, emoji, regional indicators, CJK, and
  // supplementary letters.
  private static final int[] PALETTE = {
      'a', 'b', 'Z', '7', ' ', '.', ',', '!', '-', '_', '\'', '"',
      0x09, 0x0A, 0x0D,
      0x00A0, 0x2003, 0x2028, 0x2029, 0x3000,
      0x2010, 0x2013, 0x2014, 0x2212, 0x10EAD,
      0x0301, 0x0308,
      0x200C, 0x200D,
      0x1F600, 0x1F468, 0x1F469, 0x1F427,
      0x1F1E6, 0x1F1F8, 0x1F1FA, 0x1F1F7,
      0x4E2D, 0x6587, 0x3042, 0x3044, 0x30A2, 0x30A4, 0xAC00, 0xB098, 0x0E01, 0x0E02,
      0x10000, 0x1D400,
  };

  private static final int FUZZ_ITERATIONS = 2000;
  private static final long SEED = 20260621L;

  private static String s(int... cps) {
    final StringBuilder sb = new StringBuilder();
    for (final int cp : cps) {
      sb.appendCodePoint(cp);
    }
    return sb.toString();
  }

  private static String describe(String text) {
    final StringBuilder sb = new StringBuilder("[");
    text.codePoints().forEach(c -> sb.append("U+").append(Integer.toHexString(c)).append(' '));
    return sb.append("len=").append(text.length()).append(']').toString();
  }

  @Test
  void invariantsHoldAcrossCorpusAndFuzz() {
    final List<String> failures = new ArrayList<>();

    for (final String text : corpus()) {
      check(text, failures);
    }

    final Random random = new Random(SEED);
    for (int t = 0; t < FUZZ_ITERATIONS; t++) {
      final int n = random.nextInt(13);
      final StringBuilder sb = new StringBuilder();
      for (int i = 0; i < n; i++) {
        sb.appendCodePoint(PALETTE[random.nextInt(PALETTE.length)]);
      }
      check(sb.toString(), failures);
    }

    assertTrue(failures.isEmpty(), failures.size() + " invariant violation(s):\n"
        + String.join("\n", failures.subList(0, Math.min(failures.size(), 40))));
  }

  private static List<String> corpus() {
    final List<String> corpus = new ArrayList<>();
    corpus.add("");
    corpus.add(" ");
    corpus.add("   ");
    corpus.add("a");
    corpus.add("Hello, world!");
    corpus.add("don't can't");
    corpus.add("3.14 1,000");
    corpus.add(s('c', 'a', 'f', 0x00E9));            // precomposed e-acute
    corpus.add(s('c', 'a', 'f', 'e', 0x0301));       // decomposed
    corpus.add(s(0x0301, 'a', 'b', 'c'));            // leading combining mark
    corpus.add(s('a', 0x00A0, 0x3000, 'b'));         // mixed Unicode spaces
    corpus.add("tab\tand\nnewline\r\nhere");
    corpus.add(s('U', 0x1F1FA, 0x1F1F8, 'S'));       // text, flag, text
    corpus.add(s(0x1F1FA, 0x1F1F8, 0x1F1EB, 0x1F1F7)); // two flags
    corpus.add(s(0x1F1FA, 0x1F1F8, 0x1F1EB));        // odd run of regional indicators
    corpus.add(s(0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467)); // family ZWJ emoji
    corpus.add(s('x', 0x10EAD, 'y'));                // supplementary dash
    corpus.add(s(0x10000, 0x1D400));                 // supplementary letters
    return corpus;
  }

  private static void check(String text, List<String> failures) {
    checkSegments(text, failures);
    checkTokens(text, failures);
    checkChopping(text, failures);
    checkOffsets(CharClass.whitespace(), text, true, failures);
    checkOffsets(CharClass.whitespace(), text, false, failures);
    checkOffsets(CharClass.dashes(), text, true, failures);
    checkOffsets(CharClass.dashes(), text, false, failures);
  }

  // segments() must partition [0, length) into contiguous, non-empty spans that reconstruct the
  // text, and boundaries()/forEachSegment() must agree with it.
  private static void checkSegments(String text, List<String> failures) {
    final List<Span> segments = WordSegmenter.segments(text);
    final int[] boundaries = WordSegmenter.boundaries(text);
    final List<Span> viaForEach = new ArrayList<>();
    WordSegmenter.forEachSegment(text, (start, end) -> viaForEach.add(new Span(start, end)));

    if (text.isEmpty()) {
      if (!segments.isEmpty()) {
        failures.add("empty text produced segments " + describe(text));
      }
      if (!Arrays.equals(boundaries, new int[] {0})) {
        failures.add("empty text boundaries not [0] " + describe(text));
      }
      return;
    }

    if (!segments.equals(viaForEach)) {
      failures.add("segments() != forEachSegment() for " + describe(text));
    }
    if (segments.get(0).getStart() != 0) {
      failures.add("first segment does not start at 0 for " + describe(text));
    }
    if (segments.get(segments.size() - 1).getEnd() != text.length()) {
      failures.add("last segment does not end at length for " + describe(text));
    }
    final StringBuilder reconstructed = new StringBuilder();
    for (int i = 0; i < segments.size(); i++) {
      final Span segment = segments.get(i);
      if (segment.getStart() >= segment.getEnd()) {
        failures.add("empty or inverted segment " + segment + " for " + describe(text));
      }
      if (i > 0 && segments.get(i - 1).getEnd() != segment.getStart()) {
        failures.add("non-contiguous segments at " + i + " for " + describe(text));
      }
      reconstructed.append(text, segment.getStart(), segment.getEnd());
    }
    if (!reconstructed.toString().equals(text)) {
      failures.add("segments do not reconstruct text for " + describe(text));
    }

    final int[] expected = new int[segments.size() + 1];
    for (int i = 0; i < segments.size(); i++) {
      expected[i + 1] = segments.get(i).getEnd();
    }
    if (!Arrays.equals(boundaries, expected)) {
      failures.add("boundaries() disagree with segment ends for " + describe(text));
    }
  }

  // Every token must be a non-empty, ordered span whose text matches the covered range and that
  // lies wholly within one segment (filtering never splits across a boundary).
  private static void checkTokens(String text, List<String> failures) {
    final WordTokenizer tokenizer = new WordTokenizer();
    final Span[] spans = tokenizer.tokenizePos(text);
    final String[] tokens = tokenizer.tokenize(text);
    final List<Span> segments = WordSegmenter.segments(text);

    if (spans.length != tokens.length) {
      failures.add("tokenizePos/tokenize length mismatch for " + describe(text));
      return;
    }
    int previousEnd = 0;
    for (int i = 0; i < spans.length; i++) {
      final Span span = spans[i];
      if (span.getStart() < previousEnd || span.getEnd() > text.length()
          || span.getStart() >= span.getEnd()) {
        failures.add("token span " + span + " out of order or range for " + describe(text));
      }
      previousEnd = span.getEnd();
      if (!tokens[i].equals(text.substring(span.getStart(), span.getEnd()))) {
        failures.add("token string does not match covered text at " + i + " for " + describe(text));
      }
      if (!withinOneSegment(span, segments)) {
        failures.add("token span " + span + " crosses a segment boundary for " + describe(text));
      }
    }
  }

  // Chopping at small limits must stay surrogate-safe, keep each piece within one segment and at
  // most the limit (unless a single code point exceeds it), and reconstruct the unchopped tokens.
  private static void checkChopping(String text, List<String> failures) {
    final StringBuilder unchopped = new StringBuilder();
    for (final String token : new WordTokenizer().tokenize(text)) {
      unchopped.append(token);
    }
    final List<Span> segments = WordSegmenter.segments(text);
    for (final int max : new int[] {1, 2, 3, 5}) {
      final Span[] spans = new WordTokenizer(max).tokenizePos(text);
      final StringBuilder concatenated = new StringBuilder();
      for (final Span span : spans) {
        final String piece = text.substring(span.getStart(), span.getEnd());
        concatenated.append(piece);
        if (Character.isLowSurrogate(piece.charAt(0))
            || Character.isHighSurrogate(piece.charAt(piece.length() - 1))) {
          failures.add("chop(" + max + ") split a surrogate pair " + span + " for "
              + describe(text));
        }
        if (piece.length() > max && piece.codePointCount(0, piece.length()) > 1) {
          failures.add("chop(" + max + ") piece exceeds the limit " + span + " for "
              + describe(text));
        }
        if (!withinOneSegment(span, segments)) {
          failures.add("chop(" + max + ") piece crosses a segment boundary " + span + " for "
              + describe(text));
        }
      }
      if (!concatenated.toString().equals(unchopped.toString())) {
        failures.add("chop(" + max + ") does not reconstruct the unchopped tokens for "
            + describe(text));
      }
    }
  }

  // The offset map must be monotonic, expose exact endpoints, and satisfy both the floor contract
  // (a forward-mapped offset's source is at or before the query) and the reverse round-trip bound.
  private static void checkOffsets(CharClass charClass, String text, boolean collapse,
                                   List<String> failures) {
    final NormalizedText normalized =
        collapse ? charClass.collapseMapped(text) : charClass.normalizeMapped(text);
    final OffsetMap map = normalized.offsets();
    final int normalizedLength = map.normalizedLength();
    final int originalLength = map.originalLength();
    final String op = collapse ? "collapseMapped" : "normalizeMapped";

    if (normalizedLength != normalized.normalized().length()) {
      failures.add(op + " normalizedLength mismatch for " + describe(text));
    }
    if (originalLength != text.length()) {
      failures.add(op + " originalLength mismatch for " + describe(text));
    }
    int previousOriginal = -1;
    for (int k = 0; k <= normalizedLength; k++) {
      final int original = map.toOriginalOffset(k);
      if (original < 0 || original > originalLength || original < previousOriginal) {
        failures.add(op + " toOriginalOffset(" + k + ")=" + original + " invalid for "
            + describe(text));
      }
      previousOriginal = original;
    }
    if (map.toOriginalOffset(normalizedLength) != originalLength) {
      failures.add(op + " toOriginalOffset(normalizedLength) != originalLength for "
          + describe(text));
    }
    int previousNormalized = -1;
    for (int o = 0; o <= originalLength; o++) {
      final int k = map.toNormalizedOffset(o);
      if (k < 0 || k > normalizedLength || k < previousNormalized) {
        failures.add(op + " toNormalizedOffset(" + o + ")=" + k + " invalid for " + describe(text));
      }
      previousNormalized = k;
      // Floor: the forward-mapped index's source is at or before o (when any character survives).
      if (normalizedLength > 0 && map.toOriginalOffset(k) > o) {
        failures.add(op + " floor contract broken at " + o + " for " + describe(text));
      }
    }
    for (int k = 0; k <= normalizedLength; k++) {
      // Floor reverse: re-mapping a normalized index's source lands at or after that index.
      if (map.toNormalizedOffset(map.toOriginalOffset(k)) < k) {
        failures.add(op + " reverse round-trip broken at " + k + " for " + describe(text));
      }
    }
  }

  private static boolean withinOneSegment(Span span, List<Span> segments) {
    for (final Span segment : segments) {
      if (segment.getStart() <= span.getStart() && span.getEnd() <= segment.getEnd()) {
        return true;
      }
    }
    return false;
  }
}
