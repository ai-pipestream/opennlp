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

import java.util.ArrayList;
import java.util.List;

import opennlp.tools.tokenize.uax29.ExtendedPictographic;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * The built-in {@link ArtifactDetector}: single cursor passes over the code points with
 * no regular expression, reporting maximal runs per artifact type.
 *
 * <p>Zero-width characters are only artifacts outside their orthographic uses, so they
 * are reported with context: a single joiner adjacent to an emoji or variation selector
 * (emoji sequences), or a single zero-width character between two letters (joining
 * scripts and scripts that mark line-break opportunities invisibly), is not reported.
 * Runs of two or more zero-width characters and occurrences without such context
 * are.</p>
 *
 * <p>Mojibake is reported when a maximal run of non-ASCII, single-byte-encodable
 * characters maps to a byte sequence that is entirely valid UTF-8 encoding at least one
 * non-ASCII character. Text damaged by reading UTF-8 through a legacy single-byte
 * decoding satisfies this by construction, while ordinarily accented words do not: their
 * bytes are not valid UTF-8 sequences.</p>
 *
 * <p>The detector is stateless and safe for concurrent use by multiple threads.</p>
 *
 * @since 3.0.0
 */
public final class CursorArtifactDetector implements ArtifactDetector {

  private static final int REPLACEMENT = 0xFFFD;
  private static final int ZERO_WIDTH_SPACE = 0x200B;
  private static final int ZERO_WIDTH_NON_JOINER = 0x200C;
  private static final int ZERO_WIDTH_JOINER = 0x200D;
  private static final int WORD_JOINER = 0x2060;
  private static final int ZERO_WIDTH_NO_BREAK_SPACE = 0xFEFF;
  private static final int VARIATION_SELECTOR_16 = 0xFE0F;

  /**
   * The characters the single-byte encoding places at 0x80-0x9F, indexed by byte value
   * minus 0x80; -1 marks the five bytes it leaves undefined. All other characters up to
   * U+00FF encode as their own code point.
   */
  private static final int[] SINGLE_BYTE_SPECIALS = {
      0x20AC, -1, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
      0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, -1, 0x017D, -1,
      -1, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
      0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, -1, 0x017E, 0x0178,
  };

  @Override
  public List<TextArtifact> detect(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<TextArtifact> artifacts = new ArrayList<>();
    scanClasses(text, artifacts);
    scanMojibake(text, artifacts);
    artifacts.sort((a, b) -> Integer.compare(a.span().getStart(), b.span().getStart()));
    return artifacts;
  }

  /**
   * One pass emitting maximal runs of the per-code-point classes and the
   * context-resolved zero-width occurrences.
   *
   * @param text The text to scan.
   * @param artifacts The list to add findings to.
   */
  private void scanClasses(CharSequence text, List<TextArtifact> artifacts) {
    final int length = text.length();
    String runType = null;
    int runStart = 0;
    int i = 0;
    while (i < length) {
      final char c = text.charAt(i);
      final int codePoint;
      final int width;
      final String type;
      if (Character.isHighSurrogate(c)
          && i + 1 < length && Character.isLowSurrogate(text.charAt(i + 1))) {
        codePoint = Character.toCodePoint(c, text.charAt(i + 1));
        width = 2;
        type = classify(codePoint);
      } else if (Character.isSurrogate(c)) {
        codePoint = c;
        width = 1;
        type = TextArtifact.TYPE_UNPAIRED_SURROGATE;
      } else {
        codePoint = c;
        width = 1;
        type = classify(codePoint);
      }
      if (runType != null && !runType.equals(type)) {
        artifacts.add(new TextArtifact(new Span(runStart, i), runType));
        runType = null;
      }
      if (type != null && runType == null) {
        runType = type;
        runStart = i;
      }
      if (type == null && isZeroWidth(codePoint)) {
        i = flushZeroWidth(text, i, artifacts);
        continue;
      }
      i += width;
    }
    if (runType != null) {
      artifacts.add(new TextArtifact(new Span(runStart, length), runType));
    }
  }

  /**
   * Classifies one code point, ignoring context.
   *
   * @param codePoint The code point.
   * @return The artifact type, or {@code null} for an ordinary code point. Zero-width
   *         characters classify as {@code null} here because they are context-resolved.
   */
  private static String classify(int codePoint) {
    if (codePoint == REPLACEMENT) {
      return TextArtifact.TYPE_REPLACEMENT;
    }
    if ((codePoint < 0x20 || (codePoint >= 0x7F && codePoint <= 0x9F))
        && !StringUtil.isUnicodeWhitespace(codePoint)) {
      return TextArtifact.TYPE_CONTROL;
    }
    if ((codePoint >= 0xFDD0 && codePoint <= 0xFDEF) || (codePoint & 0xFFFE) == 0xFFFE) {
      return TextArtifact.TYPE_NONCHARACTER;
    }
    if ((codePoint >= 0xE000 && codePoint <= 0xF8FF)
        || (codePoint >= 0xF0000 && codePoint <= 0xFFFFD)
        || (codePoint >= 0x100000 && codePoint <= 0x10FFFD)) {
      return TextArtifact.TYPE_PRIVATE_USE;
    }
    if ((codePoint >= 0x202A && codePoint <= 0x202E)
        || (codePoint >= 0x2066 && codePoint <= 0x2069)
        || codePoint == 0x200E || codePoint == 0x200F || codePoint == 0x061C) {
      return TextArtifact.TYPE_BIDI_CONTROL;
    }
    return null;
  }

  /**
   * Whether the code point is invisible and zero-width, the context-resolved class.
   *
   * @param codePoint The code point.
   * @return {@code true} for the zero-width characters this detector resolves.
   */
  private static boolean isZeroWidth(int codePoint) {
    return codePoint == ZERO_WIDTH_SPACE || codePoint == ZERO_WIDTH_NON_JOINER
        || codePoint == ZERO_WIDTH_JOINER || codePoint == WORD_JOINER
        || codePoint == ZERO_WIDTH_NO_BREAK_SPACE;
  }

  /**
   * Resolves the maximal zero-width run starting at {@code start}: a run of two or more
   * is always an artifact; a single occurrence is orthographic when it is a joiner in an
   * emoji sequence or any zero-width character between two letters, and an artifact
   * otherwise.
   *
   * @param text The text.
   * @param start The index of the first zero-width character.
   * @param artifacts The list to add a finding to, when the run is one.
   * @return The index of the first character after the run.
   */
  private int flushZeroWidth(CharSequence text, int start, List<TextArtifact> artifacts) {
    final int length = text.length();
    int end = start;
    int count = 0;
    int only = -1;
    while (end < length) {
      final int codePoint = Character.codePointAt(text, end);
      if (!isZeroWidth(codePoint)) {
        break;
      }
      only = codePoint;
      count++;
      end += Character.charCount(codePoint);
    }
    if (count >= 2) {
      artifacts.add(new TextArtifact(new Span(start, end), TextArtifact.TYPE_ZERO_WIDTH));
      return end;
    }
    final int previous = before(text, start);
    final int next = end < length ? Character.codePointAt(text, end) : -1;
    if (only == ZERO_WIDTH_JOINER && (isEmojiContext(previous) || isEmojiContext(next))) {
      return end;
    }
    if (previous >= 0 && next >= 0
        && Character.isLetter(previous) && Character.isLetter(next)) {
      return end;
    }
    artifacts.add(new TextArtifact(new Span(start, end), TextArtifact.TYPE_ZERO_WIDTH));
    return end;
  }

  /**
   * Whether a neighbor code point puts a joiner in an emoji sequence.
   *
   * @param neighbor The neighboring code point, or a negative value when absent.
   * @return {@code true} if the neighbor is extended pictographic or the emoji
   *         variation selector.
   */
  private static boolean isEmojiContext(int neighbor) {
    return neighbor >= 0
        && (ExtendedPictographic.is(neighbor) || neighbor == VARIATION_SELECTOR_16);
  }

  /**
   * Reads the code point ending at an index.
   *
   * @param text The text.
   * @param index The exclusive end index.
   * @return The code point before {@code index}, or -1 at the text start.
   */
  private static int before(CharSequence text, int index) {
    return index > 0 ? Character.codePointBefore(text, index) : -1;
  }

  /**
   * One pass finding mojibake: maximal runs of non-ASCII characters the single-byte
   * encoding can represent, whose bytes decode as valid UTF-8 with at least one
   * non-ASCII result.
   *
   * @param text The text to scan.
   * @param artifacts The list to add findings to.
   */
  private void scanMojibake(CharSequence text, List<TextArtifact> artifacts) {
    final int length = text.length();
    int i = 0;
    while (i < length) {
      if (singleByte(text.charAt(i)) < 0x80) {
        i++;
        continue;
      }
      final int start = i;
      while (i < length && singleByte(text.charAt(i)) >= 0x80) {
        i++;
      }
      final byte[] bytes = new byte[i - start];
      for (int b = 0; b < bytes.length; b++) {
        bytes[b] = (byte) singleByte(text.charAt(start + b));
      }
      if (isUtf8(bytes)) {
        artifacts.add(new TextArtifact(new Span(start, i), TextArtifact.TYPE_MOJIBAKE));
      }
    }
  }

  /**
   * Maps a character to its single-byte encoding.
   *
   * @param c The character.
   * @return The byte value 0x00-0xFF, or -1 when the encoding has no byte for it.
   */
  private static int singleByte(char c) {
    if (c < 0x80) {
      return c;
    }
    if (c >= 0xA0 && c <= 0xFF) {
      return c;
    }
    for (int b = 0; b < SINGLE_BYTE_SPECIALS.length; b++) {
      if (SINGLE_BYTE_SPECIALS[b] == c) {
        return 0x80 + b;
      }
    }
    return -1;
  }

  /**
   * Validates the bytes as one or more complete, strictly well-formed UTF-8 sequences
   * encoding at least one non-ASCII code point. Overlong forms, surrogate encodings,
   * and values past U+10FFFF are rejected, as the Unicode specification requires.
   *
   * @param bytes The candidate bytes.
   * @return {@code true} if the whole array is well-formed multi-byte UTF-8.
   */
  private static boolean isUtf8(byte[] bytes) {
    boolean multiByte = false;
    int i = 0;
    while (i < bytes.length) {
      final int lead = bytes[i] & 0xFF;
      if (lead < 0x80) {
        i++;
        continue;
      }
      final int continuations;
      final int min;
      final int max;
      if (lead >= 0xC2 && lead <= 0xDF) {
        continuations = 1;
        min = 0x80;
        max = 0x7FF;
      } else if (lead >= 0xE0 && lead <= 0xEF) {
        continuations = 2;
        min = 0x800;
        max = 0xFFFF;
      } else if (lead >= 0xF0 && lead <= 0xF4) {
        continuations = 3;
        min = 0x10000;
        max = 0x10FFFF;
      } else {
        return false;
      }
      if (i + continuations >= bytes.length) {
        return false;
      }
      int codePoint = lead & (0x3F >> continuations);
      for (int k = 1; k <= continuations; k++) {
        final int continuation = bytes[i + k] & 0xFF;
        if (continuation < 0x80 || continuation > 0xBF) {
          return false;
        }
        codePoint = (codePoint << 6) | (continuation & 0x3F);
      }
      if (codePoint < min || codePoint > max
          || (codePoint >= 0xD800 && codePoint <= 0xDFFF)) {
        return false;
      }
      multiByte = true;
      i += 1 + continuations;
    }
    return multiByte;
  }
}
