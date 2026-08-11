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
import java.util.Set;

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
 * <p>Mojibake is reported when a maximal run of non-ASCII characters that Windows-1252
 * can encode maps to a byte sequence that is entirely valid UTF-8 encoding at least one
 * non-ASCII character. Text damaged by reading UTF-8 through that decoding satisfies this
 * by construction, while ordinarily accented words do not: their bytes are not valid
 * UTF-8 sequences.</p>
 *
 * <p>All types are reported by default; the {@link #CursorArtifactDetector(Set)}
 * constructor limits detection to a subset.</p>
 *
 * <p>The detector is stateless and safe for concurrent use by multiple threads.</p>
 *
 * @since 3.0.0
 */
public final class CursorArtifactDetector implements ArtifactDetector {

  private static final Set<String> ALL_TYPES = Set.of(
      TextArtifact.TYPE_REPLACEMENT,
      TextArtifact.TYPE_CONTROL,
      TextArtifact.TYPE_NONCHARACTER,
      TextArtifact.TYPE_UNPAIRED_SURROGATE,
      TextArtifact.TYPE_PRIVATE_USE,
      TextArtifact.TYPE_BIDI_CONTROL,
      TextArtifact.TYPE_ZERO_WIDTH,
      TextArtifact.TYPE_UNICODE_TAG,
      TextArtifact.TYPE_MOJIBAKE);

  private static final int REPLACEMENT = 0xFFFD;
  private static final int ZERO_WIDTH_SPACE = 0x200B;
  private static final int ZERO_WIDTH_NON_JOINER = 0x200C;
  private static final int ZERO_WIDTH_JOINER = 0x200D;
  private static final int WORD_JOINER = 0x2060;
  private static final int ZERO_WIDTH_NO_BREAK_SPACE = 0xFEFF;
  private static final int VARIATION_SELECTOR_16 = 0xFE0F;
  private static final int WAVING_BLACK_FLAG = 0x1F3F4;
  private static final int TAG_OFFSET = 0xE0000;
  private static final int CANCEL_TAG = 0xE007F;

  private final Set<String> types;

  /**
   * The characters <a href="https://www.unicode.org/Public/MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1252.TXT">
   * Windows-1252</a> places at 0x80-0x9F, indexed by byte value minus 0x80; -1 marks the
   * five bytes it leaves undefined. All other characters up to U+00FF encode as their own
   * code point.
   */
  private static final int[] SINGLE_BYTE_SPECIALS = {
      0x20AC, -1, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
      0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, -1, 0x017D, -1,
      -1, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
      0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, -1, 0x017E, 0x0178,
  };

  /** Initializes a detector that reports every built-in type. */
  public CursorArtifactDetector() {
    types = ALL_TYPES;
  }

  /**
   * Initializes a detector limited to selected artifact types.
   *
   * @param types The types to report, drawn from the {@code TYPE_*} constants on
   *              {@link TextArtifact}. Must not be {@code null} or empty and must not
   *              contain a type this detector does not recognize.
   * @throws IllegalArgumentException Thrown if {@code types} is {@code null} or empty,
   *         or contains an unrecognized type.
   */
  public CursorArtifactDetector(Set<String> types) {
    if (types == null || types.isEmpty()) {
      throw new IllegalArgumentException("types must not be null or empty");
    }
    for (final String type : types) {
      if (!ALL_TYPES.contains(type)) {
        throw new IllegalArgumentException("types contains an unrecognized type: " + type);
      }
    }
    this.types = Set.copyOf(types);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The reported spans never overlap, so a caller can apply them to the text in one
   * pass.</p>
   */
  @Override
  public List<TextArtifact> detect(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<TextArtifact> classes = new ArrayList<>();
    final List<TextArtifact> mojibake = new ArrayList<>();
    if (types.size() > 1 || !types.contains(TextArtifact.TYPE_MOJIBAKE)) {
      scanClasses(text, classes);
    }
    if (types.contains(TextArtifact.TYPE_MOJIBAKE)) {
      scanMojibake(text, mojibake);
    }
    return merge(classes, mojibake);
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
        type = enabled(TextArtifact.TYPE_UNPAIRED_SURROGATE);
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
      if (type == null && types.contains(TextArtifact.TYPE_ZERO_WIDTH)
          && isZeroWidth(codePoint)) {
        i = flushZeroWidth(text, i, artifacts);
        continue;
      }
      if (type == null && types.contains(TextArtifact.TYPE_UNICODE_TAG)
          && isUnicodeTag(codePoint)) {
        i = flushUnicodeTags(text, i, artifacts);
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
  private String classify(int codePoint) {
    if (codePoint == REPLACEMENT) {
      return enabled(TextArtifact.TYPE_REPLACEMENT);
    }
    if ((codePoint < 0x20 || (codePoint >= 0x7F && codePoint <= 0x9F))
        && !StringUtil.isUnicodeWhitespace(codePoint)) {
      return enabled(TextArtifact.TYPE_CONTROL);
    }
    if ((codePoint >= 0xFDD0 && codePoint <= 0xFDEF) || (codePoint & 0xFFFE) == 0xFFFE) {
      return enabled(TextArtifact.TYPE_NONCHARACTER);
    }
    if ((codePoint >= 0xE000 && codePoint <= 0xF8FF)
        || (codePoint >= 0xF0000 && codePoint <= 0xFFFFD)
        || (codePoint >= 0x100000 && codePoint <= 0x10FFFD)) {
      return enabled(TextArtifact.TYPE_PRIVATE_USE);
    }
    if ((codePoint >= 0x202A && codePoint <= 0x202E)
        || (codePoint >= 0x2066 && codePoint <= 0x2069)
        || codePoint == 0x200E || codePoint == 0x200F || codePoint == 0x061C) {
      return enabled(TextArtifact.TYPE_BIDI_CONTROL);
    }
    return null;
  }

  /**
   * Applies the configured type filter to one classified type.
   *
   * @param type The built-in type.
   * @return The type when enabled, otherwise {@code null}.
   */
  private String enabled(String type) {
    return types.contains(type) ? type : null;
  }

  /**
   * Whether the code point is invisible and zero-width, the context-resolved class.
   *
   * @param codePoint The code point.
   * @return {@code true} for the zero-width characters this detector resolves.
   */
  private boolean isZeroWidth(int codePoint) {
    return codePoint == ZERO_WIDTH_SPACE || codePoint == ZERO_WIDTH_NON_JOINER
        || codePoint == ZERO_WIDTH_JOINER || codePoint == WORD_JOINER
        || codePoint == ZERO_WIDTH_NO_BREAK_SPACE;
  }

  /**
   * Whether a code point belongs to the Unicode Tags block.
   *
   * @param codePoint The code point.
   * @return {@code true} from U+E0000 through U+E007F.
   */
  private boolean isUnicodeTag(int codePoint) {
    return codePoint >= TAG_OFFSET && codePoint <= CANCEL_TAG;
  }

  /**
   * Resolves a maximal Unicode tag run. A well-formed subdivision flag is
   * orthographic; every other use is reported as hidden tag text.
   *
   * @param text The text.
   * @param start The first tag character.
   * @param artifacts The list to add a finding to.
   * @return The first index after the tag run.
   */
  private int flushUnicodeTags(CharSequence text, int start,
      List<TextArtifact> artifacts) {
    int end = start;
    while (end < text.length()) {
      final int codePoint = Character.codePointAt(text, end);
      if (!isUnicodeTag(codePoint)) {
        break;
      }
      end += Character.charCount(codePoint);
    }
    if (!isEmojiTagFlag(text, start, end)) {
      artifacts.add(new TextArtifact(new Span(start, end), TextArtifact.TYPE_UNICODE_TAG));
    }
    return end;
  }

  /**
   * Tests one tag run against the UTS #51 subdivision-flag shape.
   *
   * @param text The source text.
   * @param start The first tag character.
   * @param end The first index after the tag run.
   * @return {@code true} for a terminated black-flag tag sequence with a two-letter
   *         region and a non-empty subdivision suffix.
   */
  private boolean isEmojiTagFlag(CharSequence text, int start, int end) {
    if (before(text, start) != WAVING_BLACK_FLAG) {
      return false;
    }
    int i = start;
    int count = 0;
    while (i < end) {
      final int codePoint = Character.codePointAt(text, i);
      i += Character.charCount(codePoint);
      if (codePoint == CANCEL_TAG) {
        return i == end && count >= 3;
      }
      final int ascii = codePoint - TAG_OFFSET;
      final boolean letter = ascii >= 'a' && ascii <= 'z';
      if ((!letter && (ascii < '0' || ascii > '9')) || (count < 2 && !letter)) {
        return false;
      }
      count++;
    }
    return false;
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
  private boolean isEmojiContext(int neighbor) {
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
  private int before(CharSequence text, int index) {
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
      if (isUtf8(text, start, i)) {
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
  private int singleByte(char c) {
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
    if (c >= 0x80 && c <= 0x9F) {
      return c;
    }
    return -1;
  }

  /**
   * Validates the bytes as one or more complete, strictly well-formed UTF-8 sequences
   * encoding at least one non-ASCII code point. Overlong forms, surrogate encodings, and
   * values past U+10FFFF are rejected, as
   * <a href="https://www.rfc-editor.org/rfc/rfc3629#section-3">RFC 3629, section 3</a>
   * requires.
   *
   * @param text The source text.
   * @param start The candidate start.
   * @param end The candidate end.
   * @return {@code true} if the whole range has a well-formed multi-byte UTF-8 image.
   */
  private boolean isUtf8(CharSequence text, int start, int end) {
    boolean multiByte = false;
    int i = start;
    while (i < end) {
      final int lead = singleByte(text.charAt(i));
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
      if (i + continuations >= end) {
        return false;
      }
      int codePoint = lead & (0x3F >> continuations);
      for (int k = 1; k <= continuations; k++) {
        final int continuation = singleByte(text.charAt(i + k));
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

  /**
   * Merges ordered class and mojibake findings. Mojibake is the more specific
   * explanation when its Latin-1 image contains C1 controls.
   *
   * @param classes The ordered per-code-point class findings.
   * @param mojibake The ordered mojibake findings.
   * @return The ordered, non-overlapping findings.
   */
  private List<TextArtifact> merge(List<TextArtifact> classes,
      List<TextArtifact> mojibake) {
    final List<TextArtifact> merged = new ArrayList<>(classes.size() + mojibake.size());
    int c = 0;
    int m = 0;
    while (c < classes.size() && m < mojibake.size()) {
      final TextArtifact classified = classes.get(c);
      final TextArtifact damaged = mojibake.get(m);
      if (classified.span().getEnd() <= damaged.span().getStart()) {
        merged.add(classified);
        c++;
      } else if (damaged.span().getEnd() <= classified.span().getStart()) {
        merged.add(damaged);
        m++;
      } else {
        merged.add(damaged);
        m++;
        final int damageEnd = damaged.span().getEnd();
        while (c < classes.size() && classes.get(c).span().getStart() < damageEnd) {
          c++;
        }
      }
    }
    merged.addAll(classes.subList(c, classes.size()));
    merged.addAll(mojibake.subList(m, mojibake.size()));
    merged.sort((a, b) -> Integer.compare(a.span().getStart(), b.span().getStart()));
    return merged;
  }
}
