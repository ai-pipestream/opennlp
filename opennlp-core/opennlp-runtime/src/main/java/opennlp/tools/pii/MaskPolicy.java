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
 * How {@link Masker} replaces the characters of a masked span. The base policy from
 * {@link #of(char)} masks every character; {@link #keepingFormat()} leaves separators
 * and punctuation visible so the shape of the value survives; {@link #keepingTrailing(int)}
 * leaves the trailing letters or digits readable, the customary style for payment card
 * receipts.
 *
 * <p>Every policy is length preserving in UTF-16 units: a masked code point outside the
 * basic plane becomes two mask characters, so the spans of every other layer remain
 * valid for the masked text.</p>
 *
 * <p>Instances are immutable and safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class MaskPolicy {

  private final char mask;
  private final boolean keepFormat;
  private final int keepTrailing;

  private MaskPolicy(char mask, boolean keepFormat, int keepTrailing) {
    this.mask = mask;
    this.keepFormat = keepFormat;
    this.keepTrailing = keepTrailing;
  }

  /**
   * Creates the base policy: every character of the span becomes the mask character.
   *
   * @param mask The replacement character. Must not be a surrogate.
   * @return The policy. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code mask} is a surrogate character.
   */
  public static MaskPolicy of(char mask) {
    if (Character.isSurrogate(mask)) {
      throw new IllegalArgumentException("mask must not be a surrogate character");
    }
    return new MaskPolicy(mask, false, 0);
  }

  /**
   * Returns a policy that masks only letters and digits, leaving separators such as
   * spaces, hyphens, parentheses, {@code @}, and dots visible. A phone number keeps its
   * grouping and an email address keeps its {@code @} and dots, which makes the kind of
   * value recognizable without revealing it.
   *
   * @return A new policy; this instance is unchanged. Never {@code null}.
   */
  public MaskPolicy keepingFormat() {
    return new MaskPolicy(mask, true, keepTrailing);
  }

  /**
   * Returns a policy that leaves the last {@code count} letters or digits of each span
   * readable, masking the rest under this policy's other rules. Counting is by code
   * point, so the kept tail is never half a character.
   *
   * @param count The number of trailing letters or digits to keep. Must not be
   *              negative. A span with fewer letters or digits stays fully readable in
   *              those positions.
   * @return A new policy; this instance is unchanged. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code count} is negative.
   */
  public MaskPolicy keepingTrailing(int count) {
    if (count < 0) {
      throw new IllegalArgumentException("count must not be negative");
    }
    return new MaskPolicy(mask, keepFormat, count);
  }

  /**
   * Applies the policy to one span's text.
   *
   * @param spanText The original text covered by the span.
   * @return The replacement, always the same length as {@code spanText} in UTF-16
   *         units.
   */
  String apply(String spanText) {
    int alphanumeric = 0;
    for (int i = 0; i < spanText.length(); ) {
      final int cp = spanText.codePointAt(i);
      if (Character.isLetterOrDigit(cp)) {
        alphanumeric++;
      }
      i += Character.charCount(cp);
    }
    final int firstKept = alphanumeric - keepTrailing;
    final StringBuilder out = new StringBuilder(spanText.length());
    int seen = 0;
    for (int i = 0; i < spanText.length(); ) {
      final int cp = spanText.codePointAt(i);
      final int units = Character.charCount(cp);
      if (Character.isLetterOrDigit(cp)) {
        if (seen >= firstKept) {
          out.appendCodePoint(cp);
        } else {
          out.append(mask);
          if (units == 2) {
            out.append(mask);
          }
        }
        seen++;
      } else if (keepFormat) {
        out.appendCodePoint(cp);
      } else {
        out.append(mask);
        if (units == 2) {
          out.append(mask);
        }
      }
      i += units;
    }
    return out.toString();
  }
}
