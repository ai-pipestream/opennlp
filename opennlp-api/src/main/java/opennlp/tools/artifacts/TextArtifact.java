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

import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * One damaged or suspicious character sequence in a text: the {@link Span} it covers in
 * the original text and its artifact type.
 *
 * <p>The type is an open string so detectors can introduce new types without an API
 * change; the constants on this record name the types the built-in detector reports.
 * Detection never alters the text: an artifact only states where and what the problem
 * is, so a caller can mask, remove, or repair with full knowledge of the original
 * offsets.</p>
 *
 * @param span The location of the artifact in the original text. Must not be
 *             {@code null}.
 * @param type The artifact type, for example {@link #TYPE_MOJIBAKE}. Must not be
 *             {@code null} or blank.
 *
 * @since 3.0.0
 */
public record TextArtifact(Span span, String type) {

  /** A run of U+FFFD replacement characters, left behind by a failed decode. */
  public static final String TYPE_REPLACEMENT = "replacement";

  /** A run of C0 or C1 control characters that are not whitespace. */
  public static final String TYPE_CONTROL = "control";

  /** A run of Unicode noncharacters, code points reserved for internal use. */
  public static final String TYPE_NONCHARACTER = "noncharacter";

  /** A run of unpaired UTF-16 surrogates, impossible in well-formed text. */
  public static final String TYPE_UNPAIRED_SURROGATE = "unpaired-surrogate";

  /** A run of private-use code points, meaningless without an out-of-band agreement. */
  public static final String TYPE_PRIVATE_USE = "private-use";

  /**
   * A run of explicit bidirectional control characters, which can visually reorder
   * text away from its logical order.
   */
  public static final String TYPE_BIDI_CONTROL = "bidi-control";

  /**
   * Zero-width characters outside the contexts where they are orthographic, such as
   * runs of them or occurrences with no letter or emoji context.
   */
  public static final String TYPE_ZERO_WIDTH = "zero-width";

  /**
   * A sequence that reads as UTF-8 bytes shown through a single-byte decoding, the
   * classic double-decode damage.
   */
  public static final String TYPE_MOJIBAKE = "mojibake";

  /**
   * Validates the artifact.
   *
   * @throws IllegalArgumentException Thrown if {@code span} is {@code null}, or
   *         {@code type} is {@code null} or blank.
   */
  public TextArtifact {
    if (span == null) {
      throw new IllegalArgumentException("span must not be null");
    }
    if (type == null || StringUtil.isBlank(type)) {
      throw new IllegalArgumentException("type must not be null or blank");
    }
  }
}
