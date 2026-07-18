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

package opennlp.tools.assets;

import java.util.Base64;

import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * One binary asset embedded in a text as encoded characters: where it sits, where its
 * payload characters are, and what the decoded bytes are.
 *
 * <p>The format is an open string so detectors can introduce new formats without an API
 * change; the constants on this record name the formats the built-in detector
 * recognizes. Detection never alters the text: an asset states exact offsets over the
 * original, so a caller can extract the bytes, replace the span with text, or drop it,
 * without losing where it was.</p>
 *
 * @param span The full location in the original text, including any URI prefix around
 *             the payload. Must not be {@code null}.
 * @param payload The location of the encoded payload characters within {@code span}.
 *                Must not be {@code null} and must lie inside {@code span}.
 * @param format The format, for example {@link #FORMAT_PNG}: sniffed from the decoded
 *               header when it carries a known magic, otherwise as the carrying URI
 *               declared it. Must not be {@code null} or blank.
 * @param mediaType The media type, either as declared by the carrying URI or derived
 *                  from the format. Must not be {@code null} or blank.
 * @param decodedLength The length of the decoded bytes.
 * @param width The pixel width when the format's header carries one, or -1.
 * @param height The pixel height when the format's header carries one, or -1.
 *
 * @since 3.0.0
 */
public record EmbeddedAsset(Span span, Span payload, String format, String mediaType,
                            long decodedLength, int width, int height) {

  /** A PNG image. */
  public static final String FORMAT_PNG = "png";

  /** A JPEG image. */
  public static final String FORMAT_JPEG = "jpeg";

  /** A GIF image. */
  public static final String FORMAT_GIF = "gif";

  /** A WebP image. */
  public static final String FORMAT_WEBP = "webp";

  /** A RIFF wave audio file. */
  public static final String FORMAT_WAV = "wav";

  /** A PDF document. */
  public static final String FORMAT_PDF = "pdf";

  /** A zip archive, which is also the container of the common office formats. */
  public static final String FORMAT_ZIP = "zip";

  /** A TIFF image, in either byte order. */
  public static final String FORMAT_TIFF = "tiff";

  /** A gzip-compressed stream. */
  public static final String FORMAT_GZIP = "gzip";

  /** A 7z archive. */
  public static final String FORMAT_SEVEN_ZIP = "7z";

  /** A RAR archive. */
  public static final String FORMAT_RAR = "rar";

  /** A FLAC audio stream. */
  public static final String FORMAT_FLAC = "flac";

  /** An Ogg container, the usual carrier of Vorbis and Opus audio. */
  public static final String FORMAT_OGG = "ogg";

  /** A standard MIDI file. */
  public static final String FORMAT_MIDI = "midi";

  /** A SQLite database file. */
  public static final String FORMAT_SQLITE = "sqlite";

  /** An ELF executable or shared object. */
  public static final String FORMAT_ELF = "elf";

  /** A Windows Portable Executable. */
  public static final String FORMAT_PE = "pe";

  /** A Java class file. */
  public static final String FORMAT_CLASS = "class";

  /** A WOFF font. */
  public static final String FORMAT_WOFF = "woff";

  /** A WOFF2 font. */
  public static final String FORMAT_WOFF2 = "woff2";

  /**
   * Validates the asset.
   *
   * @throws IllegalArgumentException Thrown if a span is {@code null}, {@code payload}
   *         lies outside {@code span}, {@code format} or {@code mediaType} is
   *         {@code null} or blank, or {@code decodedLength} is negative.
   */
  public EmbeddedAsset {
    if (span == null || payload == null) {
      throw new IllegalArgumentException("span and payload must not be null");
    }
    if (payload.getStart() < span.getStart() || payload.getEnd() > span.getEnd()) {
      throw new IllegalArgumentException("payload must lie inside the span");
    }
    if (format == null || StringUtil.isBlank(format)) {
      throw new IllegalArgumentException("format must not be null or blank");
    }
    if (mediaType == null || StringUtil.isBlank(mediaType)) {
      throw new IllegalArgumentException("mediaType must not be null or blank");
    }
    if (decodedLength < 0) {
      throw new IllegalArgumentException("decodedLength must not be negative");
    }
  }

  /**
   * Decodes the payload from the text the asset was detected in.
   *
   * @param text The original text this asset's spans refer to. Must not be
   *             {@code null}.
   * @return The decoded bytes. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code text} is {@code null}, shorter
   *         than the payload span, or the payload characters are not valid base64.
   */
  public byte[] decode(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    if (text.length() < payload.getEnd()) {
      throw new IllegalArgumentException("text is shorter than the payload span");
    }
    return Base64.getDecoder().decode(
        text.subSequence(payload.getStart(), payload.getEnd()).toString());
  }
}
