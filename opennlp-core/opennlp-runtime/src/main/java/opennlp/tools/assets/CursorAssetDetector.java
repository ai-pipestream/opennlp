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

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import opennlp.tools.util.Span;

/**
 * The built-in {@link AssetDetector}: a single cursor pass with no regular expression,
 * finding base64-embedded binaries both as {@code data:} URIs and as bare payload runs.
 *
 * <p>A bare run is only reported when three independent checks agree: it is long enough
 * to be a real payload, its first characters are the base64 image of a known format's
 * magic bytes (a format's leading bytes always encode to the same characters at payload
 * start), and its decoded header actually carries those magic bytes. A {@code data:}
 * URI declares its media type, so it is reported at any payload length, with the format
 * sniffed from the decoded header when it is a known one.</p>
 *
 * <p>Payload runs must be unbroken: a payload wrapped across lines, as transported mail
 * bodies wrap it, is out of scope here and a candidate for a later tier.</p>
 *
 * <p>The detector is stateless and safe for concurrent use by multiple threads.</p>
 *
 * @since 3.0.0
 */
public final class CursorAssetDetector implements AssetDetector {

  /** Bare runs shorter than this are never reported; real payloads are far longer. */
  private static final int MIN_BARE_PAYLOAD = 32;

  private static final String DATA_URI_SCHEME = "data:";
  private static final String BASE64_MARKER = ";base64,";

  /**
   * The base64 images of the known magic bytes at payload start, longest match first
   * where prefixes overlap.
   */
  private static final String[] MAGIC_PREFIXES = {
      "iVBORw0KGgo", // PNG \x89PNG\r\n\x1a\n
      "/9j/",        // JPEG \xFF\xD8\xFF
      "R0lGODdh",    // GIF87a
      "R0lGODlh",    // GIF89a
      "UklGR",       // RIFF (WebP or wave)
      "JVBERi0",     // PDF %PDF-
      "UEsDB",       // zip PK\x03\x04
  };

  @Override
  public List<EmbeddedAsset> detect(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<EmbeddedAsset> assets = new ArrayList<>();
    final int length = text.length();
    int i = 0;
    while (i < length) {
      final int uriEnd = tryDataUri(text, i, assets);
      if (uriEnd > i) {
        i = uriEnd;
        continue;
      }
      if (isBase64Char(text.charAt(i))) {
        final int runEnd = payloadEnd(text, i);
        tryBareRun(text, i, runEnd, assets);
        i = runEnd;
        continue;
      }
      i++;
    }
    return assets;
  }

  /**
   * Recognizes a {@code data:<mediatype>;base64,<payload>} URI starting at an index.
   *
   * @param text The text.
   * @param start The candidate start.
   * @param assets The list to add a finding to.
   * @return The index after the URI when one was recognized, or {@code start}.
   */
  private int tryDataUri(CharSequence text, int start, List<EmbeddedAsset> assets) {
    if (!matches(text, start, DATA_URI_SCHEME)) {
      return start;
    }
    int i = start + DATA_URI_SCHEME.length();
    final int mediaTypeStart = i;
    final int length = text.length();
    while (i < length && !matches(text, i, BASE64_MARKER)) {
      final char c = text.charAt(i);
      // A media type with optional parameters; anything else means this is not a
      // base64 data URI and the scheme text is left to ordinary scanning.
      if (c <= ' ' || c == '"' || c == '\'' || i - mediaTypeStart > 127) {
        return start;
      }
      i++;
    }
    if (i >= length) {
      return start;
    }
    final String declared = text.subSequence(mediaTypeStart, i).toString();
    final int payloadStart = i + BASE64_MARKER.length();
    final int payloadStop = payloadEnd(text, payloadStart);
    if (payloadStop == payloadStart) {
      return start;
    }
    final byte[] header = decodeHeader(text, payloadStart, payloadStop);
    if (header == null) {
      return start;
    }
    final String sniffed = formatOf(header);
    final String mediaType = declared.indexOf('/') > 0 ? declared : mediaTypeOf(sniffed);
    if (mediaType == null) {
      return start;
    }
    assets.add(asset(text, start, payloadStart, payloadStop,
        sniffed != null ? sniffed : subtype(mediaType), mediaType, header));
    return payloadStop;
  }

  /**
   * Recognizes a bare payload run, demanding a known magic prefix and a verifying
   * decoded header.
   *
   * @param text The text.
   * @param start The run start.
   * @param end The run end.
   * @param assets The list to add a finding to.
   */
  private void tryBareRun(CharSequence text, int start, int end,
      List<EmbeddedAsset> assets) {
    if (end - start < MIN_BARE_PAYLOAD) {
      return;
    }
    boolean magic = false;
    for (final String prefix : MAGIC_PREFIXES) {
      if (matches(text, start, prefix)) {
        magic = true;
        break;
      }
    }
    if (!magic) {
      return;
    }
    final byte[] header = decodeHeader(text, start, end);
    if (header == null) {
      return;
    }
    final String format = formatOf(header);
    if (format == null) {
      return;
    }
    assets.add(asset(text, start, start, end, format, mediaTypeOf(format), header));
  }

  /**
   * Builds the asset record: spans, dimensions from the header where the format
   * carries them, and the decoded length from the payload length.
   *
   * @param text The text.
   * @param spanStart The asset start, at the URI scheme or the payload.
   * @param payloadStart The payload start.
   * @param payloadEnd The payload end, including padding.
   * @param format The decoded format.
   * @param mediaType The media type.
   * @param header The decoded leading bytes.
   * @return The asset. Never {@code null}.
   */
  private static EmbeddedAsset asset(CharSequence text, int spanStart, int payloadStart,
      int payloadEnd, String format, String mediaType, byte[] header) {
    if ((payloadEnd - payloadStart) % 4 == 1) {
      // A length of 1 mod 4 cannot end a base64 payload; the run is truncated, so the
      // stray last character is left out to keep the payload decodable.
      payloadEnd--;
    }
    int padding = 0;
    for (int i = payloadEnd - 1; i >= payloadStart && text.charAt(i) == '='; i--) {
      padding++;
    }
    final int payloadLength = payloadEnd - payloadStart;
    long decodedLength = payloadLength / 4L * 3 - padding;
    if (payloadLength % 4 == 2) {
      decodedLength += 1;
    } else if (payloadLength % 4 == 3) {
      decodedLength += 2;
    }
    int width = -1;
    int height = -1;
    if (EmbeddedAsset.FORMAT_PNG.equals(format) && header.length >= 24) {
      width = readInt(header, 16);
      height = readInt(header, 20);
    } else if (EmbeddedAsset.FORMAT_GIF.equals(format) && header.length >= 10) {
      width = (header[6] & 0xFF) | ((header[7] & 0xFF) << 8);
      height = (header[8] & 0xFF) | ((header[9] & 0xFF) << 8);
    }
    return new EmbeddedAsset(new Span(spanStart, payloadEnd),
        new Span(payloadStart, payloadEnd), format, mediaType, decodedLength,
        width, height);
  }

  /**
   * Finds the end of an unbroken base64 run: payload characters, then optional
   * {@code =} padding.
   *
   * @param text The text.
   * @param start The run start.
   * @return The exclusive end index.
   */
  private static int payloadEnd(CharSequence text, int start) {
    final int length = text.length();
    int i = start;
    while (i < length && isBase64Char(text.charAt(i))) {
      i++;
    }
    int padding = 0;
    while (i < length && padding < 2 && text.charAt(i) == '=') {
      i++;
      padding++;
    }
    return i;
  }

  /**
   * Decodes the leading payload characters into header bytes.
   *
   * @param text The text.
   * @param start The payload start.
   * @param end The payload end.
   * @return The decoded leading bytes, or {@code null} when fewer than four whole
   *         payload characters are available.
   */
  private static byte[] decodeHeader(CharSequence text, int start, int end) {
    final int usable = Math.min(end - start, 32) & ~3;
    if (usable < 4) {
      return null;
    }
    final String head = text.subSequence(start, start + usable).toString();
    // The run consists of alphabet characters by construction, so this cannot throw.
    return Base64.getDecoder().decode(head);
  }

  /**
   * Identifies the format of decoded header bytes.
   *
   * @param header The decoded leading bytes.
   * @return The format constant, or {@code null} when the bytes match no known magic.
   */
  private static String formatOf(byte[] header) {
    if (startsWith(header, 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A)) {
      return EmbeddedAsset.FORMAT_PNG;
    }
    if (startsWith(header, 0xFF, 0xD8, 0xFF)) {
      return EmbeddedAsset.FORMAT_JPEG;
    }
    if (startsWith(header, 'G', 'I', 'F', '8', '7', 'a')
        || startsWith(header, 'G', 'I', 'F', '8', '9', 'a')) {
      return EmbeddedAsset.FORMAT_GIF;
    }
    if (startsWith(header, 'R', 'I', 'F', 'F') && header.length >= 12) {
      if (header[8] == 'W' && header[9] == 'E' && header[10] == 'B' && header[11] == 'P') {
        return EmbeddedAsset.FORMAT_WEBP;
      }
      if (header[8] == 'W' && header[9] == 'A' && header[10] == 'V' && header[11] == 'E') {
        return EmbeddedAsset.FORMAT_WAV;
      }
      return null;
    }
    if (startsWith(header, '%', 'P', 'D', 'F', '-')) {
      return EmbeddedAsset.FORMAT_PDF;
    }
    if (startsWith(header, 'P', 'K', 0x03, 0x04)) {
      return EmbeddedAsset.FORMAT_ZIP;
    }
    return null;
  }

  /**
   * Maps a format to its media type.
   *
   * @param format The format constant, or {@code null}.
   * @return The media type, or {@code null} for an unknown format.
   */
  private static String mediaTypeOf(String format) {
    if (format == null) {
      return null;
    }
    return switch (format) {
      case EmbeddedAsset.FORMAT_PNG -> "image/png";
      case EmbeddedAsset.FORMAT_JPEG -> "image/jpeg";
      case EmbeddedAsset.FORMAT_GIF -> "image/gif";
      case EmbeddedAsset.FORMAT_WEBP -> "image/webp";
      case EmbeddedAsset.FORMAT_WAV -> "audio/wav";
      case EmbeddedAsset.FORMAT_PDF -> "application/pdf";
      case EmbeddedAsset.FORMAT_ZIP -> "application/zip";
      default -> null;
    };
  }

  /**
   * Takes the subtype of a media type as the format name for a declared type the
   * header does not identify.
   *
   * @param mediaType The media type.
   * @return The subtype.
   */
  private static String subtype(String mediaType) {
    final int slash = mediaType.indexOf('/');
    return slash >= 0 && slash + 1 < mediaType.length()
        ? mediaType.substring(slash + 1) : mediaType;
  }

  /**
   * Whether the text matches a literal at an index.
   *
   * @param text The text.
   * @param at The index.
   * @param literal The literal.
   * @return {@code true} on a full match within bounds.
   */
  private static boolean matches(CharSequence text, int at, String literal) {
    if (at + literal.length() > text.length()) {
      return false;
    }
    for (int i = 0; i < literal.length(); i++) {
      if (text.charAt(at + i) != literal.charAt(i)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Whether the bytes start with the values.
   *
   * @param bytes The bytes.
   * @param values The expected leading values, given as int for readability.
   * @return {@code true} on a full match within bounds.
   */
  private static boolean startsWith(byte[] bytes, int... values) {
    if (bytes.length < values.length) {
      return false;
    }
    for (int i = 0; i < values.length; i++) {
      if ((bytes[i] & 0xFF) != values[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Reads a big-endian int.
   *
   * @param bytes The bytes.
   * @param at The offset.
   * @return The value.
   */
  private static int readInt(byte[] bytes, int at) {
    return ((bytes[at] & 0xFF) << 24) | ((bytes[at + 1] & 0xFF) << 16)
        | ((bytes[at + 2] & 0xFF) << 8) | (bytes[at + 3] & 0xFF);
  }

  /**
   * Whether the character belongs to the base64 alphabet, padding excluded.
   *
   * @param c The character.
   * @return {@code true} for the 64 payload characters.
   */
  private static boolean isBase64Char(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
        || c == '+' || c == '/';
  }
}
