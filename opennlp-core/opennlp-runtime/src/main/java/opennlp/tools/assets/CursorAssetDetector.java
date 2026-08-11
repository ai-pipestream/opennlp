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

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import opennlp.tools.util.Span;

/**
 * The built-in {@link AssetDetector}: a single cursor pass with no regular expression,
 * finding base64-embedded binaries as {@code data:} URIs, bare payload runs,
 * <a href="https://www.rfc-editor.org/rfc/rfc7468.html">RFC 7468</a> PEM envelopes,
 * and <a href="https://www.rfc-editor.org/rfc/rfc7519.html">RFC 7519</a> compact JSON
 * Web Tokens.
 *
 * <p>A bare run is only reported when three independent checks agree: it is long enough
 * to be a real payload, its first characters are the base64 image of a known format's
 * magic bytes (a format's leading bytes always encode to the same characters at payload
 * start), and its decoded header actually carries those magic bytes. A {@code data:}
 * URI declares its media type, so it is reported at any payload length, with the format
 * sniffed from the decoded header when it is a known one.</p>
 *
 * <p>Bare standard-base64 payloads may use the 76-column MIME or 64-column PEM line
 * width with CRLF or LF separators. The
 * <a href="https://www.rfc-editor.org/rfc/rfc4648.html#section-5">base64url</a>
 * alphabet is accepted for unwrapped payloads. PEM labels and end boundaries are
 * verified before an envelope is reported.
 * A compact JWT is reported only when it has three base64url segments, a UTF-8 JSON
 * object header with a non-empty {@code alg} member, and a UTF-8 JSON object claims
 * set. Signature validity is deliberately out of scope.</p>
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
  private static final String JWT_MEDIA_TYPE = "application/jwt";
  private static final int MAX_JWT_SEGMENT_LENGTH = 16 * 1024;

  private static final List<PemType> PEM_TYPES = List.of(
      pemType("CERTIFICATE", "pem-cert", "application/x-x509-cert"),
      pemType("X509 CERTIFICATE", "pem-cert", "application/x-x509-cert"),
      pemType("X509 CRL", "pem-crl", "application/pkix-crl"),
      pemType("CERTIFICATE REQUEST", "pem-request", "application/pkcs10"),
      pemType("NEW CERTIFICATE REQUEST", "pem-request", "application/pkcs10"),
      pemType("PKCS7", "pem-pkcs7", "application/pkcs7-signature"),
      pemType("CMS", "pem-cms", "application/pkcs7-mime"),
      pemType("PRIVATE KEY", "pem-key", "application/x-x509-key"),
      pemType("ENCRYPTED PRIVATE KEY", "pem-key", "application/x-x509-key"),
      pemType("RSA PRIVATE KEY", "pem-key", "application/x-x509-key"),
      pemType("DSA PRIVATE KEY", "pem-key", "application/x-x509-key"),
      pemType("EC PRIVATE KEY", "pem-key", "application/x-x509-key"),
      pemType("PUBLIC KEY", "pem-key", "application/x-x509-key"),
      pemType("RSA PUBLIC KEY", "pem-key", "application/x-x509-key"),
      pemType("ATTRIBUTE CERTIFICATE", "pem-attribute-cert",
          "application/x-x509-attribute-cert"),
      pemType("DSA PARAMETERS", "pem-parameters", "application/x-x509-dsa-parameters"),
      pemType("EC PARAMETERS", "pem-parameters", "application/x-x509-ec-parameters"),
      pemType("DH PARAMETERS", "pem-parameters", "application/x-x509-dh-parameters"));

  /**
   * The base64 image of the RIFF container magic. RIFF is recognized apart from the
   * {@link KnownMagics} table because its file type sits at offset eight, behind the
   * chunk size, and is resolved from the decoded header.
   */
  private static final String RIFF_PREFIX = "UklGR";

  /** The RIFF container magic, and the offset its four-character form type sits at. */
  private static final String RIFF_MAGIC = "RIFF";
  private static final int RIFF_FORM_TYPE = 8;

  /**
   * {@inheritDoc}
   *
   * <p>The returned assets never overlap: the scan resumes behind the payload of the
   * asset it just reported.</p>
   */
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
      final int pemEnd = tryPem(text, i, assets);
      if (pemEnd > i) {
        i = pemEnd;
        continue;
      }
      final int jwtEnd = tryJwt(text, i, assets);
      if (jwtEnd > i) {
        i = jwtEnd;
        continue;
      }
      if (isAnyBase64Char(text.charAt(i))) {
        final Payload payload = payload(text, i, true);
        tryBareRun(text, payload, assets);
        i = Math.max(payload.end(), i + 1);
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
    final Payload payload = payload(text, payloadStart, true);
    if (payload.end() == payloadStart) {
      return start;
    }
    final byte[] header = decodeHeader(text, payload);
    if (header == null) {
      return start;
    }
    final String sniffed = formatOf(header);
    final String mediaType = declared.indexOf('/') > 0 ? declared : mediaTypeOf(sniffed);
    if (mediaType == null) {
      return start;
    }
    assets.add(asset(start, payload.end(), payload,
        sniffed != null ? sniffed : subtype(mediaType), mediaType, header));
    return payload.end();
  }

  /**
   * Recognizes a bare payload run, demanding a known magic prefix and a verifying
   * decoded header.
   *
   * @param text The text.
   * @param payload The scanned payload.
   * @param assets The list to add a finding to.
   */
  private void tryBareRun(CharSequence text, Payload payload, List<EmbeddedAsset> assets) {
    if (payload.encodedLength() < MIN_BARE_PAYLOAD) {
      return;
    }
    boolean magic = matchesBase64Prefix(text, payload.start(), RIFF_PREFIX);
    if (!magic) {
      for (final String prefix : KnownMagics.PREFIXES) {
        if (matchesBase64Prefix(text, payload.start(), prefix)) {
          magic = true;
          break;
        }
      }
    }
    if (!magic) {
      return;
    }
    final byte[] header = decodeHeader(text, payload);
    if (header == null) {
      return;
    }
    final String format = formatOf(header);
    if (format == null) {
      return;
    }
    assets.add(asset(payload.start(), payload.end(), payload,
        format, mediaTypeOf(format), header));
  }

  /**
   * Recognizes one RFC 7468 textual envelope with a supported label.
   *
   * @param text The text being scanned.
   * @param start The candidate boundary marker start.
   * @param assets The list to add a finding to.
   * @return The end of the envelope, or {@code start} when none begins here.
   */
  private int tryPem(CharSequence text, int start, List<EmbeddedAsset> assets) {
    if (!lineBoundaryBefore(text, start)) {
      return start;
    }
    for (final PemType type : PEM_TYPES) {
      if (!matches(text, start, type.begin())) {
        continue;
      }
      final int bodyStart = lineBreakEnd(text, start + type.begin().length());
      if (bodyStart == start + type.begin().length()) {
        return start;
      }
      int i = bodyStart;
      int encodedLength = 0;
      int padding = 0;
      int payloadEnd = bodyStart;
      boolean sawBody = false;
      while (i < text.length()) {
        if (matches(text, i, type.end())) {
          final int spanEnd = i + type.end().length();
          if (!sawBody || !lineBoundaryAfter(text, spanEnd)
              || encodedLength % 4 != 0) {
            return start;
          }
          final Payload payload = new Payload(bodyStart, payloadEnd, encodedLength,
              padding, false);
          final byte[] header = decodeHeader(text, payload);
          if (header == null) {
            return start;
          }
          assets.add(asset(start, spanEnd, payload, type.format(),
              type.mediaType(), header));
          return spanEnd;
        }
        if (matches(text, i, "-----END ")) {
          return start;
        }
        final int lineEnd = lineEnd(text, i);
        final int lineLength = lineEnd - i;
        if (lineLength == 0 || lineLength > 64) {
          return start;
        }
        boolean paddedLine = false;
        for (int at = i; at < lineEnd; at++) {
          final char c = text.charAt(at);
          if (isBase64Char(c) && !paddedLine) {
            encodedLength++;
          } else if (c == '=' && padding < 2) {
            padding++;
            encodedLength++;
            paddedLine = true;
          } else {
            return start;
          }
        }
        sawBody = true;
        payloadEnd = lineEnd;
        final int next = lineBreakEnd(text, lineEnd);
        if (next == lineEnd || (paddedLine && !matches(text, next, type.end()))) {
          return start;
        }
        i = next;
      }
      return start;
    }
    return start;
  }

  /**
   * Recognizes a compact JSON Web Token and exposes its claims segment as payload.
   * This identifies structure only; it does not authenticate the signature.
   *
   * @param text The text being scanned.
   * @param start The candidate header-segment start.
   * @param assets The list to add a finding to.
   * @return The token end, or {@code start} when no JWT begins here.
   */
  private int tryJwt(CharSequence text, int start, List<EmbeddedAsset> assets) {
    if (!tokenBoundaryBefore(text, start)) {
      return start;
    }
    final int headerEnd = urlSegmentEnd(text, start);
    if (headerEnd == start || headerEnd >= text.length() || text.charAt(headerEnd) != '.') {
      return start;
    }
    final int claimsStart = headerEnd + 1;
    final int claimsEnd = urlSegmentEnd(text, claimsStart);
    if (claimsEnd == claimsStart || claimsEnd >= text.length()
        || text.charAt(claimsEnd) != '.') {
      return start;
    }
    final int signatureStart = claimsEnd + 1;
    final int tokenEnd = urlSegmentEnd(text, signatureStart);
    if (!tokenBoundaryAfter(text, tokenEnd)) {
      return start;
    }
    if (headerEnd - start > MAX_JWT_SEGMENT_LENGTH
        || claimsEnd - claimsStart > MAX_JWT_SEGMENT_LENGTH
        || tokenEnd - signatureStart > MAX_JWT_SEGMENT_LENGTH) {
      return start;
    }
    final byte[] header = decodeUrlSegment(text, start, headerEnd);
    final byte[] claims = decodeUrlSegment(text, claimsStart, claimsEnd);
    if (header == null || claims == null) {
      return start;
    }
    final String headerJson = utf8(header);
    final String claimsJson = utf8(claims);
    if (!jsonObject(headerJson) || !jsonObject(claimsJson)) {
      return start;
    }
    final String algorithm = jsonStringMember(headerJson, "alg");
    if (algorithm == null
        || ("none".equals(algorithm) != (tokenEnd == signatureStart))) {
      return start;
    }
    final Payload payload = new Payload(claimsStart, claimsEnd,
        claimsEnd - claimsStart, 0, true);
    assets.add(asset(start, tokenEnd, payload, EmbeddedAsset.FORMAT_JWT,
        JWT_MEDIA_TYPE, claims));
    return tokenEnd;
  }

  /**
   * Reads one unpadded base64url segment.
   *
   * @param text The source text.
   * @param start The segment start.
   * @return The exclusive segment end.
   */
  private int urlSegmentEnd(CharSequence text, int start) {
    int i = start;
    final int limit = Math.min(text.length(), start + MAX_JWT_SEGMENT_LENGTH + 1);
    while (i < limit && isBase64UrlChar(text.charAt(i))) {
      i++;
    }
    return i;
  }

  /**
   * Decodes a bounded base64url segment.
   *
   * @param text The source text.
   * @param start The segment start.
   * @param end The segment end.
   * @return The bytes, or {@code null} for an impossible encoded length.
   */
  private byte[] decodeUrlSegment(CharSequence text, int start, int end) {
    if ((end - start) % 4 == 1) {
      return null;
    }
    try {
      return Base64.getUrlDecoder().decode(text.subSequence(start, end).toString());
    } catch (IllegalArgumentException e) {
      return null;
    }
  }

  /**
   * Decodes UTF-8 without accepting replacement characters.
   *
   * @param bytes The bytes to decode.
   * @return The decoded text, or {@code null} when malformed.
   */
  private String utf8(byte[] bytes) {
    try {
      return StandardCharsets.UTF_8.newDecoder()
          .onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT)
          .decode(ByteBuffer.wrap(bytes)).toString();
    } catch (CharacterCodingException e) {
      return null;
    }
  }

  /**
   * Tests whether a decoded JWT segment is a JSON object at its outer boundary.
   *
   * @param value The decoded UTF-8 text, possibly {@code null}.
   * @return {@code true} when the first and last non-whitespace characters are braces.
   */
  private boolean jsonObject(String value) {
    if (value == null) {
      return false;
    }
    int start = 0;
    int end = value.length();
    while (start < end && Character.isWhitespace(value.charAt(start))) {
      start++;
    }
    while (end > start && Character.isWhitespace(value.charAt(end - 1))) {
      end--;
    }
    return end - start >= 2 && value.charAt(start) == '{' && value.charAt(end - 1) == '}';
  }

  /**
   * Finds a non-empty string-valued member in a small JSON object. This is a structural
   * JWT guard, not a general JSON parser.
   *
   * @param json The decoded object text.
   * @param name The member name.
   * @return The member value, or {@code null} when absent or not a string.
   */
  private String jsonStringMember(String json, String name) {
    final String key = "\"" + name + "\"";
    for (int i = 0; i + key.length() < json.length(); i++) {
      if (!matches(json, i, key)) {
        continue;
      }
      int before = i - 1;
      while (before >= 0 && Character.isWhitespace(json.charAt(before))) {
        before--;
      }
      if (before < 0 || (json.charAt(before) != '{' && json.charAt(before) != ',')) {
        continue;
      }
      int at = i + key.length();
      while (at < json.length() && Character.isWhitespace(json.charAt(at))) {
        at++;
      }
      if (at >= json.length() || json.charAt(at++) != ':') {
        continue;
      }
      while (at < json.length() && Character.isWhitespace(json.charAt(at))) {
        at++;
      }
      if (at >= json.length() || json.charAt(at++) != '"') {
        continue;
      }
      final int valueStart = at;
      boolean escaped = false;
      while (at < json.length()) {
        final char c = json.charAt(at);
        if (c == '"' && !escaped) {
          return at > valueStart ? json.substring(valueStart, at) : null;
        }
        escaped = c == '\\' && !escaped;
        if (c != '\\') {
          escaped = false;
        }
        at++;
      }
    }
    return null;
  }

  /**
   * Builds the asset record: spans, dimensions from the header where the format
   * carries them, and the decoded length from the payload length.
   *
   * @param spanStart The asset start, at the URI scheme or the payload.
   * @param spanEnd The exclusive end of the complete asset.
   * @param payload The encoded payload and its accounting.
   * @param format The decoded format.
   * @param mediaType The media type.
   * @param header The decoded leading bytes.
   * @return The asset. Never {@code null}.
   */
  private EmbeddedAsset asset(int spanStart, int spanEnd,
      Payload payload, String format, String mediaType, byte[] header) {
    final long decodedLength = decodedLength(payload.encodedLength(), payload.padding());
    int width = -1;
    int height = -1;
    if (EmbeddedAsset.FORMAT_PNG.equals(format) && header.length >= 24) {
      width = readInt(header, 16);
      height = readInt(header, 20);
    } else if (EmbeddedAsset.FORMAT_GIF.equals(format) && header.length >= 10) {
      width = (header[6] & 0xFF) | ((header[7] & 0xFF) << 8);
      height = (header[8] & 0xFF) | ((header[9] & 0xFF) << 8);
    }
    return new EmbeddedAsset(new Span(spanStart, spanEnd),
        new Span(payload.start(), payload.end()), format, mediaType, decodedLength,
        width, height);
  }

  /**
   * Scans one base64 or base64url payload. Standard-base64 runs may cross CRLF or LF
   * when every continued line uses the MIME width of 76 or PEM width of 64.
   *
   * @param text The text.
   * @param start The run start.
   * @param allowWrap Whether conventional standard-base64 line wrapping is allowed.
   * @return The payload accounting. Never {@code null}.
   */
  private Payload payload(CharSequence text, int start, boolean allowWrap) {
    final int length = text.length();
    int i = start;
    int encodedLength = 0;
    int lineLength = 0;
    int wrapWidth = 0;
    int padding = 0;
    boolean urlSafe = false;
    while (i < length) {
      final char c = text.charAt(i);
      if (isAnyBase64Char(c) && padding == 0) {
        urlSafe |= c == '-' || c == '_';
        encodedLength++;
        lineLength++;
        i++;
        continue;
      }
      if (c == '=' && padding < 2) {
        padding++;
        encodedLength++;
        lineLength++;
        i++;
        continue;
      }
      final int next = allowWrap && !urlSafe && padding == 0
          ? lineBreakEnd(text, i) : i;
      if (next > i && (lineLength == 64 || lineLength == 76)
          && (wrapWidth == 0 || wrapWidth == lineLength)
          && next < length && isBase64Char(text.charAt(next))) {
        wrapWidth = lineLength;
        lineLength = 0;
        i = next;
        continue;
      }
      break;
    }
    if (encodedLength % 4 == 1 && padding == 0) {
      i--;
      encodedLength--;
    }
    return new Payload(start, i, encodedLength, padding, urlSafe);
  }

  /**
   * Decodes the leading payload characters into header bytes.
   *
   * @param text The text.
   * @param payload The scanned payload.
   * @return The decoded leading bytes, or {@code null} when fewer than four whole
   *         payload characters are available.
   */
  private byte[] decodeHeader(CharSequence text, Payload payload) {
    final int usable = Math.min(payload.encodedLength() - payload.padding(), 32) & ~3;
    if (usable < 4) {
      return null;
    }
    final StringBuilder head = new StringBuilder(usable);
    for (int i = payload.start(); i < payload.end() && head.length() < usable; i++) {
      final char c = text.charAt(i);
      if (isAnyBase64Char(c)) {
        head.append(c);
      }
    }
    try {
      return (payload.urlSafe() ? Base64.getUrlDecoder() : Base64.getDecoder())
          .decode(head.toString());
    } catch (IllegalArgumentException e) {
      return null;
    }
  }

  /**
   * Identifies the format of decoded header bytes: the longest matching magic in
   * {@link KnownMagics}, then the RIFF container resolved by its file type.
   *
   * @param header The decoded leading bytes.
   * @return The format tag, or {@code null} when the bytes match no known magic.
   */
  private String formatOf(byte[] header) {
    final String known = KnownMagics.formatOf(header);
    if (known != null) {
      return known;
    }
    if (carries(header, 0, RIFF_MAGIC)) {
      if (carries(header, RIFF_FORM_TYPE, "WEBP")) {
        return EmbeddedAsset.FORMAT_WEBP;
      }
      if (carries(header, RIFF_FORM_TYPE, "WAVE")) {
        return EmbeddedAsset.FORMAT_WAV;
      }
      if (carries(header, RIFF_FORM_TYPE, "AVI ")) {
        return EmbeddedAsset.FORMAT_AVI;
      }
    }
    return null;
  }

  /**
   * Maps a format to its media type through {@link KnownMagics}, covering the
   * RIFF-carried formats that table cannot hold.
   *
   * @param format The format tag, or {@code null}.
   * @return The media type, or {@code null} for an unknown format.
   */
  private String mediaTypeOf(String format) {
    if (format == null) {
      return null;
    }
    return switch (format) {
      case EmbeddedAsset.FORMAT_WEBP -> "image/webp";
      case EmbeddedAsset.FORMAT_WAV -> "audio/wav";
      case EmbeddedAsset.FORMAT_AVI -> "video/x-msvideo";
      default -> KnownMagics.mediaTypeOf(format);
    };
  }

  /**
   * Takes the subtype of a media type as the format name for a declared type the
   * header does not identify.
   *
   * @param mediaType The media type.
   * @return The subtype.
   */
  private String subtype(String mediaType) {
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
  private boolean matches(CharSequence text, int at, String literal) {
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
   * Matches a standard base64 prefix against either base64 alphabet.
   *
   * @param text The encoded text.
   * @param at The candidate prefix start.
   * @param standardPrefix The standard-base64 prefix.
   * @return {@code true} when all characters match, allowing {@code -} for {@code +}
   *         and {@code _} for {@code /}.
   */
  private boolean matchesBase64Prefix(CharSequence text, int at, String standardPrefix) {
    if (at + standardPrefix.length() > text.length()) {
      return false;
    }
    for (int i = 0; i < standardPrefix.length(); i++) {
      final char expected = standardPrefix.charAt(i);
      final char actual = text.charAt(at + i);
      if (actual != expected && !(expected == '+' && actual == '-')
          && !(expected == '/' && actual == '_')) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns the index behind one CRLF or LF line break.
   *
   * @param text The source text.
   * @param at The candidate line-break start.
   * @return The index behind the break, or {@code at} when none starts there.
   */
  private int lineBreakEnd(CharSequence text, int at) {
    if (at >= text.length()) {
      return at;
    }
    if (text.charAt(at) == '\n') {
      return at + 1;
    }
    if (text.charAt(at) == '\r' && at + 1 < text.length() && text.charAt(at + 1) == '\n') {
      return at + 2;
    }
    return at;
  }

  /**
   * Finds the next CR or LF, or the end of the text.
   *
   * @param text The source text.
   * @param start The line start.
   * @return The exclusive line end.
   */
  private int lineEnd(CharSequence text, int start) {
    int i = start;
    while (i < text.length() && text.charAt(i) != '\r' && text.charAt(i) != '\n') {
      i++;
    }
    return i;
  }

  /**
   * Tests the boundary before an RFC 7468 marker.
   *
   * @param text The source text.
   * @param at The marker start.
   * @return {@code true} at text start or immediately after a line break.
   */
  private boolean lineBoundaryBefore(CharSequence text, int at) {
    return at == 0 || text.charAt(at - 1) == '\r' || text.charAt(at - 1) == '\n';
  }

  /**
   * Tests the boundary after an RFC 7468 marker.
   *
   * @param text The source text.
   * @param at The index behind the marker.
   * @return {@code true} at text end or immediately before a line break.
   */
  private boolean lineBoundaryAfter(CharSequence text, int at) {
    return at == text.length() || text.charAt(at) == '\r' || text.charAt(at) == '\n';
  }

  /**
   * Tests the boundary before a compact token.
   *
   * @param text The source text.
   * @param at The token start.
   * @return {@code true} when no base64url or segment character precedes it.
   */
  private boolean tokenBoundaryBefore(CharSequence text, int at) {
    return at == 0 || (!isBase64UrlChar(text.charAt(at - 1)) && text.charAt(at - 1) != '.');
  }

  /**
   * Tests the boundary after a compact token.
   *
   * @param text The source text.
   * @param at The token end.
   * @return {@code true} when no base64url or segment character follows it.
   */
  private boolean tokenBoundaryAfter(CharSequence text, int at) {
    return at == text.length() || (!isBase64UrlChar(text.charAt(at)) && text.charAt(at) != '.');
  }

  /**
   * Computes decoded bytes from encoded character and padding counts.
   *
   * @param encodedLength The encoded character count, including padding.
   * @param padding The number of trailing padding characters.
   * @return The decoded byte count.
   */
  private long decodedLength(int encodedLength, int padding) {
    long decoded = encodedLength / 4L * 3 - padding;
    if (encodedLength % 4 == 2) {
      decoded++;
    } else if (encodedLength % 4 == 3) {
      decoded += 2;
    }
    return decoded;
  }

  /**
   * Whether the bytes carry an ASCII tag at an offset.
   *
   * @param bytes The bytes.
   * @param at The offset.
   * @param tag The expected characters, all of them ASCII.
   * @return {@code true} on a full match within bounds.
   */
  private boolean carries(byte[] bytes, int at, String tag) {
    if (at + tag.length() > bytes.length) {
      return false;
    }
    for (int i = 0; i < tag.length(); i++) {
      if ((bytes[at + i] & 0xFF) != tag.charAt(i)) {
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
  private int readInt(byte[] bytes, int at) {
    return ((bytes[at] & 0xFF) << 24) | ((bytes[at + 1] & 0xFF) << 16)
        | ((bytes[at + 2] & 0xFF) << 8) | (bytes[at + 3] & 0xFF);
  }

  /**
   * Whether the character belongs to the base64 alphabet, padding excluded.
   *
   * @param c The character.
   * @return {@code true} for the 64 payload characters.
   */
  private boolean isBase64Char(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
        || c == '+' || c == '/';
  }

  /**
   * Whether the character belongs to the URL-safe base64 alphabet.
   *
   * @param c The character.
   * @return {@code true} for a URL-safe payload character.
   */
  private boolean isBase64UrlChar(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
        || c == '-' || c == '_';
  }

  /**
   * Whether the character belongs to either base64 alphabet.
   *
   * @param c The character.
   * @return {@code true} for a standard or URL-safe payload character.
   */
  private boolean isAnyBase64Char(char c) {
    return isBase64Char(c) || c == '-' || c == '_';
  }

  /**
   * Precomputes an RFC 7468 type's boundary markers once rather than during scans.
   *
   * @param label The textual envelope label.
   * @param format The asset format tag.
   * @param mediaType The asset media type.
   * @return The immutable type metadata.
   */
  private static PemType pemType(String label, String format, String mediaType) {
    return new PemType("-----BEGIN " + label + "-----", "-----END " + label + "-----",
        format, mediaType);
  }

  /** One scanned encoded payload and the counts needed for exact decoding metadata. */
  private record Payload(int start, int end, int encodedLength, int padding,
                         boolean urlSafe) {
  }

  /** One supported RFC 7468 boundary pair and the asset tags it implies. */
  private record PemType(String begin, String end, String format, String mediaType) {
  }
}
