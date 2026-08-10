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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * A deterministic {@link PiiExtractor} for credentials that leak into text: forward scans
 * over the text, no regular expressions, recognizing AWS access key identifiers, GitHub
 * access tokens, JSON Web Tokens, and credentials embedded in a URL. Every form is
 * anchored by a fixed prefix or by a structure that must parse, so no candidate rests on
 * length alone. This extractor is opt-in.
 *
 * <p>Recognized forms:</p>
 * <ul>
 *   <li>AWS access key: the identifier prefixes {@code AKIA} for a long-term key and
 *   {@code ASIA} for a temporary one, followed by 16 uppercase letters and digits, as
 *   <a href="https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_identifiers.html">
 *   the IAM identifier reference</a> describes. The prefixes that mark a user, role, or
 *   policy identifier rather than a key are not reported, since those are not
 *   secrets.</li>
 *   <li>GitHub token: the prefixes {@code ghp_}, {@code gho_}, {@code ghu_},
 *   {@code ghs_}, and {@code ghr_} followed by at least 36 token characters, or
 *   {@code github_pat_} followed by at least 82 token characters. Both forms accept
 *   letters, digits, and underscores and are capped at 255 characters, following the
 *   <a href="https://github.blog/2021-04-05-behind-githubs-new-authentication-token-formats/">
 *   documented token formats</a>. The prefixes are case sensitive.</li>
 *   <li>JSON Web Token: three
 *   <a href="https://datatracker.ietf.org/doc/html/rfc4648#section-5">base64url</a>
 *   segments separated by dots, the compact serialization of
 *   <a href="https://datatracker.ietf.org/doc/html/rfc7519">RFC 7519</a>. The first
 *   segment is decoded and must be a JSON object carrying the {@code alg} header that
 *   <a href="https://datatracker.ietf.org/doc/html/rfc7515">RFC 7515</a> requires, so a
 *   dotted run of base64url characters is not enough.</li>
 *   <li>URL credential: the userinfo component of a URL, as
 *   <a href="https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1">RFC 3986</a>
 *   defines it, when it carries a password: a user name, a colon, and a non-empty
 *   password before the {@code @}. Only the credential is reported, not the whole URL, so
 *   masking it leaves the scheme and host readable. A userinfo without a password is not
 *   reported, since a bare user name in a URL is not a secret.</li>
 * </ul>
 *
 * <p>Normalized forms: every type keeps the credential exactly as written, since a
 * credential has no formatting to remove and comparing two occurrences character by
 * character is what a caller needs. A mention therefore carries the secret; use
 * {@link HmacTokenizer} or {@link PiiAuditReport} rather than the normalized form when
 * building an artifact that must not hold the secret itself.</p>
 *
 * <p>All four types are reported by default; the {@link #SecretsPiiExtractor(Set)}
 * constructor limits extraction to a subset.</p>
 *
 * <p>The extractor holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class SecretsPiiExtractor implements PiiExtractor {

  private static final Set<String> ALL_TYPES = Set.of(PiiMention.TYPE_AWS_ACCESS_KEY,
      PiiMention.TYPE_GITHUB_TOKEN, PiiMention.TYPE_JWT, PiiMention.TYPE_URL_CREDENTIAL);

  /**
   * The AWS identifier prefixes that mark an access key rather than a resource, from the IAM
   * identifier reference as of 2026-08-10. Unlike a checksum, this table is a record of what
   * a vendor issues today: a new key prefix means a value this scanner will not report until
   * the table is updated, so the date matters.
   */
  private static final String[] AWS_KEY_PREFIXES = {"AKIA", "ASIA"};

  /**
   * The GitHub token prefixes of the 40-character form, from the token format announcement
   * as of 2026-08-10. Read the note on {@link #AWS_KEY_PREFIXES} before relying on it.
   */
  private static final String[] GITHUB_PREFIXES = {"ghp_", "gho_", "ghu_", "ghs_", "ghr_"};

  private static final String GITHUB_FINE_GRAINED_PREFIX = "github_pat_";

  private static final int AWS_BODY_LENGTH = 16;
  private static final int GITHUB_BODY_LENGTH = 36;
  private static final int GITHUB_FINE_GRAINED_BODY_LENGTH = 82;
  private static final int GITHUB_MAX_LENGTH = 255;

  private static final int JWT_SEGMENTS = 3;
  private static final int JWT_HEADER_MIN_LENGTH = 8;
  private static final int JWT_PAYLOAD_MIN_LENGTH = 4;
  private static final int JWT_SIGNATURE_MIN_LENGTH = 4;

  /** The base64url prefix of a JSON object, that is of {@code {"}. */
  private static final String JWT_HEADER_PREFIX = "eyJ";

  /** As many header characters as any {@code alg} declaration needs to be visible in. */
  private static final int JWT_HEADER_SCAN_LENGTH = 88;

  /**
   * The header parameter every JWS header must carry, as it is written in the JSON: with
   * its quotes, so that a longer member name ending in those three letters, {@code notalg}
   * for instance, is not mistaken for it.
   */
  private static final String JWT_ALGORITHM_PARAMETER = "\"alg\"";

  private static final String SCHEME_SEPARATOR = "://";

  private final Set<String> types;

  /**
   * Initializes an extractor that reports all four types.
   */
  public SecretsPiiExtractor() {
    this.types = ALL_TYPES;
  }

  /**
   * Initializes an extractor limited to a subset of the types, for a caller that scans
   * commit messages for cloud keys, for example, without flagging every signed token.
   *
   * @param types The types to report, drawn from
   *              {@link PiiMention#TYPE_AWS_ACCESS_KEY},
   *              {@link PiiMention#TYPE_GITHUB_TOKEN}, {@link PiiMention#TYPE_JWT}, and
   *              {@link PiiMention#TYPE_URL_CREDENTIAL}. Must not be {@code null} or
   *              empty and must not contain a type this extractor does not recognize.
   * @throws IllegalArgumentException Thrown if {@code types} is {@code null} or empty, or
   *         contains an unrecognized type.
   */
  public SecretsPiiExtractor(Set<String> types) {
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
   * <p>Each enabled type is scanned for independently; overlapping candidates are then
   * reduced to a non-overlapping set, leftmost and longest first. A token inside a URL
   * credential is therefore reported once, as the credential that contains it.</p>
   */
  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hits.Hit> hits = new ArrayList<>();
    if (types.contains(PiiMention.TYPE_URL_CREDENTIAL)) {
      scanUrlCredentials(text, hits);
    }
    if (types.contains(PiiMention.TYPE_JWT)) {
      scanJwts(text, hits);
    }
    if (types.contains(PiiMention.TYPE_AWS_ACCESS_KEY)) {
      scanAwsKeys(text, hits);
    }
    if (types.contains(PiiMention.TYPE_GITHUB_TOKEN)) {
      scanGithubTokens(text, hits);
    }
    return Hits.resolve(hits);
  }

  /**
   * Finds AWS access key identifiers.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanAwsKeys(CharSequence text, List<Hits.Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      if (text.charAt(i) != 'A' || !Boundaries.onWordStart(text, i)) {
        continue;
      }
      for (final String prefix : AWS_KEY_PREFIXES) {
        if (!startsWith(text, i, prefix)) {
          continue;
        }
        final int end = i + prefix.length() + AWS_BODY_LENGTH;
        if (end > text.length() || !onTokenEnd(text, end)) {
          continue;
        }
        boolean body = true;
        for (int p = i + prefix.length(); p < end; p++) {
          final char c = text.charAt(p);
          body &= Ascii.isUpper(c) || Ascii.isDigit(c);
        }
        if (body) {
          Hits.add(hits, i, end, PiiMention.TYPE_AWS_ACCESS_KEY,
              text.subSequence(i, end).toString());
          // The loop increment resumes the scan at the exclusive match end.
          i = end - 1;
          break;
        }
      }
    }
  }

  /**
   * Finds GitHub access tokens in the short and the fine-grained form.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanGithubTokens(CharSequence text, List<Hits.Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      if (text.charAt(i) != 'g' || !Boundaries.onWordStart(text, i)) {
        continue;
      }
      int end = -1;
      if (startsWith(text, i, GITHUB_FINE_GRAINED_PREFIX)) {
        end = tokenEnd(text, i + GITHUB_FINE_GRAINED_PREFIX.length(),
            GITHUB_FINE_GRAINED_BODY_LENGTH, i);
      } else {
        for (final String prefix : GITHUB_PREFIXES) {
          if (startsWith(text, i, prefix)) {
            end = tokenEnd(text, i + prefix.length(), GITHUB_BODY_LENGTH, i);
            break;
          }
        }
      }
      if (end < 0) {
        continue;
      }
      Hits.add(hits, i, end, PiiMention.TYPE_GITHUB_TOKEN, text.subSequence(i, end).toString());
      // The loop increment resumes the scan at the exclusive match end.
      i = end - 1;
    }
  }

  /**
   * Reads a variable-length GitHub token body and checks its bounds.
   *
   * @param text The text being scanned.
   * @param start The first body character.
   * @param minimumLength The minimum number of body characters for this prefix.
   * @param tokenStart The first character of the prefix.
   * @return The exclusive end offset of the token, or {@code -1} if the body does not
   *         have the prescribed form.
   */
  private int tokenEnd(CharSequence text, int start, int minimumLength, int tokenStart) {
    int end = start;
    while (end < text.length()) {
      final char c = text.charAt(end);
      if (!Ascii.isLetterOrDigit(c) && c != '_') {
        break;
      }
      end++;
    }
    if (end - start < minimumLength || end - tokenStart > GITHUB_MAX_LENGTH
        || !onTokenEnd(text, end)) {
      return -1;
    }
    return end;
  }

  /**
   * Finds JSON Web Tokens in compact serialization.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanJwts(CharSequence text, List<Hits.Hit> hits) {
    final int[] minimum =
        {JWT_HEADER_MIN_LENGTH, JWT_PAYLOAD_MIN_LENGTH, JWT_SIGNATURE_MIN_LENGTH};
    for (int i = 0; i < text.length(); i++) {
      if (!startsWith(text, i, JWT_HEADER_PREFIX) || !Boundaries.onWordStart(text, i)) {
        continue;
      }
      int p = i;
      int headerEnd = -1;
      boolean segments = true;
      for (int segment = 0; segment < JWT_SEGMENTS && segments; segment++) {
        if (segment > 0) {
          if (p >= text.length() || text.charAt(p) != '.') {
            segments = false;
            break;
          }
          p++;
        }
        final int segmentStart = p;
        while (p < text.length() && isBase64UrlChar(text.charAt(p))) {
          p++;
        }
        if (p - segmentStart < minimum[segment]) {
          segments = false;
        } else if (segment == 0) {
          headerEnd = p;
        }
      }
      if (!segments || !onTokenEnd(text, p) || !isJwsHeader(text, i, headerEnd)) {
        continue;
      }
      Hits.add(hits, i, p, PiiMention.TYPE_JWT, text.subSequence(i, p).toString());
      // The loop increment resumes the scan at the exclusive match end.
      i = p - 1;
    }
  }

  /**
   * Decodes a candidate's first segment and checks that it is a JSON object carrying the
   * {@code alg} header parameter, which is what tells a token from any other dotted run
   * of base64url characters.
   *
   * <p>The parameter has to appear as a member name, quoted and followed by its colon.
   * Looking for the three letters anywhere would accept a header whose only member is
   * named {@code notalg} or {@code algorithm}, and a base64url segment that happens to
   * decode to text containing them.</p>
   *
   * @param text The text being scanned.
   * @param start The first header character.
   * @param end The exclusive end of the header segment.
   * @return {@code true} if the segment decodes to a JWS header.
   */
  private boolean isJwsHeader(CharSequence text, int start, int end) {
    final int scanEnd = Math.min(end, start + JWT_HEADER_SCAN_LENGTH);
    final byte[] header = decodeBase64Url(text, start, scanEnd);
    if (header.length == 0 || header[0] != '{') {
      return false;
    }
    for (int i = 0; i + JWT_ALGORITHM_PARAMETER.length() <= header.length; i++) {
      boolean match = true;
      for (int j = 0; j < JWT_ALGORITHM_PARAMETER.length(); j++) {
        match &= header[i + j] == (byte) JWT_ALGORITHM_PARAMETER.charAt(j);
      }
      if (match && followedByColon(header, i + JWT_ALGORITHM_PARAMETER.length())) {
        return true;
      }
    }
    return false;
  }

  /**
   * Tests whether a member name is followed by its colon, across the whitespace JSON
   * allows there.
   *
   * @param header The decoded header bytes.
   * @param from The first byte after the member name.
   * @return {@code true} if a colon follows.
   */
  private boolean followedByColon(byte[] header, int from) {
    for (int i = from; i < header.length; i++) {
      final byte b = header[i];
      if (b == ':') {
        return true;
      }
      if (b != ' ' && b != '\t' && b != '\n' && b != '\r') {
        return false;
      }
    }
    return false;
  }

  /**
   * Decodes base64url characters to bytes, ignoring a trailing group too short to form
   * one more byte.
   *
   * @param text The text being scanned.
   * @param start The first character to decode.
   * @param end The exclusive end of the characters to decode.
   * @return The decoded bytes. Never {@code null}.
   */
  private byte[] decodeBase64Url(CharSequence text, int start, int end) {
    final byte[] decoded = new byte[(end - start) * 3 / 4 + 1];
    int accumulator = 0;
    int bits = 0;
    int length = 0;
    for (int i = start; i < end; i++) {
      accumulator = accumulator << 6 | base64UrlValue(text.charAt(i));
      bits += 6;
      if (bits >= 8) {
        bits -= 8;
        decoded[length++] = (byte) (accumulator >> bits & 0xFF);
      }
    }
    return Arrays.copyOf(decoded, length);
  }

  /**
   * Finds credentials in the userinfo component of a URL.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanUrlCredentials(CharSequence text, List<Hits.Hit> hits) {
    // A scheme is at least one letter long, so no credential can start before offset 1.
    for (int i = 1; i + SCHEME_SEPARATOR.length() < text.length(); i++) {
      if (!startsWith(text, i, SCHEME_SEPARATOR) || !Ascii.isLetter(text.charAt(i - 1))) {
        continue;
      }
      final int start = i + SCHEME_SEPARATOR.length();
      int p = start;
      int colon = -1;
      while (p < text.length() && isUserinfoChar(text.charAt(p))) {
        if (text.charAt(p) == ':' && colon < 0) {
          colon = p;
        }
        p++;
      }
      if (p >= text.length() || text.charAt(p) != '@' || colon <= start || colon + 1 >= p) {
        continue;
      }
      Hits.add(hits, start, p, PiiMention.TYPE_URL_CREDENTIAL,
          text.subSequence(start, p).toString());
      // The loop increment resumes the scan at the exclusive credential end.
      i = p - 1;
    }
  }

  /**
   * Tests for a character allowed in the userinfo component of a URL: the unreserved
   * characters, the sub-delimiters, the percent sign of an escape, and the colon that
   * separates the user name from the password.
   *
   * @param c The character.
   * @return {@code true} if the character may appear in a userinfo component.
   */
  private boolean isUserinfoChar(char c) {
    if (Ascii.isLetterOrDigit(c)) {
      return true;
    }
    return switch (c) {
      case '-', '.', '_', '~', '%', '!', '$', '&', '\'', '(', ')', '*', '+', ',', ';',
           '=', ':' -> true;
      default -> false;
    };
  }

  /**
   * Tests for a base64url character.
   *
   * @param c The character.
   * @return {@code true} for a letter, a digit, {@code -}, or {@code _}.
   */
  private boolean isBase64UrlChar(char c) {
    return Ascii.isLetterOrDigit(c) || c == '-' || c == '_';
  }

  /**
   * Reads the value of a base64url character.
   *
   * @param c The character, which must be a base64url character.
   * @return The value {@code 0} to {@code 63}.
   */
  private int base64UrlValue(char c) {
    if (Ascii.isUpper(c)) {
      return c - 'A';
    }
    if (Ascii.isLower(c)) {
      return c - 'a' + 26;
    }
    if (Ascii.isDigit(c)) {
      return c - '0' + 52;
    }
    return c == '-' ? 62 : 63;
  }

  /**
   * Checks that a token ends at {@code end}: no letter, digit, or underscore follows, so
   * a prefix of a longer identifier is never reported.
   *
   * @param text The text being scanned.
   * @param end The candidate end, exclusive.
   * @return {@code true} if the token may end here.
   */
  private boolean onTokenEnd(CharSequence text, int end) {
    return Boundaries.onEnd(text, end)
        && (end >= text.length() || text.charAt(end) != '_');
  }

  /**
   * Tests whether a literal occurs at an offset.
   *
   * @param text The text being scanned.
   * @param start The offset to compare at.
   * @param literal The literal to look for.
   * @return {@code true} if the text carries the literal at that offset.
   */
  private boolean startsWith(CharSequence text, int start, String literal) {
    if (start + literal.length() > text.length()) {
      return false;
    }
    for (int i = 0; i < literal.length(); i++) {
      if (text.charAt(start + i) != literal.charAt(i)) {
        return false;
      }
    }
    return true;
  }
}
