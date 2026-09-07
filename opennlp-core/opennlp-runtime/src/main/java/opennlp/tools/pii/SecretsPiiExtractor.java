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

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Extracts AWS access key identifiers, GitHub tokens, JWT candidates and URL credentials.
 * This detector is opt-in.
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
 *   <li>JWT candidate: three non-empty, unpadded
 *   <a href="https://datatracker.ietf.org/doc/html/rfc4648#section-5">base64url</a>
 *   segments separated by dots. Encodings must have valid lengths and zero unused bits.
 *   The complete UTF-8 header must be a JSON object with one top-level {@code alg} member
 *   containing a non-empty ASCII string. JSON whitespace, escaped names and nested
 *   values are supported. Signatures, claims and other JOSE parameter semantics are not
 *   verified. Unsigned tokens with empty signatures and five-part encrypted tokens are
 *   excluded. See <a href="https://datatracker.ietf.org/doc/html/rfc7515">RFC 7515</a>
 *   and <a href="https://datatracker.ietf.org/doc/html/rfc7519">RFC 7519</a>.</li>
 *   <li>URL credential: the userinfo component of a URL, as
 *   <a href="https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1">RFC 3986</a>
 *   defines it: a non-empty username, a colon and a non-empty password before {@code @}.
 *   Schemes start with an ASCII letter and may include letters, digits, {@code +},
 *   {@code -} and {@code .}. Percent escapes must contain two hexadecimal digits.
 *   Only userinfo is reported; masking preserves the scheme, host and path. The scanner
 *   does not validate the host or scheme-specific rules.</li>
 * </ul>
 *
 * <p>Normalized values preserve the original text, including URL percent escapes.
 * These values can contain credentials. Use {@link HmacTokenizer} or
 * {@link PiiAuditReport} when output must exclude the credential text.</p>

 * <p>AWS, GitHub and JWT candidates cannot start or end within a run of Unicode
 * letters, digits or underscores. A hyphen also prevents a JWT candidate start.</p>
 *
 * <p>All four types are reported by default; the {@link #SecretsPiiExtractor(Set)}
 * constructor limits extraction to a subset.</p>
 *
 * <p>Instances have no per-call state and may be shared between threads.</p>
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

  private static final String SCHEME_SEPARATOR = "://";
  private static final int PERCENT_ESCAPE_LENGTH = 3;

  private final Set<String> types;

  /**
   * Initializes an extractor that reports all four types.
   */
  public SecretsPiiExtractor() {
    this.types = ALL_TYPES;
  }

  /**
   * Initializes an extractor for selected credential types.
   *
   * @param types The types to report: {@link PiiMention#TYPE_AWS_ACCESS_KEY},
   *              {@link PiiMention#TYPE_GITHUB_TOKEN}, {@link PiiMention#TYPE_JWT}, and
   *              {@link PiiMention#TYPE_URL_CREDENTIAL}. Must be non-null and non-empty,
   *              without null or unrecognized entries.
   * @throws IllegalArgumentException Thrown if {@code types} is {@code null} or empty, or
   *         contains a null or unrecognized type.
   */
  public SecretsPiiExtractor(Set<String> types) {
    if (types == null || types.isEmpty()) {
      throw new IllegalArgumentException("types must not be null or empty");
    }
    for (final String type : types) {
      if (type == null || !ALL_TYPES.contains(type)) {
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
      if (text.charAt(i) != 'A' || !onTokenStart(text, i, false)) {
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
      if (text.charAt(i) != 'g' || !onTokenStart(text, i, false)) {
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
    while (end < text.length() && end - tokenStart <= GITHUB_MAX_LENGTH) {
      final char c = text.charAt(end);
      if (!Ascii.isLetterOrDigit(c) && c != '_') {
        break;
      }
      end++;
    }
    if (end < text.length()) {
      final char c = text.charAt(end);
      if (Ascii.isLetterOrDigit(c) || c == '_') {
        return -1;
      }
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
    for (int i = 0; i < text.length(); i++) {
      if (!canStartJsonHeader(text.charAt(i)) || !onJwtStart(text, i)) {
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
        if (!isBase64UrlEncoding(text, segmentStart, p)) {
          segments = false;
        } else if (segment == 0) {
          headerEnd = p;
        }
      }
      if (!segments || !onJwtEnd(text, p) || !isJwsHeader(text, i, headerEnd)) {
        continue;
      }
      Hits.add(hits, i, p, PiiMention.TYPE_JWT, text.subSequence(i, p).toString());
      // The loop increment resumes the scan at the exclusive match end.
      i = p - 1;
    }
  }

  /**
   * Decodes the complete UTF-8 header and checks JSON syntax and its algorithm member.
   *
   * @param text The text being scanned.
   * @param start The first header character.
   * @param end The exclusive end of the header segment.
   * @return {@code true} if the header passes the candidate syntax checks.
   */
  private boolean isJwsHeader(CharSequence text, int start, int end) {
    final byte[] header = decodeBase64Url(text, start, end);
    try {
      return new JwsHeader(StandardCharsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(header)))
          .isValid();
    } catch (CharacterCodingException e) {
      return false;
    }
  }

  /**
   * Checks the first base64url character for a JSON object or JSON whitespace byte.
   *
   * @param c The first encoded character.
   * @return {@code true} for an encoded opening brace, space, tab, CR or LF.
   */
  private boolean canStartJsonHeader(char c) {
    return c == 'e' || c == 'I' || c == 'C' || c == 'D';
  }

  /**
   * Checks a non-empty unpadded base64url run, including unused low bits.
   *
   * @param text The candidate text; the run contains only base64url characters.
   * @param start The run start.
   * @param end The exclusive run end.
   * @return {@code true} for an encoding of a non-empty byte sequence.
   */
  private boolean isBase64UrlEncoding(CharSequence text, int start, int end) {
    final int length = end - start;
    if (length == 0 || length % 4 == 1) {
      return false;
    }
    final int unusedMask = switch (length % 4) {
      case 2 -> 0x0f;
      case 3 -> 0x03;
      default -> 0;
    };
    return (base64UrlValue(text.charAt(end - 1)) & unusedMask) == 0;
  }

  /**
   * Rejects candidates inside a longer base64url or dotted value.
   *
   * @param text The text being scanned.
   * @param start The candidate start.
   * @return {@code true} if the candidate may start here.
   */
  private boolean onJwtStart(CharSequence text, int start) {
    int p = start;
    while (p > 0 && text.charAt(p - 1) == '.') {
      p--;
    }
    return onTokenStart(text, start, true) && onTokenStart(text, p, true);
  }

  /**
   * Rejects extra segments, padding and non-URL base64 continuations.
   *
   * @param text The text being scanned.
   * @param end The exclusive candidate end.
   * @return {@code true} if the candidate may end here.
   */
  private boolean onJwtEnd(CharSequence text, int end) {
    if (!onTokenEnd(text, end)) {
      return false;
    }
    if (end < text.length()) {
      final char c = text.charAt(end);
      if (c == '=' || c == '+' || c == '/') {
        return false;
      }
    }
    int p = end;
    while (p < text.length() && text.charAt(p) == '.') {
      p++;
    }
    return onTokenEnd(text, p) && (p == text.length() || text.charAt(p) != '-');
  }

  /**
   * Decodes an already checked, unpadded base64url run.
   *
   * @param text The text being scanned.
   * @param start The first character to decode.
   * @param end The exclusive end of the characters to decode.
   * @return The decoded bytes.
   */
  private byte[] decodeBase64Url(CharSequence text, int start, int end) {
    final byte[] decoded = new byte[(int) ((long) (end - start) * 3 / 4)];
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
    return decoded;
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
      if (!startsWith(text, i, SCHEME_SEPARATOR) || !hasScheme(text, i)) {
        continue;
      }
      final int start = i + SCHEME_SEPARATOR.length();
      int p = start;
      int colon = -1;
      while (p < text.length() && isUserinfoChar(text.charAt(p))) {
        final char c = text.charAt(p);
        if (c == '%') {
          if (text.length() - p < PERCENT_ESCAPE_LENGTH || !Ascii.isHexDigit(text.charAt(p + 1))
              || !Ascii.isHexDigit(text.charAt(p + 2))) {
            break;
          }
          p += PERCENT_ESCAPE_LENGTH;
          continue;
        }
        if (c == ':' && colon < 0) {
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
   * Checks the complete scheme before an authority separator.
   *
   * @param text The text being scanned.
   * @param end The offset of the scheme's colon.
   * @return {@code true} for an ASCII scheme starting at an identifier boundary.
   */
  private boolean hasScheme(CharSequence text, int end) {
    int start = end;
    while (start > 0 && isSchemeChar(text.charAt(start - 1))) {
      start--;
    }
    return start < end && Ascii.isLetter(text.charAt(start)) && onTokenStart(text, start, false);
  }

  /**
   * Checks a character in the scheme after its initial ASCII letter.
   *
   * @param c The character to check.
   * @return {@code true} for an ASCII letter, digit, plus sign, hyphen or dot.
   */
  private boolean isSchemeChar(char c) {
    return Ascii.isLetterOrDigit(c) || c == '+' || c == '-' || c == '.';
  }

  /**
   * Checks userinfo characters and percent signs. The scanner validates escape digits.
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
   * Rejects starts after Unicode letters/digits or underscores. Base64url also excludes
   * a preceding hyphen.
   *
   * @param text The text being scanned.
   * @param start The candidate start.
   * @param base64Url Whether hyphen is part of the candidate alphabet.
   * @return {@code true} if the candidate may start here.
   */
  private boolean onTokenStart(CharSequence text, int start, boolean base64Url) {
    if (start == 0) {
      return true;
    }
    final char previous = text.charAt(start - 1);
    return Boundaries.onWordStart(text, start) && previous != '_'
        && (!base64Url || previous != '-');
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
