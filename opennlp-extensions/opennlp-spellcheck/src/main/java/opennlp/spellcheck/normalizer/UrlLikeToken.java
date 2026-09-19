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

package opennlp.spellcheck.normalizer;

/** Recognizes whole tokens that should bypass spelling correction. */
final class UrlLikeToken {

  private static final String HTTP = "http://";
  private static final String HTTPS = "https://";
  private static final String WWW = "www.";

  private UrlLikeToken() {
  }

  /**
   * Tests a punctuation-stripped whole token. This is an exclusion heuristic rather than
   * full URL, email, DNS, or IP address validation.
   *
   * @param token the whole token to test
   * @return whether the token has a supported URL, email, or domain-like shape
   * @throws IllegalArgumentException if {@code token} is {@code null}
   */
  static boolean matches(String token) {
    if (token == null) {
      throw new IllegalArgumentException("token must not be null");
    }
    if (token.isEmpty() || containsDelimiter(token)) {
      return false;
    }
    if (startsWithAsciiIgnoreCase(token, HTTPS)) {
      return networkTail(token, HTTPS.length(), false);
    }
    if (startsWithAsciiIgnoreCase(token, HTTP)) {
      return networkTail(token, HTTP.length(), false);
    }
    if (startsWithAsciiIgnoreCase(token, WWW)) {
      return networkTail(token, WWW.length(), true);
    }
    final int at = token.indexOf('@');
    if (at >= 0 && at == token.lastIndexOf('@') && validLocalPart(token, at)
        && validHost(token, at + 1, token.length(), true)) {
      return true;
    }
    return networkTail(token, 0, true);
  }

  /** Checks the authority; an opaque path, query or fragment may contain at-signs. */
  private static boolean networkTail(String token, int authorityStart, boolean requireDomain) {
    int authorityEnd = authorityStart;
    while (authorityEnd < token.length()) {
      final char c = token.charAt(authorityEnd);
      if (c == '/' || c == '?' || c == '#') {
        break;
      }
      authorityEnd++;
    }
    if (authorityEnd == authorityStart) {
      return false;
    }
    int hostStart = authorityStart;
    final int at = token.lastIndexOf('@', authorityEnd - 1);
    if (at >= authorityStart) {
      if (requireDomain || token.indexOf('@', authorityStart) != at
          || !validUserInfo(token, authorityStart, at)) {
        return false;
      }
      hostStart = at + 1;
    }
    return validHostAndPort(token, hostStart, authorityEnd, requireDomain);
  }

  /** Separates an optional decimal port from a host or bracketed HTTP address. */
  private static boolean validHostAndPort(String token, int start, int end,
      boolean requireDomain) {
    if (start >= end) {
      return false;
    }
    if (token.charAt(start) == '[') {
      final int close = token.indexOf(']', start + 1);
      if (requireDomain || close < 0 || close >= end
          || !bracketedAddress(token, start + 1, close)) {
        return false;
      }
      return close + 1 == end || token.charAt(close + 1) == ':'
          && validPort(token, close + 2, end);
    }
    final int colon = token.lastIndexOf(':', end - 1);
    final int hostEnd;
    if (colon >= start) {
      if (token.indexOf(':', start) != colon || !validPort(token, colon + 1, end)) {
        return false;
      }
      hostEnd = colon;
    } else {
      hostEnd = end;
    }
    return validHost(token, start, hostEnd, requireDomain);
  }

  /** Checks nonempty domain labels and the optional bare-domain suffix requirement. */
  private static boolean validHost(String token, int start, int end, boolean requireDomain) {
    if (start >= end) {
      return false;
    }
    int labelStart = start;
    int dots = 0;
    for (int i = start; i < end;) {
      final int codePoint = token.codePointAt(i);
      if (codePoint == '.') {
        if (!validLabel(token, labelStart, i)) {
          return false;
        }
        dots++;
        labelStart = i + 1;
      }
      i += Character.charCount(codePoint);
    }
    if (!validLabel(token, labelStart, end)) {
      return false;
    }
    return !requireDomain || dots > 0 && alphabeticFinalLabel(token, labelStart, end);
  }

  /** Accepts letters and digits, interior hyphens, and marks following a base. */
  private static boolean validLabel(String token, int start, int end) {
    if (start >= end || token.charAt(start) == '-' || token.charAt(end - 1) == '-') {
      return false;
    }
    boolean hasBase = false;
    boolean markAllowed = false;
    for (int i = start; i < end;) {
      final int codePoint = token.codePointAt(i);
      if (Character.isLetterOrDigit(codePoint)) {
        hasBase = true;
        markAllowed = true;
      } else if (codePoint == '-') {
        // Allowed only inside a label, as checked above.
        markAllowed = false;
      } else if (isMark(codePoint) && markAllowed) {
        // International labels may contain combining marks after a base character.
      } else {
        return false;
      }
      i += Character.charCount(codePoint);
    }
    return hasBase;
  }

  /** Requires two letters, allowing their combining continuations. */
  private static boolean alphabeticFinalLabel(String token, int start, int end) {
    int letters = 0;
    boolean hasBase = false;
    for (int i = start; i < end;) {
      final int codePoint = token.codePointAt(i);
      if (Character.isLetter(codePoint)) {
        hasBase = true;
        letters++;
      } else if (isMark(codePoint) && hasBase) {
        // A mark extends its preceding letter but does not satisfy the two-letter minimum.
      } else {
        return false;
      }
      i += Character.charCount(codePoint);
    }
    return letters >= 2;
  }

  /** Checks a dot-separated email-like local part without empty atoms. */
  private static boolean validLocalPart(String token, int end) {
    if (end == 0 || token.charAt(0) == '.' || token.charAt(end - 1) == '.') {
      return false;
    }
    boolean previousDot = false;
    boolean hasBase = false;
    boolean markAllowed = false;
    for (int i = 0; i < end;) {
      final int codePoint = token.codePointAt(i);
      if (codePoint == '.') {
        if (previousDot) {
          return false;
        }
        previousDot = true;
        markAllowed = false;
      } else if (Character.isLetterOrDigit(codePoint)) {
        hasBase = true;
        previousDot = false;
        markAllowed = true;
      } else if (isMark(codePoint) && markAllowed) {
        previousDot = false;
      } else if (codePoint < 128 && isAtextPunctuation((char) codePoint)) {
        previousDot = false;
        markAllowed = false;
      } else {
        return false;
      }
      i += Character.charCount(codePoint);
    }
    return hasBase;
  }

  /** Checks the nonempty user information of an explicitly prefixed HTTP URL. */
  private static boolean validUserInfo(String token, int start, int end) {
    if (start >= end) {
      return false;
    }
    boolean markAllowed = false;
    for (int i = start; i < end;) {
      final int codePoint = token.codePointAt(i);
      if (Character.isLetterOrDigit(codePoint)) {
        markAllowed = true;
      } else if (isMark(codePoint) && markAllowed) {
        // International userinfo may contain a mark after its base character.
      } else if (codePoint < 128 && isUserInfoPunctuation((char) codePoint)) {
        markAllowed = false;
      } else {
        return false;
      }
      i += Character.charCount(codePoint);
    }
    return true;
  }

  /** Recognizes the ASCII punctuation admitted in an unquoted email atom. */
  private static boolean isAtextPunctuation(char c) {
    return switch (c) {
      case '!', '#', '$', '%', '&', '\'', '*', '+', '-', '/', '=', '?', '^', '_', '`',
          '{', '|', '}', '~' -> true;
      default -> false;
    };
  }

  /** Recognizes ASCII punctuation in the user-information heuristic. */
  private static boolean isUserInfoPunctuation(char c) {
    return switch (c) {
      case '-', '.', '_', '~', '!', '$', '&', '\'', '(', ')', '*', '+', ',', ';', '=', ':',
          '%' -> true;
      default -> false;
    };
  }

  /** Recognizes a colon-containing address shape without DNS or IP validation. */
  private static boolean bracketedAddress(String token, int start, int end) {
    boolean colon = false;
    for (int i = start; i < end; i++) {
      final char c = token.charAt(i);
      if (c == ':') {
        colon = true;
      } else if (!isAsciiHex(c) && c != '.') {
        return false;
      }
    }
    return start < end && colon;
  }

  /** Checks an ASCII decimal port within the unsigned 16-bit range. */
  private static boolean validPort(String token, int start, int end) {
    if (start >= end || end - start > 5) {
      return false;
    }
    int port = 0;
    for (int i = start; i < end; i++) {
      final char c = token.charAt(i);
      if (c < '0' || c > '9') {
        return false;
      }
      port = port * 10 + c - '0';
    }
    return port <= 65_535;
  }

  /** Rejects whitespace, controls and malformed UTF-16 anywhere in the token. */
  private static boolean containsDelimiter(String token) {
    for (int i = 0; i < token.length();) {
      final int codePoint = token.codePointAt(i);
      final int type = Character.getType(codePoint);
      if (Character.isWhitespace(codePoint) || Character.isSpaceChar(codePoint)
          || Character.isISOControl(codePoint)
          || type == Character.SURROGATE) {
        return true;
      }
      i += Character.charCount(codePoint);
    }
    return false;
  }

  /** Recognizes Unicode combining mark categories. */
  private static boolean isMark(int codePoint) {
    final int type = Character.getType(codePoint);
    return type == Character.NON_SPACING_MARK || type == Character.COMBINING_SPACING_MARK
        || type == Character.ENCLOSING_MARK;
  }

  /** Folds only ASCII scheme letters, excluding Unicode case lookalikes. */
  private static boolean startsWithAsciiIgnoreCase(String value, String prefix) {
    if (value.length() < prefix.length()) {
      return false;
    }
    for (int i = 0; i < prefix.length(); i++) {
      char c = value.charAt(i);
      if (c >= 'A' && c <= 'Z') {
        c += 'a' - 'A';
      }
      if (c != prefix.charAt(i)) {
        return false;
      }
    }
    return true;
  }

  /** Recognizes the literal hexadecimal alphabet used in bracketed address shapes. */
  private static boolean isAsciiHex(char c) {
    return c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F';
  }
}
