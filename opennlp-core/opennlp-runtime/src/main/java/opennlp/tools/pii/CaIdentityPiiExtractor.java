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
import java.util.List;

/**
 * A deterministic {@link PiiExtractor} for Canadian Social Insurance Numbers: forward
 * scans over the text, no regular expressions. This extractor is opt-in.
 *
 * <p>A candidate is nine digits, compact or in three groups separated consistently by
 * spaces or hyphens, and must pass the Luhn check. It is reported only when directly
 * preceded by an ASCII case-insensitive {@code SIN}, {@code Social Insurance Number}, or
 * {@code Social Insurance No.} label. The check establishes a valid number shape, not
 * that the Government of Canada issued the number.</p>
 *
 * <p>The required label is the primary false-positive control: many unrelated nine-digit
 * values pass Luhn by chance.</p>
 *
 * <p>Normalized form: the nine digits without separators.</p>
 *
 * <p>The extractor holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class CaIdentityPiiExtractor implements PiiExtractor {

  private static final String[] LABELS =
      {"social insurance number", "social insurance no", "sin"};
  private static final int DIGITS = 9;
  private static final int GROUP_DIGITS = 3;
  private static final int CHECK_MODULUS = 10;

  /** Initializes an extractor for Canadian Social Insurance Numbers. */
  public CaIdentityPiiExtractor() {
  }

  /**
   * {@inheritDoc}
   *
   * <p>Reports one mention per labeled, checksum-valid SIN, in text order.</p>
   */
  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hits.Hit> hits = new ArrayList<>();
    for (int i = 0; i < text.length(); i++) {
      if (!Ascii.isDigit(text.charAt(i)) || !Boundaries.onNumberStart(text, i)
          || !hasLabel(text, i)) {
        continue;
      }
      final StringBuilder normalized = new StringBuilder(DIGITS);
      final int end = readDigits(text, i, normalized);
      if (end < 0 || !luhnValid(normalized)) {
        continue;
      }
      Hits.add(hits, i, end, PiiMention.TYPE_CA_SIN, normalized.toString());
      i = end - 1;
    }
    return Hits.resolve(hits);
  }

  /**
   * Checks for one of the required labels immediately before a candidate.
   *
   * @param text The text being scanned.
   * @param start The first digit of the candidate.
   * @return {@code true} if an isolated recognized label precedes the candidate.
   */
  private boolean hasLabel(CharSequence text, int start) {
    int end = start;
    while (end > 0 && isLabelSeparator(text.charAt(end - 1))) {
      end--;
    }
    for (final String label : LABELS) {
      final int labelStart = end - label.length();
      if (labelStart >= 0 && equalsAsciiIgnoreCase(text, labelStart, label)
          && (labelStart == 0 || !Ascii.isLetterOrDigit(text.charAt(labelStart - 1)))) {
        return true;
      }
    }
    return false;
  }

  /**
   * Reads a compact or consistently grouped nine-digit candidate.
   *
   * @param text The text being scanned.
   * @param start The first digit.
   * @param normalized The digit collector.
   * @return The exclusive candidate end, or {@code -1} if its form is invalid.
   */
  private int readDigits(CharSequence text, int start, StringBuilder normalized) {
    int p = start;
    char separator = 0;
    while (p < text.length() && normalized.length() < DIGITS) {
      final char c = text.charAt(p);
      if (Ascii.isDigit(c)) {
        normalized.append(c);
        p++;
      } else if ((c == '-' || c == ' ') && normalized.length() % GROUP_DIGITS == 0
          && p + 1 < text.length() && Ascii.isDigit(text.charAt(p + 1))) {
        if (separator != 0 && separator != c) {
          return -1;
        }
        separator = c;
        p++;
      } else {
        break;
      }
    }
    if (normalized.length() != DIGITS || !Boundaries.onEnd(text, p)) {
      return -1;
    }
    if (p + 1 < text.length() && (text.charAt(p) == '-' || text.charAt(p) == ' ')
        && Ascii.isDigit(text.charAt(p + 1))) {
      return -1;
    }
    return p;
  }

  /**
   * Applies the Luhn check over a normalized candidate.
   *
   * @param digits The nine digits.
   * @return {@code true} if the check digit holds.
   */
  private boolean luhnValid(CharSequence digits) {
    int sum = 0;
    boolean doubled = false;
    for (int i = digits.length() - 1; i >= 0; i--) {
      int digit = digits.charAt(i) - '0';
      if (doubled) {
        digit *= 2;
        if (digit > 9) {
          digit -= 9;
        }
      }
      sum += digit;
      doubled = !doubled;
    }
    return sum % CHECK_MODULUS == 0;
  }

  /**
   * Tests for punctuation permitted between a label and its value.
   *
   * @param c The character.
   * @return {@code true} for accepted label separators.
   */
  private boolean isLabelSeparator(char c) {
    return Character.isWhitespace(c) || c == ':' || c == '#' || c == '=' || c == '.';
  }

  /**
   * Compares an ASCII literal without allocating a folded copy.
   *
   * @param text The text being scanned.
   * @param start The candidate literal start.
   * @param literal The lowercase literal.
   * @return {@code true} if the literal matches ignoring ASCII case.
   */
  private boolean equalsAsciiIgnoreCase(CharSequence text, int start, String literal) {
    for (int i = 0; i < literal.length(); i++) {
      if (Ascii.toLower(text.charAt(start + i)) != literal.charAt(i)) {
        return false;
      }
    }
    return true;
  }
}
