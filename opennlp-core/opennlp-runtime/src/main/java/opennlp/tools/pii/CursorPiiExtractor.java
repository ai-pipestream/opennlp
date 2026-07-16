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
import java.util.Locale;

import opennlp.tools.util.Span;

/**
 * A deterministic {@link PiiExtractor}: forward scans over the text, no regular
 * expressions, recognizing email addresses, phone numbers, IBANs, and payment card
 * numbers. IBANs and card numbers are checksum validated and phone numbers must show a
 * {@code +} prefix or visible formatting, so a random digit run is rejected rather than
 * reported.
 *
 * <p>Recognized forms:</p>
 * <ul>
 *   <li>Email: a local part of ASCII letters, digits, and {@code . _ % + -} followed by
 *   {@code @} and a dotted domain whose final label is alphabetic with at least two
 *   letters.</li>
 *   <li>Phone: an international form with {@code +} and 8 to 15 digits, or a domestic
 *   form with 10 or 11 digits that shows formatting evidence, at least one space,
 *   hyphen, or parenthesis between the digits. A bare digit run is never a phone
 *   number. Dots are not accepted as separators, which keeps decimal numbers out.</li>
 *   <li>IBAN: two uppercase letters, two check digits, and 11 to 30 more uppercase
 *   letters or digits, 15 to 34 characters in total, optionally in space-separated
 *   groups, validated with the mod-97 check. Country-specific length tables are not
 *   applied.</li>
 *   <li>Card: 13 to 19 digits, optionally separated by single spaces or hyphens,
 *   validated with the Luhn check and required to start with a digit between 2 and 6,
 *   the range that covers the major card networks.</li>
 * </ul>
 *
 * <p>When candidates overlap, the leftmost wins, then the longest, then the more
 * specific type in the order email, IBAN, card, phone; the reported mentions never
 * overlap. All candidates are checked against word boundaries so nothing is reported
 * from inside a longer alphanumeric run.</p>
 *
 * <p>The extractor holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class CursorPiiExtractor implements PiiExtractor {

  private static final int PRIORITY_EMAIL = 0;
  private static final int PRIORITY_IBAN = 1;
  private static final int PRIORITY_CARD = 2;
  private static final int PRIORITY_PHONE = 3;

  private static final int IBAN_MIN_LENGTH = 15;
  private static final int IBAN_MAX_LENGTH = 34;
  private static final int CARD_MIN_DIGITS = 13;
  private static final int CARD_MAX_DIGITS = 19;
  private static final int PHONE_MIN_INTERNATIONAL_DIGITS = 8;
  private static final int PHONE_MAX_DIGITS = 15;

  /**
   * One candidate found by a scanner, held until overlap resolution decides which
   * candidates survive.
   *
   * @param start The candidate start offset in the scanned text, inclusive.
   * @param end The candidate end offset in the scanned text, exclusive.
   * @param priority The type priority that breaks exact-span ties; a lower value is the
   *                 more specific type.
   * @param mention The mention to report if this candidate survives.
   */
  private record Hit(int start, int end, int priority, PiiMention mention) {
  }

  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hit> hits = new ArrayList<>();
    scanEmails(text, hits);
    scanIbans(text, hits);
    scanCards(text, hits);
    scanPhones(text, hits);
    return resolveOverlaps(hits);
  }

  /**
   * Finds email addresses by expanding around each {@code @}.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private static void scanEmails(CharSequence text, List<Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      if (text.charAt(i) != '@') {
        continue;
      }
      int start = i;
      while (start > 0 && isLocalChar(text.charAt(start - 1))) {
        start--;
      }
      if (start == i || !validLocalPart(text, start, i)) {
        continue;
      }
      int end = i + 1;
      while (end < text.length() && isDomainChar(text.charAt(end))) {
        end++;
      }
      while (end > i + 1 && (text.charAt(end - 1) == '.' || text.charAt(end - 1) == '-')) {
        end--;
      }
      if (end == i + 1
          || !validDomain(text.subSequence(i + 1, end).toString())
          || (start > 0 && Character.isLetterOrDigit(text.charAt(start - 1)))
          || (end < text.length() && Character.isLetterOrDigit(text.charAt(end)))) {
        continue;
      }
      final String normalized =
          text.subSequence(start, end).toString().toLowerCase(Locale.ROOT);
      hits.add(new Hit(start, end, PRIORITY_EMAIL,
          new PiiMention(new Span(start, end), PiiMention.TYPE_EMAIL, normalized)));
    }
  }

  /**
   * Checks a local part: not empty, no leading, trailing, or doubled dot.
   *
   * @param text The text being scanned.
   * @param start The local part start, inclusive.
   * @param at The position of the {@code @}.
   * @return {@code true} if the local part is acceptable.
   */
  private static boolean validLocalPart(CharSequence text, int start, int at) {
    if (text.charAt(start) == '.' || text.charAt(at - 1) == '.') {
      return false;
    }
    for (int i = start + 1; i < at; i++) {
      if (text.charAt(i) == '.' && text.charAt(i - 1) == '.') {
        return false;
      }
    }
    return true;
  }

  /**
   * Checks a domain: at least two labels, each 1 to 63 characters without a leading or
   * trailing hyphen, and a final label of at least two ASCII letters.
   *
   * @param domain The domain without the {@code @}.
   * @return {@code true} if the domain is acceptable.
   */
  private static boolean validDomain(String domain) {
    int labels = 0;
    int labelStart = 0;
    for (int i = 0; i <= domain.length(); i++) {
      if (i == domain.length() || domain.charAt(i) == '.') {
        final int length = i - labelStart;
        if (length < 1 || length > 63
            || domain.charAt(labelStart) == '-' || domain.charAt(i - 1) == '-') {
          return false;
        }
        labels++;
        labelStart = i + 1;
      }
    }
    if (labels < 2) {
      return false;
    }
    final int tldStart = domain.lastIndexOf('.') + 1;
    if (domain.length() - tldStart < 2) {
      return false;
    }
    for (int i = tldStart; i < domain.length(); i++) {
      if (!isAsciiLetter(domain.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Finds IBANs: a run of uppercase letters and digits in optional space-separated
   * groups, tried longest first at group boundaries until the mod-97 check passes.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private static void scanIbans(CharSequence text, List<Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      if (!isAsciiUpper(text.charAt(i))
          || (i > 0 && Character.isLetterOrDigit(text.charAt(i - 1)))
          || i + 3 >= text.length()
          || !isAsciiUpper(text.charAt(i + 1))
          || !isAsciiDigit(text.charAt(i + 2))
          || !isAsciiDigit(text.charAt(i + 3))) {
        continue;
      }
      final StringBuilder compact = new StringBuilder();
      final List<int[]> groupEnds = new ArrayList<>();
      int p = i;
      while (p < text.length() && compact.length() <= IBAN_MAX_LENGTH) {
        final char c = text.charAt(p);
        if (isAsciiUpper(c) || isAsciiDigit(c)) {
          compact.append(c);
          p++;
        } else if (c == ' ' && p + 1 < text.length()
            && (isAsciiUpper(text.charAt(p + 1)) || isAsciiDigit(text.charAt(p + 1)))) {
          groupEnds.add(new int[] {p, compact.length()});
          p++;
        } else {
          break;
        }
      }
      groupEnds.add(new int[] {p, compact.length()});
      for (int g = groupEnds.size() - 1; g >= 0; g--) {
        final int textEnd = groupEnds.get(g)[0];
        final int length = groupEnds.get(g)[1];
        if (length < IBAN_MIN_LENGTH || length > IBAN_MAX_LENGTH
            || (textEnd < text.length() && Character.isLetterOrDigit(text.charAt(textEnd)))) {
          continue;
        }
        final String candidate = compact.substring(0, length);
        if (mod97(candidate) == 1) {
          hits.add(new Hit(i, textEnd, PRIORITY_IBAN,
              new PiiMention(new Span(i, textEnd), PiiMention.TYPE_IBAN, candidate)));
          i = textEnd;
          break;
        }
      }
    }
  }

  /**
   * Finds payment card numbers: a digit run with optional single space or hyphen
   * separators, an accepted leading digit, and a passing Luhn check.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private static void scanCards(CharSequence text, List<Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      if (!isAsciiDigit(text.charAt(i)) || !onNumberStartBoundary(text, i)) {
        continue;
      }
      final StringBuilder digits = new StringBuilder();
      int lastDigit = -1;
      boolean previousSeparator = false;
      int p = i;
      while (p < text.length() && digits.length() <= CARD_MAX_DIGITS) {
        final char c = text.charAt(p);
        if (isAsciiDigit(c)) {
          digits.append(c);
          lastDigit = p;
          previousSeparator = false;
          p++;
        } else if ((c == ' ' || c == '-') && !previousSeparator) {
          previousSeparator = true;
          p++;
        } else {
          break;
        }
      }
      final int end = lastDigit + 1;
      final char first = digits.charAt(0);
      if (digits.length() < CARD_MIN_DIGITS || digits.length() > CARD_MAX_DIGITS
          || first < '2' || first > '6'
          || (end < text.length() && Character.isLetterOrDigit(text.charAt(end)))
          || !luhnValid(digits.toString())) {
        continue;
      }
      hits.add(new Hit(i, end, PRIORITY_CARD,
          new PiiMention(new Span(i, end), PiiMention.TYPE_CARD, digits.toString())));
      i = end;
    }
  }

  /**
   * Finds phone numbers: an international form starting with {@code +}, or a domestic
   * form whose digits are visibly formatted with spaces, hyphens, or parentheses.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private static void scanPhones(CharSequence text, List<Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      final char c = text.charAt(i);
      final boolean plus = c == '+';
      if ((!plus && !isAsciiDigit(c) && c != '(') || !onNumberStartBoundary(text, i)
          || (i > 0 && text.charAt(i - 1) == '+')) {
        continue;
      }
      int digits = 0;
      int lastDigit = -1;
      int open = 0;
      int close = 0;
      boolean separated = false;
      boolean previousSeparator = false;
      int p = plus ? i + 1 : i;
      while (p < text.length() && digits <= PHONE_MAX_DIGITS) {
        final char ch = text.charAt(p);
        if (isAsciiDigit(ch)) {
          if (digits > 0 && p > i && !isAsciiDigit(text.charAt(p - 1))) {
            separated = true;
          }
          digits++;
          lastDigit = p;
          previousSeparator = false;
          p++;
        } else if ((ch == ' ' || ch == '-') && !previousSeparator) {
          previousSeparator = true;
          p++;
        } else if (ch == '(' && open == 0) {
          open++;
          previousSeparator = false;
          p++;
        } else if (ch == ')' && close == 0 && open == 1) {
          close++;
          previousSeparator = false;
          p++;
        } else {
          break;
        }
      }
      if (digits == 0 || open != close) {
        continue;
      }
      final int end = lastDigit + 1;
      final boolean lengthOk = plus
          ? digits >= PHONE_MIN_INTERNATIONAL_DIGITS && digits <= PHONE_MAX_DIGITS
          : (digits == 10 || digits == 11) && separated;
      if (!lengthOk
          || (end < text.length() && Character.isLetterOrDigit(text.charAt(end)))
          || (end + 1 < text.length() && text.charAt(end) == '.'
              && isAsciiDigit(text.charAt(end + 1)))) {
        continue;
      }
      final StringBuilder normalized = new StringBuilder(plus ? "+" : "");
      for (int q = i; q < end; q++) {
        if (isAsciiDigit(text.charAt(q))) {
          normalized.append(text.charAt(q));
        }
      }
      hits.add(new Hit(i, end, PRIORITY_PHONE,
          new PiiMention(new Span(i, end), PiiMention.TYPE_PHONE, normalized.toString())));
      i = end;
    }
  }

  /**
   * Checks that a numeric candidate does not continue a word, a number, a decimal
   * fraction, or a comma-grouped number to its left.
   *
   * @param text The text being scanned.
   * @param start The candidate start.
   * @return {@code true} if the candidate may start here.
   */
  private static boolean onNumberStartBoundary(CharSequence text, int start) {
    if (start == 0) {
      return true;
    }
    final char previous = text.charAt(start - 1);
    if (Character.isLetterOrDigit(previous)) {
      return false;
    }
    return (previous != '.' && previous != ',')
        || start < 2 || !isAsciiDigit(text.charAt(start - 2));
  }

  /**
   * Resolves overlapping candidates: leftmost first, then longest, then the more
   * specific type.
   *
   * @param hits The raw candidates.
   * @return The surviving mentions in text order. Never {@code null}.
   */
  private static List<PiiMention> resolveOverlaps(List<Hit> hits) {
    hits.sort((a, b) -> {
      if (a.start() != b.start()) {
        return Integer.compare(a.start(), b.start());
      }
      if (a.end() != b.end()) {
        return Integer.compare(b.end(), a.end());
      }
      return Integer.compare(a.priority(), b.priority());
    });
    final List<PiiMention> mentions = new ArrayList<>();
    int lastEnd = 0;
    for (final Hit hit : hits) {
      if (hit.start() >= lastEnd) {
        mentions.add(hit.mention());
        lastEnd = hit.end();
      }
    }
    return mentions;
  }

  /**
   * Computes the IBAN mod-97 remainder of a compact candidate.
   *
   * @param compact The candidate without spaces, uppercase letters and digits only.
   * @return The remainder; {@code 1} for a valid IBAN.
   */
  private static int mod97(String compact) {
    final String rearranged = compact.substring(4) + compact.substring(0, 4);
    int remainder = 0;
    for (int i = 0; i < rearranged.length(); i++) {
      final char c = rearranged.charAt(i);
      if (isAsciiDigit(c)) {
        remainder = (remainder * 10 + (c - '0')) % 97;
      } else {
        remainder = (remainder * 100 + (c - 'A' + 10)) % 97;
      }
    }
    return remainder;
  }

  /**
   * Applies the Luhn check to a digit string.
   *
   * @param digits The digits to check.
   * @return {@code true} if the checksum passes.
   */
  private static boolean luhnValid(String digits) {
    int sum = 0;
    boolean twice = false;
    for (int i = digits.length() - 1; i >= 0; i--) {
      int d = digits.charAt(i) - '0';
      if (twice) {
        d *= 2;
        if (d > 9) {
          d -= 9;
        }
      }
      sum += d;
      twice = !twice;
    }
    return sum % 10 == 0;
  }

  /**
   * Tests for a character allowed in an email local part.
   *
   * @param c The character.
   * @return {@code true} for ASCII letters, digits, and {@code . _ % + -}.
   */
  private static boolean isLocalChar(char c) {
    return isAsciiLetter(c) || isAsciiDigit(c)
        || c == '.' || c == '_' || c == '%' || c == '+' || c == '-';
  }

  /**
   * Tests for a character allowed in an email domain.
   *
   * @param c The character.
   * @return {@code true} for ASCII letters, digits, dot, and hyphen.
   */
  private static boolean isDomainChar(char c) {
    return isAsciiLetter(c) || isAsciiDigit(c) || c == '.' || c == '-';
  }

  /**
   * Tests for an ASCII letter.
   *
   * @param c The character.
   * @return {@code true} for {@code A-Z} and {@code a-z}.
   */
  private static boolean isAsciiLetter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  }

  /**
   * Tests for an uppercase ASCII letter.
   *
   * @param c The character.
   * @return {@code true} for {@code A-Z}.
   */
  private static boolean isAsciiUpper(char c) {
    return c >= 'A' && c <= 'Z';
  }

  /**
   * Tests for an ASCII digit.
   *
   * @param c The character.
   * @return {@code true} for {@code 0-9}.
   */
  private static boolean isAsciiDigit(char c) {
    return c >= '0' && c <= '9';
  }
}
