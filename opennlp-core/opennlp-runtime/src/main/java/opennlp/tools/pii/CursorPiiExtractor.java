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
import java.util.Set;

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
 *   {@code @} and a dotted domain of at most {@link #DOMAIN_MAX_LENGTH} characters whose
 *   final label is an
 *   <a href="https://data.iana.org/TLD/tlds-alpha-by-domain.txt">IANA-registered</a>
 *   top-level domain, including punycode forms. Private-use suffixes such as
 *   {@code .internal} or {@code .local} are not reported. The ASCII local part is limited
 *   to 64 characters and the complete mailbox to 254 characters.</li>
 *   <li>Phone: an international form with {@code +} whose digits split into an
 *   assigned calling code and a national number of a length some territory under that
 *   code assigns, or a domestic form with 10 or 11 digits that shows formatting
 *   evidence, at least one space, hyphen, or parenthesis between the digits. A bare
 *   digit run is never a phone number. Dots are not accepted as separators, which
 *   keeps decimal numbers out.</li>
 *   <li>IBAN: two uppercase letters, two check digits, and more uppercase letters or
 *   digits, optionally in space-separated groups, validated with the
 *   <a href="https://en.wikipedia.org/wiki/International_Bank_Account_Number">ISO 13616</a>
 *   mod-97 check. The country code must be in the ISO 13616 registry and the candidate
 *   must have exactly the length that registry entry assigns, so a checksum-passing run
 *   with an unregistered country or a wrong length is rejected.</li>
 *   <li>Card: 13 to 19 digits, optionally separated by single spaces or hyphens,
 *   validated with the <a href="https://en.wikipedia.org/wiki/Luhn_algorithm">Luhn</a>
 *   check and required to start with a digit between 2 and 6,
 *   the range that covers the major card networks. When the full run fails the check,
 *   shorter separator-delimited prefixes are tried longest first, so a trailing
 *   separated digit group, such as an expiry date, does not hide the card before
 *   it.</li>
 * </ul>
 *
 * <p>When candidates overlap, the leftmost wins, then the longest, then the more
 * specific type in the order email, IBAN, card, phone; the reported mentions never
 * overlap. All candidates are checked against word boundaries so nothing is reported
 * from inside a longer alphanumeric run.</p>
 *
 * <p>Normalized forms: email addresses are lowercased, IBANs keep their uppercase
 * letters and digits with separators removed, and phone and card numbers keep digits
 * only, with a leading {@code +} preserved for phone numbers.</p>
 *
 * <p>All four types are reported by default; the {@link #CursorPiiExtractor(Set)}
 * constructor limits extraction to a subset.</p>
 *
 * <p>The extractor holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class CursorPiiExtractor implements PiiExtractor {

  private static final Set<String> ALL_TYPES = Set.of(PiiMention.TYPE_EMAIL,
      PiiMention.TYPE_PHONE, PiiMention.TYPE_IBAN, PiiMention.TYPE_CARD);

  private static final int IBAN_MAX_LENGTH = 34;
  private static final int IBAN_MODULUS = 97;
  private static final int IBAN_ROTATION = 4;
  private static final int CARD_MIN_DIGITS = 13;
  private static final int CARD_MAX_DIGITS = 19;
  private static final int PHONE_MAX_DIGITS = 15;
  private static final int PHONE_DOMESTIC_MIN_DIGITS = 10;
  private static final int PHONE_DOMESTIC_MAX_DIGITS = 11;
  private static final int DOMAIN_LABEL_MAX_LENGTH = 63;
  private static final int EMAIL_LOCAL_PART_MAX_LENGTH = 64;
  private static final int EMAIL_MAX_LENGTH = 254;

  /**
   * Maximum domain length in presentation form
   * (<a href="https://datatracker.ietf.org/doc/html/rfc1035">RFC 1035</a>: 255 octets on
   * the wire, 253 in presentation form).
   */
  static final int DOMAIN_MAX_LENGTH = 253;

  private final Set<String> types;

  /**
   * Initializes an extractor that reports all four types.
   */
  public CursorPiiExtractor() {
    this.types = ALL_TYPES;
  }

  /**
   * Initializes an extractor limited to a subset of the types, for a caller that wants
   * only payment data masked, for example, without flagging every email address.
   *
   * @param types The types to report, drawn from the {@code TYPE_*} constants on
   *              {@link PiiMention}. Must not be {@code null} or empty and must not
   *              contain a type this extractor does not recognize.
   * @throws IllegalArgumentException Thrown if {@code types} is {@code null} or empty,
   *         or contains an unrecognized type.
   */
  public CursorPiiExtractor(Set<String> types) {
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
   * reduced to the non-overlapping set this class describes.</p>
   */
  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hits.Hit> hits = new ArrayList<>();
    if (types.contains(PiiMention.TYPE_EMAIL)) {
      scanEmails(text, hits);
    }
    if (types.contains(PiiMention.TYPE_IBAN)) {
      scanIbans(text, hits);
    }
    if (types.contains(PiiMention.TYPE_CARD)) {
      scanCards(text, hits);
    }
    if (types.contains(PiiMention.TYPE_PHONE)) {
      scanPhones(text, hits);
    }
    return Hits.resolve(hits);
  }

  /**
   * Finds email addresses by expanding around each {@code @}.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanEmails(CharSequence text, List<Hits.Hit> hits) {
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
          || end - start > EMAIL_MAX_LENGTH
          || !validDomain(text.subSequence(i + 1, end).toString())
          || (start > 0 && Character.isLetterOrDigit(Character.codePointBefore(text, start)))
          || !Boundaries.onEnd(text, end)) {
        continue;
      }
      final String normalized =
          text.subSequence(start, end).toString().toLowerCase(Locale.ROOT);
      Hits.add(hits, start, end, PiiMention.TYPE_EMAIL, normalized);
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
  private boolean validLocalPart(CharSequence text, int start, int at) {
    if (at - start > EMAIL_LOCAL_PART_MAX_LENGTH
        || text.charAt(start) == '.' || text.charAt(at - 1) == '.') {
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
   * Checks a domain: at most {@link #DOMAIN_MAX_LENGTH} characters, at least two labels,
   * each 1 to 63 characters without a leading or trailing hyphen, and a final label that
   * is an {@link IanaTlds IANA-registered} top-level domain.
   *
   * @param domain The domain without the {@code @}.
   * @return {@code true} if the domain is acceptable.
   */
  private boolean validDomain(String domain) {
    if (domain.length() > DOMAIN_MAX_LENGTH) {
      return false;
    }
    int labels = 0;
    int labelStart = 0;
    for (int i = 0; i <= domain.length(); i++) {
      if (i == domain.length() || domain.charAt(i) == '.') {
        final int length = i - labelStart;
        if (length < 1 || length > DOMAIN_LABEL_MAX_LENGTH
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
    return IanaTlds.registered(domain, tldStart, domain.length());
  }

  /**
   * Finds IBANs: a run of uppercase letters and digits in optional space-separated
   * groups, accepted at the group boundary whose length equals the country's
   * {@link IbanLengths registry entry} and whose mod-97 check passes.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanIbans(CharSequence text, List<Hits.Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      if (!Ascii.isUpper(text.charAt(i))
          || (i > 0 && Character.isLetterOrDigit(Character.codePointBefore(text, i)))
          || i + 3 >= text.length()
          || !Ascii.isUpper(text.charAt(i + 1))
          || !Ascii.isDigit(text.charAt(i + 2))
          || !Ascii.isDigit(text.charAt(i + 3))) {
        continue;
      }
      final StringBuilder compact = new StringBuilder();
      final List<int[]> groupEnds = new ArrayList<>();
      int p = i;
      while (p < text.length() && compact.length() <= IBAN_MAX_LENGTH) {
        final char c = text.charAt(p);
        if (Ascii.isUpper(c) || Ascii.isDigit(c)) {
          compact.append(c);
          p++;
        } else if (c == ' ' && p + 1 < text.length()
            && (Ascii.isUpper(text.charAt(p + 1)) || Ascii.isDigit(text.charAt(p + 1)))) {
          groupEnds.add(new int[] {p, compact.length()});
          p++;
        } else {
          break;
        }
      }
      groupEnds.add(new int[] {p, compact.length()});
      final int registeredLength =
          IbanLengths.registeredLength(text.charAt(i), text.charAt(i + 1));
      for (int g = groupEnds.size() - 1; g >= 0; g--) {
        final int textEnd = groupEnds.get(g)[0];
        final int length = groupEnds.get(g)[1];
        if (length != registeredLength || !Boundaries.onEnd(text, textEnd)) {
          continue;
        }
        if (mod97(compact, length) == 1) {
          final String candidate = compact.substring(0, length);
          Hits.add(hits, i, textEnd, PiiMention.TYPE_IBAN, candidate);
          // The loop increment resumes the scan at the exclusive match end.
          i = textEnd - 1;
          break;
        }
      }
    }
  }

  /**
   * Finds payment card numbers: a digit run with optional single space or hyphen
   * separators, an accepted leading digit, and a passing Luhn check. Candidates are
   * tried longest first at separator boundaries until the Luhn check passes, so a card
   * directly followed by another separated digit group, such as an expiry date, is
   * still found instead of being swallowed into one over-long rejected candidate.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanCards(CharSequence text, List<Hits.Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      final char first = text.charAt(i);
      if (first < '2' || first > '6' || !Boundaries.onNumberStart(text, i)) {
        continue;
      }
      final StringBuilder digits = new StringBuilder();
      final List<int[]> groupEnds = new ArrayList<>();
      int lastDigit = -1;
      boolean previousSeparator = false;
      int p = i;
      while (p < text.length() && digits.length() <= CARD_MAX_DIGITS) {
        final char c = text.charAt(p);
        if (Ascii.isDigit(c)) {
          digits.append(c);
          lastDigit = p;
          previousSeparator = false;
          p++;
        } else if ((c == ' ' || c == '-') && !previousSeparator) {
          groupEnds.add(new int[] {lastDigit + 1, digits.length()});
          previousSeparator = true;
          p++;
        } else {
          break;
        }
      }
      groupEnds.add(new int[] {lastDigit + 1, digits.length()});
      for (int g = groupEnds.size() - 1; g >= 0; g--) {
        final int end = groupEnds.get(g)[0];
        final int length = groupEnds.get(g)[1];
        if (length < CARD_MIN_DIGITS || length > CARD_MAX_DIGITS
            || !Boundaries.onEnd(text, end)
            || !luhnValid(digits, length)) {
          continue;
        }
        final String candidate = digits.substring(0, length);
        Hits.add(hits, i, end, PiiMention.TYPE_CARD, candidate);
        // The loop increment resumes the scan at the exclusive match end.
        i = end - 1;
        break;
      }
    }
  }

  /**
   * Finds phone numbers: an international form starting with {@code +} and validated
   * against {@link PhoneNumberLengths}, or a domestic form whose digits are visibly
   * formatted with spaces, hyphens, or parentheses. Candidates are tried longest first
   * at separator boundaries until the length and form checks pass, exactly like the
   * card scan, so a phone directly followed by another separated digit group, such as
   * an extension or a count, is still found instead of being swallowed into one
   * over-long rejected candidate.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanPhones(CharSequence text, List<Hits.Hit> hits) {
    int lastEnd = -1;
    for (int i = 0; i < text.length(); i++) {
      final char c = text.charAt(i);
      final boolean plus = c == '+';
      // A position where a reported phone just ended is a fresh start boundary even
      // though the character before it is a digit of that phone.
      if ((!plus && !Ascii.isDigit(c) && c != '(')
          || (i != lastEnd && !Boundaries.onNumberStart(text, i))
          || (i > 0 && text.charAt(i - 1) == '+')) {
        continue;
      }
      int digits = 0;
      int lastDigit = -1;
      int open = 0;
      int close = 0;
      boolean separated = false;
      boolean previousSeparator = false;
      final StringBuilder digitRun = new StringBuilder();
      // Each entry is a candidate cut at a separator boundary: the exclusive text end,
      // the digit count, whether the digits were visibly separated, and the
      // parenthesis counts up to the cut, so every prefix is judged by its own form.
      final List<int[]> groups = new ArrayList<>();
      int p = plus ? i + 1 : i;
      while (p < text.length() && digits <= PHONE_MAX_DIGITS) {
        final char ch = text.charAt(p);
        if (Ascii.isDigit(ch)) {
          if (digits > 0 && p > i && !Ascii.isDigit(text.charAt(p - 1))) {
            separated = true;
          }
          digits++;
          digitRun.append(ch);
          lastDigit = p;
          previousSeparator = false;
          p++;
        } else if ((ch == ' ' || ch == '-') && !previousSeparator) {
          groups.add(new int[] {lastDigit + 1, digits, separated ? 1 : 0, open, close});
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
      groups.add(new int[] {lastDigit + 1, digits, separated ? 1 : 0, open, close});
      for (int g = groups.size() - 1; g >= 0; g--) {
        final int end = groups.get(g)[0];
        final int count = groups.get(g)[1];
        final boolean visiblySeparated = groups.get(g)[2] == 1;
        if (count == 0 || groups.get(g)[3] != groups.get(g)[4]) {
          continue;
        }
        final String candidate = digitRun.substring(0, count);
        final boolean lengthOk = plus
            ? count <= PHONE_MAX_DIGITS && PhoneNumberLengths.plausibleInternational(candidate)
            : count >= PHONE_DOMESTIC_MIN_DIGITS && count <= PHONE_DOMESTIC_MAX_DIGITS
                && visiblySeparated;
        if (!lengthOk
            || !Boundaries.onEnd(text, end)
            || (end + 1 < text.length() && text.charAt(end) == '.'
                && Ascii.isDigit(text.charAt(end + 1)))) {
          continue;
        }
        Hits.add(hits, i, end, PiiMention.TYPE_PHONE,
            plus ? "+" + candidate : candidate);
        lastEnd = end;
        // The loop increment resumes the scan at the exclusive match end.
        i = end - 1;
        break;
      }
    }
  }

  /**
   * Computes the mod-97 remainder that
   * <a href="https://en.wikipedia.org/wiki/International_Bank_Account_Number">ISO 13616</a>
   * prescribes for a compact IBAN candidate. The four leading characters are read last,
   * which is the rearrangement the check prescribes.
   *
   * @param compact The candidate characters without spaces, uppercase letters and digits
   *                only.
   * @param length The number of leading characters that form the candidate; must be
   *               longer than the four characters that are rotated.
   * @return The remainder; {@code 1} for a valid IBAN.
   */
  private int mod97(CharSequence compact, int length) {
    int remainder = 0;
    for (int i = 0; i < length; i++) {
      final char c = compact.charAt((i + IBAN_ROTATION) % length);
      if (Ascii.isDigit(c)) {
        remainder = (remainder * 10 + (c - '0')) % IBAN_MODULUS;
      } else {
        remainder = (remainder * 100 + (c - 'A' + 10)) % IBAN_MODULUS;
      }
    }
    return remainder;
  }

  /**
   * Applies the <a href="https://en.wikipedia.org/wiki/Luhn_algorithm">Luhn</a> check
   * to a digit sequence.
   *
   * @param digits The digits to check.
   * @param length The number of leading digits that form the candidate.
   * @return {@code true} if the checksum passes.
   */
  private boolean luhnValid(CharSequence digits, int length) {
    int sum = 0;
    boolean twice = false;
    for (int i = length - 1; i >= 0; i--) {
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
  private boolean isLocalChar(char c) {
    return Ascii.isLetter(c) || Ascii.isDigit(c)
        || c == '.' || c == '_' || c == '%' || c == '+' || c == '-';
  }

  /**
   * Tests for a character allowed in an email domain.
   *
   * @param c The character.
   * @return {@code true} for ASCII letters, digits, dot, and hyphen.
   */
  private boolean isDomainChar(char c) {
    return Ascii.isLetter(c) || Ascii.isDigit(c) || c == '.' || c == '-';
  }
}
