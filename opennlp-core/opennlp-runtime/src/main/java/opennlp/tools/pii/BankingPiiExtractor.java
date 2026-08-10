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
 * A deterministic {@link PiiExtractor} for United States bank account routing: forward
 * scans over the text, no regular expressions, recognizing
 * <a href="https://en.wikipedia.org/wiki/ABA_routing_transit_number">ABA routing transit
 * numbers</a>. This extractor is opt-in.
 *
 * <p>A candidate is nine digits standing on their own, with a routing symbol in one of the
 * assigned ranges, {@code 01} to {@code 12} for the Federal Reserve districts, {@code 21}
 * to {@code 32} for thrift institutions, {@code 61} to {@code 72} for electronic
 * transactions, and {@code 80} for traveler's checks, and a passing check digit under the
 * weights three, seven, and one that the ABA prescribes.</p>
 *
 * <p>Both tests together still leave a routing number weakly evidenced: about one nine-digit
 * run in ten passes the check digit by chance and about two prefixes in five are assigned,
 * so roughly one arbitrary nine-digit run in twenty five is reported. A run that looks like
 * a routing number is a routing number as far as any character test can tell, so a caller
 * that scans text full of nine-digit identifiers should expect false positives and use the
 * surrounding context, not this extractor alone, to act on them. This is why routing
 * numbers are not part of the default extractor.</p>
 *
 * <p>Only the plain nine-digit form is recognized, the form a payment instruction and a
 * cheque carry; a run broken up by spaces or hyphens is not a routing number.</p>
 *
 * <p>Normalized form: the nine digits as written.</p>
 *
 * <p>The extractor holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class BankingPiiExtractor implements PiiExtractor {

  private static final int ROUTING_DIGITS = 9;

  /** The weights of the ABA check digit, repeating over the nine digits. */
  private static final int[] WEIGHTS = {3, 7, 1};

  private static final int CHECK_MODULUS = 10;

  /**
   * The assigned routing symbol ranges as inclusive pairs: the Federal Reserve districts,
   * the thrift institutions, the electronic transaction ranges, and traveler's checks.
   */
  private static final int[][] ASSIGNED_PREFIXES = {{1, 12}, {21, 32}, {61, 72}, {80, 80}};

  /**
   * Initializes an extractor for ABA routing numbers.
   */
  public BankingPiiExtractor() {
  }

  /**
   * {@inheritDoc}
   *
   * <p>Reports one mention per accepted nine-digit run, in text order.</p>
   */
  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hits.Hit> hits = new ArrayList<>();
    for (int i = 0; i + ROUTING_DIGITS <= text.length(); i++) {
      if (!Ascii.isDigit(text.charAt(i)) || !Boundaries.onNumberStart(text, i)) {
        continue;
      }
      final int end = i + ROUTING_DIGITS;
      if (!onNumberEnd(text, end) || !allDigits(text, i, end)) {
        continue;
      }
      if (!assignedPrefix(text, i) || !checkDigitValid(text, i)) {
        continue;
      }
      Hits.add(hits, i, end, PiiMention.TYPE_ABA_ROUTING,
          text.subSequence(i, end).toString());
      // The loop increment resumes the scan at the exclusive match end.
      i = end - 1;
    }
    return Hits.resolve(hits);
  }

  /**
   * Checks that every character of a range is an ASCII digit.
   *
   * @param text The text being scanned.
   * @param start The first character of the range.
   * @param end The exclusive end of the range.
   * @return {@code true} if the range holds digits only.
   */
  private boolean allDigits(CharSequence text, int start, int end) {
    for (int i = start; i < end; i++) {
      if (!Ascii.isDigit(text.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Checks that a numeric candidate ends at {@code end} and does not continue into a
   * decimal fraction or a comma-grouped number.
   *
   * @param text The text being scanned.
   * @param end The candidate end, exclusive.
   * @return {@code true} if the candidate may end here.
   */
  private boolean onNumberEnd(CharSequence text, int end) {
    if (!Boundaries.onEnd(text, end)) {
      return false;
    }
    if (end + 1 >= text.length()) {
      return true;
    }
    final char next = text.charAt(end);
    return (next != '.' && next != ',') || !Ascii.isDigit(text.charAt(end + 1));
  }

  /**
   * Checks the routing symbol, the first two digits, against the assigned ranges.
   *
   * @param text The text being scanned.
   * @param start The first digit of the candidate.
   * @return {@code true} if the routing symbol is assigned.
   */
  private boolean assignedPrefix(CharSequence text, int start) {
    final int prefix =
        (text.charAt(start) - '0') * 10 + (text.charAt(start + 1) - '0');
    for (final int[] range : ASSIGNED_PREFIXES) {
      if (prefix >= range[0] && prefix <= range[1]) {
        return true;
      }
    }
    return false;
  }

  /**
   * Applies the ABA check digit: the weighted sum of the nine digits under the repeating
   * weights three, seven, and one must be a multiple of ten.
   *
   * @param text The text being scanned.
   * @param start The first digit of the candidate.
   * @return {@code true} if the check digit holds.
   */
  private boolean checkDigitValid(CharSequence text, int start) {
    int sum = 0;
    for (int i = 0; i < ROUTING_DIGITS; i++) {
      sum += WEIGHTS[i % WEIGHTS.length] * (text.charAt(start + i) - '0');
    }
    return sum % CHECK_MODULUS == 0;
  }
}
