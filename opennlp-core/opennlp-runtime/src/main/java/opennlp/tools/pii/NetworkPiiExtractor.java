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
import java.util.Set;

/**
 * A deterministic {@link PiiExtractor} for network addresses: forward scans over the
 * text, no regular expressions, recognizing IPv4 addresses, IPv6 addresses, and MAC
 * addresses. Addresses identify devices and subscribers and are personal data under most
 * privacy regimes, which is why they are found here rather than in the default extractor;
 * this extractor is opt-in.
 *
 * <p>Recognized forms:</p>
 * <ul>
 *   <li>IPv4: four dot-separated decimal octets of {@code 0} to {@code 255}, as
 *   <a href="https://datatracker.ietf.org/doc/html/rfc791">RFC 791</a> addresses are
 *   written. An octet with a leading zero is rejected, since a leading zero means octal
 *   in some resolvers and decimal in others, and a fifth dotted group rejects the
 *   candidate, so a dotted version or a hostname is not reported.</li>
 *   <li>IPv6: the text representation of
 *   <a href="https://datatracker.ietf.org/doc/html/rfc4291#section-2.2">RFC 4291</a>,
 *   with at most one {@code ::} run and an optional embedded IPv4 part in the last 32
 *   bits.</li>
 *   <li>MAC: six colon-separated or six hyphen-separated pairs of hexadecimal digits, or
 *   the three dot-separated quadruples that network equipment prints, as
 *   <a href="https://standards.ieee.org/products-programs/regauth/">IEEE 802</a> 48-bit
 *   addresses are written. The separator must be the same throughout.</li>
 * </ul>
 *
 * <p>Two rules keep the ordinary shapes of technical text out of the results. An IPv6
 * candidate whose every group is one or two hexadecimal digits is only reported in the
 * full eight-group form, because short groups joined by colons are also how clock times,
 * timestamps, and namespace operators are written; a compressed candidate must therefore
 * show a group of three or more digits or an embedded IPv4 part. The unspecified and
 * broadcast addresses and the all-zero and broadcast MAC addresses are never reported,
 * since none of them identifies anything.</p>
 *
 * <p>A dotted quad remains genuinely ambiguous: a four-part software version is written
 * exactly like an address, and no test on the characters can separate the two. Such a
 * version is reported as an IPv4 address.</p>
 *
 * <p>Normalized forms: an IPv4 address keeps its dotted decimal form, an IPv6 address is
 * put in the lowercase compressed form recommended by
 * <a href="https://datatracker.ietf.org/doc/html/rfc5952">RFC 5952</a> with an embedded
 * IPv4 part folded into hexadecimal groups, and a MAC address becomes lowercase pairs
 * separated by colons, so the same address written in any accepted form normalizes to one
 * string.</p>
 *
 * <p>All three types are reported by default; the {@link #NetworkPiiExtractor(Set)}
 * constructor limits extraction to a subset.</p>
 *
 * <p>The extractor holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class NetworkPiiExtractor implements PiiExtractor {

  private static final Set<String> ALL_TYPES =
      Set.of(PiiMention.TYPE_IPV4, PiiMention.TYPE_IPV6, PiiMention.TYPE_MAC);

  private static final int IPV4_OCTETS = 4;
  private static final int IPV4_OCTET_MAX = 255;
  private static final int IPV4_OCTET_MAX_DIGITS = 3;
  private static final int IPV6_GROUPS = 8;
  private static final int IPV6_GROUP_MAX_DIGITS = 4;

  /** The number of hexadecimal digit pairs of an IEEE 802 48-bit address. */
  private static final int MAC_PAIRS = 6;

  /** The number of dot-separated quadruples of the equipment-style MAC form. */
  private static final int MAC_QUADS = 3;

  /**
   * The shortest group of three or more hexadecimal digits that lets a compressed IPv6
   * candidate be told from a clock time or a namespace operator.
   */
  private static final int IPV6_STRONG_GROUP_DIGITS = 3;

  private final Set<String> types;

  /**
   * Initializes an extractor that reports all three types.
   */
  public NetworkPiiExtractor() {
    this.types = ALL_TYPES;
  }

  /**
   * Initializes an extractor limited to a subset of the types, for a caller that wants
   * only hardware addresses found, for example, without flagging every server address in
   * a log line.
   *
   * @param types The types to report, drawn from {@link PiiMention#TYPE_IPV4},
   *              {@link PiiMention#TYPE_IPV6}, and {@link PiiMention#TYPE_MAC}. Must not
   *              be {@code null} or empty and must not contain a type this extractor does
   *              not recognize.
   * @throws IllegalArgumentException Thrown if {@code types} is {@code null} or empty, or
   *         contains an unrecognized type.
   */
  public NetworkPiiExtractor(Set<String> types) {
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
   * reduced to a non-overlapping set, leftmost and longest first.</p>
   */
  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hits.Hit> hits = new ArrayList<>();
    if (types.contains(PiiMention.TYPE_MAC)) {
      scanMacs(text, hits);
    }
    if (types.contains(PiiMention.TYPE_IPV6)) {
      scanIpv6(text, hits);
    }
    if (types.contains(PiiMention.TYPE_IPV4)) {
      scanIpv4(text, hits);
    }
    return Hits.resolve(hits);
  }

  /**
   * Finds IPv4 addresses in dotted decimal form.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanIpv4(CharSequence text, List<Hits.Hit> hits) {
    final int[] octets = new int[IPV4_OCTETS];
    for (int i = 0; i < text.length(); i++) {
      if (!Ascii.isDigit(text.charAt(i)) || !Boundaries.onNumberStart(text, i)) {
        continue;
      }
      final int end = parseIpv4(text, i, octets);
      if (end < 0 || !Boundaries.onEndBefore(text, end, '.') || isReservedIpv4(octets)) {
        continue;
      }
      final StringBuilder normalized = new StringBuilder();
      for (int o = 0; o < IPV4_OCTETS; o++) {
        if (o > 0) {
          normalized.append('.');
        }
        normalized.append(octets[o]);
      }
      Hits.add(hits, i, end, PiiMention.TYPE_IPV4, normalized.toString());
      // The loop increment resumes the scan at the exclusive match end.
      i = end - 1;
    }
  }

  /**
   * Reads a dotted decimal IPv4 address.
   *
   * @param text The text being scanned.
   * @param start The offset to read from.
   * @param octets Receives the four octet values when the read succeeds.
   * @return The exclusive end offset of the address, or {@code -1} if no address starts
   *         at {@code start}.
   */
  private int parseIpv4(CharSequence text, int start, int[] octets) {
    int p = start;
    for (int o = 0; o < IPV4_OCTETS; o++) {
      if (o > 0) {
        if (p >= text.length() || text.charAt(p) != '.') {
          return -1;
        }
        p++;
      }
      if (p >= text.length() || !Ascii.isDigit(text.charAt(p))) {
        return -1;
      }
      final boolean leadingZero = text.charAt(p) == '0';
      int digits = 0;
      int value = 0;
      while (p < text.length() && Ascii.isDigit(text.charAt(p))
          && digits < IPV4_OCTET_MAX_DIGITS) {
        value = value * 10 + (text.charAt(p) - '0');
        digits++;
        p++;
      }
      if ((digits > 1 && leadingZero) || value > IPV4_OCTET_MAX
          || (p < text.length() && Ascii.isDigit(text.charAt(p)))) {
        return -1;
      }
      octets[o] = value;
    }
    return p;
  }

  /**
   * Tests for the two IPv4 addresses that identify no host: the unspecified address and
   * the limited broadcast address.
   *
   * @param octets The four octet values.
   * @return {@code true} if the address must not be reported.
   */
  private boolean isReservedIpv4(int[] octets) {
    boolean allZero = true;
    boolean allMax = true;
    for (final int octet : octets) {
      allZero &= octet == 0;
      allMax &= octet == IPV4_OCTET_MAX;
    }
    return allZero || allMax;
  }

  /**
   * One IPv6 candidate read from the text.
   *
   * @param end The exclusive end offset of the candidate.
   * @param groups The eight 16-bit groups of the address.
   * @param written The number of groups the text spelled out, not counting the groups an
   *                embedded IPv4 part contributed.
   * @param longestGroup The digit count of the longest group the text spelled out.
   * @param embeddedIpv4 Whether the last 32 bits were written as a dotted quad.
   */
  private record Ipv6(int end, int[] groups, int written, int longestGroup,
                      boolean embeddedIpv4) {
  }

  /**
   * Finds IPv6 addresses in text representation.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanIpv6(CharSequence text, List<Hits.Hit> hits) {
    for (int i = 0; i < text.length(); i++) {
      final char c = text.charAt(i);
      final boolean doubleColon = c == ':' && i + 1 < text.length() && text.charAt(i + 1) == ':';
      if ((!Ascii.isHexDigit(c) && !doubleColon) || !onGroupStart(text, i)) {
        continue;
      }
      final Ipv6 candidate = parseIpv6(text, i);
      if (candidate == null || !Boundaries.onEnd(text, candidate.end())
          || !plausibleIpv6(candidate)) {
        continue;
      }
      Hits.add(hits, i, candidate.end(), PiiMention.TYPE_IPV6,
          formatIpv6(candidate.groups()));
      // The loop increment resumes the scan at the exclusive match end.
      i = candidate.end() - 1;
    }
  }

  /**
   * Reads an IPv6 address: hexadecimal groups separated by single colons, with at most
   * one {@code ::} run standing for one or more all-zero groups, and an optional dotted
   * quad in the last 32 bits.
   *
   * @param text The text being scanned.
   * @param start The offset to read from.
   * @return The candidate, or {@code null} if no address starts at {@code start}.
   */
  private Ipv6 parseIpv6(CharSequence text, int start) {
    final List<Integer> head = new ArrayList<>();
    final List<Integer> tail = new ArrayList<>();
    final int[] octets = new int[IPV4_OCTETS];
    boolean compressed = false;
    boolean embedded = false;
    int written = 0;
    int longestGroup = 0;
    int p = start;
    if (text.charAt(p) == ':') {
      compressed = true;
      p += 2;
    }
    while (p < text.length()) {
      final List<Integer> target = compressed ? tail : head;
      if ((!head.isEmpty() || compressed) && Ascii.isDigit(text.charAt(p))) {
        final int quadEnd = parseIpv4(text, p, octets);
        if (quadEnd > 0) {
          target.add(octets[0] << 8 | octets[1]);
          target.add(octets[2] << 8 | octets[3]);
          embedded = true;
          p = quadEnd;
          break;
        }
      }
      int digits = 0;
      int value = 0;
      while (p < text.length() && Ascii.isHexDigit(text.charAt(p))
          && digits < IPV6_GROUP_MAX_DIGITS) {
        value = value * 16 + Ascii.hexValue(text.charAt(p));
        digits++;
        p++;
      }
      if (digits == 0 || (p < text.length() && Ascii.isHexDigit(text.charAt(p)))) {
        return null;
      }
      target.add(value);
      written++;
      longestGroup = Math.max(longestGroup, digits);
      if (p >= text.length() || text.charAt(p) != ':') {
        break;
      }
      if (p + 1 < text.length() && text.charAt(p + 1) == ':') {
        if (compressed) {
          return null;
        }
        compressed = true;
        p += 2;
        continue;
      }
      if (p + 1 < text.length() && (Ascii.isHexDigit(text.charAt(p + 1)))) {
        p++;
        continue;
      }
      break;
    }
    final int total = head.size() + tail.size();
    if (compressed ? total >= IPV6_GROUPS : total != IPV6_GROUPS) {
      return null;
    }
    // Nothing may follow an embedded quad: it holds the last 32 bits of the address.
    if (embedded && p < text.length() && text.charAt(p) == ':') {
      return null;
    }
    final int[] groups = new int[IPV6_GROUPS];
    for (int g = 0; g < head.size(); g++) {
      groups[g] = head.get(g);
    }
    for (int g = 0; g < tail.size(); g++) {
      groups[IPV6_GROUPS - tail.size() + g] = tail.get(g);
    }
    return new Ipv6(p, groups, written, longestGroup, embedded);
  }

  /**
   * Judges whether an IPv6 candidate carries enough evidence to be reported: an embedded
   * IPv4 part, or at least two groups of which one is three or more digits long, or the
   * full eight groups.
   *
   * @param candidate The candidate read from the text.
   * @return {@code true} if the candidate is reported.
   */
  private boolean plausibleIpv6(Ipv6 candidate) {
    if (candidate.embeddedIpv4()) {
      return true;
    }
    return candidate.written() >= 2
        && (candidate.longestGroup() >= IPV6_STRONG_GROUP_DIGITS
            || candidate.written() == IPV6_GROUPS);
  }

  /**
   * Formats an IPv6 address in the form
   * <a href="https://datatracker.ietf.org/doc/html/rfc5952">RFC 5952</a> recommends:
   * lowercase, no leading zeros in a group, and the longest run of two or more all-zero
   * groups replaced by {@code ::}.
   *
   * @param groups The eight 16-bit groups.
   * @return The normalized form. Never {@code null}.
   */
  private String formatIpv6(int[] groups) {
    int runStart = -1;
    int runLength = 0;
    for (int i = 0; i < IPV6_GROUPS; ) {
      if (groups[i] != 0) {
        i++;
        continue;
      }
      int j = i;
      while (j < IPV6_GROUPS && groups[j] == 0) {
        j++;
      }
      if (j - i > runLength) {
        runLength = j - i;
        runStart = i;
      }
      i = j;
    }
    if (runLength < 2) {
      runStart = -1;
    }
    final StringBuilder out = new StringBuilder();
    int i = 0;
    while (i < IPV6_GROUPS) {
      if (i == runStart) {
        out.append("::");
        i += runLength;
        continue;
      }
      if (out.length() > 0 && out.charAt(out.length() - 1) != ':') {
        out.append(':');
      }
      out.append(Integer.toHexString(groups[i]));
      i++;
    }
    return out.length() == 0 ? "::" : out.toString();
  }

  /**
   * Finds MAC addresses in the colon, hyphen, and dotted forms.
   *
   * @param text The text to scan.
   * @param hits The candidate collector.
   */
  private void scanMacs(CharSequence text, List<Hits.Hit> hits) {
    final int[] bytes = new int[MAC_PAIRS];
    for (int i = 0; i < text.length(); i++) {
      if (!Ascii.isHexDigit(text.charAt(i)) || !onGroupStart(text, i)) {
        continue;
      }
      int end = parseMac(text, i, ':', 2, MAC_PAIRS, bytes);
      char separator = ':';
      if (end < 0) {
        end = parseMac(text, i, '-', 2, MAC_PAIRS, bytes);
        separator = '-';
      }
      if (end < 0) {
        end = parseMac(text, i, '.', 4, MAC_QUADS, bytes);
        separator = '.';
      }
      if (end < 0 || !Boundaries.onEndBefore(text, end, separator) || isReservedMac(bytes)) {
        continue;
      }
      final StringBuilder normalized = new StringBuilder();
      for (int b = 0; b < MAC_PAIRS; b++) {
        if (b > 0) {
          normalized.append(':');
        }
        normalized.append(hexDigit(bytes[b] >> 4));
        normalized.append(hexDigit(bytes[b] & 0xF));
      }
      Hits.add(hits, i, end, PiiMention.TYPE_MAC, normalized.toString());
      // The loop increment resumes the scan at the exclusive match end.
      i = end - 1;
    }
  }

  /**
   * Reads a MAC address written as a fixed number of equally long hexadecimal groups.
   *
   * @param text The text being scanned.
   * @param start The offset to read from.
   * @param separator The separator between the groups.
   * @param digitsPerGroup The number of hexadecimal digits in each group.
   * @param groups The number of groups.
   * @param bytes Receives the six address bytes when the read succeeds.
   * @return The exclusive end offset of the address, or {@code -1} if no address in this
   *         form starts at {@code start}.
   */
  private int parseMac(CharSequence text, int start, char separator, int digitsPerGroup,
      int groups, int[] bytes) {
    int p = start;
    int digits = 0;
    for (int g = 0; g < groups; g++) {
      if (g > 0) {
        if (p >= text.length() || text.charAt(p) != separator) {
          return -1;
        }
        p++;
      }
      for (int d = 0; d < digitsPerGroup; d++) {
        if (p >= text.length() || !Ascii.isHexDigit(text.charAt(p))) {
          return -1;
        }
        final int value = Ascii.hexValue(text.charAt(p));
        if (digits % 2 == 0) {
          bytes[digits / 2] = value << 4;
        } else {
          bytes[digits / 2] |= value;
        }
        digits++;
        p++;
      }
      if (p < text.length() && Ascii.isHexDigit(text.charAt(p))) {
        return -1;
      }
    }
    return p;
  }

  /**
   * Tests for the two MAC addresses that identify no device: the all-zero address and
   * the broadcast address.
   *
   * @param bytes The six address bytes.
   * @return {@code true} if the address must not be reported.
   */
  private boolean isReservedMac(int[] bytes) {
    boolean allZero = true;
    boolean allMax = true;
    for (final int value : bytes) {
      allZero &= value == 0;
      allMax &= value == 0xFF;
    }
    return allZero || allMax;
  }

  /**
   * Checks that a grouped candidate does not continue a longer grouped value to its
   * left, so nothing is reported from inside an address.
   *
   * @param text The text being scanned.
   * @param start The candidate start.
   * @return {@code true} if the candidate may start here.
   */
  private boolean onGroupStart(CharSequence text, int start) {
    if (!Boundaries.onWordStart(text, start)) {
      return false;
    }
    if (start == 0) {
      return true;
    }
    final char previous = text.charAt(start - 1);
    return (previous != ':' && previous != '-' && previous != '.')
        || start < 2 || !Ascii.isHexDigit(text.charAt(start - 2));
  }

  /**
   * Renders one hexadecimal digit.
   *
   * @param value The value {@code 0} to {@code 15}.
   * @return The digit character.
   */
  private char hexDigit(int value) {
    return value < 10 ? (char) ('0' + value) : (char) ('a' + value - 10);
  }
}
