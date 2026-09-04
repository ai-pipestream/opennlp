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
package opennlp.tools.util.normalizer;

import opennlp.tools.util.StringUtil;

/**
 * A {@link CharSequenceNormalizer} that undoes line-break hyphenation: when a letter is
 * followed by a hyphen, a line break, optional horizontal whitespace, and another letter,
 * the hyphen, the line break, and the whitespace run are deleted, joining the two letter
 * runs into one word. Text extracted from PDFs and other typeset sources breaks long words
 * at the right margin ({@code "litiga-\ntion"}), and this is what lets a document carrying
 * the broken form and a query carrying the whole word ({@code "litigation"}) agree on term
 * identity.
 *
 * <p>Recognized hyphens are {@code U+002D HYPHEN-MINUS}, {@code U+00AD SOFT HYPHEN}, and
 * {@code U+2010 HYPHEN}, the forms typesetters and PDF extractors (Poppler, PDFBox) actually
 * emit at a break. {@code U+2011 NON-BREAKING HYPHEN} is deliberately excluded: its whole
 * meaning is that the word was <em>not</em> broken at that hyphen, so a line break after it
 * is not a hyphenation break and joining across it would be wrong. Recognized line breaks
 * are the forced-break members of {@link UnicodeWhitespace#lineBreaks()}:
 * {@code U+000A LINE FEED} (also as {@code U+000D U+000A}), {@code U+000B VERTICAL TAB},
 * {@code U+000C FORM FEED}, {@code U+000D CARRIAGE RETURN}, {@code U+0085 NEXT LINE},
 * {@code U+2028 LINE SEPARATOR}, and {@code U+2029 PARAGRAPH SEPARATOR}. The whitespace run
 * after the break covers every character for which {@link StringUtil#isUnicodeWhitespace(char)}
 * holds and that is not itself one of those line breaks, so an indented continuation line
 * ({@code "com-\r\n  plete"}) joins to {@code "complete"}, while a second break, such as the
 * form feed of a page break ({@code "litiga-\n\fPage 3\ntion"}), stops the join instead of
 * fusing the word with the page header.</p>
 *
 * <p>The letter test on both sides of the break is supplementary-aware
 * ({@link Character#codePointBefore(CharSequence, int)} /
 * {@link Character#codePointAt(CharSequence, int)} with {@link Character#isLetter(int)}),
 * while every offset and length remains a UTF-16 code unit index, as everywhere else in
 * OpenNLP. The scan is a manual cursor walk; no regular expression is involved.</p>
 *
 * <p>The hyphen is always deleted when the pattern matches. That is the typesetting-artifact
 * case, but it is also the known trade-off of a dictionary-free rule: a genuinely hyphenated
 * compound that happens to be split at its hyphen ({@code "well-\nknown"}) joins to
 * {@code "wellknown"}. Telling the two apart needs a dictionary of valid compounds, which is
 * deliberately out of scope for this normalizer; callers that need it must disambiguate
 * before this step.</p>
 *
 * <p>Because the edit changes the token count of the surrounding text, offset consumers that
 * roll tokens up to terms cannot map the two original tokens forward onto the joined word;
 * use a re-tokenizing consumer (such as the retokenizing term vector annotator) that
 * tokenizes the normalized form and maps the joined term's span back through this
 * normalizer's {@link Alignment}.</p>
 *
 * <p>Texts without a matching break are returned without copying, like the sibling
 * normalizers.</p>
 *
 * @since 3.0.0
 */
public class DehyphenationCharSequenceNormalizer implements OffsetAwareNormalizer {

  private static final long serialVersionUID = 3659585484250150798L;

  private static final char HYPHEN_MINUS = '-';
  private static final char SOFT_HYPHEN = '\u00AD';

  /**
   * {@code U+2010 HYPHEN}, the dedicated typographic hyphen that PDF extractors such as
   * Poppler and PDFBox emit for typeset hyphens. {@code U+2011 NON-BREAKING HYPHEN} is
   * deliberately not a joinable hyphen: its whole meaning is that the word was <em>not</em>
   * broken at that hyphen, so a line break after it is not a hyphenation break.
   */
  private static final char TYPESET_HYPHEN = '\u2010';

  /**
   * The forced line-break characters, derived from the authoritative
   * {@link UnicodeWhitespace#lineBreaks()} table: line feed, vertical tab, form feed,
   * carriage return, next line, line separator, and paragraph separator.
   */
  private static final CodePointSet LINE_BREAKS = CodePointSet.of(
      UnicodeWhitespace.lineBreaks().stream()
          .mapToInt(UnicodeWhitespace.WhitespaceCharacter::codePoint)
          .toArray());

  private static final DehyphenationCharSequenceNormalizer INSTANCE =
      new DehyphenationCharSequenceNormalizer();

  /** {@return the shared, stateless instance} */
  public static DehyphenationCharSequenceNormalizer getInstance() {
    return INSTANCE;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Returns {@code text} itself when no line-break hyphenation is present.</p>
   */
  @Override
  public CharSequence normalize(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("The text must not be null.");
    }
    final int length = text.length();
    StringBuilder out = null;
    int cursor = 0;
    int i = 0;
    while (i < length) {
      final int joinEnd = joinEditEnd(text, i);
      if (joinEnd >= 0) {
        if (out == null) {
          out = new StringBuilder(length);
        }
        out.append(text, cursor, i);
        cursor = i = joinEnd;
      } else {
        i++;
      }
    }
    return out == null ? text : out.append(text, cursor, length).toString();
  }

  /**
   * {@inheritDoc}
   *
   * <p>Each join is recorded as an {@code equal} run up to the hyphen followed by a
   * {@code replace(runLength, 0)} deletion of the hyphen, the line break, and the
   * horizontal-whitespace run, so the joined word's normalized span maps back to the exact
   * original range covering both halves and the deleted break between them.</p>
   */
  @Override
  public AlignedText normalizeAligned(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("The text must not be null.");
    }
    final int length = text.length();
    final StringBuilder out = new StringBuilder(length);
    final Alignment.Builder alignment = new Alignment.Builder(length);
    int cursor = 0;
    int i = 0;
    while (i < length) {
      final int joinEnd = joinEditEnd(text, i);
      if (joinEnd >= 0) {
        out.append(text, cursor, i);
        alignment.equal(i - cursor);
        alignment.replace(joinEnd - i, 0);
        cursor = i = joinEnd;
      } else {
        i++;
      }
    }
    out.append(text, cursor, length);
    alignment.equal(length - cursor);
    return new AlignedText(text, out.toString(), alignment.build(length));
  }

  /**
   * Determines whether {@code text} has a joinable line-break hyphenation starting at
   * {@code hyphen}: a hyphen-minus, soft hyphen, or typeset hyphen ({@code U+2010}) with a
   * letter immediately before it, a
   * line break immediately after it, then an optional horizontal-whitespace run, then a
   * letter.
   *
   * @param text The text being scanned.
   * @param hyphen The offset of the candidate hyphen character.
   * @return The offset of the first letter after the break, that is, the end of the run to
   *     delete, or {@code -1} when {@code hyphen} does not start a join.
   */
  private static int joinEditEnd(CharSequence text, int hyphen) {
    final char c = text.charAt(hyphen);
    if (c != HYPHEN_MINUS && c != SOFT_HYPHEN && c != TYPESET_HYPHEN) {
      return -1;
    }
    if (hyphen == 0 || !Character.isLetter(Character.codePointBefore(text, hyphen))) {
      return -1;
    }
    final int length = text.length();
    int end = hyphen + 1;
    if (end >= length) {
      return -1;
    }
    final char breakStart = text.charAt(end);
    if (!isLineBreak(breakStart)) {
      return -1;
    }
    end++;
    if (breakStart == '\r' && end < length && text.charAt(end) == '\n') {
      end++;  // CRLF is one break, not two
    }
    while (end < length && isHorizontalWhitespace(text.charAt(end))) {
      end++;
    }
    if (end >= length || !Character.isLetter(Character.codePointAt(text, end))) {
      return -1;
    }
    return end;
  }

  /**
   * Determines whether {@code c} is a forced line break per {@link #LINE_BREAKS}; the caller
   * consumes a line feed following a carriage return together with it as one break.
   */
  private static boolean isLineBreak(char c) {
    return LINE_BREAKS.contains(c);
  }

  /**
   * Determines whether {@code c} is horizontal whitespace: a Unicode {@code White_Space}
   * character that is not itself one of the {@link #LINE_BREAKS forced line breaks}, so the
   * indentation of a continuation line is consumed but a second line break, such as the form
   * feed of a page break, is not.
   */
  private static boolean isHorizontalWhitespace(char c) {
    return StringUtil.isUnicodeWhitespace(c) && !isLineBreak(c);
  }
}
