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

package opennlp.tools.tokenize;

import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import opennlp.tools.ResetWhitespaceMode;
import opennlp.tools.util.Span;
import opennlp.tools.util.WhitespaceMode;

/**
 * Checks retained line breaks and token offsets at whitespace boundaries.
 */
@ResetWhitespaceMode
class TokenizerLineBreakBoundaryTest {

  @AfterEach
  void resetSimpleTokenizer() {
    SimpleTokenizer.INSTANCE.setKeepNewLines(false);
  }

  @ParameterizedTest(name = "{0}, {1}, keep={2}, {3}")
  @MethodSource("lineBreakBoundaries")
  void testLineBreakBoundaries(String tokenizerName, WhitespaceMode mode, boolean keep,
                              String description, String input, Span[] expected) {
    WhitespaceMode.setActive(mode);
    AbstractTokenizer tokenizer = "simple".equals(tokenizerName)
        ? SimpleTokenizer.INSTANCE : WhitespaceTokenizer.newInstance(keep);
    tokenizer.setKeepNewLines(keep);

    Assertions.assertArrayEquals(expected, tokenizer.tokenizePos(input));
    Assertions.assertArrayEquals(Span.spansToStrings(expected, input), tokenizer.tokenize(input));
  }

  /**
   * Supplies explicit UTF-16 offsets for each boundary shape and whitespace mode.
   *
   * @return The tokenizer, mode, setting, description, input, and expected spans.
   */
  static Stream<Arguments> lineBreakBoundaries() {
    return Stream.of("simple", "whitespace").flatMap(tokenizer ->
        Stream.of(WhitespaceMode.UNICODE, WhitespaceMode.LEGACY).flatMap(mode ->
            Stream.of(false, true).flatMap(keep -> {
              Stream<Boundary> ascii = Stream.of(
                  new Boundary("leading CRLF", "\r\nab", spans(0, 1, 1, 2, 2, 4), spans(2, 4)),
                  new Boundary("trailing CRLF", "ab\r\n", spans(0, 2, 2, 3, 3, 4), spans(0, 2)),
                  new Boundary("internal CRLF", "ab\r\ncd",
                      spans(0, 2, 2, 3, 3, 4, 4, 6), spans(0, 2, 4, 6)),
                  new Boundary("CRLF after space and tab", "ab \t\r\n cd",
                      spans(0, 2, 4, 5, 5, 6, 7, 9), spans(0, 2, 7, 9)),
                  new Boundary("only CRLF", "\r\n", spans(0, 1, 1, 2), spans()),
                  new Boundary("breaks separated by spaces", "ab\n  \ncd",
                      spans(0, 2, 2, 3, 5, 6, 6, 8), spans(0, 2, 6, 8)));
              Stream<Arguments> asciiArguments = ascii.map(boundary -> Arguments.of(
                  tokenizer, mode, keep, boundary.description(), boundary.input(),
                  keep ? boundary.kept() : boundary.dropped()));
              Stream<Arguments> characterArguments = Stream.of('\n', '\r', '\u000B', '\f',
                  '\u0085', '\u2028', '\u2029').flatMap(separator -> {
                    boolean legacyNel = mode == WhitespaceMode.LEGACY && separator == '\u0085';
                    boolean retained = keep && (mode == WhitespaceMode.UNICODE
                        || separator == '\n' || separator == '\r');
                    return characterBoundaries(separator).map(boundary -> {
                      Span[] expected = retained ? boundary.kept() : boundary.dropped();
                      if (legacyNel) {
                        expected = "simple".equals(tokenizer)
                            ? boundary.legacySimple() : boundary.legacyWhitespace();
                      }
                      return Arguments.of(tokenizer, mode, keep,
                          "U+" + Integer.toHexString(separator) + " " + boundary.description(),
                          boundary.input(), expected);
                    });
                  });
              return Stream.concat(asciiArguments, characterArguments);
            })));
  }

  /**
   * Supplies boundary shapes, including the ordinary-character behavior of LEGACY NEL.
   *
   * @param separator The character placed at each boundary.
   * @return Inputs with explicitly specified token spans.
   */
  private static Stream<Boundary> characterBoundaries(char separator) {
    return Stream.of(
        new Boundary("leading", separator + "ab", spans(0, 1, 1, 3), spans(1, 3),
            spans(0, 1, 1, 3), spans(0, 3)),
        new Boundary("trailing", "ab" + separator, spans(0, 2, 2, 3), spans(0, 2),
            spans(0, 2, 2, 3), spans(0, 3)),
        new Boundary("consecutive", "ab" + separator + separator + "cd",
            spans(0, 2, 2, 3, 3, 4, 4, 6), spans(0, 2, 4, 6),
            spans(0, 2, 2, 4, 4, 6), spans(0, 6)),
        new Boundary("after space", "ab " + separator + " cd",
            spans(0, 2, 3, 4, 5, 7), spans(0, 2, 5, 7),
            spans(0, 2, 3, 4, 5, 7), spans(0, 2, 3, 4, 5, 7)),
        new Boundary("only breaks", "" + separator + separator, spans(0, 1, 1, 2), spans(),
            spans(0, 2), spans(0, 2)));
  }

  /**
   * Converts explicit start/end pairs into expected spans without consulting the tokenizer.
   *
   * @param offsets Alternating start and exclusive end offsets.
   * @return The expected token spans.
   */
  private static Span[] spans(int... offsets) {
    Span[] spans = new Span[offsets.length / 2];
    for (int i = 0; i < spans.length; i++) {
      spans[i] = new Span(offsets[2 * i], offsets[2 * i + 1]);
    }
    return spans;
  }

  private record Boundary(String description, String input, Span[] kept, Span[] dropped,
                          Span[] legacySimple, Span[] legacyWhitespace) {
    private Boundary(String description, String input, Span[] kept, Span[] dropped) {
      this(description, input, kept, dropped, null, null);
    }
  }
}
