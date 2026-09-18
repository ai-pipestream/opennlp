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

package opennlp.tools.formats.ad;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.namefind.NameSample;
import opennlp.tools.tokenize.DetokenizationDictionary;
import opennlp.tools.tokenize.DetokenizationDictionary.Operation;
import opennlp.tools.tokenize.DictionaryDetokenizer;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/** Checks compound preservation and optional splitting without detaching combining marks. */
class ADHyphenatedTokenTest {

  @TempDir
  Path directory;

  private static Stream<Arguments> hyphenatedWords() {
    return Stream.of(
        Arguments.of("x-caf\u00E9", new String[] {"x", "-", "caf\u00E9"}),
        Arguments.of("x-cafe\u0301", new String[] {"x", "-", "cafe\u0301"}),
        Arguments.of("S\u00E3o-Paulo", new String[] {"S\u00E3o", "-", "Paulo"}),
        Arguments.of("Sa\u0303o-Paulo", new String[] {"Sa\u0303o", "-", "Paulo"}),
        Arguments.of("x-e\u0323\u0301", new String[] {"x", "-", "e\u0323\u0301"}),
        Arguments.of("\uD801\uDC12\u0301-a",
            new String[] {"\uD801\uDC12\u0301", "-", "a"}),
        Arguments.of("a-b\u0301-c", new String[] {"a", "-", "b\u0301", "-", "c"}),
        Arguments.of("\u0301a-b", new String[] {"\u0301a-b"}));
  }

  private static Stream<Arguments> conversionCases() {
    return hyphenatedWords().flatMap(argument -> {
      String input = (String) argument.get()[0];
      String[] split = (String[]) argument.get()[1];
      return Stream.of(
          Arguments.of(input, "default", new String[] {input}),
          Arguments.of(input, "false", new String[] {input}),
          Arguments.of(input, "true", split));
    });
  }

  /** Both constructor modes retain each combining sequence and the full entity span. */
  @ParameterizedTest(name = "{0}")
  @MethodSource("hyphenatedWords")
  void testStreamPreservesOrSplitsWholeWords(String input, String[] split) throws IOException {
    for (boolean enabled : new boolean[] {false, true}) {
      try (ADNameSampleStream stream = new ADNameSampleStream(
          ObjectStreamUtils.createObjectStream(sentence(input)), enabled)) {
        assertSample(enabled ? split : new String[] {input}, stream.read());
        assertNull(stream.read());
      }
    }
  }

  /** An omitted CLI option preserves compounds, while an explicit true still splits. */
  @ParameterizedTest(name = "{0}, split={1}")
  @MethodSource("conversionCases")
  void testNameConverter(String input, String option, String[] expected) throws IOException {
    ADNameSampleStreamFactory factory =
        new ADNameSampleStreamFactory(ADNameSampleStreamFactory.Parameters.class);
    try (ObjectStream<NameSample> stream = factory.create(arguments(input, option))) {
      assertSample(expected, stream.read());
      assertNull(stream.read());
    }
  }

  /** Token conversion inherits the same default and explicit option as name conversion. */
  @ParameterizedTest(name = "{0}, split={1}")
  @MethodSource("conversionCases")
  void testTokenConverter(String input, String option, String[] expected) throws IOException {
    ADNameSampleStreamFactory.registerFactory();
    ADTokenSampleStreamFactory factory =
        new ADTokenSampleStreamFactory(ADTokenSampleStreamFactory.Parameters.class);
    try (ObjectStream<TokenSample> stream = factory.create(tokenArguments(input, option))) {
      TokenSample sample = stream.read();
      assertNotNull(sample);
      assertArrayEquals(expected, Span.spansToStrings(sample.getTokenSpans(), sample.getText()));
      assertNull(stream.read());
    }
  }

  /** AD attaches the clitic's hyphen to the verb, even when the verb stays one token. */
  @ParameterizedTest
  @ValueSource(strings = {"default", "false", "true"})
  void testCliticSurfaceTextAndBoundaries(String option) throws IOException {
    ADNameSampleStreamFactory.registerFactory();
    String[] args = tokenArguments("ofereceu-me", option);
    Files.write(directory.resolve("sample.ad"), List.of("<s>", "SOURCE: ref=\"x\"",
        "1001 ofereceu-me", "STA:fcl", "=P:v-fin(\"oferecer\" PS 3S IND VFIN)\tofereceu-",
        "=DAT:pron-pers(\"eu\" M/F 1S DAT)\tme", "</s>"), StandardCharsets.UTF_8);
    ADTokenSampleStreamFactory factory =
        new ADTokenSampleStreamFactory(ADTokenSampleStreamFactory.Parameters.class);
    try (ObjectStream<TokenSample> stream = factory.create(args)) {
      TokenSample sample = stream.read();
      assertNotNull(sample);
      assertEquals("ofereceu-me", sample.getText());
      String[] expected = option.equals("true")
          ? new String[] {"ofereceu", "-", "me"} : new String[] {"ofereceu-", "me"};
      assertArrayEquals(expected, Span.spansToStrings(sample.getTokenSpans(), sample.getText()));
      assertNull(stream.read());
    }
  }

  /** Markers separate annotated tokens, never a word from its own trailing hyphen. */
  @Test
  void testDetokenizationMarkers() {
    ADDetokenizer detokenizer = new ADDetokenizer(new DictionaryDetokenizer(
        new DetokenizationDictionary(new String[] {"-"}, new Operation[] {Operation.MOVE_BOTH})));
    String[] tokens = {"ofereceu-", "me", "x-caf\u00E9"};
    assertEquals("ofereceu-me x-caf\u00E9", detokenizer.detokenize(tokens, null));
    assertEquals("ofereceu-|me x-caf\u00E9", detokenizer.detokenize(tokens, "|"));
    assertEquals("ofereceu-", detokenizer.detokenize(new String[] {"ofereceu-"}, "|"));
    assertEquals("", detokenizer.detokenize(new String[0], "|"));
  }

  /**
   * The corpus also leaves a trailing hyphen on names and codes whose second part is annotated
   * on the next line, where the character before the hyphen is a digit or another hyphen.
   * Every such token in FlorestaVirgem joins to the right, as the clitics do.
   */
  @ParameterizedTest(name = "{0}")
  @MethodSource("corpusTrailingHyphens")
  void testCorpusTrailingHyphensJoinRight(String left, String right, String expected) {
    ADDetokenizer detokenizer = new ADDetokenizer(new DictionaryDetokenizer(
        new DetokenizationDictionary(new String[] {"-"}, new Operation[] {Operation.MOVE_BOTH})));
    assertEquals(expected, detokenizer.detokenize(new String[] {left, right}, null));
  }

  private static Stream<Arguments> corpusTrailingHyphens() {
    return Stream.of(
        // a clitic verb
        Arguments.of("ofereceu-", "me", "ofereceu-me"),
        // a postal code, so the character before the hyphen is a digit
        Arguments.of("CEP_01290-", "900", "CEP_01290-900"),
        // a name the corpus writes with two hyphens
        Arguments.of("Projeto_Baleia--", "Jubarte", "Projeto_Baleia--Jubarte"),
        Arguments.of("Boutros_Boutros--", "Ghali", "Boutros_Boutros--Ghali"));
  }

  /** A hyphen of its own stays a token and follows the dictionary, not the trailing-hyphen rule. */
  @Test
  void testBareHyphenFollowsTheDictionary() {
    ADDetokenizer detokenizer = new ADDetokenizer(new DictionaryDetokenizer(
        new DetokenizationDictionary(new String[] {"-"}, new Operation[] {Operation.MOVE_BOTH})));
    assertEquals("a-b", detokenizer.detokenize(new String[] {"a", "-", "b"}, null));
    assertEquals("a|-|b", detokenizer.detokenize(new String[] {"a", "-", "b"}, "|"));
  }

  /** A custom dictionary still controls whether a hyphen joins to the following token. */
  @Test
  void testDetokenizationHonorsDictionary() {
    ADDetokenizer detokenizer = new ADDetokenizer(new DictionaryDetokenizer(
        new DetokenizationDictionary(new String[] {"-"}, new Operation[] {Operation.MOVE_LEFT})));
    assertEquals("ofereceu- me", detokenizer.detokenize(new String[] {"ofereceu-", "me"}, null));
  }

  private String[] tokenArguments(String input, String option) throws IOException {
    List<String> args = new ArrayList<>(List.of(arguments(input, option)));
    Path detokenizer = directory.resolve("detokenizer.xml");
    try (var resource = ADHyphenatedTokenTest.class.getResourceAsStream(
        "/opennlp/tools/tokenize/latin-detokenizer.xml")) {
      assertNotNull(resource);
      Files.copy(resource, detokenizer);
    }
    args.addAll(List.of("-detokenizer", detokenizer.toString()));
    return args.toArray(String[]::new);
  }

  private String[] arguments(String input, String option) throws IOException {
    Path data = directory.resolve("sample.ad");
    Files.write(data, sentence(input), StandardCharsets.UTF_8);
    List<String> args = new ArrayList<>(List.of("-lang", "por", "-encoding", "UTF-8",
        "-data", data.toString()));
    if (!option.equals("default")) {
      args.addAll(List.of("-splitHyphenatedTokens", option));
    }
    return args.toArray(String[]::new);
  }

  private List<String> sentence(String input) {
    return List.of("<s>", "SOURCE: ref=\"x\"", "1001 " + input, "STA:fcl",
        "=H:n(\"x\" <NER:hum> M S)\t" + input, "</s>");
  }

  private void assertSample(String[] expected, NameSample sample) {
    assertNotNull(sample);
    assertArrayEquals(expected, sample.getSentence());
    assertArrayEquals(new Span[] {new Span(0, expected.length, "person")}, sample.getNames());
  }
}
