/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
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

package opennlp.tools.stemmer.hunspell;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.Assertions;

import opennlp.tools.stemmer.hunspell.HunspellDictionary.LoadMode;

/** Shared in-memory fixtures and loading helpers for the Hunspell tests. */
final class HunspellTestDictionaries {

  /** Declares UTF-8 as the encoding of an affix file. */
  static final String SET_UTF8 = "SET UTF-8\n";

  /** A suffix rule with flag {@code A} that adds a plural {@code s}. */
  static final String PLURAL = "SFX A Y 1\nSFX A 0 s .\n";

  /** Lets entries with flag {@code C} form compounds of parts with one or more letters. */
  static final String COMPOUND = "COMPOUNDFLAG C\nCOMPOUNDMIN 1\n";

  /** A recorded acceptance by Hunspell's spell checker. */
  static final boolean ACCEPTED = true;

  /** A recorded rejection by Hunspell's spell checker. */
  static final boolean REJECTED = false;

  /** A recorded empty result of Hunspell's analyzer. */
  static final List<String> NO_STEMS = List.of();

  /** Prevents instantiation. */
  private HunspellTestDictionaries() {
  }

  /**
   * Encodes fixture text as a UTF-8 stream.
   *
   * @param content The fixture text.
   * @return The encoded stream.
   */
  static ByteArrayInputStream stream(String content) {
    return new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Loads an in-memory dictionary under the default loading policy.
   *
   * @param affix The affix file content.
   * @param words The word list content.
   * @return The loaded dictionary.
   * @throws IOException Thrown if the content fails to load.
   */
  static HunspellDictionary load(String affix, String words) throws IOException {
    return HunspellDictionary.load(stream(affix), stream(words));
  }

  /**
   * Loads an in-memory dictionary under the given loading policy.
   *
   * @param affix The affix file content.
   * @param words The word list content.
   * @param mode The loading policy.
   * @return The loaded dictionary.
   * @throws IOException Thrown if the content fails to load.
   */
  static HunspellDictionary load(String affix, String words, LoadMode mode) throws IOException {
    return HunspellDictionary.load(stream(affix), stream(words), mode);
  }

  /**
   * An original dictionary with the expected OpenNLP stems and the results recorded
   * from Hunspell for its input with {@code dev/hunspell-record.cxx}.
   *
   * @param name The test identifier.
   * @param affix The affix content, without the encoding declaration.
   * @param words The dictionary content.
   * @param input The input form.
   * @param expected The expected OpenNLP stems, with identity for unknown input.
   * @param hunspellAccepts Whether Hunspell's spell checker accepted the input.
   * @param hunspellStems The stems Hunspell's analyzer returned for the input.
   */
  record Fixture(String name, String affix, String words, String input, List<String> expected,
                 boolean hunspellAccepts, List<String> hunspellStems) {

    /**
     * Loads the fixture dictionary as UTF-8 under the default loading policy.
     *
     * @return The loaded dictionary.
     * @throws IOException Thrown if the fixture fails to load.
     */
    HunspellDictionary load() throws IOException {
      return HunspellTestDictionaries.load(SET_UTF8 + affix, words);
    }

    /**
     * Stems the input with the fixture dictionary.
     *
     * @return The OpenNLP stems.
     * @throws IOException Thrown if the fixture fails to load.
     */
    List<String> stems() throws IOException {
      return new HunspellStemmer(load()).stemAll(input).stream().map(CharSequence::toString).toList();
    }

    /**
     * Tells whether the OpenNLP stems can be compared with Hunspell's: Hunspell's spell
     * checker accepts the input and OpenNLP returns a single stem. For a compound or a
     * word-break form OpenNLP returns the stem of each part, where Hunspell returns one
     * stem of the whole word or none, and Hunspell's analyzer may return stems for an
     * input its spell checker rejects.
     *
     * @return {@code true} if the stems are compared.
     */
    boolean hasComparableStems() {
      return hunspellAccepts && expected.size() == 1;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
      return name;
    }
  }

  /**
   * Asserts that the stemmer recognizes a fixture input exactly when Hunspell's spell
   * checker accepted it. With a deviation, the two must differ instead, so a deviation
   * that no longer applies fails.
   *
   * @param fixture The fixture.
   * @param deviation The reason recognition differs, or {@code null} if it must not.
   * @throws IOException Thrown if the fixture fails to load.
   */
  static void assertRecognition(Fixture fixture, String deviation) throws IOException {
    final boolean recognized = !new HunspellStemmer(fixture.load()).analyze(fixture.input()).isEmpty();
    if (deviation == null) {
      Assertions.assertEquals(fixture.hunspellAccepts(), recognized,
          "recognition differs from Hunspell's spell checker");
    } else {
      Assertions.assertNotEquals(fixture.hunspellAccepts(), recognized,
          "listed deviation no longer applies: " + deviation);
    }
  }

  /**
   * Asserts that the OpenNLP stems of a fixture equal the stems recorded from Hunspell's
   * analyzer. With a deviation, the two must differ instead, so a deviation that no
   * longer applies fails.
   *
   * @param fixture The fixture.
   * @param deviation The reason the stems differ, or {@code null} if they must not.
   * @throws IOException Thrown if the fixture fails to load.
   */
  static void assertStems(Fixture fixture, String deviation) throws IOException {
    final List<String> actual = fixture.stems();
    if (deviation == null) {
      Assertions.assertEquals(fixture.hunspellStems(), actual, "stems differ from Hunspell's");
    } else {
      Assertions.assertNotEquals(fixture.hunspellStems(), actual,
          "listed deviation no longer applies: " + deviation);
    }
  }
}
