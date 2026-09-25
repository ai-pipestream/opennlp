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

package opennlp.tools.stemmer.hunspell;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.stemmer.hunspell.HunspellDictionary.LoadMode;

/** Tests how the dictionary parser reads directives, tables, and morphology aliases. */
class HunspellDictionaryInternalsTest {

  private static final String WORDS = "1\ndog/A\n";
  private static final String MORPHOLOGY_ALIASES = "AM 1\nAM po:noun\n";
  private static final String REPLACEMENTS = "REP 1\nREP coat boat\n";
  private static final String CHECK_REPLACEMENTS = "CHECKCOMPOUNDREP\n";

  /**
   * Rejects a table count that is not written in ASCII digits, although
   * {@link Integer#parseInt(String)} reads it as 1.
   *
   * @param count The count of a table with one entry.
   */
  @ParameterizedTest
  @ValueSource(strings = {"١", "１", "+1"})
  void testTableCountRequiresAsciiDigits(String count) {
    for (LoadMode mode : LoadMode.values()) {
      Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
          stream("BREAK " + count + "\nBREAK -\n"), stream(WORDS), mode));
      Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
          stream("AM " + count + "\nAM po:noun\n"), stream(WORDS), mode));
    }
  }

  /**
   * Rejects a morphology alias reference that is not written in ASCII digits, although
   * {@link Integer#parseInt(String)} reads it as 1.
   *
   * @param reference The alias reference.
   */
  @ParameterizedTest
  @ValueSource(strings = {"١", "１", "+1"})
  void testMorphologyAliasRequiresAsciiDigits(String reference) {
    for (LoadMode mode : LoadMode.values()) {
      Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
          stream(MORPHOLOGY_ALIASES), stream("1\ndog\t" + reference + "\n"), mode));
      Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
          stream(MORPHOLOGY_ALIASES + "SFX A Y 1\nSFX A 0 s . " + reference + "\n"),
          stream(WORDS), mode));
    }
  }

  /**
   * Keeps the {@code REP} table when {@code CHECKCOMPOUNDREP} comes before or after it.
   *
   * @param checkFirst Whether {@code CHECKCOMPOUNDREP} precedes the table.
   * @throws IOException Thrown if loading fails.
   */
  @ParameterizedTest
  @ValueSource(booleans = {true, false})
  void testReplacementTableKeptWithCheckCompoundRep(boolean checkFirst) throws IOException {
    final String affix = checkFirst ? CHECK_REPLACEMENTS + REPLACEMENTS
        : REPLACEMENTS + CHECK_REPLACEMENTS;
    final HunspellDictionary dictionary = HunspellDictionary.load(stream(affix), stream(WORDS));
    Assertions.assertTrue(dictionary.rejectsCompoundReplacement("raincoat", "rainboat"::equals));
  }

  /**
   * Skips a {@code REP} table without {@code CHECKCOMPOUNDREP}, including text that is not
   * valid in the declared encoding.
   *
   * @throws IOException Thrown if loading fails.
   */
  @Test
  void testReplacementTableSkippedWithoutCheckCompoundRep() throws IOException {
    final byte[] invalid = {'R', 'E', 'P', ' ', 'c', (byte) 0xe9, ' ', 'x', '\n'};
    final byte[] affix = concat(("SET UTF-8\n" + REPLACEMENTS).replace("REP 1", "REP 2"),
        invalid);
    final HunspellDictionary dictionary = HunspellDictionary.load(
        new ByteArrayInputStream(affix), stream(WORDS));
    Assertions.assertFalse(dictionary.rejectsCompoundReplacement("raincoat", "rainboat"::equals));
    final byte[] checked = concat(CHECK_REPLACEMENTS, affix);
    Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
        new ByteArrayInputStream(checked), stream(WORDS)));
  }

  /**
   * Joins UTF-8 text and raw bytes.
   *
   * @param text The leading text.
   * @param bytes The trailing bytes.
   * @return The text's UTF-8 bytes followed by the given bytes.
   */
  private static byte[] concat(String text, byte[] bytes) {
    final byte[] head = text.getBytes(StandardCharsets.UTF_8);
    final byte[] joined = new byte[head.length + bytes.length];
    System.arraycopy(head, 0, joined, 0, head.length);
    System.arraycopy(bytes, 0, joined, head.length, bytes.length);
    return joined;
  }

  /**
   * Encodes test content as UTF-8.
   *
   * @param content The file content.
   * @return A stream over the encoded content.
   */
  private static ByteArrayInputStream stream(String content) {
    return new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));
  }
}
