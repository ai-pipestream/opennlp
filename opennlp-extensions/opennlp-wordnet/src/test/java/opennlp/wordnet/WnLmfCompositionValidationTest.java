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
package opennlp.wordnet;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Named;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.wordnet.LexicalKnowledgeBase;
import opennlp.tools.wordnet.Synset;
import opennlp.tools.wordnet.WordNetPOS;
import opennlp.tools.wordnet.WordNetRelation;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Tests independent base validation and sense membership during composition. */
class WnLmfCompositionValidationTest {

  private static final String BASE_SOURCE = "base-source.xml";
  private static final String TOP_SOURCE = "extension-source.xml";
  private static final String MIDDLE_SOURCE = "middle-source.xml";
  private static final String BASE_SYNSET = "base-synset";
  private static final String BASE_ENTRY = """
      <LexicalEntry id="base-entry">
        <Lemma writtenForm="cat" partOfSpeech="n"/>
        <Sense id="base-sense" synset="base-synset"/>
      </LexicalEntry>
      """;
  private static final String BASE_SYNSET_ELEMENT =
      "<Synset id=\"base-synset\" partOfSpeech=\"n\"/>";

  /**
   * Rejects the same malformed base independently and through an extension.
   *
   * @param baseBody The invalid base content.
   * @param additions Content that could otherwise hide the invalid reference.
   * @param reference The identifier expected in the error.
   */
  @ParameterizedTest
  @MethodSource("invalidBases")
  void testExtensionCannotRepairInvalidBase(String baseBody, String additions, String reference) {
    final String base = lexicon("base", baseBody);
    assertAll(
        () -> assertFailure(assertThrows(InvalidFormatException.class,
            () -> WnLmfReader.read(bytes(base), BASE_SOURCE)), BASE_SOURCE, reference),
        () -> assertFailure(assertThrows(InvalidFormatException.class,
            () -> compose(extension("ext", "base", additions), base)), BASE_SOURCE, reference));
  }

  /**
   * Supplies invalid bases and additions that must not make them loadable.
   *
   * @return The named base and extension pairs.
   */
  private static Stream<Arguments> invalidBases() {
    return Stream.of(
        Arguments.of(Named.of("sense to missing synset", """
            <LexicalEntry id="base-entry">
              <Lemma writtenForm="cat" partOfSpeech="n"/>
              <Sense id="base-sense" synset="missing"/>
            </LexicalEntry>
            """), "<Synset id=\"missing\" partOfSpeech=\"n\"/>", "missing"),
        Arguments.of(Named.of("relation to missing synset", BASE_ENTRY + """
            <Synset id="base-synset" partOfSpeech="n">
              <SynsetRelation relType="also" target="missing"/>
            </Synset>
            """), "<Synset id=\"missing\" partOfSpeech=\"n\"/>", "missing"),
        Arguments.of(Named.of("relation to missing sense", """
            <LexicalEntry id="base-entry">
              <Lemma writtenForm="cat" partOfSpeech="n"/>
              <Sense id="base-sense" synset="base-synset">
                <SenseRelation relType="also" target="missing-sense"/>
              </Sense>
            </LexicalEntry>
            """ + BASE_SYNSET_ELEMENT), """
            <ExternalLexicalEntry id="base-entry">
              <Sense id="missing-sense" synset="new-synset"/>
            </ExternalLexicalEntry>
            <Synset id="new-synset" partOfSpeech="n"/>
            """, "missing-sense"),
        Arguments.of(Named.of("missing member sense", BASE_ENTRY + """
            <Synset id="base-synset" partOfSpeech="n" members="missing-sense"/>
            """), """
            <ExternalLexicalEntry id="base-entry">
              <Sense id="missing-sense" synset="base-synset"/>
            </ExternalLexicalEntry>
            <ExternalSynset id="base-synset"/>
            """, "missing-sense"),
        Arguments.of(Named.of("omitted sense lemma", BASE_ENTRY + """
            <LexicalEntry id="dog-entry">
              <Lemma writtenForm="dog" partOfSpeech="n"/>
              <Sense id="dog-sense" synset="base-synset"/>
            </LexicalEntry>
            <Synset id="base-synset" partOfSpeech="n" members="base-sense"/>
            """), """
            <ExternalLexicalEntry id="dog-entry">
              <Sense id="extra-dog-sense" synset="base-synset"/>
            </ExternalLexicalEntry>
            <ExternalSynset id="base-synset"/>
            """, "dog-sense"));
  }

  /**
   * Validates an intermediate extension before applying another extension to it.
   */
  @Test
  void testOuterExtensionCannotRepairInvalidIntermediateExtension() {
    final String base = lexicon("base", BASE_ENTRY + BASE_SYNSET_ELEMENT);
    final String middle = extension("middle", "base", """
        <ExternalLexicalEntry id="base-entry">
          <Sense id="middle-sense" synset="middle-synset"/>
        </ExternalLexicalEntry>
        <Synset id="middle-synset" partOfSpeech="n" members="future-sense"/>
        """);
    final String outer = extension("outer", "middle", """
        <ExternalLexicalEntry id="base-entry">
          <Sense id="future-sense" synset="middle-synset"/>
        </ExternalLexicalEntry>
        <ExternalSynset id="middle-synset"/>
        """);
    final InvalidFormatException error = assertThrows(InvalidFormatException.class,
        () -> WnLmfReader.read(bytes(outer), TOP_SOURCE, dependency -> {
          if (dependency.ref().equals("middle")) {
            return new WnLmfSource(MIDDLE_SOURCE, bytes(middle));
          }
          assertEquals("base", dependency.ref());
          return new WnLmfSource(BASE_SOURCE, bytes(base));
        }));
    assertFailure(error, MIDDLE_SOURCE, "future-sense");
  }

  /**
   * Checks senses' parts of speech, including senses omitted from explicit members.
   *
   * @param members The explicit member form.
   */
  @ParameterizedTest
  @ValueSource(strings = {"base-sense", "base-entry"})
  void testOmittedSensePosMismatchIsAFormatError(String members) {
    final String document = lexicon("base", BASE_ENTRY + """
        <LexicalEntry id="verb-entry">
          <Lemma writtenForm="purr" partOfSpeech="v"/>
          <Sense id="verb-sense" synset="base-synset"/>
        </LexicalEntry>
        """ + "<Synset id=\"base-synset\" partOfSpeech=\"n\" members=\"" + members + "\"/>");
    final InvalidFormatException error = assertThrows(InvalidFormatException.class,
        () -> WnLmfReader.read(bytes(document), BASE_SOURCE));
    assertFailure(error, BASE_SOURCE, "verb-sense");
    assertTrue(error.getMessage().contains("part of speech"), error::getMessage);
  }

  /**
   * Accepts equivalent member lemmas after case and underscore normalization.
   *
   * @param lemma The written form in the members list.
   * @param alias The alternative written form.
   * @throws IOException If parsing fails.
   */
  @ParameterizedTest
  @CsvSource({"cat,CAT", "cat,cAt", "domestic dog,domestic_dog", "domestic dog,DOMESTIC_DOG"})
  void testOmittedAliasWithSameFoldRemainsAccepted(String lemma, String alias) throws IOException {
    final String document = lexicon("base", BASE_ENTRY.replace("cat", lemma)
        + "<LexicalEntry id=\"alias-entry\"><Lemma writtenForm=\"" + alias + "\" partOfSpeech=\"n\"/>"
        + "<Sense id=\"alias-sense\" synset=\"base-synset\"/></LexicalEntry>"
        + "<Synset id=\"base-synset\" partOfSpeech=\"n\" members=\"base-sense\"/>");
    final LexicalKnowledgeBase result = WnLmfReader.read(bytes(document), BASE_SOURCE);
    assertEquals(List.of(BASE_SYNSET), result.lookup(alias, WordNetPOS.NOUN).stream()
        .map(Synset::id).toList());
    assertEquals(List.of(lemma), result.synset(BASE_SYNSET).orElseThrow().lemmas());
    final LexicalKnowledgeBase composed = compose(extension("ext", "base", ""), document);
    assertEquals(result.lookup(alias, WordNetPOS.NOUN), composed.lookup(alias, WordNetPOS.NOUN));
  }

  /**
   * Accepts a base synset without a lexical entry as a relation target.
   *
   * @throws IOException If parsing fails.
   */
  @Test
  void testBaseSynsetWithoutSensesRemainsAccepted() throws IOException {
    final String base = lexicon("base", BASE_ENTRY + BASE_SYNSET_ELEMENT
        + "<Synset id=\"unlexicalized\" partOfSpeech=\"n\"/>");
    final LexicalKnowledgeBase result = compose(extension("ext", "base", """
        <Synset id="new-synset" partOfSpeech="n">
          <SynsetRelation relType="hypernym" target="unlexicalized"/>
        </Synset>
        <ExternalSynset id="unlexicalized"/>
        """), base);
    assertEquals(List.of(), result.synset("unlexicalized").orElseThrow().lemmas());
    assertEquals(List.of("unlexicalized"), result.synset("new-synset").orElseThrow()
        .related(WordNetRelation.HYPERNYM));
  }

  /** Reports missing sense lemmas in the final extension as a format error. */
  @Test
  void testFinalExtensionMissingLemmaIsAFormatError() {
    final String base = lexicon("base", BASE_ENTRY + BASE_SYNSET_ELEMENT);
    final String extension = extension("ext", "base", """
        <LexicalEntry id="cat-entry">
          <Lemma writtenForm="cat" partOfSpeech="n"/>
          <Sense id="cat-sense" synset="new-synset"/>
        </LexicalEntry>
        <LexicalEntry id="dog-entry">
          <Lemma writtenForm="dog" partOfSpeech="n"/>
          <Sense id="dog-sense" synset="new-synset"/>
        </LexicalEntry>
        <Synset id="new-synset" partOfSpeech="n" members="cat-sense"/>
        """);
    assertFailure(assertThrows(InvalidFormatException.class,
        () -> compose(extension, base)), TOP_SOURCE, "dog-sense");
  }

  /**
   * Validates only the selected lexicon in a resolver's multi-lexicon document.
   *
   * @throws IOException If parsing fails.
   */
  @Test
  void testUnusedResolvedLexiconDoesNotChangeSelectedBase() throws IOException {
    final String base = lexicon("base", BASE_ENTRY + BASE_SYNSET_ELEMENT);
    final String resolved = base.replace("</LexicalResource>", """
        <Lexicon id="unused" label="unused" language="en" version="1">
          <LexicalEntry id="unused-entry">
            <Lemma writtenForm="unused" partOfSpeech="n"/>
            <Sense id="unused-sense" synset="missing"/>
          </LexicalEntry>
        </Lexicon></LexicalResource>
        """);
    final LexicalKnowledgeBase result = compose(extension("ext", "base", ""), resolved);
    assertEquals(List.of(BASE_SYNSET), result.lookup("cat", WordNetPOS.NOUN).stream()
        .map(Synset::id).toList());
  }

  /**
   * Loads an extension with a supplied base document.
   *
   * @param extension The extension XML.
   * @param base The base XML.
   * @return The composed knowledge base.
   * @throws IOException If loading fails.
   */
  private LexicalKnowledgeBase compose(String extension, String base) throws IOException {
    return WnLmfReader.read(bytes(extension), TOP_SOURCE,
        dependency -> new WnLmfSource(BASE_SOURCE, bytes(base)));
  }

  /**
   * Wraps ordinary lexical content in a resource.
   *
   * @param id The lexicon identifier.
   * @param content The lexical content.
   * @return The document.
   */
  private String lexicon(String id, String content) {
    return "<LexicalResource><Lexicon id=\"" + id + "\" label=\"" + id
        + "\" language=\"en\" version=\"1\">\n" + content + "</Lexicon></LexicalResource>";
  }

  /**
   * Builds an extension document.
   *
   * @param id The extension identifier.
   * @param base The base identifier.
   * @param content The new content.
   * @return The document.
   */
  private String extension(String id, String base, String content) {
    return "<LexicalResource><LexiconExtension id=\"" + id + "\" label=\"" + id
        + "\" language=\"en\" version=\"1\"><Extends ref=\"" + base + "\" version=\"1\"/>\n"
        + content + "</LexiconExtension></LexicalResource>";
  }

  /**
   * Encodes XML using UTF-8.
   *
   * @param document The XML text.
   * @return The encoded stream.
   */
  private ByteArrayInputStream bytes(String document) {
    return new ByteArrayInputStream(document.getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Checks the diagnostic resource, reference and line.
   *
   * @param error The format error.
   * @param source The source document name.
   * @param reference The invalid reference.
   */
  private void assertFailure(InvalidFormatException error, String source, String reference) {
    assertTrue(error.getMessage().contains(source), error::getMessage);
    assertTrue(error.getMessage().contains(reference), error::getMessage);
    assertTrue(error.getMessage().contains(" at line "), error::getMessage);
  }
}
