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
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import opennlp.tools.wordnet.LexicalKnowledgeBase;
import opennlp.tools.wordnet.Synset;
import opennlp.tools.wordnet.WordNetPos;
import opennlp.tools.wordnet.WordNetRelation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class WnLmfReaderTest {

  static LexicalKnowledgeBase fixture() {
    try (InputStream in = WnLmfReaderTest.class.getResourceAsStream("mini-wn-lmf.xml")) {
      assertNotNull(in, "Fixture mini-wn-lmf.xml must be on the test classpath");
      return WnLmfReader.read(in, "mini-wn-lmf.xml");
    } catch (IOException e) {
      throw new IllegalStateException("Unexpected IOException from a classpath stream", e);
    }
  }

  private static LexicalKnowledgeBase parse(String document) {
    return WnLmfReader.read(
        new ByteArrayInputStream(document.getBytes(StandardCharsets.UTF_8)), "inline.xml");
  }

  private static String wrap(String body) {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<LexicalResource>\n"
        + "<Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">\n"
        + body + "\n</Lexicon>\n</LexicalResource>\n";
  }

  @Test
  void testLookupReturnsSynsetWithAllComponents() {
    final List<Synset> senses = fixture().lookup("dog", WordNetPos.NOUN);
    assertEquals(1, senses.size());
    final Synset dog = senses.get(0);
    assertEquals("mini-n1", dog.id());
    assertEquals(WordNetPos.NOUN, dog.pos());
    assertEquals(List.of("dog", "domestic dog"), dog.lemmas());
    assertEquals("a domesticated canid", dog.gloss());
    assertEquals(List.of("mini-n2"), dog.related(WordNetRelation.HYPERNYM));
  }

  @Test
  void testLookupFoldsCaseAndUnderscore() {
    final LexicalKnowledgeBase lexicon = fixture();
    assertEquals("mini-n1", lexicon.lookup("Domestic_Dog", WordNetPos.NOUN).get(0).id());
    assertEquals("mini-n1", lexicon.lookup("DOG", WordNetPos.NOUN).get(0).id());
  }

  @Test
  void testLookupKeepsSenseOrder() {
    final List<Synset> runSenses = fixture().lookup("run", WordNetPos.NOUN);
    assertEquals(List.of("mini-n5", "mini-n9"),
        runSenses.stream().map(Synset::id).toList());
  }

  @Test
  void testLookupIsPosScoped() {
    final LexicalKnowledgeBase lexicon = fixture();
    assertEquals(1, lexicon.lookup("run", WordNetPos.VERB).size());
    assertTrue(lexicon.lookup("dog", WordNetPos.VERB).isEmpty());
    assertFalse(lexicon.contains("walk", WordNetPos.NOUN));
    assertTrue(lexicon.contains("walk", WordNetPos.VERB));
  }

  @Test
  void testRelationNavigation() {
    final LexicalKnowledgeBase lexicon = fixture();
    assertEquals(List.of("mini-n1"), lexicon.related("mini-n2", WordNetRelation.HYPONYM));
    assertEquals(List.of("mini-v1", "mini-v2"),
        lexicon.related("mini-v4", WordNetRelation.HYPONYM));
    assertEquals(List.of("mini-v4"), lexicon.related("mini-v1", WordNetRelation.HYPERNYM));
  }

  @Test
  void testSenseRelationsAreLiftedToSynsetLevel() {
    final LexicalKnowledgeBase lexicon = fixture();
    assertEquals(List.of("mini-a2"), lexicon.related("mini-a1", WordNetRelation.ANTONYM));
    assertEquals(List.of("mini-a1"), lexicon.related("mini-a2", WordNetRelation.ANTONYM));
    assertEquals(List.of("mini-v1"),
        lexicon.related("mini-n5", WordNetRelation.DERIVATIONALLY_RELATED));
    assertEquals(List.of("mini-n5"),
        lexicon.related("mini-v1", WordNetRelation.DERIVATIONALLY_RELATED));
  }

  @Test
  void testSatelliteNormalizesToAdjective() {
    final List<Synset> senses = fixture().lookup("large", WordNetPos.ADJECTIVE);
    assertEquals(1, senses.size());
    assertEquals(WordNetPos.ADJECTIVE, senses.get(0).pos());
    assertEquals(List.of("mini-a4"), fixture().related("mini-a3", WordNetRelation.SIMILAR_TO));
    assertEquals(List.of("mini-a3"), fixture().related("mini-a4", WordNetRelation.SIMILAR_TO));
  }

  @Test
  void testUnknownLemmaOrSynsetIsEmpty() {
    final LexicalKnowledgeBase lexicon = fixture();
    assertTrue(lexicon.lookup("zebra", WordNetPos.NOUN).isEmpty());
    assertTrue(lexicon.synset("mini-n99").isEmpty());
  }

  @Test
  void testReadPath(@TempDir Path tempDir) throws IOException {
    final Path file = tempDir.resolve("tiny.xml");
    Files.writeString(file, wrap(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>"
            + "<Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>"
            + "<Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>"));
    final LexicalKnowledgeBase lexicon = WnLmfReader.read(file);
    assertEquals("a feline", lexicon.lookup("cat", WordNetPos.NOUN).get(0).gloss());
  }

  @Test
  void testReadPathRejectsNullAndMissing(@TempDir Path tempDir) {
    assertThrows(IllegalArgumentException.class, () -> WnLmfReader.read((Path) null));
    assertThrows(IllegalArgumentException.class,
        () -> WnLmfReader.read(tempDir.resolve("absent.xml")));
  }

  @Test
  void testReadStreamRejectsNulls() {
    assertThrows(IllegalArgumentException.class, () -> WnLmfReader.read(null, "x"));
    assertThrows(IllegalArgumentException.class,
        () -> WnLmfReader.read(new ByteArrayInputStream(new byte[0]), null));
  }

  @Test
  void testSkipsDoctypeDeclaration() {
    // Real Open English WordNet releases ship exactly this shape: a DOCTYPE naming the schema
    // DTD by an unreachable SYSTEM identifier (example.invalid is the RFC 2606 reserved domain
    // that must never resolve). The reader must parse past it without attempting to fetch it.
    final String document = "<?xml version=\"1.0\"?>\n"
        + "<!DOCTYPE LexicalResource SYSTEM \"http://example.invalid/WN-LMF-1.3.dtd\">\n"
        + "<LexicalResource><Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">"
        + "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>"
        + "<Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>"
        + "<Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>"
        + "</Lexicon></LexicalResource>";
    final LexicalKnowledgeBase lexicon = parse(document);
    assertEquals("a feline", lexicon.lookup("cat", WordNetPos.NOUN).get(0).gloss());
  }

  @Test
  void testInternalSubsetEntityIsNeverExpanded(@TempDir Path tempDir) throws IOException {
    // A DOCTYPE-declared internal-subset entity is the classic XXE payload: if the parser ever
    // honored it, the entity reference below would be replaced by the target file's content.
    // With SUPPORT_DTD disabled the declaration itself is never registered, so the reference is
    // undefined and parsing must fail loud rather than silently expand it.
    final Path secret = tempDir.resolve("secret.txt");
    Files.writeString(secret, "xxe-marker-should-never-appear");
    final String document = "<?xml version=\"1.0\"?>\n"
        + "<!DOCTYPE LexicalResource [<!ENTITY xxe SYSTEM \"" + secret.toUri() + "\">]>\n"
        + "<LexicalResource><Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">"
        + "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"&xxe;\" partOfSpeech=\"n\"/>"
        + "<Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>"
        + "<Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>"
        + "</Lexicon></LexicalResource>";
    final IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> parse(document));
    assertFalse(e.getMessage().contains("xxe-marker-should-never-appear"));
  }

  @Test
  void testRejectsTruncatedDocument() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
        () -> parse("<?xml version=\"1.0\"?>\n<LexicalResource><Lexicon id=\"t\""));
    assertTrue(e.getMessage().contains("inline.xml"));
  }

  @Test
  void testRejectsSenseWithoutSynsetAttribute() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(
        wrap("<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>"
            + "<Sense id=\"t-cat-n-1\"/></LexicalEntry>")));
    assertTrue(e.getMessage().contains("synset"));
  }

  @Test
  void testRejectsSenseToUndeclaredSynset() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(
        wrap("<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>"
            + "<Sense id=\"t-cat-n-1\" synset=\"t-9\"/></LexicalEntry>")));
    assertTrue(e.getMessage().contains("t-9"));
  }

  @Test
  void testRejectsRelationToUndeclaredSynset() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(
        wrap("<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>"
            + "<Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>"
            + "<Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition>"
            + "<SynsetRelation relType=\"hypernym\" target=\"t-9\"/></Synset>")));
    assertTrue(e.getMessage().contains("t-9"));
  }

  @Test
  void testRejectsUnknownRelationType() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(
        wrap("<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>"
            + "<Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>"
            + "<Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition>"
            + "<SynsetRelation relType=\"quasi_synonym\" target=\"t-1\"/></Synset>")));
    assertTrue(e.getMessage().contains("quasi_synonym"));
  }

  @Test
  void testSkipsOtherRelationType() {
    final LexicalKnowledgeBase lexicon = parse(
        wrap("<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>"
            + "<Sense id=\"t-cat-n-1\" synset=\"t-1\">"
            + "<SenseRelation relType=\"other\" target=\"t-cat-n-1\"/></Sense></LexicalEntry>"
            + "<Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>"));
    assertTrue(lexicon.synset("t-1").orElseThrow().relations().isEmpty());
  }

  @Test
  void testRejectsUnknownPartOfSpeech() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(
        wrap("<LexicalEntry id=\"t-cat-x\"><Lemma writtenForm=\"cat\" partOfSpeech=\"x\"/>"
            + "<Sense id=\"t-cat-x-1\" synset=\"t-1\"/></LexicalEntry>"
            + "<Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>")));
    assertTrue(e.getMessage().contains("x"));
  }

  @Test
  void testRejectsSynsetWithoutMembers() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse(
        wrap("<Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>orphan</Definition></Synset>")));
    assertTrue(e.getMessage().contains("t-1"));
  }
}
