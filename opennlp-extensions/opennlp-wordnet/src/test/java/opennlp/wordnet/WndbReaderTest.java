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

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.UnaryOperator;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import opennlp.tools.wordnet.Synset;
import opennlp.tools.wordnet.WordNetLexicon;
import opennlp.tools.wordnet.WordNetPos;
import opennlp.tools.wordnet.WordNetRelation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class WndbReaderTest {

  private static final String DOG_ID = "wndb-00001075-n";
  private static final String CANID_ID = "wndb-00001160-n";

  static Path fixtureDirectory() {
    final URL url = WndbReaderTest.class.getResource("mini-wndb");
    assertNotNull(url, "Fixture directory mini-wndb must be on the test classpath");
    try {
      return Path.of(url.toURI());
    } catch (URISyntaxException e) {
      throw new IllegalStateException("Unexpected fixture URI: " + url, e);
    }
  }

  static WordNetLexicon fixture() {
    return WndbReader.read(fixtureDirectory());
  }

  @Test
  void testLookupReturnsSynsetWithAllComponents() {
    final List<Synset> senses = fixture().lookup("dog", WordNetPos.NOUN);
    assertEquals(1, senses.size());
    final Synset dog = senses.get(0);
    assertEquals(DOG_ID, dog.id());
    assertEquals(WordNetPos.NOUN, dog.pos());
    assertEquals(List.of("dog", "domestic dog"), dog.lemmas());
    assertEquals("a domesticated canid", dog.gloss());
    assertEquals(List.of(CANID_ID), dog.related(WordNetRelation.HYPERNYM));
  }

  @Test
  void testLookupFoldsCaseAndUnderscore() {
    final WordNetLexicon lexicon = fixture();
    assertEquals(DOG_ID, lexicon.lookup("Domestic_Dog", WordNetPos.NOUN).get(0).id());
    assertEquals(DOG_ID, lexicon.lookup("DOG", WordNetPos.NOUN).get(0).id());
  }

  @Test
  void testLookupKeepsIndexSenseOrder() {
    assertEquals(List.of("wndb-00001427-n", "wndb-00001669-n"),
        fixture().lookup("run", WordNetPos.NOUN).stream().map(Synset::id).toList());
  }

  @Test
  void testRelationNavigation() {
    final WordNetLexicon lexicon = fixture();
    assertEquals(List.of(DOG_ID), lexicon.related(CANID_ID, WordNetRelation.HYPONYM));
    assertEquals(List.of("wndb-00001075-v", "wndb-00001171-v"),
        lexicon.related("wndb-00001324-v", WordNetRelation.HYPONYM));
    assertEquals(List.of("wndb-00001075-v"),
        lexicon.related("wndb-00001427-n", WordNetRelation.DERIVATIONALLY_RELATED));
  }

  @Test
  void testLexicalPointersSurfaceAtSynsetLevel() {
    final WordNetLexicon lexicon = fixture();
    assertEquals(List.of("wndb-00001141-a"),
        lexicon.related("wndb-00001075-a", WordNetRelation.ANTONYM));
    assertEquals(List.of("wndb-00001075-a"),
        lexicon.related("wndb-00001141-a", WordNetRelation.ANTONYM));
  }

  @Test
  void testSatelliteNormalizesToAdjectiveAndMarkerIsStripped() {
    final WordNetLexicon lexicon = fixture();
    final Synset large = lexicon.lookup("large", WordNetPos.ADJECTIVE).get(0);
    assertEquals(WordNetPos.ADJECTIVE, large.pos());
    assertEquals(List.of("wndb-00001211-a"), large.related(WordNetRelation.SIMILAR_TO));
    // short is stored as short(p); the syntactic marker is not part of the lemma.
    assertEquals(List.of("short"),
        lexicon.lookup("short", WordNetPos.ADJECTIVE).get(0).lemmas());
  }

  @Test
  void testUnknownLemmaOrSynsetIsEmpty() {
    final WordNetLexicon lexicon = fixture();
    assertTrue(lexicon.lookup("zebra", WordNetPos.NOUN).isEmpty());
    assertTrue(lexicon.synset("wndb-99999999-n").isEmpty());
  }

  @Test
  void testRejectsNullAndMissingDirectory(@TempDir Path tempDir) {
    assertThrows(IllegalArgumentException.class, () -> WndbReader.read(null));
    assertThrows(IllegalArgumentException.class,
        () -> WndbReader.read(tempDir.resolve("absent")));
  }

  @Test
  void testRejectsMissingDatabaseFile(@TempDir Path tempDir) throws IOException {
    copyFixture(tempDir);
    Files.delete(tempDir.resolve("data.verb"));
    final IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> WndbReader.read(tempDir));
    assertTrue(e.getMessage().contains("data.verb"));
  }

  @Test
  void testRejectsIndexOffsetWithoutDataLine(@TempDir Path tempDir) throws IOException {
    copyFixture(tempDir);
    mutate(tempDir, "index.noun", line -> line.startsWith("berry ")
        ? line.replace("00001564", "00001565") : line);
    final IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> WndbReader.read(tempDir));
    assertTrue(e.getMessage().contains("berry"));
    assertTrue(e.getMessage().contains("00001565"));
  }

  @Test
  void testRejectsDataOffsetFieldMismatch(@TempDir Path tempDir) throws IOException {
    copyFixture(tempDir);
    mutate(tempDir, "data.noun",
        line -> line.replace("00001503 03 n 01 box", "00001504 03 n 01 box"));
    final IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> WndbReader.read(tempDir));
    assertTrue(e.getMessage().contains("disagrees"));
  }

  @Test
  void testRejectsTruncatedDataLine(@TempDir Path tempDir) throws IOException {
    copyFixture(tempDir);
    mutate(tempDir, "data.noun", line -> line.startsWith("00001564")
        ? line.substring(0, line.indexOf(" 000 |")) : line);
    final IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> WndbReader.read(tempDir));
    assertTrue(e.getMessage().contains("data.noun"));
    assertTrue(e.getMessage().contains("Truncated"));
  }

  @Test
  void testRejectsUndeclaredPointerSymbol(@TempDir Path tempDir) throws IOException {
    copyFixture(tempDir);
    mutate(tempDir, "data.noun", line -> line.replace("001 @ 00001160 n 0000",
        "001 ? 00001160 n 0000"));
    final IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> WndbReader.read(tempDir));
    assertTrue(e.getMessage().contains("Undeclared pointer symbol: ?"));
  }

  @Test
  void testRejectsPointerToNonexistentSynset(@TempDir Path tempDir) throws IOException {
    copyFixture(tempDir);
    mutate(tempDir, "data.noun", line -> line.replace("001 @ 00001160 n 0000",
        "001 @ 00009999 n 0000"));
    final IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> WndbReader.read(tempDir));
    assertTrue(e.getMessage().contains("wndb-00009999-n"));
  }

  private static void copyFixture(Path target) throws IOException {
    try (var files = Files.list(fixtureDirectory())) {
      for (final Path file : files.toList()) {
        Files.copy(file, target.resolve(file.getFileName().toString()));
      }
    }
  }

  // Applies a line transformation to one fixture file. The mutations only ever keep or shrink
  // line lengths of the affected line's own fields, so surrounding offsets stay valid.
  private static void mutate(Path directory, String fileName, UnaryOperator<String> edit)
      throws IOException {
    final Path file = directory.resolve(fileName);
    final List<String> lines = Files.readAllLines(file, StandardCharsets.ISO_8859_1);
    final StringBuilder out = new StringBuilder();
    for (final String line : lines) {
      out.append(edit.apply(line)).append('\n');
    }
    Files.writeString(file, out.toString(), StandardCharsets.ISO_8859_1);
  }
}
