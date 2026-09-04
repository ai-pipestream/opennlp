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

package opennlp.tools.formats.glossary;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.formats.AbstractFormatTest;
import opennlp.tools.glossary.GlossaryEntry;
import opennlp.tools.util.InvalidFormatException;

/**
 * Pins the TBX termbase reader: both the TBX&#160;2 martif shape and the TBX&#160;3
 * conceptEntry shape load into {@link GlossaryEntry} lists, language selection follows
 * BCP&#160;47 prefix matching, and malformed or unsafe input fails closed with
 * {@link InvalidFormatException}.
 */
public class TbxGlossaryReaderTest extends AbstractFormatTest {

  /**
   * The TBX 2 fixture carries a DOCTYPE whose SYSTEM identifier exists nowhere, so a
   * successful read also proves the external DTD is tolerated but never fetched. All
   * terms of a matching langSet are collected in file order, whether they sit in a
   * tig or in a nested ntig term group, and aliases share the entry id.
   */
  @Test
  void testReadsV2TermEntriesForLanguage() throws IOException {
    final TbxGlossaryReader reader = new TbxGlossaryReader("en");
    final List<GlossaryEntry> entries;
    try (InputStream in = getResourceStream("glossary/glossary-v2.tbx")) {
      entries = reader.read(in);
    }

    Assertions.assertEquals(3, entries.size());
    Assertions.assertEquals("c1", entries.get(0).id());
    Assertions.assertEquals("hot dog", entries.get(0).term());
    Assertions.assertEquals("c1", entries.get(1).id());
    Assertions.assertEquals("frankfurter", entries.get(1).term());
    Assertions.assertEquals("c2", entries.get(2).id());
    Assertions.assertEquals("New York City", entries.get(2).term());
  }

  /**
   * The same file serves any of its languages: selecting {@code de} yields the German
   * term and none of the English ones.
   */
  @Test
  void testReadsOtherLanguageFromSameFile() throws IOException {
    final TbxGlossaryReader reader = new TbxGlossaryReader("de");
    final List<GlossaryEntry> entries;
    try (InputStream in = getResourceStream("glossary/glossary-v2.tbx")) {
      entries = reader.read(in);
    }

    Assertions.assertEquals(1, entries.size());
    Assertions.assertEquals("c1", entries.get(0).id());
    Assertions.assertEquals("Wiener W\u00FCrstchen", entries.get(0).term());
  }

  /**
   * The TBX 3 shape (conceptEntry, langSec, termSec, ISO 30042 namespace) reads with
   * the same reader; element names are matched by local name so the namespace does
   * not matter.
   */
  @Test
  void testReadsV3ConceptEntries() throws IOException {
    final TbxGlossaryReader reader = new TbxGlossaryReader("en");
    final List<GlossaryEntry> entries;
    try (InputStream in = getResourceStream("glossary/glossary-v3.tbx")) {
      entries = reader.read(in);
    }

    Assertions.assertEquals(3, entries.size());
    Assertions.assertEquals("ML", entries.get(0).id());
    Assertions.assertEquals("machine learning", entries.get(0).term());
    Assertions.assertEquals("NN", entries.get(1).id());
    Assertions.assertEquals("neural network", entries.get(1).term());
    Assertions.assertEquals("NN", entries.get(2).id());
    Assertions.assertEquals("neural net", entries.get(2).term());
  }

  /**
   * Language selection is case-insensitive and prefix-based like BCP 47 lookup:
   * {@code en} matches {@code en-US}, {@code EN-us} matches it exactly, and
   * {@code en-GB} does not match {@code en-US}.
   */
  @Test
  void testLanguageMatchingIsCaseInsensitiveAndPrefixBased() throws IOException {
    try (InputStream in = getResourceStream("glossary/glossary-v2.tbx")) {
      Assertions.assertEquals(3, new TbxGlossaryReader("EN-us").read(in).size());
    }
    try (InputStream in = getResourceStream("glossary/glossary-v2.tbx")) {
      Assertions.assertTrue(new TbxGlossaryReader("en-GB").read(in).isEmpty());
    }
  }

  /**
   * A language absent from the file reads as an empty list; the matcher constructors
   * fail loud on an empty glossary, so the miss cannot pass silently downstream.
   */
  @Test
  void testAbsentLanguageYieldsEmptyList() throws IOException {
    final TbxGlossaryReader reader = new TbxGlossaryReader("sv");
    try (InputStream in = getResourceStream("glossary/glossary-v2.tbx")) {
      Assertions.assertTrue(reader.read(in).isEmpty());
    }
  }

  /**
   * Internal entities declared in the internal DTD subset resolve normally; real
   * termbases use them for recurring names.
   */
  @Test
  void testInternalDtdEntityResolves() throws IOException {
    final String doc = "<?xml version=\"1.0\"?>"
        + "<!DOCTYPE martif [<!ENTITY co \"Acme Corp\">]>"
        + "<martif type=\"TBX\"><text><body>"
        + "<termEntry id=\"c1\"><langSet xml:lang=\"en\">"
        + "<tig><term>&co; press</term></tig>"
        + "</langSet></termEntry>"
        + "</body></text></martif>";

    final List<GlossaryEntry> entries = new TbxGlossaryReader("en")
        .read(new ByteArrayInputStream(doc.getBytes(StandardCharsets.UTF_8)));

    Assertions.assertEquals(1, entries.size());
    Assertions.assertEquals("Acme Corp press", entries.get(0).term());
  }

  /**
   * An external entity reference fails closed as {@link InvalidFormatException}
   * instead of touching the file system.
   */
  @Test
  void testExternalEntityFailsClosed() {
    final String doc = "<?xml version=\"1.0\"?>"
        + "<!DOCTYPE martif [<!ENTITY xxe SYSTEM \"file:///nonexistent-entity\">]>"
        + "<martif type=\"TBX\"><text><body>"
        + "<termEntry id=\"c1\"><langSet xml:lang=\"en\">"
        + "<tig><term>&xxe;</term></tig>"
        + "</langSet></termEntry>"
        + "</body></text></martif>";

    Assertions.assertThrows(InvalidFormatException.class, () -> new TbxGlossaryReader("en")
        .read(new ByteArrayInputStream(doc.getBytes(StandardCharsets.UTF_8))));
  }

  /**
   * A term entry without an id attribute fails loud: the id becomes the
   * {@link GlossaryEntry} identifier, and inventing one would silently corrupt
   * downstream joins.
   */
  @Test
  void testMissingEntryIdFailsLoud() {
    final String doc = "<?xml version=\"1.0\"?>"
        + "<martif type=\"TBX\"><text><body>"
        + "<termEntry><langSet xml:lang=\"en\">"
        + "<tig><term>orphan</term></tig>"
        + "</langSet></termEntry>"
        + "</body></text></martif>";

    final InvalidFormatException ex = Assertions.assertThrows(InvalidFormatException.class,
        () -> new TbxGlossaryReader("en")
            .read(new ByteArrayInputStream(doc.getBytes(StandardCharsets.UTF_8))));
    Assertions.assertTrue(ex.getMessage().contains("id"), ex.getMessage());
  }

  /**
   * A blank term element inside a selected language fails loud rather than producing
   * an entry the matcher constructors would reject with less context.
   */
  @Test
  void testBlankTermFailsLoud() {
    final String doc = "<?xml version=\"1.0\"?>"
        + "<martif type=\"TBX\"><text><body>"
        + "<termEntry id=\"c1\"><langSet xml:lang=\"en\">"
        + "<tig><term>   </term></tig>"
        + "</langSet></termEntry>"
        + "</body></text></martif>";

    Assertions.assertThrows(InvalidFormatException.class, () -> new TbxGlossaryReader("en")
        .read(new ByteArrayInputStream(doc.getBytes(StandardCharsets.UTF_8))));
  }

  /**
   * Truncated XML surfaces as {@link InvalidFormatException}, not as an unchecked
   * parser exception.
   */
  @Test
  void testMalformedXmlFailsAsInvalidFormat() {
    final String doc = "<?xml version=\"1.0\"?><martif type=\"TBX\"><text><body>"
        + "<termEntry id=\"c1\"><langSet xml:lang=\"en\"><tig><term>cut off";

    Assertions.assertThrows(InvalidFormatException.class, () -> new TbxGlossaryReader("en")
        .read(new ByteArrayInputStream(doc.getBytes(StandardCharsets.UTF_8))));
  }

  /**
   * A well-formed document that is not a termbase (wrong root element) is rejected
   * with a message naming the expected roots.
   */
  @Test
  void testWrongRootElementFailsLoud() {
    final String doc = "<?xml version=\"1.0\"?><html><body><p>not a termbase</p></body></html>";

    final InvalidFormatException ex = Assertions.assertThrows(InvalidFormatException.class,
        () -> new TbxGlossaryReader("en")
            .read(new ByteArrayInputStream(doc.getBytes(StandardCharsets.UTF_8))));
    Assertions.assertTrue(ex.getMessage().contains("martif")
        || ex.getMessage().contains("tbx"), ex.getMessage());
  }

  /**
   * Constructor and read-side argument validation fail with
   * {@link IllegalArgumentException}.
   */
  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class, () -> new TbxGlossaryReader(null));
    Assertions.assertThrows(IllegalArgumentException.class, () -> new TbxGlossaryReader(""));
    Assertions.assertThrows(IllegalArgumentException.class, () -> new TbxGlossaryReader(" "));
    final TbxGlossaryReader reader = new TbxGlossaryReader("en");
    Assertions.assertThrows(IllegalArgumentException.class, () -> reader.read(null));
  }
}
