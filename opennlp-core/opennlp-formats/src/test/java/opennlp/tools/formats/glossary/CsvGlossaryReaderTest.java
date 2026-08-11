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

import opennlp.tools.glossary.GlossaryEntry;
import opennlp.tools.util.InvalidFormatException;

/**
 * Pins the CSV term list reader: RFC&#160;4180 quoting, delimiter and header options,
 * BOM tolerance, and loud failures with line numbers for malformed rows.
 */
public class CsvGlossaryReaderTest {

  private static InputStream utf8(String content) {
    return new ByteArrayInputStream(content.getBytes(StandardCharsets.UTF_8));
  }

  /**
   * The minimal shape: one row per entry, first column id, second column term,
   * returned in file order.
   */
  @Test
  void testReadsSimpleCommaSeparatedRows() throws IOException {
    final List<GlossaryEntry> entries = new CsvGlossaryReader()
        .read(utf8("Q1,hot dog\nQ2,New York City\n"));

    Assertions.assertEquals(2, entries.size());
    Assertions.assertEquals("Q1", entries.get(0).id());
    Assertions.assertEquals("hot dog", entries.get(0).term());
    Assertions.assertEquals("Q2", entries.get(1).id());
    Assertions.assertEquals("New York City", entries.get(1).term());
  }

  /**
   * With header skipping on, the first row is dropped whatever it contains.
   */
  @Test
  void testSkipsHeaderWhenConfigured() throws IOException {
    final List<GlossaryEntry> entries = new CsvGlossaryReader(',', true)
        .read(utf8("id,term\nQ1,hot dog\n"));

    Assertions.assertEquals(1, entries.size());
    Assertions.assertEquals("Q1", entries.get(0).id());
  }

  /**
   * RFC 4180 quoting: a quoted field may contain the delimiter, an escaped quote
   * (doubled), and even a line break.
   */
  @Test
  void testQuotedFieldsCarryDelimitersQuotesAndNewlines() throws IOException {
    final List<GlossaryEntry> entries = new CsvGlossaryReader()
        .read(utf8("Q1,\"hot, dog\"\nQ2,\"say \"\"hi\"\"\"\nQ3,\"line one\nline two\"\n"));

    Assertions.assertEquals(3, entries.size());
    Assertions.assertEquals("hot, dog", entries.get(0).term());
    Assertions.assertEquals("say \"hi\"", entries.get(1).term());
    Assertions.assertEquals("line one\nline two", entries.get(2).term());
  }

  /**
   * Tab is a first-class delimiter for TSV term lists.
   */
  @Test
  void testTabDelimiter() throws IOException {
    final List<GlossaryEntry> entries = new CsvGlossaryReader('\t', false)
        .read(utf8("Q1\thot dog\nQ2\tNew York City\n"));

    Assertions.assertEquals(2, entries.size());
    Assertions.assertEquals("New York City", entries.get(1).term());
  }

  /**
   * CRLF line ends read like LF, blank lines anywhere are skipped, and a missing
   * final newline is fine.
   */
  @Test
  void testCrlfBlankLinesAndMissingFinalNewline() throws IOException {
    final List<GlossaryEntry> entries = new CsvGlossaryReader()
        .read(utf8("Q1,hot dog\r\n\r\nQ2,New York City"));

    Assertions.assertEquals(2, entries.size());
    Assertions.assertEquals("Q2", entries.get(1).id());
  }

  /**
   * A UTF-8 byte order mark (the Excel export signature) is stripped rather than
   * glued onto the first id.
   */
  @Test
  void testUtf8ByteOrderMarkIsStripped() throws IOException {
    final List<GlossaryEntry> entries = new CsvGlossaryReader()
        .read(utf8("\uFEFFQ1,hot dog\n"));

    Assertions.assertEquals(1, entries.size());
    Assertions.assertEquals("Q1", entries.get(0).id());
  }

  /**
   * Columns beyond the second are ignored, leaving room for metadata columns the
   * reader does not model.
   */
  @Test
  void testExtraColumnsAreIgnored() throws IOException {
    final List<GlossaryEntry> entries = new CsvGlossaryReader()
        .read(utf8("Q1,hot dog,en,approved\n"));

    Assertions.assertEquals(1, entries.size());
    Assertions.assertEquals("Q1", entries.get(0).id());
    Assertions.assertEquals("hot dog", entries.get(0).term());
  }

  /**
   * A row with fewer than two columns fails loud with its line number.
   */
  @Test
  void testTooFewColumnsFailsWithLineNumber() {
    final InvalidFormatException ex = Assertions.assertThrows(InvalidFormatException.class,
        () -> new CsvGlossaryReader().read(utf8("Q1,hot dog\nQ2\n")));
    Assertions.assertTrue(ex.getMessage().contains("line 2"), ex.getMessage());
  }

  /**
   * Blank ids and blank terms fail loud with the offending line number instead of
   * surfacing as a bare IllegalArgumentException from {@link GlossaryEntry}.
   */
  @Test
  void testBlankIdOrTermFailsWithLineNumber() {
    final InvalidFormatException blankId = Assertions.assertThrows(InvalidFormatException.class,
        () -> new CsvGlossaryReader().read(utf8(" ,hot dog\n")));
    Assertions.assertTrue(blankId.getMessage().contains("line 1"), blankId.getMessage());

    final InvalidFormatException blankTerm = Assertions.assertThrows(InvalidFormatException.class,
        () -> new CsvGlossaryReader().read(utf8("Q1,hot dog\nQ2,\"\"\n")));
    Assertions.assertTrue(blankTerm.getMessage().contains("line 2"), blankTerm.getMessage());
  }

  /**
   * A quote opened and never closed fails loud instead of silently swallowing the
   * rest of the file into one field.
   */
  @Test
  void testUnclosedQuoteFailsLoud() {
    Assertions.assertThrows(InvalidFormatException.class,
        () -> new CsvGlossaryReader().read(utf8("Q1,\"never closed\n")));
  }

  /**
   * Empty input reads as an empty list; the matcher constructors reject an empty
   * glossary downstream, so nothing silently matches nothing.
   */
  @Test
  void testEmptyInputYieldsEmptyList() throws IOException {
    Assertions.assertTrue(new CsvGlossaryReader().read(utf8("")).isEmpty());
    Assertions.assertTrue(new CsvGlossaryReader(',', true).read(utf8("id,term\n")).isEmpty());
  }

  /**
   * Constructor and read-side argument validation: the delimiter may not be the
   * quote or a line-break character, and the stream must not be {@code null}.
   */
  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CsvGlossaryReader('"', false));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CsvGlossaryReader('\n', false));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CsvGlossaryReader('\r', false));
    final CsvGlossaryReader reader = new CsvGlossaryReader();
    Assertions.assertThrows(IllegalArgumentException.class, () -> reader.read(null));
  }
}
