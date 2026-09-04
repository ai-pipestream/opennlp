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
import opennlp.tools.glossary.AhoCorasickGlossaryMatcher;
import opennlp.tools.glossary.GlossaryEntry;
import opennlp.tools.glossary.GlossaryMatch;

/**
 * Mirrors the glossary-loading examples in the manual: a TBX termbase and a CSV term
 * list load into {@link GlossaryEntry} lists that feed an
 * {@link AhoCorasickGlossaryMatcher} directly.
 */
public class GlossaryLoadingExampleTest extends AbstractFormatTest {

  /**
   * Mirrors the TBX example in {@code glossary.xml}: the English terms of a TBX
   * termbase become the glossary, and an alias registered under the same concept id
   * reports that id.
   */
  @Test
  void testTbxManualExample() throws IOException {
    final TbxGlossaryReader reader = new TbxGlossaryReader("en");
    final List<GlossaryEntry> glossary;
    try (InputStream in = getResourceStream("glossary/glossary-v2.tbx")) {
      glossary = reader.read(in);
    }

    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(glossary, true);
    final String text = "A frankfurter, please.";
    final List<GlossaryMatch> hits = matcher.match(text);

    Assertions.assertEquals(1, hits.size());
    Assertions.assertEquals("c1", hits.get(0).id());
    Assertions.assertEquals("frankfurter",
        text.substring(hits.get(0).span().getStart(), hits.get(0).span().getEnd()));
  }

  /**
   * Mirrors the CSV example in {@code glossary.xml}: a two-column term list with a
   * header loads and matches.
   */
  @Test
  void testCsvManualExample() throws IOException {
    final String csv = "id,term\nQ60,New York City\nQ11299,Manhattan\n";
    final CsvGlossaryReader reader = new CsvGlossaryReader(',', true);
    final List<GlossaryEntry> glossary =
        reader.read(new ByteArrayInputStream(csv.getBytes(StandardCharsets.UTF_8)));

    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(glossary, true);
    final List<GlossaryMatch> hits = matcher.match("From Manhattan to New York City.");

    Assertions.assertEquals(2, hits.size());
    Assertions.assertEquals("Q11299", hits.get(0).id());
    Assertions.assertEquals("Q60", hits.get(1).id());
  }
}
