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

package opennlp.tools.formats.conllu;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.coref.CorefAnnotator;
import opennlp.tools.coref.CorefMention;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.Span;

public class ConlluCorefDocumentStreamTest {

  /** Two documents: nested and rich entity brackets, a speaker, a multiword range. */
  private static final String CONLLU = """
      # newdoc id = doc1
      # sent_id = doc1-1
      # speaker = Ann
      1\tByron\tByron\tPROPN\tNNP\t_\t2\tnsubj\t_\tEntity=(1)
      2\tmet\tmeet\tVERB\tVBD\t_\t0\troot\t_\t_
      3\this\the\tPRON\tPRP$\t_\t4\tnmod:poss\t_\tEntity=(2-person-new(1)
      4\tmother\tmother\tNOUN\tNN\t_\t2\tobj\t_\tEntity=2)|SpaceAfter=No
      5\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_

      # sent_id = doc1-2
      1-2\tShe's\t_\t_\t_\t_\t_\t_\t_\t_
      1\tShe\tshe\tPRON\tPRP\t_\t3\tnsubj\t_\tEntity=(2)
      2\t's\tbe\tAUX\tVBZ\t_\t3\tcop\t_\t_
      3\thome\thome\tNOUN\tNN\t_\t0\troot\t_\tSpaceAfter=No
      4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

      # newdoc id = doc2
      1\tAcme\tAcme\tPROPN\tNNP\t_\t2\tnsubj\t_\tEntity=(7)
      2\tgrew\tgrow\tVERB\tVBD\t_\t0\troot\t_\tSpaceAfter=No
      3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_
      """;

  private static ConlluCorefDocumentStream stream(String conllu, ConlluTagset tagset)
      throws IOException {
    return new ConlluCorefDocumentStream(
        () -> new ByteArrayInputStream(conllu.getBytes(StandardCharsets.UTF_8)), tagset);
  }

  @Test
  void testReadsDocumentsWithLayersAndGoldChains() throws IOException {
    try (ConlluCorefDocumentStream stream = stream(CONLLU, ConlluTagset.X)) {
      final Document first = stream.read();
      Assertions.assertEquals("Byron met his mother.\nShe 's home.", first.text());
      Assertions.assertEquals(List.of("Byron met his mother.", "She 's home."),
          first.get(Layers.SENTENCES).stream().map(Annotation::value).toList());
      Assertions.assertEquals(List.of("Byron", "met", "his", "mother", ".", "She", "'s",
          "home", "."), first.get(Layers.TOKENS).stream().map(Annotation::value).toList());
      Assertions.assertEquals(List.of("NNP", "VBD", "PRP$", "NN", ".", "PRP", "VBZ", "NN",
          "."), first.get(Layers.POS_TAGS).stream().map(Annotation::value).toList());
      Assertions.assertEquals(List.of(new Annotation<>(new Span(0, 21), "Ann")),
          first.get(CorefAnnotator.SPEAKERS));

      final List<Annotation<CorefMention>> chains = first.get(CorefAnnotator.GOLD_CHAINS);
      Assertions.assertEquals(List.of("Byron", "his mother", "his", "She"),
          chains.stream().map(a -> first.text().subSequence(
              a.span().getStart(), a.span().getEnd()).toString()).toList());
      Assertions.assertEquals(List.of(0, 1, 0, 1),
          chains.stream().map(a -> a.value().chain()).toList());
      Assertions.assertEquals(CorefMention.KIND_GOLD, chains.get(0).value().kind());

      final Document second = stream.read();
      Assertions.assertEquals("Acme grew.", second.text());
      Assertions.assertFalse(second.layers().contains(CorefAnnotator.SPEAKERS));
      Assertions.assertEquals(1, second.get(CorefAnnotator.GOLD_CHAINS).size());
      Assertions.assertNull(stream.read());

      stream.reset();
      Assertions.assertEquals("Byron met his mother.\nShe 's home.", stream.read().text());
    }
  }

  @Test
  void testUniversalTagsetReadsTheUposColumn() throws IOException {
    try (ConlluCorefDocumentStream stream = stream(CONLLU, ConlluTagset.U)) {
      Assertions.assertEquals("PROPN",
          stream.read().get(Layers.POS_TAGS).get(0).value());
    }
  }

  @Test
  void testFileWithoutNewdocIsOneDocument() throws IOException {
    final String single = "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\tEntity=(3)\n";
    try (ConlluCorefDocumentStream stream = stream(single, ConlluTagset.X)) {
      Assertions.assertEquals("Acme", stream.read().text());
      Assertions.assertNull(stream.read());
    }
  }

  @Test
  void testDiscontinuousMentionPartsJoinTheirEntity() throws IOException {
    final String parts = "1\tKim\tKim\tPROPN\tNNP\t_\t0\troot\t_\tEntity=(e5[1/2]-person-1)\n"
        + "2\tand\tand\tCCONJ\tCC\t_\t3\tcc\t_\t_\n"
        + "3\tLee\tLee\tPROPN\tNNP\t_\t1\tconj\t_\tEntity=(e5[2/2]-person-1)\n";
    try (ConlluCorefDocumentStream stream = stream(parts, ConlluTagset.X)) {
      final List<Annotation<CorefMention>> chains =
          stream.read().get(CorefAnnotator.GOLD_CHAINS);
      Assertions.assertEquals(2, chains.size());
      Assertions.assertEquals(chains.get(0).value().chain(), chains.get(1).value().chain());
    }
  }

  @Test
  void testRejectsUnbalancedBracketsAndShortLines() throws IOException {
    try (ConlluCorefDocumentStream open = stream(
        "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\tEntity=(3\n", ConlluTagset.X)) {
      Assertions.assertThrows(InvalidFormatException.class, open::read);
    }
    try (ConlluCorefDocumentStream closed = stream(
        "1\tAcme\tAcme\tPROPN\tNNP\t_\t0\troot\t_\tEntity=3)\n", ConlluTagset.X)) {
      Assertions.assertThrows(InvalidFormatException.class, closed::read);
    }
    try (ConlluCorefDocumentStream shortLine = stream("1\tAcme\n", ConlluTagset.X)) {
      Assertions.assertThrows(InvalidFormatException.class, shortLine::read);
    }
  }

  @Test
  void testRejectsNullArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new ConlluCorefDocumentStream(null, ConlluTagset.X));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new ConlluCorefDocumentStream(
            () -> new ByteArrayInputStream(new byte[0]), null));
  }
}
