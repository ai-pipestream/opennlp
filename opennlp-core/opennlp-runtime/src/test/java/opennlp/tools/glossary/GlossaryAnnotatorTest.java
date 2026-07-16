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

package opennlp.tools.glossary;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;

public class GlossaryAnnotatorTest {

  @Test
  void testProvidesGlossaryLayer() {
    final GlossaryAnnotator annotator = new GlossaryAnnotator(
        new AhoCorasickGlossaryMatcher(
            List.of(new GlossaryEntry("Q60", "New York City")), true));

    final Document document =
        annotator.annotate(Document.of("Prices in new york city keep climbing."));

    final List<Annotation<GlossaryMatch>> hits = document.get(GlossaryAnnotator.GLOSSARY);
    Assertions.assertEquals(1, hits.size());
    Assertions.assertEquals("Q60", hits.get(0).value().id());
    Assertions.assertEquals("new york city", document.text().subSequence(
        hits.get(0).span().getStart(), hits.get(0).span().getEnd()).toString());
    Assertions.assertTrue(document.layers().contains(GlossaryAnnotator.GLOSSARY));
  }

  /**
   * Verifies that a text without any glossary hit, including the empty text, still gets
   * a glossary layer: the layer key is present on the document and its annotation list
   * is empty, so consumers can distinguish "annotator ran, found nothing" from
   * "annotator never ran".
   */
  @Test
  void testNoHitsStillProvidesEmptyGlossaryLayer() {
    final GlossaryAnnotator annotator = new GlossaryAnnotator(
        new AhoCorasickGlossaryMatcher(
            List.of(new GlossaryEntry("Q60", "New York City")), false));

    final Document noHits = annotator.annotate(Document.of("Nothing to see here."));
    Assertions.assertTrue(noHits.layers().contains(GlossaryAnnotator.GLOSSARY));
    Assertions.assertTrue(noHits.get(GlossaryAnnotator.GLOSSARY).isEmpty());

    final Document emptyText = annotator.annotate(Document.of(""));
    Assertions.assertTrue(emptyText.layers().contains(GlossaryAnnotator.GLOSSARY));
    Assertions.assertTrue(emptyText.get(GlossaryAnnotator.GLOSSARY).isEmpty());
  }

  /**
   * Verifies that annotating the same document twice fails loud: the second run tries
   * to add the glossary layer again and the document rejects the duplicate layer with
   * an {@link IllegalArgumentException}.
   */
  @Test
  void testAnnotateTwiceRejectsDuplicateGlossaryLayer() {
    final GlossaryAnnotator annotator = new GlossaryAnnotator(
        new AhoCorasickGlossaryMatcher(
            List.of(new GlossaryEntry("Q60", "New York City")), false));

    final Document once = annotator.annotate(Document.of("New York City"));
    Assertions.assertEquals(1, once.get(GlossaryAnnotator.GLOSSARY).size());
    Assertions.assertThrows(IllegalArgumentException.class, () -> annotator.annotate(once));
  }

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class, () -> new GlossaryAnnotator(null));
    final GlossaryAnnotator annotator = new GlossaryAnnotator(
        new AhoCorasickGlossaryMatcher(List.of(new GlossaryEntry("T", "term")), false));
    Assertions.assertThrows(IllegalArgumentException.class, () -> annotator.annotate(null));
  }
}
