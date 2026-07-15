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

package opennlp.tools.pii;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Document;
import opennlp.tools.glossary.AhoCorasickGlossaryMatcher;
import opennlp.tools.glossary.GlossaryAnnotator;
import opennlp.tools.glossary.GlossaryEntry;

public class MaskerTest {

  @Test
  void testMasksPiiLayerLengthPreserving() {
    final PiiAnnotator annotator = new PiiAnnotator(new CursorPiiExtractor());
    final String text = "Contact jane@example.com or call +44 20 7946 0958.";
    final Document document = annotator.annotate(Document.of(text));

    final String masked = Masker.mask(document, PiiAnnotator.PII, '*');

    Assertions.assertEquals(text.length(), masked.length());
    Assertions.assertEquals("Contact **************** or call ****************.", masked);
  }

  @Test
  void testMasksSeveralLayersAtOnce() {
    final String text = "Send the Widget Press manual to jane@example.com now.";
    Document document = new PiiAnnotator(new CursorPiiExtractor())
        .annotate(Document.of(text));
    document = new GlossaryAnnotator(new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ACME-1", "widget press")), true)).annotate(document);

    final String masked = Masker.mask(document,
        List.of(PiiAnnotator.PII, GlossaryAnnotator.GLOSSARY), '#');

    Assertions.assertEquals(text.length(), masked.length());
    Assertions.assertEquals("Send the ############ manual to ################ now.", masked);
  }

  @Test
  void testInvalidArguments() {
    final Document document = Document.of("some text");
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> Masker.mask(null, PiiAnnotator.PII, '*'));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> Masker.mask(document, (opennlp.tools.document.LayerKey<?>) null, '*'));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> Masker.mask(document, List.of(), '*'));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> Masker.mask(document, PiiAnnotator.PII, '*'));
  }
}
