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

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.LayerKey;
import opennlp.tools.util.Span;

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

  /**
   * Verifies that several layers are masked in one pass: the PII layer produced by the
   * annotator and a directly built custom span layer, since the masker works with any
   * span layer, not only PII.
   */
  @Test
  void testMasksSeveralLayersAtOnce() {
    final String text = "Send the Widget Press manual to jane@example.com now.";
    final LayerKey<String> terms = LayerKey.of("term", String.class);
    Document document = new PiiAnnotator(new CursorPiiExtractor())
        .annotate(Document.of(text));
    document = document.with(terms,
        List.of(new Annotation<>(new Span(9, 21), "widget press")));

    final String masked = Masker.mask(document, List.of(PiiAnnotator.PII, terms), '#');

    Assertions.assertEquals(text.length(), masked.length());
    Assertions.assertEquals("Send the ############ manual to ################ now.", masked);
  }

  /**
   * Verifies that masking a document whose PII layer is present but holds no
   * detections returns the text completely unchanged.
   */
  @Test
  void testMaskWithoutDetectionsLeavesTextIdentical() {
    final String text = "Nothing sensitive here.";
    final Document document =
        new PiiAnnotator(new CursorPiiExtractor()).annotate(Document.of(text));

    Assertions.assertEquals(text, Masker.mask(document, PiiAnnotator.PII, '*'));
  }

  /**
   * Verifies that a card number followed by its space-separated expiry date is fully
   * masked while the expiry stays readable: the redacted copy must not leak a single
   * card digit.
   */
  @Test
  void testMasksCardFollowedByExpiry() {
    final String text = "Card 4111111111111111 12/26";
    final Document document =
        new PiiAnnotator(new CursorPiiExtractor()).annotate(Document.of(text));

    Assertions.assertEquals("Card **************** 12/26",
        Masker.mask(document, PiiAnnotator.PII, '*'));
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
