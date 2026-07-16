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

/**
 * Demonstrates the end-to-end PII flow on one realistic text that contains an email
 * address, a phone number, and a payment card number: annotate the document, read the
 * PII layer back with exact spans, types, and normalized forms, and finally produce a
 * masked copy of the text in which only the detected spans are replaced.
 */
public class PiiUsageExampleTest {

  /**
   * The example text; it holds exactly one mention of each of the three types.
   */
  private static final String TEXT =
      "Contact jane@example.com, call (555) 123-4567, or charge card 4111 1111 1111 1111.";

  /**
   * Annotates the example text and verifies the complete PII layer: three mentions in
   * text order, each with its exact span offsets, the exact text the span covers, the
   * expected type constant, and the exact normalized form.
   */
  @Test
  void testAnnotateAndReadPiiLayer() {
    final PiiAnnotator annotator = new PiiAnnotator(new CursorPiiExtractor());

    final Document document = annotator.annotate(Document.of(TEXT));

    Assertions.assertTrue(document.layers().contains(PiiAnnotator.PII));
    final List<Annotation<PiiMention>> mentions = document.get(PiiAnnotator.PII);
    Assertions.assertEquals(3, mentions.size());
    assertMention(mentions.get(0), 8, 24,
        PiiMention.TYPE_EMAIL, "jane@example.com", "jane@example.com");
    assertMention(mentions.get(1), 31, 45,
        PiiMention.TYPE_PHONE, "(555) 123-4567", "5551234567");
    assertMention(mentions.get(2), 62, 81,
        PiiMention.TYPE_CARD, "4111 1111 1111 1111", "4111111111111111");
  }

  /**
   * Masks the annotated document and verifies the exact redacted string: every
   * character inside a detected span becomes the mask character, every character
   * outside the spans is unchanged, and the overall length is preserved.
   */
  @Test
  void testMaskProducesExactRedactedText() {
    final PiiAnnotator annotator = new PiiAnnotator(new CursorPiiExtractor());
    final Document document = annotator.annotate(Document.of(TEXT));

    final String masked = Masker.mask(document, PiiAnnotator.PII, '*');

    Assertions.assertEquals(TEXT.length(), masked.length());
    Assertions.assertEquals(
        "Contact ****************, call **************, or charge card *******************.",
        masked);
  }

  /**
   * Verifies one annotation of the PII layer against its expected span offsets in
   * {@link #TEXT}, the text those offsets cover, and the type and normalized form of
   * the carried {@link PiiMention}. Also verifies that the annotation span and the
   * mention span agree, since downstream consumers may read either one.
   *
   * @param annotation The annotation to verify. Must not be {@code null}.
   * @param start The expected span start, inclusive.
   * @param end The expected span end, exclusive.
   * @param type The expected mention type.
   * @param covered The exact text the span is expected to cover.
   * @param normalized The exact expected normalized form.
   * @throws IllegalArgumentException Thrown if {@code annotation} is {@code null}.
   */
  private static void assertMention(Annotation<PiiMention> annotation, int start, int end,
      String type, String covered, String normalized) {
    if (annotation == null) {
      throw new IllegalArgumentException("annotation must not be null");
    }
    Assertions.assertEquals(start, annotation.span().getStart());
    Assertions.assertEquals(end, annotation.span().getEnd());
    Assertions.assertEquals(annotation.span(), annotation.value().span());
    Assertions.assertEquals(covered, TEXT.substring(start, end));
    Assertions.assertEquals(type, annotation.value().type());
    Assertions.assertEquals(normalized, annotation.value().normalized());
  }
}
