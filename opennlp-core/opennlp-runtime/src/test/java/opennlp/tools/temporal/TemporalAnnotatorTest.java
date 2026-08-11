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

package opennlp.tools.temporal;

import java.time.LocalDate;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.util.Span;

/**
 * Pins the reference date relative expressions resolve against in the document
 * pipeline: a document that dates itself in a dateline resolves the relative
 * expressions that follow, while a document that dates itself nowhere leaves them
 * unreported rather than guessing against the wall clock.
 */
public class TemporalAnnotatorTest {

  private final TemporalAnnotator annotator =
      new TemporalAnnotator(new CursorTemporalExtractor());

  private static List<Annotation<TemporalExpression>> temporals(TemporalAnnotator annotator,
      String text) {
    return annotator.annotate(Document.of(text)).get(TemporalAnnotator.TEMPORALS);
  }

  private static Span spanOf(String text, String fragment) {
    final int start = text.indexOf(fragment);
    Assertions.assertTrue(start >= 0, "fragment not found: " + fragment);
    return new Span(start, start + fragment.length());
  }

  /**
   * Pins the headline behavior: a dateline dates the document, so a relative expression
   * later in the same text resolves against it and is reported with its own span and
   * its resolved value.
   */
  @Test
  void testDatelineResolvesALaterRelativeExpression() {
    final String text = "Berlin, 14 July 2026. The buyer paid yesterday.";

    final List<Annotation<TemporalExpression>> mentions = temporals(annotator, text);

    Assertions.assertEquals(2, mentions.size());
    Assertions.assertEquals(spanOf(text, "14 July 2026"), mentions.get(0).span());
    Assertions.assertEquals("2026-07-14", mentions.get(0).value().value());
    Assertions.assertEquals(spanOf(text, "yesterday"), mentions.get(1).span());
    Assertions.assertEquals("2026-07-13", mentions.get(1).value().value());
    Assertions.assertEquals(TemporalExpression.Granularity.DAY,
        mentions.get(1).value().granularity());
  }

  /**
   * Pins the election rule as the dateline rule of {@link DocumentDateAnnotator}: the
   * first day-granularity mention in text order supplies the reference, later days do
   * not.
   */
  @Test
  void testFirstDayMentionSuppliesTheReference() {
    final String text = "Filed 2026-07-10. Hearing on 2026-09-02. Amended yesterday.";

    final List<Annotation<TemporalExpression>> mentions = temporals(annotator, text);

    Assertions.assertEquals(3, mentions.size());
    Assertions.assertEquals(spanOf(text, "yesterday"), mentions.get(2).span());
    Assertions.assertEquals("2026-07-09", mentions.get(2).value().value());
  }

  /**
   * Pins the resolution of every supported relative shape once a dateline is present: a
   * day word, a counted offset, and a coarser unit that keeps its own granularity.
   */
  @Test
  void testCountedAndCoarserRelativesResolveAgainstTheDateline() {
    final String text = "Berlin, 14 July 2026. Shipped 3 days ago, audited last month, "
        + "due tomorrow.";

    final List<Annotation<TemporalExpression>> mentions = temporals(annotator, text);

    Assertions.assertEquals(4, mentions.size());
    Assertions.assertEquals("2026-07-11", mentions.get(1).value().value());
    Assertions.assertEquals(spanOf(text, "3 days ago"), mentions.get(1).span());
    Assertions.assertEquals("2026-06", mentions.get(2).value().value());
    Assertions.assertEquals(TemporalExpression.Granularity.MONTH,
        mentions.get(2).value().granularity());
    Assertions.assertEquals("2026-07-15", mentions.get(3).value().value());
  }

  /**
   * Pins that a text without a day-granularity mention leaves relative expressions
   * unreported: a month mention is too coarse to date the document, exactly as it is
   * too coarse to elect the document date.
   */
  @Test
  void testTextWithoutADayMentionLeavesRelativesUnreported() {
    final String text = "The July 2026 report was filed yesterday.";

    final List<Annotation<TemporalExpression>> mentions = temporals(annotator, text);

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(spanOf(text, "July 2026"), mentions.get(0).span());
    Assertions.assertEquals("2026-07", mentions.get(0).value().value());
  }

  /**
   * Pins that a text holding nothing but a relative expression yields the temporal
   * layer present and empty rather than a mention resolved against the wall clock.
   */
  @Test
  void testRelativeWithoutAnyDateYieldsAnEmptyLayer() {
    final Document document = annotator.annotate(Document.of("we shipped it yesterday"));

    Assertions.assertTrue(document.get(TemporalAnnotator.TEMPORALS).isEmpty());
  }

  /**
   * Pins that the annotator and {@link DocumentDateAnnotator} agree in a pipeline: the
   * mention that resolves the relative expressions is the one that elects the document
   * date.
   */
  @Test
  void testPipelineElectsTheSameDateThatResolvedTheRelatives() {
    final String text = "Berlin, 14 July 2026. The buyer paid yesterday.";
    final Document document = DocumentAnalyzer.builder()
        .add(new TemporalAnnotator(new CursorTemporalExtractor()))
        .add(new DocumentDateAnnotator())
        .build()
        .analyze(text);

    final List<Annotation<LocalDate>> dates =
        document.get(DocumentDateAnnotator.DOCUMENT_DATE);
    Assertions.assertEquals(1, dates.size());
    Assertions.assertEquals(LocalDate.of(2026, 7, 14), dates.get(0).value());
    Assertions.assertEquals(spanOf(text, "14 July 2026"), dates.get(0).span());
    Assertions.assertEquals("2026-07-13",
        document.get(TemporalAnnotator.TEMPORALS).get(1).value().value());
  }

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new TemporalAnnotator(null));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> annotator.annotate(null));
  }
}
