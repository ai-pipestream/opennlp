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
 * Tests the document date election rule: the first day-granularity temporal mention in
 * the text wins, coarser mentions never elect a date, and the electing mention's span
 * stays on the annotation for auditability.
 */
public class DocumentDateAnnotatorTest {

  @Test
  void testFirstDayMentionElectsTheDate() {
    final Document document = Document.of("2026-07 report, filed 2026-07-10, due 2026-07-20.")
        .with(TemporalAnnotator.TEMPORALS, List.of(
            new Annotation<>(new Span(0, 7), new TemporalExpression(
                new Span(0, 7), "2026-07", TemporalExpression.Granularity.MONTH)),
            new Annotation<>(new Span(22, 32), new TemporalExpression(
                new Span(22, 32), "2026-07-10", TemporalExpression.Granularity.DAY)),
            new Annotation<>(new Span(38, 48), new TemporalExpression(
                new Span(38, 48), "2026-07-20", TemporalExpression.Granularity.DAY))));

    final Document dated = new DocumentDateAnnotator().annotate(document);

    final List<Annotation<LocalDate>> dates =
        dated.get(DocumentDateAnnotator.DOCUMENT_DATE);
    Assertions.assertEquals(1, dates.size());
    Assertions.assertEquals(LocalDate.parse("2026-07-10"), dates.get(0).value());
    Assertions.assertEquals(new Span(22, 32), dates.get(0).span());
  }

  @Test
  void testCoarserMentionsNeverElectADate() {
    final Document document = Document.of("the 2026-07 report and 2024-Q3 numbers")
        .with(TemporalAnnotator.TEMPORALS, List.of(
            new Annotation<>(new Span(4, 11), new TemporalExpression(
                new Span(4, 11), "2026-07", TemporalExpression.Granularity.MONTH)),
            new Annotation<>(new Span(23, 30), new TemporalExpression(
                new Span(23, 30), "2024-Q3", TemporalExpression.Granularity.QUARTER))));

    Assertions.assertTrue(new DocumentDateAnnotator().annotate(document)
        .get(DocumentDateAnnotator.DOCUMENT_DATE).isEmpty());
  }

  /**
   * Verifies the election rule on conflicting dates through the real extractor: when a
   * document mentions two different days, the first one in text order wins.
   */
  @Test
  void testFirstOfConflictingDatesWinsThroughThePipeline() {
    final Document document = DocumentAnalyzer.builder()
        .add(new TemporalAnnotator(new CursorTemporalExtractor()))
        .add(new DocumentDateAnnotator())
        .build()
        .analyze("Filed 2026-07-10. Hearing on 2026-09-02.");

    final List<Annotation<LocalDate>> dates =
        document.get(DocumentDateAnnotator.DOCUMENT_DATE);
    Assertions.assertEquals(1, dates.size());
    Assertions.assertEquals(LocalDate.parse("2026-07-10"), dates.get(0).value());
    Assertions.assertEquals(new Span(6, 16), dates.get(0).span());
  }

  /**
   * Verifies that a document without a temporal layer gets an empty date layer, because
   * an absent layer reads as an empty list.
   */
  @Test
  void testMissingTemporalLayerElectsNothing() {
    final Document dated =
        new DocumentDateAnnotator().annotate(Document.of("no dates here"));
    Assertions.assertTrue(dated.get(DocumentDateAnnotator.DOCUMENT_DATE).isEmpty());
  }

  /**
   * Verifies that a day-granularity mention whose value is not an ISO 8601 date, as a
   * third-party extractor may supply, fails loud as {@link IllegalArgumentException}
   * naming the offending value and its span, not as a raw parse exception.
   */
  @Test
  void testNonIsoDayValueFailsLoudWithValueAndSpan() {
    final Document document = Document.of("Filed July 14, 2026.")
        .with(TemporalAnnotator.TEMPORALS, List.of(
            new Annotation<>(new Span(6, 19), new TemporalExpression(
                new Span(6, 19), "July 14, 2026", TemporalExpression.Granularity.DAY))));

    final IllegalArgumentException e = Assertions.assertThrows(
        IllegalArgumentException.class,
        () -> new DocumentDateAnnotator().annotate(document));
    Assertions.assertEquals(
        "not an ISO 8601 day value at [6..19): July 14, 2026", e.getMessage());
  }

  /**
   * Verifies that a value in {@code yyyy-MM-dd} shape that is not a real calendar day,
   * such as a February 30th, fails loud the same way as a value that is not ISO-shaped
   * at all: the javadoc promises rejection of any value that is not an ISO 8601
   * calendar date, not merely of un-ISO-shaped text.
   */
  @Test
  void testIsoShapedImpossibleDayFailsLoudWithValueAndSpan() {
    final Document document = Document.of("Filed 2026-02-30.")
        .with(TemporalAnnotator.TEMPORALS, List.of(
            new Annotation<>(new Span(6, 16), new TemporalExpression(
                new Span(6, 16), "2026-02-30", TemporalExpression.Granularity.DAY))));

    final IllegalArgumentException e = Assertions.assertThrows(
        IllegalArgumentException.class,
        () -> new DocumentDateAnnotator().annotate(document));
    Assertions.assertEquals(
        "not an ISO 8601 day value at [6..16): 2026-02-30", e.getMessage());
  }

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new DocumentDateAnnotator().annotate(null));
  }
}
