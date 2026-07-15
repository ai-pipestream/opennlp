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

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.temporal.TemporalExpression.Granularity;
import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests the recognized calendar shapes, their normalized values, and the reject cases,
 * in particular calendar validation and the month-name words that double as ordinary
 * English.
 */
public class CursorTemporalExtractorTest {

  private final CursorTemporalExtractor extractor = new CursorTemporalExtractor();

  private TemporalExpression single(String text) {
    final List<TemporalExpression> mentions = extractor.extract(text);
    assertEquals(1, mentions.size(), "expected one mention in: " + text);
    return mentions.get(0);
  }

  @ParameterizedTest
  @CsvSource(delimiter = ';', value = {
      "2026-07-14; 2026-07-14; DAY",
      "July 14, 2026; 2026-07-14; DAY",
      "July 14 2026; 2026-07-14; DAY",
      "Jul 14 2026; 2026-07-14; DAY",
      "14 July 2026; 2026-07-14; DAY",
      "14th July 2026; 2026-07-14; DAY",
      "march 3rd, 1999; 1999-03-03; DAY",
      "July 2026; 2026-07; MONTH",
      "SEPTEMBER 2024; 2024-09; MONTH",
      "Q3 2024; 2024-Q3; QUARTER",
      "q1 1999; 1999-Q1; QUARTER"
  })
  void testRecognizedShapesCoverTheFullMention(String text, String value, String granularity) {
    final TemporalExpression mention = single(text);
    assertEquals(new Span(0, text.length()), mention.span(), text);
    assertEquals(value, mention.value(), text);
    assertEquals(Granularity.valueOf(granularity), mention.granularity(), text);
  }

  @Test
  void testMentionInsideSentenceHasExactSpan() {
    final String text = "the release shipped on 2026-07-14 after review";
    final TemporalExpression mention = single(text);
    assertEquals("2026-07-14",
        mention.span().getCoveredText(text).toString());
  }

  @Test
  void testMultipleMentions() {
    final List<TemporalExpression> mentions =
        extractor.extract("from July 2025 to Q2 2026");
    assertEquals(2, mentions.size());
    assertEquals("2025-07", mentions.get(0).value());
    assertEquals("2026-Q2", mentions.get(1).value());
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "2026-13-01",           // no thirteenth month
      "2026-02-30",           // calendar-validated
      "February 30, 2026",    // calendar-validated
      "July 32 2026",         // no thirty-second day
      "it may be 100",        // month word without a calendar context
      "March onwards",        // month word without a year
      "in 2026 alone",        // bare years are out of scope
      "Q5 2024",              // no fifth quarter
      "Q3 24",                // two-digit years are out of scope
      "2026-07",              // ISO month form is out of scope
      "12026-07-14"           // year is part of a longer digit run
  })
  void testRejectedShapes(String text) {
    assertTrue(extractor.extract(text).isEmpty(), text);
  }

  @Test
  void testNullTextThrows() {
    assertThrows(IllegalArgumentException.class, () -> extractor.extract(null));
  }

  @Test
  void testTemporalExpressionValidation() {
    assertThrows(IllegalArgumentException.class,
        () -> new TemporalExpression(null, "2026-07", Granularity.MONTH));
    assertThrows(IllegalArgumentException.class,
        () -> new TemporalExpression(new Span(0, 1), " ", Granularity.MONTH));
    assertThrows(IllegalArgumentException.class,
        () -> new TemporalExpression(new Span(0, 1), "2026-07", null));
  }

  @Test
  void testAnnotatorProvidesTheTemporalLayer() {
    final Document document = DocumentAnalyzer.builder()
        .add(new TemporalAnnotator(extractor))
        .build()
        .analyze("earnings land on July 24, 2026 before Q3 2026 begins");
    final List<Annotation<TemporalExpression>> temporals =
        document.get(TemporalAnnotator.TEMPORALS);
    assertEquals(2, temporals.size());
    assertEquals("2026-07-24", temporals.get(0).value().value());
    assertEquals(Granularity.QUARTER, temporals.get(1).value().granularity());
  }

  @Test
  void testAnnotatorValidation() {
    assertThrows(IllegalArgumentException.class, () -> new TemporalAnnotator(null));
    assertThrows(IllegalArgumentException.class,
        () -> new TemporalAnnotator(extractor).annotate(null));
  }
}
