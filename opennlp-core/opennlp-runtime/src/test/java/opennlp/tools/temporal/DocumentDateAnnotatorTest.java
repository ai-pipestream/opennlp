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
import opennlp.tools.util.Span;

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

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new DocumentDateAnnotator().annotate(null));
  }
}
