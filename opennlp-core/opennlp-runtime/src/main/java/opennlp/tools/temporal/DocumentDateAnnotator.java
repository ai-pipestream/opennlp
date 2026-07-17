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
import java.time.format.DateTimeParseException;
import java.util.List;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;

/**
 * Elects the document's reference date from the temporal layer and provides
 * {@link #DOCUMENT_DATE}: the first day-granularity mention in the text wins, following
 * the dateline convention that a document dates itself up front.
 *
 * <p>The annotation covers the winning mention's span, so the choice stays auditable.
 * A document without a day-granularity mention gets an empty layer; coarser mentions
 * such as months and quarters never elect a date silently. Downstream consumers such as
 * currency conversion read this layer to anchor time-dependent lookups.</p>
 *
 * <p>The annotator holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class DocumentDateAnnotator implements DocumentAnnotator {

  /**
   * The document's reference date: at most one annotation, covering the mention that
   * elected it.
   */
  public static final LayerKey<LocalDate> DOCUMENT_DATE =
      Layers.key("document.date", LocalDate.class);

  /**
   * Elects the document date from the first day-granularity temporal mention.
   *
   * @param document The document to annotate. Must not be {@code null}.
   * @return The document with the {@link #DOCUMENT_DATE} layer added. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, or if
   *         the electing day-granularity mention carries a value that is not an ISO 8601
   *         calendar date in the {@code yyyy-MM-dd} form, as a third-party
   *         {@link TemporalExtractor} may supply.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    for (final Annotation<TemporalExpression> mention
        : document.get(TemporalAnnotator.TEMPORALS)) {
      if (mention.value().granularity() == TemporalExpression.Granularity.DAY) {
        final LocalDate date;
        try {
          date = LocalDate.parse(mention.value().value());
        } catch (DateTimeParseException e) {
          throw new IllegalArgumentException("not an ISO 8601 day value at "
              + mention.span() + ": " + mention.value().value(), e);
        }
        return document.with(DOCUMENT_DATE,
            List.of(new Annotation<>(mention.span(), date)));
      }
    }
    return document.with(DOCUMENT_DATE, List.of());
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(TemporalAnnotator.TEMPORALS);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(DOCUMENT_DATE);
  }
}
