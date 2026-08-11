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
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;

/**
 * Adapts a {@link TemporalExtractor} to the document pipeline: scans the document text
 * and provides {@link #TEMPORALS}, one annotation per mention carrying its
 * {@link TemporalExpression}.
 *
 * <p>Relative expressions such as {@code yesterday} need a reference date, and this
 * annotator finds one in the document itself. The text is scanned twice: the first pass
 * reports absolute mentions only, and if one of them is a day-granularity mention, it
 * dates the document and a second pass resolves the relative expressions against it
 * through {@link TemporalExtractor#extract(CharSequence, LocalDate)}. The mentions of
 * the second pass are the ones reported, so a dateline makes every relative expression
 * behind it resolvable. The electing mention is the first day-granularity mention in
 * text order, the same dateline rule {@link DocumentDateAnnotator} applies, so the two
 * annotators never disagree about the date a document carries.</p>
 *
 * <p>A document that dates itself nowhere keeps the absolute-only mentions: relative
 * expressions stay unreported rather than being guessed against the wall clock, since a
 * document read a year after it was written would otherwise resolve them wrongly and
 * silently. A caller who knows the date from metadata rather than from the text supplies
 * it through {@link #TemporalAnnotator(TemporalExtractor, LocalDate)}, which skips the
 * election and wins over any dateline.</p>
 *
 * <p>The extractor works on the raw text, so this annotator requires no other layer and
 * can run anywhere in a pipeline.</p>
 *
 * @since 3.0.0
 */
public class TemporalAnnotator implements DocumentAnnotator {

  /**
   * Temporal mentions; each annotation covers one mention and carries its normalized
   * {@link TemporalExpression}.
   */
  public static final LayerKey<TemporalExpression> TEMPORALS =
      Layers.key("temporals", TemporalExpression.class);

  private final TemporalExtractor extractor;

  /** The caller-fixed reference date, or {@code null} to elect one per document. */
  private final LocalDate reference;

  /**
   * Initializes the adapter, electing the reference date from each document's own
   * dateline.
   *
   * @param extractor The extractor to delegate to. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code extractor} is {@code null}.
   */
  public TemporalAnnotator(TemporalExtractor extractor) {
    this.extractor = requireExtractor(extractor);
    this.reference = null;
  }

  /**
   * Initializes the adapter with a fixed reference date, for documents whose date is
   * known from metadata rather than from the text.
   *
   * @param extractor The extractor to delegate to. Must not be {@code null}.
   * @param reference The date relative expressions resolve against, used for every
   *                  document and in preference to any dateline in it. Must not be
   *                  {@code null}.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null}.
   */
  public TemporalAnnotator(TemporalExtractor extractor, LocalDate reference) {
    this.extractor = requireExtractor(extractor);
    if (reference == null) {
      throw new IllegalArgumentException("reference must not be null");
    }
    this.reference = reference;
  }

  /**
   * Scans the document text and adds the {@link #TEMPORALS} layer.
   *
   * <p>No other layer is read, so a document without any layer yields the temporal layer
   * present and, if the text holds no calendar mention, empty. Relative expressions are
   * resolved against the fixed reference date, or against the document's own dateline
   * when none was fixed; a document with neither reports its absolute mentions only.</p>
   *
   * @param document The document to annotate. Must not be {@code null}.
   * @return A new {@link Document} with the {@link #TEMPORALS} layer added. Never
   *         {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final List<Annotation<TemporalExpression>> mentions = new ArrayList<>();
    for (final TemporalExpression expression : resolved(document.text())) {
      mentions.add(new Annotation<>(expression.span(), expression));
    }
    return document.with(TEMPORALS, mentions);
  }

  /**
   * Extracts the mentions of a text with the reference date the document supplies, if
   * any.
   *
   * @param text The document text. Must not be {@code null}.
   * @return The mentions in text order. Never {@code null}.
   */
  private List<TemporalExpression> resolved(CharSequence text) {
    if (reference != null) {
      return extractor.extract(text, reference);
    }
    final List<TemporalExpression> absolute = extractor.extract(text);
    final LocalDate dateline = dateline(absolute);
    return dateline == null ? absolute : extractor.extract(text, dateline);
  }

  /**
   * Elects the document's date from its absolute mentions: the first day-granularity
   * mention wins, following the dateline convention that a document dates itself up
   * front.
   *
   * <p>A day-granularity value that is not an ISO 8601 calendar date, as a third-party
   * {@link TemporalExtractor} may supply, elects nothing; the mentions then stay the
   * absolute ones. {@link DocumentDateAnnotator} reports such a value as an error when
   * it reaches the same mention, so the pipeline still fails loud rather than silently
   * dating the document wrongly.</p>
   *
   * @param mentions The absolute mentions in text order. Must not be {@code null}.
   * @return The elected date, or {@code null} when the text holds no usable
   *         day-granularity mention.
   */
  private LocalDate dateline(List<TemporalExpression> mentions) {
    for (final TemporalExpression mention : mentions) {
      if (mention.granularity() == TemporalExpression.Granularity.DAY) {
        try {
          return LocalDate.parse(mention.value());
        } catch (DateTimeParseException e) {
          return null;
        }
      }
    }
    return null;
  }

  /**
   * Validates the extractor a constructor was given.
   *
   * @param extractor The extractor to validate.
   * @return {@code extractor}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code extractor} is {@code null}.
   */
  private static TemporalExtractor requireExtractor(TemporalExtractor extractor) {
    if (extractor == null) {
      throw new IllegalArgumentException("extractor must not be null");
    }
    return extractor;
  }

  /** {@inheritDoc} */
  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(TEMPORALS);
  }
}
