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

package opennlp.tools.money;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.temporal.DocumentDateAnnotator;

/**
 * Converts the money layer into one target currency: reads {@link MoneyAnnotator#MONEY}
 * and provides {@link #CONVERTED_MONEY}, one annotation per convertible mention with
 * the amount restated in the target currency on the mention's original span.
 *
 * <p>The conversion is as-of dated, from either a fixed date or the document's own
 * reference date. In document-dated mode the annotator reads
 * {@link DocumentDateAnnotator#DOCUMENT_DATE}, so a dateline in the text anchors its
 * conversions; a document without an elected date converts nothing. Mentions without a
 * usable rate are left out of the converted layer and logged at debug level; the
 * original mention stays in the money layer either way, so nothing is lost.</p>
 *
 * @since 3.0.0
 */
public class MoneyConversionAnnotator implements DocumentAnnotator {

  /**
   * Money mentions restated in the target currency; aligned with the money layer by
   * span, omitting mentions without a usable rate.
   */
  public static final LayerKey<MoneyAmount> CONVERTED_MONEY =
      Layers.key("money.converted", MoneyAmount.class);

  private static final Logger logger =
      LoggerFactory.getLogger(MoneyConversionAnnotator.class);

  private static final String MISSING_LAYER = "document lacks the required layer ";

  private final FxRates rates;
  private final String target;
  private final LocalDate asOf;

  /**
   * Initializes the annotator.
   *
   * @param rates The rate provider. Must not be {@code null}.
   * @param target The ISO 4217 code of the target currency. Must not be {@code null} or
   *               blank.
   * @param asOf The date the conversions should hold for. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if any parameter is {@code null} or
   *         {@code target} is blank.
   */
  public MoneyConversionAnnotator(FxRates rates, String target, LocalDate asOf) {
    checkProviderAndTarget(rates, target);
    if (asOf == null) {
      throw new IllegalArgumentException("asOf must not be null");
    }
    this.rates = rates;
    this.target = target;
    this.asOf = asOf;
  }

  /**
   * Initializes the annotator in document-dated mode: each document's conversions are
   * anchored on its {@link DocumentDateAnnotator#DOCUMENT_DATE} layer.
   *
   * @param rates The rate provider. Must not be {@code null}.
   * @param target The ISO 4217 code of the target currency. Must not be {@code null} or
   *               blank.
   * @throws IllegalArgumentException Thrown if {@code rates} is {@code null} or
   *         {@code target} is {@code null} or blank.
   */
  public MoneyConversionAnnotator(FxRates rates, String target) {
    checkProviderAndTarget(rates, target);
    this.rates = rates;
    this.target = target;
    this.asOf = null;
  }

  /**
   * Validates the arguments both constructors share.
   *
   * @param rates The rate provider.
   * @param target The ISO 4217 code of the target currency.
   * @throws IllegalArgumentException Thrown if {@code rates} is {@code null} or
   *         {@code target} is {@code null} or blank.
   */
  private static void checkProviderAndTarget(FxRates rates, String target) {
    if (rates == null) {
      throw new IllegalArgumentException("rates must not be null");
    }
    if (target == null || target.isBlank()) {
      throw new IllegalArgumentException("target must not be null or blank");
    }
  }

  /**
   * Restates the money layer in the target currency and adds the
   * {@link #CONVERTED_MONEY} layer.
   *
   * <p>The required layers must be present, but they may be empty. A mention without a
   * usable rate is left out of the converted layer, so the converted layer is a subset
   * of the money layer, aligned with it by span.</p>
   *
   * @param document The document to annotate. Must not be {@code null} and must carry
   *                 the {@link MoneyAnnotator#MONEY} layer, plus the
   *                 {@link DocumentDateAnnotator#DOCUMENT_DATE} layer in document-dated
   *                 mode.
   * @return A new {@link Document} with the {@link #CONVERTED_MONEY} layer added. Never
   *         {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null} or a
   *         required layer is absent.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    if (!document.layers().contains(MoneyAnnotator.MONEY)) {
      throw new IllegalArgumentException(MISSING_LAYER + MoneyAnnotator.MONEY);
    }
    if (asOf == null
        && !document.layers().contains(DocumentDateAnnotator.DOCUMENT_DATE)) {
      throw new IllegalArgumentException(MISSING_LAYER
          + DocumentDateAnnotator.DOCUMENT_DATE);
    }
    final LocalDate date = asOf != null ? asOf : documentDate(document);
    final List<Annotation<MoneyAmount>> converted = new ArrayList<>();
    if (date == null) {
      logger.debug("No document date elected; converting nothing");
      return document.with(CONVERTED_MONEY, converted);
    }
    for (final Annotation<MoneyAmount> mention : document.get(MoneyAnnotator.MONEY)) {
      final Optional<MoneyAmount> restated = rates.convert(mention.value(), target, date);
      if (restated.isPresent()) {
        converted.add(new Annotation<>(mention.span(), restated.get()));
      } else {
        logger.debug("No {} rate as of {} for mention {}", target, date, mention.value());
      }
    }
    return document.with(CONVERTED_MONEY, converted);
  }

  /**
   * Reads the document's elected reference date.
   *
   * @param document The document being annotated. Must not be {@code null}.
   * @return The date, or {@code null} when the layer is empty.
   */
  private LocalDate documentDate(Document document) {
    final List<Annotation<LocalDate>> dates =
        document.get(DocumentDateAnnotator.DOCUMENT_DATE);
    return dates.isEmpty() ? null : dates.get(0).value();
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return asOf != null
        ? Set.of(MoneyAnnotator.MONEY)
        : Set.of(MoneyAnnotator.MONEY, DocumentDateAnnotator.DOCUMENT_DATE);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(CONVERTED_MONEY);
  }
}
