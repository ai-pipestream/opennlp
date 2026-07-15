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

/**
 * Converts the money layer into one target currency: reads {@link MoneyAnnotator#MONEY}
 * and provides {@link #CONVERTED_MONEY}, one annotation per convertible mention with
 * the amount restated in the target currency on the mention's original span.
 *
 * <p>The conversion is as-of dated. Mentions without a usable rate are left out of the
 * converted layer and logged at debug level; the original mention stays in the money
 * layer either way, so nothing is lost.</p>
 *
 * @since 3.0.0
 */
public class MoneyConversionAnnotator implements DocumentAnnotator {

  /**
   * Money mentions restated in the target currency; aligned with the money layer by
   * span, omitting mentions without a usable rate.
   */
  public static final LayerKey<MoneyAmount> CONVERTED_MONEY =
      LayerKey.of("money.converted", MoneyAmount.class);

  private static final Logger logger =
      LoggerFactory.getLogger(MoneyConversionAnnotator.class);

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
    if (rates == null || asOf == null) {
      throw new IllegalArgumentException("rates and asOf must not be null");
    }
    if (target == null || target.isBlank()) {
      throw new IllegalArgumentException("target must not be null or blank");
    }
    this.rates = rates;
    this.target = target;
    this.asOf = asOf;
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final List<Annotation<MoneyAmount>> converted = new ArrayList<>();
    for (final Annotation<MoneyAmount> mention : document.get(MoneyAnnotator.MONEY)) {
      final Optional<MoneyAmount> restated = rates.convert(mention.value(), target, asOf);
      if (restated.isPresent()) {
        converted.add(new Annotation<>(mention.span(), restated.get()));
      } else {
        logger.debug("No {} rate as of {} for mention {}", target, asOf, mention.value());
      }
    }
    return document.with(CONVERTED_MONEY, converted);
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(MoneyAnnotator.MONEY);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(CONVERTED_MONEY);
  }
}
