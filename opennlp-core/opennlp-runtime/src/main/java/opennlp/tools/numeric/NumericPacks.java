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

package opennlp.tools.numeric;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.extraction.NumberNotation;
import opennlp.tools.money.CursorMoneyExtractor;
import opennlp.tools.money.MoneyAnnotator;
import opennlp.tools.money.MoneyConversionAnnotator;
import opennlp.tools.quantity.CursorQuantityExtractor;
import opennlp.tools.quantity.QuantityAnnotator;
import opennlp.tools.temporal.CursorTemporalExtractor;
import opennlp.tools.temporal.DocumentDateAnnotator;
import opennlp.tools.temporal.TemporalAnnotator;

/**
 * Ready-made numeric pipelines, one per kind of number a caller usually looks for: money
 * amounts, measured quantities, calendar mentions, and all of them at once.
 *
 * <p>A pack saves the caller from naming individual annotators and, more importantly, from
 * ordering them wrongly. The order matters in one place that is easy to miss: the temporal
 * annotator must run before {@link DocumentDateAnnotator}, and it is the temporal annotator
 * that elects the dateline resolving relative expressions such as {@code yesterday}. A pack
 * wires that once and correctly.</p>
 *
 * <p>The regional variants read a document the way its region writes numbers: an amount in
 * a German document groups digits with dots, and a dollar sign in an Australian one denotes
 * Australian dollars. Both decisions come from JDK locale data through
 * {@link CursorMoneyExtractor#forRegion(Locale)} and
 * {@link NumberNotation#forLocale(Locale)}.</p>
 *
 * <p>Every pack returns a new analyzer built from stateless extractors, so a caller may keep
 * one in a static field and share it between threads. A pipeline that needs more than a pack
 * offers, currency conversion through {@link MoneyConversionAnnotator} for instance, starts
 * from {@link #annotators(Locale)} and adds to it.</p>
 *
 * @since 3.0.0
 */
public final class NumericPacks {

  private NumericPacks() {
    // This class holds static factories only and is never instantiated.
  }

  /**
   * Money amounts, read in {@link NumberNotation#LATIN_US} with the default symbol table.
   *
   * @return A new analyzer providing {@link MoneyAnnotator#MONEY}. Never {@code null}.
   */
  public static DocumentAnalyzer money() {
    return DocumentAnalyzer.builder()
        .add(new MoneyAnnotator(new CursorMoneyExtractor()))
        .build();
  }

  /**
   * Money amounts, read the way a region writes and denominates them.
   *
   * @param region A locale with a country component. Must not be {@code null} and must
   *               name a region with a currency.
   * @return A new analyzer providing {@link MoneyAnnotator#MONEY}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code region} is {@code null} or has no
   *         currency.
   */
  public static DocumentAnalyzer money(Locale region) {
    return DocumentAnalyzer.builder()
        .add(new MoneyAnnotator(CursorMoneyExtractor.forRegion(region)))
        .build();
  }

  /**
   * Measured quantities and percentages, read in {@link NumberNotation#LATIN_US} with the
   * default unit set.
   *
   * @return A new analyzer providing {@link QuantityAnnotator#QUANTITIES}. Never
   *         {@code null}.
   */
  public static DocumentAnalyzer quantity() {
    return DocumentAnalyzer.builder()
        .add(new QuantityAnnotator(new CursorQuantityExtractor()))
        .build();
  }

  /**
   * Measured quantities and percentages, read in the number notation of a region.
   *
   * @param region The locale whose notation the document follows. Must not be
   *               {@code null}.
   * @return A new analyzer providing {@link QuantityAnnotator#QUANTITIES}. Never
   *         {@code null}.
   * @throws IllegalArgumentException Thrown if {@code region} is {@code null}.
   */
  public static DocumentAnalyzer quantity(Locale region) {
    return DocumentAnalyzer.builder()
        .add(new QuantityAnnotator(
            new CursorQuantityExtractor(NumberNotation.forLocale(region))))
        .build();
  }

  /**
   * Calendar mentions and the document date they elect: a dateline in the text supplies
   * the reference date, so relative expressions behind it are resolved.
   *
   * @return A new analyzer providing {@link TemporalAnnotator#TEMPORALS} and
   *         {@link DocumentDateAnnotator#DOCUMENT_DATE}. Never {@code null}.
   */
  public static DocumentAnalyzer temporal() {
    return DocumentAnalyzer.builder()
        .add(new TemporalAnnotator(new CursorTemporalExtractor()))
        .add(new DocumentDateAnnotator())
        .build();
  }

  /**
   * Calendar mentions and an absolute document date, with the reference for relative
   * expressions fixed by the caller. A fixed reference is not synthesized as a text span,
   * so the document-date layer stays empty unless the text contains an absolute day.
   *
   * @param reference The date relative expressions resolve against. Must not be
   *                  {@code null}.
   * @return A new analyzer providing {@link TemporalAnnotator#TEMPORALS} and
   *         {@link DocumentDateAnnotator#DOCUMENT_DATE}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code reference} is {@code null}.
   */
  public static DocumentAnalyzer temporal(LocalDate reference) {
    return DocumentAnalyzer.builder()
        .add(new TemporalAnnotator(new CursorTemporalExtractor(), reference))
        .add(new DocumentDateAnnotator())
        .build();
  }

  /**
   * Every numeric layer at once: calendar mentions, the elected document date, money
   * amounts, and measured quantities.
   *
   * @return A new analyzer providing all four numeric layers. Never {@code null}.
   */
  public static DocumentAnalyzer fullPipeline() {
    return analyzer(annotators());
  }

  /**
   * Every numeric layer at once, read the way a region writes and denominates numbers.
   *
   * @param region A locale with a country component. Must not be {@code null} and must
   *               name a region with a currency.
   * @return A new analyzer providing all four numeric layers. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code region} is {@code null} or has no
   *         currency.
   */
  public static DocumentAnalyzer fullPipeline(Locale region) {
    return analyzer(annotators(region));
  }

  /**
   * The annotators of {@link #fullPipeline()} in execution order, for callers that add
   * further annotators of their own: appending a {@link MoneyConversionAnnotator} to this
   * list yields a pipeline that also restates every amount in one currency as of the
   * document date.
   *
   * @return A new, modifiable list of annotators in execution order. Never {@code null}.
   */
  public static List<DocumentAnnotator> annotators() {
    return pipeline(new CursorMoneyExtractor(), new CursorQuantityExtractor());
  }

  /**
   * The annotators of {@link #fullPipeline(Locale)} in execution order, for callers that
   * add further annotators of their own.
   *
   * @param region A locale with a country component. Must not be {@code null} and must
   *               name a region with a currency.
   * @return A new, modifiable list of annotators in execution order. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code region} is {@code null} or has no
   *         currency.
   */
  public static List<DocumentAnnotator> annotators(Locale region) {
    if (region == null) {
      throw new IllegalArgumentException("region must not be null");
    }
    return pipeline(CursorMoneyExtractor.forRegion(region),
        new CursorQuantityExtractor(NumberNotation.forLocale(region)));
  }

  /**
   * Assembles the full pipeline around the extractors the caller's region decided on.
   *
   * @param money The money extractor to use. Must not be {@code null}.
   * @param quantity The quantity extractor to use. Must not be {@code null}.
   * @return The annotators in execution order. Never {@code null}.
   */
  private static List<DocumentAnnotator> pipeline(CursorMoneyExtractor money,
      CursorQuantityExtractor quantity) {
    return new ArrayList<>(List.of(
        new TemporalAnnotator(new CursorTemporalExtractor()),
        new DocumentDateAnnotator(),
        new MoneyAnnotator(money),
        new QuantityAnnotator(quantity)));
  }

  /**
   * Builds an analyzer from annotators in execution order.
   *
   * @param annotators The annotators to run. Must not be {@code null} or empty.
   * @return The analyzer. Never {@code null}.
   */
  private static DocumentAnalyzer analyzer(List<DocumentAnnotator> annotators) {
    final DocumentAnalyzer.Builder builder = DocumentAnalyzer.builder();
    for (final DocumentAnnotator annotator : annotators) {
      builder.add(annotator);
    }
    return builder.build();
  }
}
