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

import java.util.ArrayList;
import java.util.Currency;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.geo.DocumentRegionAnnotator;
import opennlp.tools.geo.RegionVote;

/**
 * Money extraction that resolves ambiguous currency symbols per document: reads the
 * region ballot of {@link DocumentRegionAnnotator}, picks the symbol table for the
 * winning country at annotation time, and provides {@link MoneyAnnotator#MONEY}, so in
 * a document that speaks from Australia {@code $} is identified as {@code AUD}.
 *
 * <p>This exists because a pipeline is assembled before any document is seen, while
 * the right symbol table is per-document evidence. Extractors are built through
 * {@link CursorMoneyExtractor#forRegion(Locale)} and cached per country. A document
 * with an empty ballot, or a winning country without a suitable locale or single
 * currency-sign symbol, falls back to the default table.</p>
 *
 * <p>The annotator is safe to share between threads: the per-country cache is a
 * concurrent map and the extractors themselves hold no per-call state.</p>
 *
 * @since 3.0.0
 */
public class RegionAwareMoneyAnnotator implements DocumentAnnotator {

  private final CursorMoneyExtractor defaultExtractor = new CursorMoneyExtractor();
  private final Map<String, CursorMoneyExtractor> byCountry = new ConcurrentHashMap<>();

  /**
   * Initializes the annotator. There is nothing to configure: the symbol table for
   * each document is derived from that document's region ballot at annotation time.
   */
  public RegionAwareMoneyAnnotator() {
    // extractors are derived per document from the region ballot
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    final CursorMoneyExtractor extractor = ballot.isEmpty()
        ? defaultExtractor
        : extractorFor(ballot.get(0).value().countryCode());
    final List<Annotation<MoneyAmount>> mentions = new ArrayList<>();
    for (final MoneyAmount amount : extractor.extract(document.text())) {
      mentions.add(new Annotation<>(amount.span(), amount));
    }
    return document.with(MoneyAnnotator.MONEY, mentions);
  }

  /**
   * Resolves and caches the extractor for a winning country. Fallback outcomes are
   * cached too: a country with no locale whose currency symbol is a single currency
   * sign maps to the default extractor, so the locale scan runs at most once per
   * country.
   */
  private CursorMoneyExtractor extractorFor(String countryCode) {
    return byCountry.computeIfAbsent(countryCode, code -> {
      final Optional<Locale> locale = localeOf(code);
      if (locale.isEmpty()) {
        return defaultExtractor;
      }
      try {
        return CursorMoneyExtractor.forRegion(locale.get());
      } catch (IllegalArgumentException e) {
        return defaultExtractor;
      }
    });
  }

  /**
   * Picks a locale through which a country's currency symbol can be resolved, that is
   * one whose currency is written with a single currency-sign code point in that locale.
   * English is tried first, as {@code en} plus the country, because a country's
   * minority-language locales often write the currency with a disambiguating prefix, for
   * example the Canadian dollar as {@code CA$} rather than {@code $}. If English does not
   * yield a usable symbol, the installed locales for the country are scanned and the
   * usable candidate with the lexicographically smallest language tag wins, so the choice
   * never depends on the order in which the runtime lists its locales. If no candidate is
   * usable, as for a country whose currency is conventionally spelled with letters, the
   * result is empty and the caller keeps the default symbol table.
   *
   * @param countryCode The ISO 3166-1 alpha-2 country code. Must not be {@code null}.
   * @return A locale for the country whose currency symbol is a single currency sign, or
   *         an empty {@link Optional} if the country has none. Never {@code null}.
   */
  private static Optional<Locale> localeOf(String countryCode) {
    final Locale english = Locale.of("en", countryCode);
    if (hasResolvableSymbol(english)) {
      return Optional.of(english);
    }
    Locale best = null;
    for (final Locale locale : Locale.getAvailableLocales()) {
      if (countryCode.equals(locale.getCountry())
          && hasResolvableSymbol(locale)
          && (best == null
              || locale.toLanguageTag().compareTo(best.toLanguageTag()) < 0)) {
        best = locale;
      }
    }
    return Optional.ofNullable(best);
  }

  /**
   * Reports whether a locale's currency is written with a single currency-sign code
   * point there, which is what makes a symbol table override possible; a locale with no
   * currency at all, such as one naming an unassigned country code, is not resolvable.
   *
   * @param locale The locale to test. Must not be {@code null}.
   * @return {@code true} if the locale has a currency written as one currency sign.
   */
  private static boolean hasResolvableSymbol(Locale locale) {
    final Currency currency;
    try {
      currency = Currency.getInstance(locale);
    } catch (IllegalArgumentException e) {
      return false;
    }
    if (currency == null) {
      return false;
    }
    final String symbol = currency.getSymbol(locale);
    return symbol.codePointCount(0, symbol.length()) == 1
        && Character.getType(symbol.codePointAt(0)) == Character.CURRENCY_SYMBOL;
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(DocumentRegionAnnotator.REGIONS);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(MoneyAnnotator.MONEY);
  }
}
