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
 * @since 3.0.0
 */
public class RegionAwareMoneyAnnotator implements DocumentAnnotator {

  private final CursorMoneyExtractor defaultExtractor = new CursorMoneyExtractor();
  private final Map<String, CursorMoneyExtractor> byCountry = new ConcurrentHashMap<>();

  /**
   * Initializes the annotator.
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

  /** Resolves and caches the extractor for a winning country. */
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

  /** Picks a deterministic installed locale for a country, if any. */
  private static Optional<Locale> localeOf(String countryCode) {
    Locale best = null;
    for (final Locale locale : Locale.getAvailableLocales()) {
      if (countryCode.equals(locale.getCountry())
          && (best == null
              || locale.toLanguageTag().compareTo(best.toLanguageTag()) < 0)) {
        best = locale;
      }
    }
    return Optional.ofNullable(best);
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
