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

package opennlp.tools.geo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.normalizer.EmojiFlags;

/**
 * Votes on where a document speaks from: every location mention casts a ballot for a
 * country, and the document-scoped layer reports the ranked result as
 * {@link RegionVote} shares without spans.
 *
 * <p>Three kinds of evidence vote. A location entity whose text is an English country
 * name, recognized through JDK locale data, votes for that country at a fixed weight. A
 * location entity resolved by the {@link GeocodeAnnotator} votes for its entry's country
 * weighted by the resolution confidence, so a coherence-aware geocoder makes the ballot
 * sharper. A country flag emoji anywhere in the text, a regional indicator pair naming an
 * assigned ISO 3166-1 code, votes for its country and needs no entity layer support at
 * all; subdivision tag-sequence flags name no country and cast no vote. Mentions with no
 * kind of evidence do not vote; a document without usable evidence gets an empty
 * layer.</p>
 *
 * <p>Country-name evidence takes precedence over a gazetteer resolution for the same
 * mention: an entity spelled exactly like a country is direct evidence for that country,
 * while a gazetteer hit on the same text can be any same-named place, so a mention reading
 * {@code Georgia} votes for the country and not for the US state a gazetteer resolves it
 * to. A resolution is therefore consulted only for mentions that are not country names,
 * and every mention casts at most one vote.</p>
 *
 * <p>When the geocoder ranks several candidates for one mention, the locations layer holds
 * them in the geocoder's order, best first, and the mention votes with that best candidate
 * alone. The order is the geocoder's ranking, which is authoritative: confidence is
 * resolver-defined and not comparable across implementations, so it ranks nothing by
 * itself.</p>
 *
 * <p>The annotator holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class DocumentRegionAnnotator implements DocumentAnnotator {

  /**
   * The document's region ballot: a document-scoped layer with one span-less annotation
   * per candidate country, ordered by descending share, with equal shares ranked by
   * ascending country code so the order is deterministic.
   */
  public static final LayerKey<RegionVote> REGIONS =
      Layers.documentKey("regions", RegionVote.class);

  /**
   * The vote weight of a direct country-name mention, sitting near the top of the
   * geocoder confidence range: naming a country outright counts like one confidently
   * resolved city, without outvoting two independent mentions on its own.
   */
  private static final double COUNTRY_NAME_WEIGHT = 0.95;

  /**
   * The weight of a country flag emoji, deliberately equal to
   * {@link #COUNTRY_NAME_WEIGHT} because writing a flag is as explicit a signal as
   * writing the country's name.
   */
  private static final double FLAG_WEIGHT = 0.95;

  private static final Map<String, String> COUNTRY_NAMES = countryNames();

  private static final Set<String> ISO_COUNTRIES = Set.of(Locale.getISOCountries());

  private final Set<String> locationTypes;

  /**
   * Initializes the annotator for entities typed {@code location}.
   */
  public DocumentRegionAnnotator() {
    this(Set.of("location"));
  }

  /**
   * Initializes the annotator.
   *
   * @param locationTypes The entity type labels treated as locations, matched
   *                      case-insensitively. Must not be {@code null} or empty.
   * @throws IllegalArgumentException Thrown if {@code locationTypes} is {@code null},
   *         empty, or contains a blank entry.
   */
  public DocumentRegionAnnotator(Set<String> locationTypes) {
    if (locationTypes == null || locationTypes.isEmpty()) {
      throw new IllegalArgumentException("locationTypes must not be null or empty");
    }
    final Set<String> lowered = new HashSet<>(locationTypes.size());
    for (final String type : locationTypes) {
      if (type == null || StringUtil.isBlank(type)) {
        throw new IllegalArgumentException("locationTypes must not contain blank entries");
      }
      lowered.add(StringUtil.toLowerCase(type));
    }
    this.locationTypes = Set.copyOf(lowered);
  }

  /**
   * Annotates the document with the {@link #REGIONS} ballot.
   *
   * <p>Every location entity casts at most one vote. A mention spelled like an English
   * country name votes for that country at the country-name weight; otherwise the mention
   * is matched to the locations layer by exact span and votes for its best candidate's
   * country with that candidate's confidence. A mention that is neither, or whose best
   * candidate carries no country code, casts no vote. Country flag emoji vote directly
   * from the text, independently of any entity.</p>
   *
   * <p>The required layers must be present, but they may be empty: a document with a
   * present but empty locations layer is a document nothing geocoded, and its country-name
   * and flag evidence still votes. An absent required layer is a pipeline error rather
   * than an evidence-free document, because a missing {@link GeocodeAnnotator} stage would
   * otherwise drop every geocoded vote silently.</p>
   *
   * @param document The document to annotate. Must not be {@code null} and must carry the
   *                 {@link Layers#ENTITIES} and {@link GeocodeAnnotator#LOCATIONS} layers.
   * @return A new {@link Document} with the {@link #REGIONS} layer added. Never
   *         {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, the entity
   *         layer or the locations layer is absent, or the document already carries the
   *         {@link #REGIONS} layer.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final Set<LayerKey<?>> present = document.layers();
    if (!present.contains(Layers.ENTITIES)) {
      throw new IllegalArgumentException("document lacks the required layer "
          + Layers.ENTITIES);
    }
    if (!present.contains(GeocodeAnnotator.LOCATIONS)) {
      throw new IllegalArgumentException("document lacks the required layer "
          + GeocodeAnnotator.LOCATIONS);
    }
    final CharSequence text = document.text();
    // The layer holds a mention's candidates in the geocoder's ranking, best first, so
    // the first entry for a span is the one that votes.
    final Map<Long, GeoResolution> resolutionsBySpan = new HashMap<>();
    for (final Annotation<GeoResolution> location : document.get(GeocodeAnnotator.LOCATIONS)) {
      resolutionsBySpan.putIfAbsent(spanKey(location.span()), location.value());
    }
    final Map<String, Double> weights = new HashMap<>();
    for (final Annotation<String> entity : document.get(Layers.ENTITIES)) {
      if (!locationTypes.contains(StringUtil.toLowerCase(entity.value()))) {
        continue;
      }
      final Span span = entity.span();
      final String mention = text.subSequence(span.getStart(), span.getEnd()).toString();
      final String named = COUNTRY_NAMES.get(StringUtil.toLowerCase(mention));
      if (named != null) {
        weights.merge(named, COUNTRY_NAME_WEIGHT, Double::sum);
        continue;
      }
      final GeoResolution resolution = resolutionsBySpan.get(spanKey(span));
      if (resolution == null) {
        continue;
      }
      final String countryCode = resolution.entry().countryCode();
      if (countryCode != null) {
        weights.merge(countryCode, resolution.confidence(), Double::sum);
      }
    }
    flagVotes(text, weights);
    return document.with(REGIONS, ballot(weights));
  }

  /**
   * Adds one vote per country flag emoji in the text. Consecutive flags are segmented
   * left to right, so two adjacent flags never form a spurious middle pair. A lone
   * regional indicator with no partner to its right casts no vote, and a pair decoding
   * to a code outside the assigned ISO countries is skipped the same way.
   *
   * @param text The document text.
   * @param weights The running weight sums by country code.
   */
  private static void flagVotes(CharSequence text, Map<String, Double> weights) {
    int i = 0;
    while (i < text.length()) {
      final int first = Character.codePointAt(text, i);
      final int width = Character.charCount(first);
      if (isRegionalIndicator(first) && i + width < text.length()) {
        final int second = Character.codePointAt(text, i + width);
        if (isRegionalIndicator(second)) {
          final int end = i + width + Character.charCount(second);
          final String code = EmojiFlags.isoRegion(text.subSequence(i, end)).orElse(null);
          if (code != null && ISO_COUNTRIES.contains(code)) {
            weights.merge(code, FLAG_WEIGHT, Double::sum);
          }
          i = end;
          continue;
        }
      }
      i += width;
    }
  }

  /**
   * {@return whether {@code codePoint} is a REGIONAL INDICATOR SYMBOL LETTER (A..Z)} The pair
   * segmentation above needs the raw block check, which {@link EmojiFlags} does not expose.
   */
  private static boolean isRegionalIndicator(int codePoint) {
    return codePoint >= 0x1F1E6 && codePoint <= 0x1F1FF;
  }

  /**
   * Turns the weight sums into the ranked ballot: each country's share is its weight
   * over the total, and rows are ordered by descending share with ties broken by
   * ascending country code so the ranking is deterministic. The rows carry no spans,
   * since the ballot describes the document as a whole.
   *
   * <p>A country whose weight sums to zero carries no evidence, which a resolution at
   * confidence {@code 0.0} is entitled to report, so it casts no vote and gets no row.
   * An empty weight map, or one without a positive weight, produces an empty ballot
   * rather than a set of undefined shares.</p>
   *
   * @param weights The weight sums by country code.
   * @param length The document text length, defining the whole-document span.
   * @return The ballot annotations in rank order. Never {@code null}; empty when no
   *         country has evidence.
   */
  private static List<Annotation<RegionVote>> ballot(Map<String, Double> weights) {
    double total = 0.0;
    final List<Map.Entry<String, Double>> ranked = new ArrayList<>(weights.size());
    for (final Map.Entry<String, Double> entry : weights.entrySet()) {
      if (entry.getValue() > 0.0) {
        total += entry.getValue();
        ranked.add(entry);
      }
    }
    ranked.sort(Map.Entry.<String, Double>comparingByValue().reversed()
        .thenComparing(Map.Entry.comparingByKey()));
    final List<Annotation<RegionVote>> votes = new ArrayList<>(ranked.size());
    for (final Map.Entry<String, Double> entry : ranked) {
      votes.add(Annotation.of(new RegionVote(entry.getKey(), entry.getValue() / total)));
    }
    return votes;
  }

  /** Collapses a span to its offsets, so entity and resolution spans match by position. */
  private static long spanKey(Span span) {
    return ((long) span.getStart() << 32) | span.getEnd();
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(Layers.ENTITIES, GeocodeAnnotator.LOCATIONS);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(REGIONS);
  }

  /**
   * Maps lowercased English country display names from JDK locale data to their ISO
   * 3166-1 alpha-2 codes. Codes for which the JDK has no distinct display name are
   * skipped rather than mapped to themselves.
   */
  private static Map<String, String> countryNames() {
    final Map<String, String> names = new HashMap<>();
    for (final String code : Locale.getISOCountries()) {
      final String name = Locale.of("", code).getDisplayCountry(Locale.ENGLISH);
      if (!name.isEmpty() && !name.equals(code)) {
        names.put(StringUtil.toLowerCase(name), code);
      }
    }
    return Map.copyOf(names);
  }
}
