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

import java.io.IOException;
import java.io.UncheckedIOException;
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

/**
 * Derives the countries a document is about from its location entities and provides
 * {@link #REGIONS}, a document-scoped layer of span-less {@link RegionVote} rows ranked
 * by share.
 *
 * <p>Two kinds of evidence vote. A mention whose text is an English country name, for
 * example {@code Australia}, votes for that country directly through JDK locale data,
 * because place gazetteers often carry no country entries. Every other location mention
 * goes through the {@link Geocoder} and votes for its entry's country weighted by the
 * resolution confidence. A mention that resolves to nothing does not vote, and a
 * document without usable evidence gets an empty layer.</p>
 *
 * <p>The annotator holds no per-call state and is as thread-safe as its geocoder.</p>
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
   * The vote weight of a direct country-name mention. It sits near the top of the
   * geocoder confidence range, so one country name counts about as much as one
   * confidently resolved city and never outvotes two independent mentions on its own.
   */
  private static final double COUNTRY_NAME_WEIGHT = 0.95;

  /** The entity type label treated as a location unless the caller names others. */
  private static final String DEFAULT_LOCATION_TYPE = "location";

  private static final Map<String, String> COUNTRY_NAMES = countryNames();

  private final Geocoder geocoder;
  private final Set<String> locationTypes;

  /**
   * Initializes the annotator for entities typed {@value #DEFAULT_LOCATION_TYPE}.
   *
   * @param geocoder The geocoder resolving location mentions. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code geocoder} is {@code null}.
   */
  public DocumentRegionAnnotator(Geocoder geocoder) {
    this(geocoder, Set.of(DEFAULT_LOCATION_TYPE));
  }

  /**
   * Initializes the annotator.
   *
   * @param geocoder The geocoder resolving location mentions. Must not be {@code null}.
   * @param locationTypes The entity type labels treated as locations, matched
   *                      case-insensitively. Must not be {@code null} or empty.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null} or
   *         {@code locationTypes} is empty.
   */
  public DocumentRegionAnnotator(Geocoder geocoder, Set<String> locationTypes) {
    if (geocoder == null) {
      throw new IllegalArgumentException("geocoder must not be null");
    }
    if (locationTypes == null || locationTypes.isEmpty()) {
      throw new IllegalArgumentException("locationTypes must not be null or empty");
    }
    this.geocoder = geocoder;
    final Set<String> lowered = new HashSet<>(locationTypes.size());
    for (final String type : locationTypes) {
      if (type == null || StringUtil.isBlank(type)) {
        throw new IllegalArgumentException("locationTypes must not contain blank entries");
      }
      lowered.add(type.toLowerCase(Locale.ROOT));
    }
    this.locationTypes = Set.copyOf(lowered);
  }

  /**
   * {@inheritDoc}
   *
   * @throws UncheckedIOException Thrown if the {@link Geocoder} fails with an
   *         {@link IOException}.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final CharSequence text = document.text();
    final Map<String, Double> weights = new HashMap<>();
    final List<Span> toGeocode = new ArrayList<>();
    for (final Annotation<String> entity : document.get(Layers.ENTITIES)) {
      if (!locationTypes.contains(entity.value().toLowerCase(Locale.ROOT))) {
        continue;
      }
      final Span span = entity.span();
      final String mention = text.subSequence(span.getStart(), span.getEnd()).toString();
      final String countryCode = COUNTRY_NAMES.get(mention.toLowerCase(Locale.ROOT));
      if (countryCode != null) {
        weights.merge(countryCode, COUNTRY_NAME_WEIGHT, Double::sum);
      } else {
        toGeocode.add(span);
      }
    }
    if (!toGeocode.isEmpty()) {
      final List<GeoResolution> resolutions;
      try {
        resolutions = geocoder.resolve(text, toGeocode);
      } catch (IOException e) {
        throw new UncheckedIOException("Unable to geocode the document's location mentions", e);
      }
      for (final GeoResolution resolution : resolutions) {
        final String countryCode = resolution.entry().countryCode();
        if (countryCode != null) {
          weights.merge(countryCode, resolution.confidence(), Double::sum);
        }
      }
    }
    return document.with(REGIONS, ballot(weights));
  }

  /**
   * Turns the weight sums into a ranked ballot: each country's share is its weight over
   * the weight total, so shares sum to one, and rows are ordered by descending share
   * with ties broken by ascending country code. The rows carry no spans, since the
   * ballot describes the document as a whole. A country whose weight sums to zero
   * carries no evidence and gets no row, so a ballot without positive weights is empty
   * rather than a set of undefined shares.
   *
   * @param weights Maps a country code to the summed weight of its evidence. Must not be
   *                {@code null}.
   * @return The ranked ballot rows. Never {@code null}.
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

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(Layers.ENTITIES);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(REGIONS);
  }

  /**
   * Builds the country-name lookup from JDK locale data. Codes for which the JDK has no
   * distinct display name are skipped rather than mapped to themselves.
   *
   * @return Lowercased English country display names mapped to their ISO 3166-1 alpha-2
   *         codes. Never {@code null}.
   */
  private static Map<String, String> countryNames() {
    final Map<String, String> names = new HashMap<>();
    for (final String code : Locale.getISOCountries()) {
      final String name = Locale.of("", code).getDisplayCountry(Locale.ENGLISH);
      if (!name.isEmpty() && !name.equals(code)) {
        names.put(name.toLowerCase(Locale.ROOT), code);
      }
    }
    return Map.copyOf(names);
  }
}
