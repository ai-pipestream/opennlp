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

/**
 * Votes on where a document speaks from: every location mention casts a ballot for a
 * country, and the layer reports the ranked result as {@link RegionVote} shares over
 * the whole document span.
 *
 * <p>Two kinds of evidence vote. A location entity resolved by the
 * {@link GeocodeAnnotator} votes for its entry's country weighted by the resolution
 * confidence, so a coherence-aware geocoder makes the ballot sharper. A location
 * entity the geocoder left unresolved still votes when its text is an English country
 * name, recognized through JDK locale data, which covers gazetteers that carry no
 * country entries. Mentions with neither kind of evidence do not vote; a document
 * without usable evidence gets an empty layer.</p>
 *
 * <p>The annotator holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class DocumentRegionAnnotator implements DocumentAnnotator {

  /**
   * The document's region ballot: one annotation per candidate country over the whole
   * document span, ordered by descending share.
   */
  public static final LayerKey<RegionVote> REGIONS = LayerKey.of("regions", RegionVote.class);

  /** The weight of a direct country-name mention, matching a confident resolution. */
  private static final double COUNTRY_NAME_WEIGHT = 0.95;

  private static final Map<String, String> COUNTRY_NAMES = countryNames();

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
      if (type == null || type.isBlank()) {
        throw new IllegalArgumentException("locationTypes must not contain blank entries");
      }
      lowered.add(type.toLowerCase(Locale.ROOT));
    }
    this.locationTypes = Set.copyOf(lowered);
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final CharSequence text = document.text();
    final Map<Long, GeoResolution> resolutionsBySpan = new HashMap<>();
    for (final Annotation<GeoResolution> location : document.get(GeocodeAnnotator.LOCATIONS)) {
      resolutionsBySpan.put(spanKey(location.span()), location.value());
    }
    final Map<String, Double> weights = new HashMap<>();
    for (final Annotation<String> entity : document.get(Layers.ENTITIES)) {
      if (!locationTypes.contains(entity.value().toLowerCase(Locale.ROOT))) {
        continue;
      }
      final Span span = entity.span();
      final GeoResolution resolution = resolutionsBySpan.get(spanKey(span));
      if (resolution != null) {
        final String countryCode = resolution.entry().countryCode();
        if (countryCode != null) {
          weights.merge(countryCode, resolution.confidence(), Double::sum);
        }
        continue;
      }
      final String mention = text.subSequence(span.getStart(), span.getEnd()).toString();
      final String countryCode = COUNTRY_NAMES.get(mention.toLowerCase(Locale.ROOT));
      if (countryCode != null) {
        weights.merge(countryCode, COUNTRY_NAME_WEIGHT, Double::sum);
      }
    }
    return document.with(REGIONS, ballot(weights, text.length()));
  }

  /** Turns the weight sums into a ranked ballot of shares over the whole-document span. */
  private static List<Annotation<RegionVote>> ballot(Map<String, Double> weights, int length) {
    double total = 0.0;
    for (final double weight : weights.values()) {
      total += weight;
    }
    final List<Map.Entry<String, Double>> ranked = new ArrayList<>(weights.entrySet());
    ranked.sort(Map.Entry.<String, Double>comparingByValue().reversed()
        .thenComparing(Map.Entry.comparingByKey()));
    final Span documentSpan = new Span(0, length);
    final List<Annotation<RegionVote>> votes = new ArrayList<>(ranked.size());
    for (final Map.Entry<String, Double> entry : ranked) {
      votes.add(new Annotation<>(documentSpan,
          new RegionVote(entry.getKey(), entry.getValue() / total)));
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

  /** English country display names from JDK locale data, lowercased, to alpha-2 codes. */
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
