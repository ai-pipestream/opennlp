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
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;

/**
 * Geocodes the location entities of a document: reads {@link Layers#ENTITIES} and
 * provides {@link #LOCATIONS}, one annotation per resolved mention carrying its full
 * {@link GeoResolution}, so the gazetteer entry, coordinates, and confidence stay
 * available to every downstream consumer instead of being recomputed or discarded.
 *
 * <p>Mentions the geocoder cannot resolve simply have no annotation in the layer.</p>
 *
 * @since 3.0.0
 */
public class GeocodeAnnotator implements DocumentAnnotator {

  /**
   * Resolved location mentions; each annotation covers one mention and carries its
   * {@link GeoResolution}.
   */
  public static final LayerKey<GeoResolution> LOCATIONS =
      LayerKey.of("locations", GeoResolution.class);

  private final Geocoder geocoder;
  private final Set<String> locationTypes;

  /**
   * Initializes the annotator for entities typed {@code location}.
   *
   * @param geocoder The geocoder resolving location mentions. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code geocoder} is {@code null}.
   */
  public GeocodeAnnotator(Geocoder geocoder) {
    this(geocoder, Set.of("location"));
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
  public GeocodeAnnotator(Geocoder geocoder, Set<String> locationTypes) {
    if (geocoder == null) {
      throw new IllegalArgumentException("geocoder must not be null");
    }
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
    this.geocoder = geocoder;
    this.locationTypes = Set.copyOf(lowered);
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final List<Span> mentions = new ArrayList<>();
    for (final Annotation<String> entity : document.get(Layers.ENTITIES)) {
      if (locationTypes.contains(entity.value().toLowerCase(Locale.ROOT))) {
        mentions.add(entity.span());
      }
    }
    final List<Annotation<GeoResolution>> resolved = new ArrayList<>();
    if (!mentions.isEmpty()) {
      final List<GeoResolution> resolutions;
      try {
        resolutions = geocoder.resolve(document.text(), mentions);
      } catch (IOException e) {
        throw new UncheckedIOException("geocoding failed", e);
      }
      for (final GeoResolution resolution : resolutions) {
        resolved.add(new Annotation<>(resolution.mention(), resolution));
      }
    }
    return document.with(LOCATIONS, resolved);
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(Layers.ENTITIES);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(LOCATIONS);
  }
}
