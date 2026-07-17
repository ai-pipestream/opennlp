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
 * available to every downstream consumer.
 *
 * <p>All qualifying mentions are handed to the {@link Geocoder} in a single call, which
 * lets an implementation use co-occurring mentions to disambiguate. A mention the
 * geocoder cannot resolve simply has no annotation in the layer, and a mention the
 * geocoder ranks against several candidates gets one annotation per candidate, in the
 * geocoder's order, so the candidates of a mention stay ranked best first in the layer.</p>
 *
 * <p>The {@link Geocoder} contract binds every resolution's mention span to one of the
 * spans handed in. That is what lets a consumer match an annotation back to the entity it
 * came from by span, so a geocoder returning a span it was not given is rejected rather
 * than silently producing a layer nothing downstream can align.</p>
 *
 * <p>The annotator holds no per-call state; it is safe to share between threads
 * whenever its geocoder honors the {@link Geocoder} thread-safety contract.</p>
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
   * @throws IllegalArgumentException Thrown if a parameter is {@code null}, or if
   *         {@code locationTypes} is empty or contains a {@code null} or blank entry.
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

  /**
   * Annotates the document with the {@link #LOCATIONS} layer.
   *
   * <p>Entities whose type is in the configured set, compared case-insensitively,
   * become the mention spans of one geocoder call, and every returned resolution
   * becomes one annotation at its mention's span. A document without qualifying
   * mentions gets an empty layer without any geocoder call.</p>
   *
   * @param document The document to annotate. Must not be {@code null}; a document
   *                 without an entity layer is treated as having no entities.
   * @return A new {@link Document} with the {@link #LOCATIONS} layer added. Never
   *         {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, already
   *         carries the {@link #LOCATIONS} layer, or the geocoder returns a resolution
   *         whose mention span is not one of the spans it was given.
   * @throws UncheckedIOException Thrown if the geocoder fails with an I/O error.
   */
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
      final Set<Span> given = new HashSet<>(mentions);
      for (final GeoResolution resolution : resolutions) {
        if (!given.contains(resolution.mention())) {
          throw new IllegalArgumentException("geocoder resolved the mention span "
              + resolution.mention() + ", which is not one of the given mentions");
        }
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
