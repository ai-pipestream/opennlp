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
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests the {@link GeocodeAnnotator}: which entities become geocoder mentions, how the
 * geocoder's resolutions turn into the locations layer, and how that layer feeds the
 * region ballot downstream.
 */
public class GeocodeAnnotatorTest {

  /**
   * Builds an annotator that provides the entity layer with one {@code location} entity
   * per given name, spanning the name's first occurrence in the document text.
   *
   * @param names The entity surface forms; every name must occur in the text.
   * @return The entity-layer annotator. Never {@code null}.
   */
  private static DocumentAnnotator locationEntities(String... names) {
    return new DocumentAnnotator() {

      @Override
      public Document annotate(Document document) {
        final String text = document.text().toString();
        final List<Annotation<String>> entities = new ArrayList<>();
        for (final String name : names) {
          final int start = text.indexOf(name);
          if (start < 0) {
            throw new IllegalStateException("Fixture name not in the text: " + name);
          }
          entities.add(
              new Annotation<>(new Span(start, start + name.length()), "location"));
        }
        return document.with(Layers.ENTITIES, entities);
      }

      @Override
      public Set<LayerKey<?>> provides() {
        return Set.of(Layers.ENTITIES);
      }
    };
  }

  /**
   * Builds a geocoder that resolves mentions by a fixed name-to-country table with a
   * fixed confidence and omits every name the table does not know.
   *
   * @param countryByName The resolvable surface forms mapped to their country codes.
   * @param confidence The confidence given to every resolution.
   * @return The table-backed geocoder. Never {@code null}.
   */
  private static Geocoder tableGeocoder(Map<String, String> countryByName,
                                        double confidence) {
    return (text, mentions) -> {
      final List<GeoResolution> resolutions = new ArrayList<>();
      for (final Span mention : mentions) {
        final String name =
            text.subSequence(mention.getStart(), mention.getEnd()).toString();
        final String country = countryByName.get(name);
        if (country != null) {
          resolutions.add(new GeoResolution(mention, entry(name, country), confidence));
        }
      }
      return resolutions;
    };
  }

  /**
   * Builds a geocoder that fails the test when it is called, for documents whose
   * entities must never reach the geocoder.
   *
   * @return The failing geocoder. Never {@code null}.
   */
  private static Geocoder unreachableGeocoder() {
    return (text, mentions) -> {
      throw new IllegalStateException("The geocoder must not be called here");
    };
  }

  /**
   * Builds a minimal gazetteer entry whose record id equals its name.
   *
   * @param name The place name.
   * @param country The ISO 3166-1 alpha-2 country code.
   * @return The entry. Never {@code null}.
   */
  private static GazetteerEntry entry(String name, String country) {
    return entry(name, name, country);
  }

  /**
   * Builds a minimal gazetteer entry, with a distinct record id for same-name places.
   *
   * @param recordId The source-scoped record id.
   * @param name The place name.
   * @param country The ISO 3166-1 alpha-2 country code.
   * @return The entry. Never {@code null}.
   */
  private static GazetteerEntry entry(String recordId, String name, String country) {
    return new GazetteerEntry("test", recordId, name, List.of(), new GeoPoint(0.0, 0.0),
        country, List.of(), 1000, GazetteerEntry.FEATURE_CLASS_CITY, Map.of());
  }

  @Test
  void testUsageExamplePipelineFromEntitiesToRegionBallot() {
    // Two location entities go in; the geocoder knows Berlin and not Atlantis, so the
    // locations layer keeps exactly the one resolvable mention with its resolution, and
    // the region ballot downstream is built from that same layer.
    final Geocoder geocoder = tableGeocoder(Map.of("Berlin", "DE"), 0.9);
    final String text = "flights from Berlin to Atlantis this week";

    final Document document = DocumentAnalyzer.builder()
        .add(locationEntities("Berlin", "Atlantis"))
        .add(new GeocodeAnnotator(geocoder))
        .add(new DocumentRegionAnnotator())
        .build()
        .analyze(text);

    final List<Annotation<GeoResolution>> locations =
        document.get(GeocodeAnnotator.LOCATIONS);
    assertEquals(1, locations.size());
    assertEquals(new Span(13, 19), locations.get(0).span());
    assertEquals("Berlin", locations.get(0).span().getCoveredText(text).toString());
    assertEquals("Berlin", locations.get(0).value().entry().name());
    assertEquals("DE", locations.get(0).value().entry().countryCode());
    assertEquals(0.9, locations.get(0).value().confidence(), 1e-9);

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(1, ballot.size());
    assertEquals("DE", ballot.get(0).value().countryCode());
    assertEquals(1.0, ballot.get(0).value().share(), 1e-9);
    assertEquals(new Span(0, text.length()), ballot.get(0).span());
  }

  @Test
  void testCustomLocationTypesMatchCaseInsensitively() {
    // The set is configured mixed-case and the entities are typed upper- and
    // lower-case; every case combination of the same label qualifies.
    final Geocoder geocoder = tableGeocoder(Map.of("Paris", "FR", "Lyon", "FR"), 0.7);
    final String text = "from Paris to Lyon";
    final Document input = Document.of(text).with(Layers.ENTITIES, List.of(
        new Annotation<>(new Span(5, 10), "PLACE"),
        new Annotation<>(new Span(14, 18), "place")));

    final List<Annotation<GeoResolution>> locations =
        new GeocodeAnnotator(geocoder, Set.of("Place")).annotate(input)
            .get(GeocodeAnnotator.LOCATIONS);
    assertEquals(2, locations.size());
    assertEquals(new Span(5, 10), locations.get(0).span());
    assertEquals("Paris", locations.get(0).value().entry().name());
    assertEquals(new Span(14, 18), locations.get(1).span());
    assertEquals("Lyon", locations.get(1).value().entry().name());
  }

  @Test
  void testEntityTypesOutsideTheConfiguredSetAreIgnored() {
    // Both entities carry non-location types, so the geocoder is never called and the
    // provided layer is empty.
    final String text = "Sydney Smith met the Berlin Philharmonic";
    final Document input = Document.of(text).with(Layers.ENTITIES, List.of(
        new Annotation<>(new Span(0, 12), "person"),
        new Annotation<>(new Span(21, 40), "organization")));

    final Document document = new GeocodeAnnotator(unreachableGeocoder()).annotate(input);
    assertTrue(document.get(GeocodeAnnotator.LOCATIONS).isEmpty());
    assertTrue(document.layers().contains(GeocodeAnnotator.LOCATIONS));
  }

  @Test
  void testZeroEntitiesYieldAnEmptyLocationsLayer() {
    // A document without an entity layer reads as zero entities; the layer is still
    // provided, just empty, and the geocoder is never called.
    final Document document =
        new GeocodeAnnotator(unreachableGeocoder()).annotate(Document.of("no places"));
    assertTrue(document.get(GeocodeAnnotator.LOCATIONS).isEmpty());
    assertTrue(document.layers().contains(GeocodeAnnotator.LOCATIONS));
  }

  @Test
  void testMultipleResolutionsForOneMentionAreAllKept() {
    // A geocoder ranking two candidates for one mention yields two annotations on the
    // same span, in the geocoder's order, so the full ranked list stays available.
    final Geocoder geocoder = (text, mentions) -> {
      final List<GeoResolution> ranked = new ArrayList<>();
      for (final Span mention : mentions) {
        ranked.add(new GeoResolution(mention,
            entry("springfield-us", "Springfield", "US"), 0.6));
        ranked.add(new GeoResolution(mention,
            entry("springfield-ca", "Springfield", "CA"), 0.3));
      }
      return ranked;
    };
    final String text = "meet in Springfield today";
    final Document input = Document.of(text).with(Layers.ENTITIES,
        List.of(new Annotation<>(new Span(8, 19), "location")));

    final List<Annotation<GeoResolution>> locations =
        new GeocodeAnnotator(geocoder).annotate(input).get(GeocodeAnnotator.LOCATIONS);
    assertEquals(2, locations.size());
    assertEquals(new Span(8, 19), locations.get(0).span());
    assertEquals(new Span(8, 19), locations.get(1).span());
    assertEquals("US", locations.get(0).value().entry().countryCode());
    assertEquals(0.6, locations.get(0).value().confidence(), 1e-9);
    assertEquals("CA", locations.get(1).value().entry().countryCode());
    assertEquals(0.3, locations.get(1).value().confidence(), 1e-9);
  }

  @Test
  void testGeocoderIoFailureSurfacesAsUncheckedIOException() {
    final Geocoder failing = (text, mentions) -> {
      throw new IOException("gazetteer offline");
    };
    final String text = "a Berlin landmark";
    final Document input = Document.of(text).with(Layers.ENTITIES,
        List.of(new Annotation<>(new Span(2, 8), "location")));

    final UncheckedIOException thrown = assertThrows(UncheckedIOException.class,
        () -> new GeocodeAnnotator(failing).annotate(input));
    assertEquals("geocoding failed", thrown.getMessage());
    assertEquals("gazetteer offline", thrown.getCause().getMessage());
  }

  @Test
  void testAnnotatingTwiceRejectsTheDuplicateLayer() {
    final Geocoder geocoder = tableGeocoder(Map.of(), 0.5);
    final GeocodeAnnotator annotator = new GeocodeAnnotator(geocoder);
    final Document once = annotator.annotate(Document.of("plain text"));
    assertThrows(IllegalArgumentException.class, () -> annotator.annotate(once));
  }

  @Test
  void testResolutionOutsideTheGivenMentionsIsRejected() {
    // The Geocoder contract binds every returned mention() to one of the input spans.
    // A geocoder that trims the span it was handed breaks that contract, and the layer
    // it would produce could no longer be matched back to its entity, so the violation
    // is reported instead of being carried into the layer.
    final Geocoder trimming = (text, mentions) -> {
      final List<GeoResolution> resolutions = new ArrayList<>();
      for (final Span mention : mentions) {
        resolutions.add(new GeoResolution(
            new Span(mention.getStart(), mention.getEnd() - 1),
            entry("Sydney", "AU"), 0.8));
      }
      return resolutions;
    };
    final String text = "a Sydney landmark";
    final Document input = Document.of(text).with(Layers.ENTITIES,
        List.of(new Annotation<>(new Span(2, 8), "location")));

    final IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
        () -> new GeocodeAnnotator(trimming).annotate(input));
    assertTrue(thrown.getMessage().contains("mention"), thrown.getMessage());
  }

  @Test
  void testTypeSetValidation() {
    final Geocoder geocoder = tableGeocoder(Map.of(), 0.5);
    assertThrows(IllegalArgumentException.class,
        () -> new GeocodeAnnotator(geocoder, null));
    assertThrows(IllegalArgumentException.class,
        () -> new GeocodeAnnotator(geocoder, Set.of(" ")));
    assertThrows(IllegalArgumentException.class,
        () -> new GeocodeAnnotator(geocoder, Collections.singleton(null)));
  }
}
