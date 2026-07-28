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
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests the region ballot: geocoded mentions vote with their confidence, country-name
 * mentions vote directly through JDK locale data, and shares rank the result.
 */
public class DocumentRegionAnnotatorTest {

  /** The tolerance for a share compared against a recomputed quotient. */
  private static final double SHARE_DELTA = 1e-9;

  /** The confidence every resolution of the table geocoder reports. */
  private static final double TABLE_CONFIDENCE = 0.8;

  /** The entity type label the annotator treats as a location by default. */
  private static final String LOCATION = "location";

  /**
   * A no-break space: blank under the toolkit's whitespace definition, but not under
   * {@link String#isBlank()}.
   */
  private static final String NO_BREAK_SPACE = "\u00A0";

  /**
   * Builds a geocoder that resolves a fixed name-to-country table at
   * {@link #TABLE_CONFIDENCE} and leaves unknown mentions unresolved.
   *
   * @param countryByName Maps a mention text to its country code. Must not be
   *                      {@code null}.
   * @return A {@link Geocoder} over the table. Never {@code null}.
   */
  private static Geocoder tableGeocoder(Map<String, String> countryByName) {
    return (text, mentions) -> {
      final List<GeoResolution> resolutions = new ArrayList<>();
      for (final Span mention : mentions) {
        final String name =
            text.subSequence(mention.getStart(), mention.getEnd()).toString();
        final String country = countryByName.get(name);
        if (country != null) {
          resolutions.add(
              new GeoResolution(mention, entry(name, country), TABLE_CONFIDENCE));
        }
      }
      return resolutions;
    };
  }

  /**
   * Builds a minimal city entry for a country; only the country code matters for the
   * region ballot.
   *
   * @param name The city name. Must not be {@code null}.
   * @param country The ISO 3166-1 alpha-2 country code. Must not be {@code null}.
   * @return A {@link GazetteerEntry} for the city. Never {@code null}.
   */
  private static GazetteerEntry entry(String name, String country) {
    return new GazetteerEntry("test", name, name, List.of(), new GeoPoint(0.0, 0.0),
        country, List.of(), 1000, GazetteerEntry.FEATURE_CLASS_CITY, Map.of());
  }

  /**
   * Builds a document whose entity layer marks each given mention as a location.
   *
   * @param text The document text. Must not be {@code null}.
   * @param names The mention texts to mark. Each must occur in {@code text}.
   * @return A {@link Document} with an entity layer. Never {@code null}.
   */
  private static Document withEntities(String text, String... names) {
    final List<Annotation<String>> entities = new ArrayList<>();
    for (final String name : names) {
      final int start = text.indexOf(name);
      entities.add(new Annotation<>(new Span(start, start + name.length()), LOCATION));
    }
    return Document.of(text).with(Layers.ENTITIES, entities);
  }

  @Test
  void testGeocodedMentionsVoteWithConfidence() {
    final Geocoder geocoder = tableGeocoder(
        Map.of("Sydney", "AU", "Melbourne", "AU", "Auckland", "NZ"));
    final String text = "flights from Sydney and Melbourne to Auckland";
    final Document document = new DocumentRegionAnnotator(geocoder)
        .annotate(withEntities(text, "Sydney", "Melbourne", "Auckland"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(2, ballot.size());
    assertEquals("AU", ballot.get(0).value().countryCode());
    assertEquals(2.0 / 3.0, ballot.get(0).value().share(), SHARE_DELTA);
    assertEquals("NZ", ballot.get(1).value().countryCode());
    assertNull(ballot.get(0).span());
  }

  @Test
  void testCountryNamesVoteWithoutTheGazetteer() {
    // the geocoder resolves nothing; the country name still votes
    final Geocoder geocoder = tableGeocoder(Map.of());
    final String text = "mining exports from Australia rose";
    final Document document = new DocumentRegionAnnotator(geocoder)
        .annotate(withEntities(text, "Australia"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(1, ballot.size());
    assertEquals("AU", ballot.get(0).value().countryCode());
    assertEquals(1.0, ballot.get(0).value().share(), SHARE_DELTA);
  }

  @Test
  void testNoEvidenceMeansAnEmptyBallot() {
    final Geocoder geocoder = tableGeocoder(Map.of());
    final Document document = new DocumentRegionAnnotator(geocoder)
        .annotate(withEntities("nothing resolvable in Atlantis", "Atlantis"));
    assertTrue(document.get(DocumentRegionAnnotator.REGIONS).isEmpty());
  }

  @Test
  void testNonLocationEntitiesDoNotVote() {
    final Geocoder geocoder = tableGeocoder(Map.of("Sydney", "AU"));
    final String text = "Sydney Smith spoke";
    final Document document = Document.of(text)
        .with(Layers.ENTITIES,
            List.of(new Annotation<>(new Span(0, 12), "person")));
    assertTrue(new DocumentRegionAnnotator(geocoder).annotate(document)
        .get(DocumentRegionAnnotator.REGIONS).isEmpty());
  }

  @Test
  void testCustomLocationTypesMatchCaseInsensitively() {
    final Geocoder geocoder = tableGeocoder(Map.of("Sydney", "AU"));
    final String text = "Sydney hosts the summit";
    final Document document = Document.of(text).with(Layers.ENTITIES,
        List.of(new Annotation<>(new Span(0, 6), "GPE")));
    final Document annotated =
        new DocumentRegionAnnotator(geocoder, Set.of("gpe")).annotate(document);

    final List<Annotation<RegionVote>> ballot =
        annotated.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(1, ballot.size());
    assertEquals("AU", ballot.get(0).value().countryCode());
  }

  @Test
  void testGeocoderIoFailureSurfacesUnchecked() {
    final Geocoder failing = (text, mentions) -> {
      throw new IOException("gazetteer unavailable");
    };
    final Document document = withEntities("a dispatch from Bilbao", "Bilbao");
    assertThrows(UncheckedIOException.class,
        () -> new DocumentRegionAnnotator(failing).annotate(document));
  }

  @Test
  void testNullGeocoderIsRejected() {
    assertThrows(IllegalArgumentException.class, () -> new DocumentRegionAnnotator(null));
  }

  @Test
  void testNullDocumentIsRejected() {
    final DocumentRegionAnnotator annotator =
        new DocumentRegionAnnotator(tableGeocoder(Map.of()));
    assertThrows(IllegalArgumentException.class, () -> annotator.annotate(null));
  }

  @ParameterizedTest
  @MethodSource("invalidLocationTypes")
  void testLocationTypesAreValidated(Set<String> locationTypes) {
    final Geocoder geocoder = tableGeocoder(Map.of());
    assertThrows(IllegalArgumentException.class,
        () -> new DocumentRegionAnnotator(geocoder, locationTypes));
  }

  private static Stream<Arguments> invalidLocationTypes() {
    return Stream.of(
        Arguments.of((Set<String>) null),
        Arguments.of(Set.of()),
        Arguments.of(Collections.<String>singleton(null)),
        Arguments.of(Set.of(" ")),
        Arguments.of(Set.of(NO_BREAK_SPACE)),
        Arguments.of(Set.of(LOCATION, "")));
  }

  @ParameterizedTest
  @MethodSource("invalidVotes")
  void testRegionVoteRejectsInvalidRows(String countryCode, double share) {
    assertThrows(IllegalArgumentException.class,
        () -> new RegionVote(countryCode, share));
  }

  private static Stream<Arguments> invalidVotes() {
    return Stream.of(
        Arguments.of(null, 0.5),
        Arguments.of("", 0.5),
        Arguments.of(" ", 0.5),
        Arguments.of(NO_BREAK_SPACE, 0.5),
        Arguments.of("AU", 0.0),
        Arguments.of("AU", -0.1),
        Arguments.of("AU", 1.1),
        Arguments.of("AU", Double.NaN));
  }

  @Test
  void testRegionVoteAcceptsTheShareBounds() {
    assertEquals(1.0, new RegionVote("AU", 1.0).share(), 0.0);
    assertEquals(Double.MIN_VALUE, new RegionVote("AU", Double.MIN_VALUE).share(), 0.0);
  }
}
