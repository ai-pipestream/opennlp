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
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
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
 * Tests the locations layer and the region ballot built on it: resolved mentions keep
 * their full resolution in the layer and vote with their confidence, country-name
 * mentions the geocoder left unresolved vote through JDK locale data, and shares rank
 * the result.
 */
public class DocumentRegionAnnotatorTest {

  /** The tolerance for a share compared against a recomputed quotient. */
  private static final double SHARE_DELTA = 1e-9;

  /** The confidence every resolution of the table geocoder reports. */
  private static final double TABLE_CONFIDENCE = 0.8;

  /** Mirrors the annotator's fixed vote weight of a direct country-name mention. */
  private static final double COUNTRY_NAME_WEIGHT = 0.95;

  /**
   * A no-break space: blank under the toolkit's whitespace definition, but not under
   * {@link String#isBlank()}.
   */
  private static final String NO_BREAK_SPACE = "\u00A0";

  /**
   * Runs the location pipeline: the geocode annotator provides the locations layer and
   * the region annotator derives the ballot from it.
   *
   * @param geocoder The geocoder backing the locations layer. Must not be {@code null}.
   * @param document The document carrying an entity layer. Must not be {@code null}.
   * @return The document with the locations and regions layers added. Never
   *         {@code null}.
   */
  private static Document annotate(Geocoder geocoder, Document document) {
    return new DocumentRegionAnnotator()
        .annotate(new GeocodeAnnotator(geocoder).annotate(document));
  }

  @Test
  void testLocationsLayerKeepsTheFullResolution() {
    final Geocoder geocoder =
        GeoTestUtil.tableGeocoder(Map.of("Sydney", "AU"), TABLE_CONFIDENCE);
    final String text = "a Sydney landmark";
    final Document document = new GeocodeAnnotator(geocoder)
        .annotate(GeoTestUtil.withLocations(text, "Sydney"));

    final List<Annotation<GeoResolution>> locations =
        document.get(GeocodeAnnotator.LOCATIONS);
    assertEquals(1, locations.size());
    assertEquals("AU", locations.get(0).value().entry().countryCode());
    assertEquals(TABLE_CONFIDENCE, locations.get(0).value().confidence(), SHARE_DELTA);
    assertEquals("Sydney",
        locations.get(0).span().getCoveredText(text).toString());
  }

  @Test
  void testGeocodedMentionsVoteWithConfidence() {
    final Geocoder geocoder = GeoTestUtil.tableGeocoder(
        Map.of("Sydney", "AU", "Melbourne", "AU", "Auckland", "NZ"), TABLE_CONFIDENCE);
    final String text = "flights from Sydney and Melbourne to Auckland";
    final Document document =
        annotate(geocoder, GeoTestUtil.withLocations(text, "Sydney", "Melbourne", "Auckland"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(2, ballot.size());
    assertEquals("AU", ballot.get(0).value().countryCode());
    assertEquals(2.0 / 3.0, ballot.get(0).value().share(), SHARE_DELTA);
    assertEquals("NZ", ballot.get(1).value().countryCode());
    assertNull(ballot.get(0).span());
  }

  @Test
  void testCountryNamesVoteWhenTheGeocoderCannotResolve() {
    final Geocoder geocoder = GeoTestUtil.tableGeocoder(Map.of());
    final String text = "mining exports from Australia rose";
    final Document document = annotate(geocoder, GeoTestUtil.withLocations(text, "Australia"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(1, ballot.size());
    assertEquals("AU", ballot.get(0).value().countryCode());
    assertEquals(1.0, ballot.get(0).value().share(), SHARE_DELTA);
  }

  /**
   * Verifies that a resolved mention never consults the name table: {@code Georgia} is
   * the display name of the country {@code GE}, but the locations layer resolves the
   * mention to the US state, so the document elects {@code US} with no {@code GE} row,
   * pricing its dollars as USD rather than GEL.
   */
  @Test
  void testResolvedMentionOverridesTheCountryNameTable() {
    final Geocoder geocoder = GeoTestUtil.tableGeocoder(
        Map.of("Georgia", "US", "Atlanta", "US"), TABLE_CONFIDENCE);
    final String text = "Georgia's secretary of state confirmed the recount in Atlanta";
    final Document document =
        annotate(geocoder, GeoTestUtil.withLocations(text, "Georgia", "Atlanta"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(1, ballot.size());
    assertEquals("US", ballot.get(0).value().countryCode());
    assertEquals(1.0, ballot.get(0).value().share(), SHARE_DELTA);
  }

  /**
   * Verifies that a resolved country-name mention casts exactly one vote, at its
   * resolution confidence: {@code Australia} resolves to {@code AU} in the locations
   * layer, so it never adds a second name-table vote, and the two countries tie at
   * equal confidence with the tie broken by ascending country code.
   */
  @Test
  void testResolvedCountryNameMentionVotesOnceWithItsConfidence() {
    final Geocoder geocoder = GeoTestUtil.tableGeocoder(
        Map.of("Australia", "AU", "Auckland", "NZ"), TABLE_CONFIDENCE);
    final String text = "exports from Australia to Auckland rose";
    final Document document =
        annotate(geocoder, GeoTestUtil.withLocations(text, "Australia", "Auckland"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(2, ballot.size());
    assertEquals("AU", ballot.get(0).value().countryCode());
    assertEquals(0.5, ballot.get(0).value().share(), SHARE_DELTA);
    assertEquals("NZ", ballot.get(1).value().countryCode());
    assertEquals(0.5, ballot.get(1).value().share(), SHARE_DELTA);
  }

  /**
   * Pins the country-name weight against geocoder confidence: a country name the
   * geocoder cannot resolve votes at {@code 0.95} while a confidently geocoded city
   * votes at {@code 0.8}, so New Zealand outranks Sydney's Australia and the shares
   * are the exact weight quotients, about {@code 0.543} to {@code 0.457}.
   */
  @Test
  void testCountryNameOutweighsAConfidentlyGeocodedCity() {
    final Geocoder geocoder =
        GeoTestUtil.tableGeocoder(Map.of("Sydney", "AU"), TABLE_CONFIDENCE);
    final String text = "the Sydney office reports to New Zealand headquarters";
    final Document document =
        annotate(geocoder, GeoTestUtil.withLocations(text, "Sydney", "New Zealand"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(2, ballot.size());
    assertEquals("NZ", ballot.get(0).value().countryCode());
    assertEquals(COUNTRY_NAME_WEIGHT / (COUNTRY_NAME_WEIGHT + TABLE_CONFIDENCE),
        ballot.get(0).value().share(), SHARE_DELTA);
    assertEquals("AU", ballot.get(1).value().countryCode());
    assertEquals(TABLE_CONFIDENCE / (COUNTRY_NAME_WEIGHT + TABLE_CONFIDENCE),
        ballot.get(1).value().share(), SHARE_DELTA);
  }

  /**
   * Verifies that country-name matching tolerates the spellings real text uses: the
   * CLDR display name's apostrophe (U+2019) and the ASCII apostrophe, the accented and
   * the plain spelling of a name with diacritics, and any letter case. The geocoder
   * resolves nothing, so no mention has a locations row and each match is proven by
   * the name-table fallback's one-row ballot.
   *
   * @param mention The country mention as it appears in the text.
   * @param expectedCountry The ISO 3166-1 alpha-2 code the mention must elect.
   */
  @ParameterizedTest
  @CsvSource(quoteCharacter = '"', value = {
      "Côte d’Ivoire, CI",
      "Côte d'Ivoire, CI",
      "Cote d'Ivoire, CI",
      "México, MX",
      "AUSTRALIA, AU",
      "australia, AU"
  })
  void testCountryNameSpellingVariantsMatch(String mention, String expectedCountry) {
    final Document document = annotate(GeoTestUtil.tableGeocoder(Map.of()),
        GeoTestUtil.withLocations("new offices in " + mention + " opened", mention));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(1, ballot.size());
    assertEquals(expectedCountry, ballot.get(0).value().countryCode());
    assertEquals(1.0, ballot.get(0).value().share(), SHARE_DELTA);
  }

  @Test
  void testNoEvidenceMeansAnEmptyBallot() {
    final Geocoder geocoder = GeoTestUtil.tableGeocoder(Map.of());
    final Document document = annotate(geocoder,
        GeoTestUtil.withLocations("nothing resolvable in Atlantis", "Atlantis"));
    assertTrue(document.get(DocumentRegionAnnotator.REGIONS).isEmpty());
    assertTrue(document.get(GeocodeAnnotator.LOCATIONS).isEmpty());
  }

  @Test
  void testNonLocationEntitiesDoNotVote() {
    final Geocoder geocoder =
        GeoTestUtil.tableGeocoder(Map.of("Sydney", "AU"), TABLE_CONFIDENCE);
    final String text = "Sydney Smith spoke";
    final Document document = Document.of(text)
        .with(Layers.ENTITIES,
            List.of(new Annotation<>(new Span(0, 12), "person")));
    assertTrue(annotate(geocoder, document)
        .get(DocumentRegionAnnotator.REGIONS).isEmpty());
  }

  @Test
  void testCustomLocationTypesMatchCaseInsensitively() {
    final Geocoder geocoder =
        GeoTestUtil.tableGeocoder(Map.of("Sydney", "AU"), TABLE_CONFIDENCE);
    final String text = "Sydney hosts the summit";
    final Document document = Document.of(text).with(Layers.ENTITIES,
        List.of(new Annotation<>(new Span(0, 6), "GPE")));
    final Document annotated = new DocumentRegionAnnotator(Set.of("gpe"))
        .annotate(new GeocodeAnnotator(geocoder, Set.of("gpe")).annotate(document));

    final List<Annotation<RegionVote>> ballot =
        annotated.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(1, ballot.size());
    assertEquals("AU", ballot.get(0).value().countryCode());
  }

  /**
   * Verifies that an absent locations layer is rejected as an assembly error rather
   * than tolerated as an empty ballot, so a pipeline missing its
   * {@link GeocodeAnnotator} stage fails loudly.
   */
  @Test
  void testMissingLocationsLayerIsRejected() {
    final Document document =
        GeoTestUtil.withLocations("a dispatch from Bilbao", "Bilbao");
    assertThrows(IllegalArgumentException.class,
        () -> new DocumentRegionAnnotator().annotate(document));
  }

  @Test
  void testValidation() {
    final Geocoder geocoder = GeoTestUtil.tableGeocoder(Map.of());
    assertThrows(IllegalArgumentException.class, () -> new GeocodeAnnotator(null));
    assertThrows(IllegalArgumentException.class,
        () -> new GeocodeAnnotator(geocoder, Set.of()));
    assertThrows(IllegalArgumentException.class,
        () -> new DocumentRegionAnnotator(Set.of()));
    assertThrows(IllegalArgumentException.class,
        () -> new DocumentRegionAnnotator().annotate(null));
    assertThrows(IllegalArgumentException.class,
        () -> new GeocodeAnnotator(geocoder).annotate(null));
  }

  @Test
  void testGeocoderIoFailureSurfacesUnchecked() {
    final Geocoder failing = (text, mentions) -> {
      throw new IOException("gazetteer unavailable");
    };
    final Document document =
        GeoTestUtil.withLocations("a dispatch from Bilbao", "Bilbao");
    assertThrows(UncheckedIOException.class,
        () -> new GeocodeAnnotator(failing).annotate(document));
  }

  @ParameterizedTest
  @MethodSource("invalidLocationTypes")
  void testLocationTypesAreValidated(Set<String> locationTypes) {
    assertThrows(IllegalArgumentException.class,
        () -> new DocumentRegionAnnotator(locationTypes));
  }

  private static Stream<Arguments> invalidLocationTypes() {
    return Stream.of(
        Arguments.of((Set<String>) null),
        Arguments.of(Set.of()),
        Arguments.of(Collections.<String>singleton(null)),
        Arguments.of(Set.of(" ")),
        Arguments.of(Set.of(NO_BREAK_SPACE)),
        Arguments.of(Set.of(GeoTestUtil.LOCATION, "")));
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
