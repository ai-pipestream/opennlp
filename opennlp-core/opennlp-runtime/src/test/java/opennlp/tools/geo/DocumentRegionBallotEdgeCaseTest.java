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
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests the edges of the region ballot: the deterministic order of tied countries, the
 * country-name vote when the geocoder is never consulted, the empty ballot of a document
 * without location entities, and the confidence weighting that lets one strong mention
 * outvote several weak ones.
 */
public class DocumentRegionBallotEdgeCaseTest {

  /**
   * One geocoding outcome of the table geocoder: the country a mention resolves to and
   * the confidence of that resolution.
   *
   * @param countryCode The ISO 3166-1 alpha-2 country code. Must not be {@code null}.
   * @param confidence The resolution confidence, in {@code [0, 1]}.
   */
  private record ScoredCountry(String countryCode, double confidence) {
  }

  /**
   * Builds a geocoder that resolves each known mention text to a fixed country with a
   * fixed per-mention confidence, and leaves unknown mentions unresolved.
   *
   * @param outcomes Maps a mention text to its resolution outcome. Must not be
   *                 {@code null}.
   * @return A {@link Geocoder} over the table. Never {@code null}.
   */
  private static Geocoder tableGeocoder(Map<String, ScoredCountry> outcomes) {
    return (text, mentions) -> {
      final List<GeoResolution> resolutions = new ArrayList<>();
      for (final Span mention : mentions) {
        final String name =
            text.subSequence(mention.getStart(), mention.getEnd()).toString();
        final ScoredCountry outcome = outcomes.get(name);
        if (outcome != null) {
          resolutions.add(new GeoResolution(mention,
              entry(name, outcome.countryCode()), outcome.confidence()));
        }
      }
      return resolutions;
    };
  }

  /**
   * Builds a geocoder that must never be consulted: any call fails the test loudly.
   * Passing it proves a code path resolves all of its evidence without the gazetteer.
   *
   * @return A {@link Geocoder} that rejects every call. Never {@code null}.
   */
  private static Geocoder unreachableGeocoder() {
    return (text, mentions) -> {
      throw new IllegalStateException("the geocoder must not be consulted");
    };
  }

  /**
   * Builds a minimal city entry for a country; only the country code matters for the
   * region ballot.
   *
   * @param name The city name. Must not be {@code null}.
   * @param countryCode The ISO 3166-1 alpha-2 country code. Must not be {@code null}.
   * @return A {@link GazetteerEntry} for the city. Never {@code null}.
   */
  private static GazetteerEntry entry(String name, String countryCode) {
    return new GazetteerEntry("test", name, name, List.of(), new GeoPoint(0.0, 0.0),
        countryCode, List.of(), 1000, GazetteerEntry.FEATURE_CLASS_CITY, Map.of());
  }

  /**
   * Builds a document whose entity layer marks each given mention as a location.
   *
   * @param text The document text. Must not be {@code null}.
   * @param mentions The mention texts to mark. Each must occur in {@code text}.
   * @return A {@link Document} with an entity layer. Never {@code null}.
   * @throws IllegalArgumentException Thrown if a mention does not occur in the text.
   */
  private static Document withLocations(String text, String... mentions) {
    final List<Annotation<String>> entities = new ArrayList<>();
    for (final String mention : mentions) {
      final int start = text.indexOf(mention);
      if (start < 0) {
        throw new IllegalArgumentException("mention not in the text: " + mention);
      }
      entities.add(new Annotation<>(new Span(start, start + mention.length()), "location"));
    }
    return Document.of(text).with(Layers.ENTITIES, entities);
  }

  /**
   * Verifies the ranking rule for a tie: two countries with equal weight split the
   * ballot evenly, and the tie breaks by ascending country code, so {@code FR} ranks
   * ahead of {@code GB} regardless of mention order in the text.
   */
  @Test
  void testTieBreaksByAscendingCountryCode() {
    final Geocoder geocoder = tableGeocoder(Map.of(
        "London", new ScoredCountry("GB", 0.8),
        "Paris", new ScoredCountry("FR", 0.8)));
    final Document document = new DocumentRegionAnnotator(geocoder)
        .annotate(withLocations("trains between London and Paris", "London", "Paris"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(2, ballot.size());
    assertEquals("FR", ballot.get(0).value().countryCode());
    assertEquals(0.5, ballot.get(0).value().share(), 0.0);
    assertEquals("GB", ballot.get(1).value().countryCode());
    assertEquals(0.5, ballot.get(1).value().share(), 0.0);
  }

  /**
   * Verifies that country-name mentions carry a ballot on their own: with two English
   * country names and a geocoder that fails loudly when consulted, both names vote with
   * the fixed country-name weight, tie evenly, and rank by ascending country code.
   */
  @Test
  void testCountryNamesAloneFillTheBallotWithoutTheGeocoder() {
    final Document document = new DocumentRegionAnnotator(unreachableGeocoder())
        .annotate(withLocations("trade between Mexico and New Zealand grew",
            "Mexico", "New Zealand"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(2, ballot.size());
    assertEquals("MX", ballot.get(0).value().countryCode());
    assertEquals(0.5, ballot.get(0).value().share(), 0.0);
    assertEquals("NZ", ballot.get(1).value().countryCode());
    assertEquals(0.5, ballot.get(1).value().share(), 0.0);
  }

  /**
   * Verifies the empty case: a document without an entity layer produces a present but
   * empty region layer, and the geocoder is never consulted for it.
   */
  @Test
  void testNoLocationEntitiesYieldAnEmptyPresentLayer() {
    final Document document = new DocumentRegionAnnotator(unreachableGeocoder())
        .annotate(Document.of("nothing to locate here"));

    assertTrue(document.get(DocumentRegionAnnotator.REGIONS).isEmpty());
    assertTrue(document.layers().contains(DocumentRegionAnnotator.REGIONS));
  }

  /**
   * Verifies that confidence weights the vote rather than the mention count: two French
   * mentions at low confidence lose to one US mention at high confidence, and the
   * shares are the exact confidence sums over the ballot total.
   */
  @Test
  void testOneConfidentMentionOutvotesTwoWeakOnes() {
    final Geocoder geocoder = tableGeocoder(Map.of(
        "Nice", new ScoredCountry("FR", 0.3),
        "Nancy", new ScoredCountry("FR", 0.3),
        "Chicago", new ScoredCountry("US", 0.7)));
    final Document document = new DocumentRegionAnnotator(geocoder)
        .annotate(withLocations("flights from Nice and Nancy to Chicago",
            "Nice", "Nancy", "Chicago"));

    final List<Annotation<RegionVote>> ballot =
        document.get(DocumentRegionAnnotator.REGIONS);
    assertEquals(2, ballot.size());
    assertEquals("US", ballot.get(0).value().countryCode());
    assertEquals(0.7 / (0.7 + 0.6), ballot.get(0).value().share(), 0.0);
    assertEquals("FR", ballot.get(1).value().countryCode());
    assertEquals(0.6 / (0.7 + 0.6), ballot.get(1).value().share(), 0.0);
  }

  /**
   * Verifies that a {@code null} document is rejected with a clear exception before any
   * layer is touched.
   */
  @Test
  void testNullDocumentIsRejected() {
    final DocumentRegionAnnotator annotator =
        new DocumentRegionAnnotator(unreachableGeocoder());
    assertThrows(IllegalArgumentException.class, () -> annotator.annotate(null));
  }
}
