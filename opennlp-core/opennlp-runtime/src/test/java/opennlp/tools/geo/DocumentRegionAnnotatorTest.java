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
import java.util.Set;

import org.junit.jupiter.api.Test;

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

  /** A geocoder that resolves a fixed name-to-country table with fixed confidence. */
  private static Geocoder tableGeocoder(Map<String, String> countryByName) {
    return (text, mentions) -> {
      final List<GeoResolution> resolutions = new ArrayList<>();
      for (final Span mention : mentions) {
        final String name =
            text.subSequence(mention.getStart(), mention.getEnd()).toString();
        final String country = countryByName.get(name);
        if (country != null) {
          resolutions.add(new GeoResolution(mention, entry(name, country), 0.8));
        }
      }
      return resolutions;
    };
  }

  private static GazetteerEntry entry(String name, String country) {
    return new GazetteerEntry("test", name, name, List.of(), new GeoPoint(0.0, 0.0),
        country, List.of(), 1000, GazetteerEntry.FEATURE_CLASS_CITY, Map.of());
  }

  private static Document withEntities(String text, String... names) {
    final List<Annotation<String>> entities = new ArrayList<>();
    for (final String name : names) {
      final int start = text.indexOf(name);
      entities.add(new Annotation<>(new Span(start, start + name.length()), "location"));
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
    assertEquals(2.0 / 3.0, ballot.get(0).value().share(), 1e-9);
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
    assertEquals(1.0, ballot.get(0).value().share(), 1e-9);
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
  void testValidation() {
    final Geocoder geocoder = tableGeocoder(Map.of());
    assertThrows(IllegalArgumentException.class, () -> new DocumentRegionAnnotator(null));
    assertThrows(IllegalArgumentException.class,
        () -> new DocumentRegionAnnotator(geocoder, Set.of()));
    assertThrows(IllegalArgumentException.class,
        () -> new DocumentRegionAnnotator(geocoder).annotate(null));
  }

  @Test
  void testRegionVoteValidation() {
    assertThrows(IllegalArgumentException.class, () -> new RegionVote(" ", 0.5));
    assertThrows(IllegalArgumentException.class, () -> new RegionVote("AU", 0.0));
    assertThrows(IllegalArgumentException.class, () -> new RegionVote("AU", 1.1));
  }
}
