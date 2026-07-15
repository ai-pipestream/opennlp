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

package opennlp.geo;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.geo.AttributeValue;
import opennlp.tools.geo.ContainmentChain;
import opennlp.tools.geo.GazetteerEntry;
import opennlp.tools.geo.GeoPoint;
import opennlp.tools.geo.GeoResolution;
import opennlp.tools.geo.GeocodeAnnotator;
import opennlp.tools.geo.Geocoder;
import opennlp.tools.geo.PlaceAncestor;
import opennlp.tools.geo.PlaceHierarchy;
import opennlp.tools.util.Span;

/**
 * Runs the code from the "Geographic Resolution" manual chapter (geo.xml) and asserts the
 * behavior the chapter states. Each test mirrors an example in that chapter; if the chapter
 * and the API disagree, a test here fails.
 */
public class ManualExamplesTest {

  // ---- gazetteer entry and point accessors (tools.geo.gazetteer) ----

  @Test
  void aGazetteerEntryExposesItsPointAndAttributes() {
    final GazetteerEntry entry = parkSlopeEntry("parkslope");
    final GeoPoint point = entry.location();
    Assertions.assertEquals(40.67, point.latitude(), 1e-9);
    Assertions.assertEquals(-73.98, point.longitude(), 1e-9);
    Assertions.assertEquals("Park Slope", entry.name());
    Assertions.assertEquals("parkslope",
        entry.attributes().get(GazetteerEntry.ATTRIBUTE_KEY_WHOSONFIRST).value());
  }

  // ---- geocode annotator (tools.geo.annotator) ----

  @Test
  void theGeocodeAnnotatorDeclaresItsLayers() {
    final GeocodeAnnotator annotator = new GeocodeAnnotator(new FixedGeocoder(parkSlopeEntry("x")));
    Assertions.assertEquals(Set.of(Layers.ENTITIES), annotator.requires());
    Assertions.assertEquals(Set.of(GeocodeAnnotator.LOCATIONS), annotator.provides());
  }

  @Test
  void theGeocodeAnnotatorGeocodesLocationEntitiesByDefault() {
    final String text = "Park Slope and Ada met.";
    final Document document = Document.of(text).with(Layers.ENTITIES, List.of(
        new Annotation<>(new Span(0, 10), "location"),
        new Annotation<>(new Span(15, 18), "person")));

    final Document geocoded =
        new GeocodeAnnotator(new FixedGeocoder(parkSlopeEntry("parkslope"))).annotate(document);

    final List<Annotation<GeoResolution>> located = geocoded.get(GeocodeAnnotator.LOCATIONS);
    Assertions.assertEquals(1, located.size());
    Assertions.assertEquals("Park Slope", located.get(0).value().entry().name());
    // the location keeps the mention span
    Assertions.assertEquals(new Span(0, 10), located.get(0).span());
  }

  @Test
  void theSecondConstructorNamesTheEntityTypesToGeocode() {
    final String text = "Park Slope";
    final Document document = Document.of(text).with(Layers.ENTITIES, List.of(
        new Annotation<>(new Span(0, 10), "gpe")));
    final Geocoder geocoder = new FixedGeocoder(parkSlopeEntry("parkslope"));

    // the default set of {"location"} does not geocode a "gpe" entity
    Assertions.assertTrue(
        new GeocodeAnnotator(geocoder).annotate(document)
            .get(GeocodeAnnotator.LOCATIONS).isEmpty());
    // naming "gpe" does
    Assertions.assertEquals(1,
        new GeocodeAnnotator(geocoder, Set.of("gpe")).annotate(document)
            .get(GeocodeAnnotator.LOCATIONS).size());
  }

  // ---- containment hierarchy (tools.geo.hierarchy) ----

  private static PlaceHierarchy manualSpine() {
    return ContainmentSpine.builder()
        .add("parkslope", "brooklyn", "Park Slope", "neighbourhood")
        .add("brooklyn", "nyc", "Brooklyn", "borough")
        .add("nyc", "nystate", "New York City", "locality")
        .add("nystate", "usa", "New York", "region")
        .add("usa", null, "United States", "country")
        .build();
  }

  @Test
  void ancestorsAreTheEnclosingPlacesNearestFirstWithoutThePlaceItself() {
    final List<PlaceAncestor> chain = manualSpine().ancestors("parkslope");
    final List<String> names = new ArrayList<>();
    for (final PlaceAncestor ancestor : chain) {
      names.add(ancestor.name());
    }
    Assertions.assertEquals(
        List.of("Brooklyn", "New York City", "New York", "United States"), names);
    Assertions.assertFalse(names.contains("Park Slope"));
  }

  @Test
  void theHierarchyAnnotatorDeclaresItsLayers() {
    final HierarchyAnnotator annotator = new HierarchyAnnotator(manualSpine());
    Assertions.assertEquals(Set.of(GeocodeAnnotator.LOCATIONS), annotator.requires());
    Assertions.assertEquals(Set.of(HierarchyAnnotator.CONTAINMENT), annotator.provides());
  }

  @Test
  void theHierarchyAnnotatorAttachesTheEnclosingChain() {
    final Document document = documentWithResolvedPlace("parkslope");
    final Document withChains = new HierarchyAnnotator(manualSpine()).annotate(document);

    final List<Annotation<ContainmentChain>> chains =
        withChains.get(HierarchyAnnotator.CONTAINMENT);
    Assertions.assertEquals(1, chains.size());
    final List<String> names = new ArrayList<>();
    for (final PlaceAncestor ancestor : chains.get(0).value().ancestors()) {
      names.add(ancestor.name());
    }
    Assertions.assertEquals(
        List.of("Brooklyn", "New York City", "New York", "United States"), names);
  }

  @Test
  void aPlaceTheHierarchyDoesNotKnowGetsNoChain() {
    final Document document = documentWithResolvedPlace("unknown-id");
    final Document withChains = new HierarchyAnnotator(manualSpine()).annotate(document);
    Assertions.assertTrue(withChains.get(HierarchyAnnotator.CONTAINMENT).isEmpty());
  }

  // ---- place profiles (tools.geo.profiles) ----

  private static final String PROFILE_TABLE = String.join("\n",
      "id\tdensity\tincome\ttransit",
      "park-slope\t38000\t95000\t9",
      "brooklyn-heights\t36000\t110000\t9",
      "suburbia\t2000\t85000\t2",
      "rural-town\t150\t45000\t1",
      "");

  @Test
  void placeProfilesCompareByStandardizedProfile() throws IOException {
    final PlaceProfiles profiles = PlaceProfiles.load(
        new ByteArrayInputStream(PROFILE_TABLE.getBytes(StandardCharsets.UTF_8)));

    Assertions.assertTrue(profiles.similarity("park-slope", "brooklyn-heights") > 0.9);
    Assertions.assertTrue(profiles.similarity("park-slope", "rural-town") < 0.0);
    Assertions.assertEquals("brooklyn-heights",
        profiles.mostSimilar("park-slope", 5).get(0).id());
    Assertions.assertEquals(List.of("density", "income", "transit"), profiles.metrics());
    Assertions.assertTrue(profiles.contains("suburbia"));
    Assertions.assertFalse(profiles.contains("atlantis"));
  }

  // ---- fixtures ----

  private static GazetteerEntry parkSlopeEntry(String whosOnFirstId) {
    return new GazetteerEntry("test", "1", "Park Slope", List.of(),
        new GeoPoint(40.67, -73.98), "US", List.of(),
        62000L, GazetteerEntry.FEATURE_CLASS_CITY,
        Map.of(GazetteerEntry.ATTRIBUTE_KEY_WHOSONFIRST,
            new AttributeValue(whosOnFirstId, "test", "")));
  }

  private static Document documentWithResolvedPlace(String whosOnFirstId) {
    final Span mention = new Span(0, 10);
    final GeoResolution resolution =
        new GeoResolution(mention, parkSlopeEntry(whosOnFirstId), 1.0);
    return Document.of("Park Slope")
        .with(GeocodeAnnotator.LOCATIONS, List.of(new Annotation<>(mention, resolution)));
  }

  /** A geocoder that resolves every mention to one fixed entry. */
  private static final class FixedGeocoder implements Geocoder {
    private final GazetteerEntry entry;

    FixedGeocoder(GazetteerEntry entry) {
      this.entry = entry;
    }

    @Override
    public List<GeoResolution> resolve(CharSequence text, List<Span> locationMentions) {
      final List<GeoResolution> out = new ArrayList<>();
      for (final Span mention : locationMentions) {
        out.add(new GeoResolution(mention, entry, 1.0));
      }
      return out;
    }
  }
}
