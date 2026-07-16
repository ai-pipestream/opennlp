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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
import opennlp.tools.util.Span;

/**
 * Exercises the containment feature end to end the way a user assembles it: a document
 * with location entities flows through a {@link GeocodeAnnotator} into a
 * {@link HierarchyAnnotator}, and each resolved mention expands into the exact chain of
 * places that contain it. The hierarchy is a small self-contained
 * {@link ContainmentSpine} spanning one chain from a neighbourhood up to its country;
 * no external data is involved.
 */
public class HierarchyPipelineExampleTest {

  /**
   * Builds the example hierarchy, one containment chain of four places: the Le Marais
   * neighbourhood inside the locality Paris, inside the region Ile-de-France, inside
   * the country France, which is the root.
   *
   * @return The spine over the four example places. Never {@code null}.
   */
  private static ContainmentSpine exampleSpine() {
    return ContainmentSpine.builder()
        .add("101", "102", "Le Marais", "neighbourhood")
        .add("102", "103", "Paris", "locality")
        .add("103", "104", "Ile-de-France", "region")
        .add("104", null, "France", "country")
        .build();
  }

  /**
   * Builds a geocoder stub that resolves a mention by looking its exact covered text up
   * in a fixed table, with a fixed confidence of {@code 0.9}. Mentions whose text the
   * table does not contain are omitted from the result, as the {@link Geocoder}
   * contract requires for unresolvable mentions.
   *
   * @param entriesByName The surface-text-to-entry table. Must not be {@code null}.
   * @return The stub geocoder. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code entriesByName} is {@code null}.
   */
  private static Geocoder tableGeocoder(Map<String, GazetteerEntry> entriesByName) {
    if (entriesByName == null) {
      throw new IllegalArgumentException("entriesByName must not be null");
    }
    return (text, mentions) -> {
      final List<GeoResolution> resolutions = new ArrayList<>();
      for (final Span mention : mentions) {
        final String name =
            text.subSequence(mention.getStart(), mention.getEnd()).toString();
        final GazetteerEntry entry = entriesByName.get(name);
        if (entry != null) {
          resolutions.add(new GeoResolution(mention, entry, 0.9));
        }
      }
      return resolutions;
    };
  }

  /**
   * Builds a minimal gazetteer entry carrying the given place name and, when an
   * identifier is supplied, the conventional Who's On First attribute the
   * {@link HierarchyAnnotator} joins on by default.
   *
   * @param name The place name. Must not be {@code null} or empty.
   * @param wofId The Who's On First identifier, or {@code null} for an entry without
   *              the join attribute.
   * @return The entry. Never {@code null}.
   */
  private static GazetteerEntry entry(String name, String wofId) {
    final Map<String, AttributeValue> attributes = wofId == null ? Map.of()
        : Map.of(GazetteerEntry.ATTRIBUTE_KEY_WHOSONFIRST,
            new AttributeValue(wofId, "test", "fixture"));
    return new GazetteerEntry("example", name, name, List.of(),
        new GeoPoint(48.859, 2.361), "FR", List.of(), 0,
        GazetteerEntry.FEATURE_CLASS_CITY, attributes);
  }

  /**
   * Runs one location mention through the full pipeline and asserts the exact
   * containment layer: one chain on the mention's span, the ancestors nearest first
   * with their identifiers, names, and types, and the mentioned place itself excluded
   * from its own chain.
   */
  @Test
  void testMentionExpandsIntoItsExactContainmentChain() {
    final String text = "We wandered through Le Marais all afternoon.";
    final Span mention = new Span(20, 29);
    Assertions.assertEquals("Le Marais", mention.getCoveredText(text).toString());

    final Geocoder geocoder =
        tableGeocoder(Map.of("Le Marais", entry("Le Marais", "101")));
    final Document document = Document.of(text)
        .with(Layers.ENTITIES, List.of(new Annotation<>(mention, "location")));

    final Document annotated = new HierarchyAnnotator(exampleSpine())
        .annotate(new GeocodeAnnotator(geocoder).annotate(document));

    Assertions.assertEquals(1, annotated.get(GeocodeAnnotator.LOCATIONS).size());

    final List<Annotation<ContainmentChain>> chains =
        annotated.get(HierarchyAnnotator.CONTAINMENT);
    Assertions.assertEquals(1, chains.size());
    Assertions.assertEquals(mention, chains.get(0).span());
    Assertions.assertEquals(List.of(
        new PlaceAncestor("102", "Paris", "locality"),
        new PlaceAncestor("103", "Ile-de-France", "region"),
        new PlaceAncestor("104", "France", "country")),
        chains.get(0).value().ancestors());
    for (final PlaceAncestor ancestor : chains.get(0).value().ancestors()) {
      Assertions.assertNotEquals("101", ancestor.id());
      Assertions.assertNotEquals("Le Marais", ancestor.name());
    }
  }

  /**
   * Runs two location mentions through the full pipeline where the geocoder cannot
   * resolve one of them, and asserts that only the resolved mention gets a chain: the
   * unresolvable mention is dropped by the geocoder, so no containment annotation is
   * ever fabricated for it.
   */
  @Test
  void testUnresolvedMentionsGetNoChain() {
    final String text = "From Narnia to Le Marais.";
    final Span unresolvable = new Span(5, 11);
    final Span resolvable = new Span(15, 24);
    Assertions.assertEquals("Narnia", unresolvable.getCoveredText(text).toString());
    Assertions.assertEquals("Le Marais", resolvable.getCoveredText(text).toString());

    final Geocoder geocoder =
        tableGeocoder(Map.of("Le Marais", entry("Le Marais", "101")));
    final Document document = Document.of(text)
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(unresolvable, "location"),
            new Annotation<>(resolvable, "location")));

    final Document annotated = new HierarchyAnnotator(exampleSpine())
        .annotate(new GeocodeAnnotator(geocoder).annotate(document));

    final List<Annotation<ContainmentChain>> chains =
        annotated.get(HierarchyAnnotator.CONTAINMENT);
    Assertions.assertEquals(1, chains.size());
    Assertions.assertEquals(resolvable, chains.get(0).span());
    Assertions.assertEquals(3, chains.get(0).value().ancestors().size());
  }
}
