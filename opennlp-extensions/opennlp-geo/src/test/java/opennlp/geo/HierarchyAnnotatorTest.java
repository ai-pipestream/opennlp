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

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.geo.AttributeValue;
import opennlp.tools.geo.ContainmentChain;
import opennlp.tools.geo.GazetteerEntry;
import opennlp.tools.geo.GeoPoint;
import opennlp.tools.geo.GeoResolution;
import opennlp.tools.geo.GeocodeAnnotator;
import opennlp.tools.util.Span;

public class HierarchyAnnotatorTest {

  private static GazetteerEntry entry(String name, String wofId) {
    final Map<String, AttributeValue> attributes = wofId == null ? Map.of()
        : Map.of(GazetteerEntry.ATTRIBUTE_KEY_WHOSONFIRST,
            new AttributeValue(wofId, "test", "fixture"));
    return new GazetteerEntry("test", name, name, List.of(), new GeoPoint(0.0, 0.0),
        "US", List.of(), 1000, GazetteerEntry.FEATURE_CLASS_CITY, attributes);
  }

  private static ContainmentSpine spine() {
    return ContainmentSpine.builder()
        .add("85865587", "421205765", "Park Slope", "neighbourhood")
        .add("421205765", "85977539", "Brooklyn", "borough")
        .add("85977539", null, "New York", "locality")
        .build();
  }

  @Test
  void testResolvedMentionsExpandIntoTheirChains() {
    final String text = "A stroll through Park Slope.";
    final Span mention = new Span(17, 27);
    final Document document = Document.of(text)
        .with(GeocodeAnnotator.LOCATIONS, List.of(new Annotation<>(mention,
            new GeoResolution(mention, entry("Park Slope", "85865587"), 0.9))));

    final Document annotated = new HierarchyAnnotator(spine()).annotate(document);

    final List<Annotation<ContainmentChain>> chains =
        annotated.get(HierarchyAnnotator.CONTAINMENT);
    Assertions.assertEquals(1, chains.size());
    Assertions.assertEquals(mention, chains.get(0).span());
    Assertions.assertEquals(List.of("Brooklyn", "New York"),
        chains.get(0).value().ancestors().stream().map(a -> a.name()).toList());
  }

  @Test
  void testMentionsWithoutJoinIdOrChainAreOmitted() {
    final String text = "Atlantis and Brooklyn";
    final Document document = Document.of(text)
        .with(GeocodeAnnotator.LOCATIONS, List.of(
            new Annotation<>(new Span(0, 8),
                new GeoResolution(new Span(0, 8), entry("Atlantis", null), 0.5)),
            new Annotation<>(new Span(13, 21),
                new GeoResolution(new Span(13, 21), entry("Brooklyn", "77"), 0.5))));

    final Document annotated = new HierarchyAnnotator(spine()).annotate(document);

    Assertions.assertTrue(annotated.get(HierarchyAnnotator.CONTAINMENT).isEmpty());
  }

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new HierarchyAnnotator(null));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new HierarchyAnnotator(spine(), " "));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new HierarchyAnnotator(spine()).annotate(null));
  }
}
