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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.geo.AttributeValue;
import opennlp.tools.geo.ContainmentChain;
import opennlp.tools.geo.GazetteerEntry;
import opennlp.tools.geo.GeoResolution;
import opennlp.tools.geo.GeocodeAnnotator;
import opennlp.tools.geo.PlaceAncestor;
import opennlp.tools.geo.PlaceHierarchy;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * Expands each resolved location into the places that contain it: reads the geocoded
 * locations layer, joins each entry to a {@link PlaceHierarchy} through a configured
 * attribute key, and provides {@link #CONTAINMENT}, one annotation per expandable
 * mention carrying its {@link ContainmentChain} on the mention's span.
 *
 * <p>A mention whose entry lacks the join attribute, whose identifier the hierarchy
 * does not know, or whose place sits at the top of the hierarchy (an empty chain)
 * simply gets no annotation; nothing is invented. The default join key is the
 * conventional Who's On First attribute, matching the identifiers the bundled
 * gazetteer derivations carry.</p>
 *
 * <p>A {@link GeocodeAnnotator} emits one annotation per candidate, in the geocoder's
 * order, for a mention it ranks against several. Only the first annotation of a span
 * decides that mention's chain, so one span never carries two contradicting chains and a
 * mention whose best candidate cannot be expanded gets no annotation rather than the
 * chain of a lower-ranked candidate.</p>
 *
 * <p>The annotator holds no per-call state; it is as thread-safe as its hierarchy.</p>
 *
 * @since 3.0.0
 */
public class HierarchyAnnotator implements DocumentAnnotator {

  /**
   * Containment chains; each annotation covers one resolved mention and carries its
   * {@link ContainmentChain}.
   */
  public static final LayerKey<ContainmentChain> CONTAINMENT =
      Layers.key("containment", ContainmentChain.class);

  private final PlaceHierarchy hierarchy;
  private final String attributeKey;

  /**
   * Initializes the annotator joining on the Who's On First attribute.
   *
   * @param hierarchy The hierarchy to walk. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code hierarchy} is {@code null}.
   */
  public HierarchyAnnotator(PlaceHierarchy hierarchy) {
    this(hierarchy, GazetteerEntry.ATTRIBUTE_KEY_WHOSONFIRST);
  }

  /**
   * Initializes the annotator.
   *
   * @param hierarchy The hierarchy to walk. Must not be {@code null}.
   * @param attributeKey The gazetteer attribute holding the identifier in the
   *                     hierarchy's identifier space. Must not be {@code null} or
   *                     blank.
   * @throws IllegalArgumentException Thrown if {@code hierarchy} is {@code null} or
   *         {@code attributeKey} is {@code null} or blank.
   */
  public HierarchyAnnotator(PlaceHierarchy hierarchy, String attributeKey) {
    if (hierarchy == null) {
      throw new IllegalArgumentException("hierarchy must not be null");
    }
    if (attributeKey == null || StringUtil.isBlank(attributeKey)) {
      throw new IllegalArgumentException("attributeKey must not be null or blank");
    }
    this.hierarchy = hierarchy;
    this.attributeKey = attributeKey;
  }

  /**
   * Annotates the document with the {@link #CONTAINMENT} layer.
   *
   * <p>A mention is keyed by its character offsets alone, so a typed and an untyped span
   * over the same text are one mention and are expanded once.</p>
   *
   * <p>The locations layer must be present, but it may be empty: an absent layer is a
   * pipeline error, not a location-free document.</p>
   *
   * @param document The document to annotate. Must not be {@code null} and must carry
   *                 the {@link GeocodeAnnotator#LOCATIONS} layer.
   * @return A new {@link Document} with the {@link #CONTAINMENT} layer added, one
   *         annotation per expandable mention. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, lacks
   *         the locations layer, or already carries the {@link #CONTAINMENT} layer.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    if (!document.layers().contains(GeocodeAnnotator.LOCATIONS)) {
      throw new IllegalArgumentException("document lacks the required layer "
          + GeocodeAnnotator.LOCATIONS);
    }
    final List<Annotation<ContainmentChain>> chains = new ArrayList<>();
    final Set<Long> expanded = new HashSet<>();
    for (final Annotation<GeoResolution> location
        : document.get(GeocodeAnnotator.LOCATIONS)) {
      if (!expanded.add(offsets(location.span()))) {
        continue;
      }
      final AttributeValue joinId =
          location.value().entry().attributes().get(attributeKey);
      if (joinId == null) {
        continue;
      }
      final List<PlaceAncestor> ancestors = hierarchy.ancestors(joinId.value());
      if (!ancestors.isEmpty()) {
        chains.add(new Annotation<>(location.span(), new ContainmentChain(ancestors)));
      }
    }
    return document.with(CONTAINMENT, chains);
  }

  /** Collapses a span to its offsets, so typed and untyped spans key one mention. */
  private static long offsets(Span span) {
    return ((long) span.getStart() << 32) | span.getEnd();
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(GeocodeAnnotator.LOCATIONS);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(CONTAINMENT);
  }
}
