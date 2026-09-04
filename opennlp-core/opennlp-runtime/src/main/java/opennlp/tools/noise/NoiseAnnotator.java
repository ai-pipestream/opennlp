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

package opennlp.tools.noise;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import opennlp.tools.assets.AssetAnnotator;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.DocumentAnnotators;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;

/**
 * Adapts a {@link NoiseScorer} to the document pipeline: scores the document text and
 * provides {@link #NOISE}, one annotation per finding carrying its {@link NoiseSpan}.
 *
 * <p>In its default mode this annotator requires {@link AssetAnnotator#ASSETS} and
 * excludes those spans from scoring, so an embedded binary another detector already
 * explained is never reported a second time as noise; a document without that layer
 * is rejected loudly. The standalone mode skips that requirement and scores the whole
 * text; what this annotator requires therefore depends on the mode it was built
 * with.</p>
 *
 * @since 3.0.0
 */
public class NoiseAnnotator implements DocumentAnnotator {

  /**
   * Noisy stretches of the text; each annotation covers one finding and carries its
   * {@link NoiseSpan}.
   */
  public static final LayerKey<NoiseSpan> NOISE = Layers.key("noise", NoiseSpan.class);

  private final NoiseScorer scorer;
  private final boolean excludeAssets;

  /**
   * Initializes the adapter in the default mode: the built-in
   * {@link StructuralNoiseScorer} without a dictionary, excluding detected assets.
   */
  public NoiseAnnotator() {
    this(new StructuralNoiseScorer(), true);
  }

  /**
   * Initializes the adapter.
   *
   * @param scorer The scorer to delegate to. Must not be {@code null}.
   * @param excludeAssets {@code true} to require {@link AssetAnnotator#ASSETS} and
   *                      exclude those spans from scoring; {@code false} to run
   *                      standalone over the whole text.
   * @throws IllegalArgumentException Thrown if {@code scorer} is {@code null}.
   */
  public NoiseAnnotator(NoiseScorer scorer, boolean excludeAssets) {
    if (scorer == null) {
      throw new IllegalArgumentException("scorer must not be null");
    }
    this.scorer = scorer;
    this.excludeAssets = excludeAssets;
  }

  /**
   * Scores the document text and adds the {@link #NOISE} layer. The asset-excluding
   * mode enforces {@link #requires()} before scoring, so a pipeline that forgot the
   * asset detector fails loud instead of silently scoring every embedded payload as
   * noise.
   *
   * @param document The document to annotate. Must not be {@code null}; in the
   *                 asset-excluding mode it must carry {@link AssetAnnotator#ASSETS}.
   * @return A new {@link Document} with the {@link #NOISE} layer added. Never
   *         {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, or
   *         if the annotator excludes assets and the document lacks
   *         {@link AssetAnnotator#ASSETS}.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final List<Span> exclude = new ArrayList<>();
    if (excludeAssets) {
      DocumentAnnotators.requireLayers(document, AssetAnnotator.ASSETS);
      for (final Annotation<?> asset : document.get(AssetAnnotator.ASSETS)) {
        exclude.add(asset.span());
      }
    }
    final List<Annotation<NoiseSpan>> found = new ArrayList<>();
    for (final NoiseSpan noise : scorer.score(document.text(), exclude)) {
      found.add(new Annotation<>(noise.span(), noise));
    }
    return document.with(NOISE, found);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The default mode requires {@link AssetAnnotator#ASSETS}; the standalone mode
   * requires nothing.</p>
   */
  @Override
  public Set<LayerKey<?>> requires() {
    return excludeAssets ? Set.of(AssetAnnotator.ASSETS) : Set.of();
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(NOISE);
  }
}
