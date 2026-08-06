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

package opennlp.tools.embeddings;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.util.Span;

/**
 * Adds dense vectors to the document graph: every annotation of a source layer is
 * embedded through a {@link TextEmbedder} and provided as a parallel layer of
 * {@code float[]} values on the same spans.
 *
 * <p>The provided layer's identifier is the source layer's identifier prefixed with
 * {@code "embeddings:"}, for example {@code embeddings:opennlp:tokens} for a source
 * layer with the identifier {@code opennlp:tokens}, mirroring the {@code gold:} prefix
 * convention. Because the identifier is derived from the source, one
 * pipeline can carry token and sentence embeddings side by side by adding two instances
 * over different source layers; read each instance's provided layer through
 * {@link #layer()}.</p>
 *
 * <p>The embedder receives the covered text of each source annotation, taken from
 * {@link Document#text()} by the annotation's span; the annotation's stored value is
 * never consulted. Every resulting vector is anchored to the span it was computed from,
 * so the provided layer stays in original text coordinates.</p>
 *
 * <p>Instances are immutable, so one instance may serve concurrent pipelines whenever the
 * supplied {@link TextEmbedder} is safe for concurrent use.</p>
 *
 * @since 3.0.0
 */
public class EmbeddingAnnotator implements DocumentAnnotator {

  private final TextEmbedder embedder;
  private final LayerKey<String> source;
  private final LayerKey<float[]> layer;

  /**
   * Initializes the annotator and derives the provided layer from the source layer by
   * prefixing the source layer's identifier with {@code "embeddings:"}.
   *
   * @param embedder The embedder to delegate to. Must not be {@code null}.
   * @param source The layer whose annotations are embedded, for example the token or
   *               sentence layer. Must not be {@code null} and must be
   *               {@link LayerKey.Scope#POSITIONAL positional}, because every annotation
   *               is embedded by its span.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null} or if
   *         {@code source} is not positional.
   */
  public EmbeddingAnnotator(TextEmbedder embedder, LayerKey<String> source) {
    if (embedder == null || source == null) {
      throw new IllegalArgumentException("embedder and source must not be null");
    }
    if (source.scope() != LayerKey.Scope.POSITIONAL) {
      throw new IllegalArgumentException("source must be a positional layer: " + source);
    }
    this.embedder = embedder;
    this.source = source;
    this.layer = LayerKey.of("embeddings:" + source.id(), float[].class);
  }

  /**
   * @return The layer this instance provides: one vector per source annotation, on the
   *         source annotation's span. The layer's identifier is the source layer's
   *         identifier prefixed with {@code "embeddings:"}. Never {@code null}.
   */
  public LayerKey<float[]> layer() {
    return layer;
  }

  /**
   * Embeds the covered text of every annotation of the source layer and returns a new
   * document that additionally carries the resulting vectors under {@link #layer()}.
   *
   * <p>The source layer must be present, but it may be empty: a document with an empty
   * source layer yields a present-but-empty vector layer. Every vector is stored exactly
   * as the embedder returned it; no dimension check is applied.</p>
   *
   * <p>The embedder is invoked once per document through
   * {@link TextEmbedder#embedAll(List)} over the <em>distinct</em> covered texts, so an
   * embedder with a batched execution path batches, and spans that cover the same text
   * (common on a token layer) share one vector instance rather than embedding twice.
   * Sharing is safe because the stored vectors are treated as read-only values.</p>
   *
   * @param document The document to annotate. Must not be {@code null}, must carry the
   *                 source layer, and must not already carry the provided layer.
   * @return A new {@link Document} with the vector layer added to the input layers.
   *         Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, if the
   *         source layer is absent, if the document already carries the provided layer,
   *         if the embedder returns {@code null} for an annotation's covered text, or if
   *         the batch does not hold exactly one vector per distinct text.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    if (!document.layers().contains(source)) {
      throw new IllegalArgumentException("document lacks the required layer " + source);
    }
    final CharSequence text = document.text();
    final List<Annotation<String>> annotations = document.get(source);
    if (annotations.isEmpty()) {
      return document.with(layer, List.of());
    }
    // One batch call over the distinct covered texts: an embedder with a real batch
    // path executes it once per document, and a covered text that repeats across spans
    // (common on a token layer) is embedded once and its vector shared by every span
    // that covers it.
    final Map<String, Integer> distinctIndex = new HashMap<>();
    final List<String> distinct = new ArrayList<>();
    final int[] route = new int[annotations.size()];
    for (int i = 0; i < annotations.size(); i++) {
      final Span span = annotations.get(i).span();
      // The covered text of the span, never the annotation's stored value.
      final String covered = text.subSequence(span.getStart(), span.getEnd()).toString();
      final Integer seen = distinctIndex.putIfAbsent(covered, distinct.size());
      if (seen == null) {
        distinct.add(covered);
        route[i] = distinct.size() - 1;
      } else {
        route[i] = seen;
      }
    }
    final float[][] embedded = embedder.embedAll(distinct);
    if (embedded.length != distinct.size()) {
      throw new IllegalArgumentException("embedder returned " + embedded.length
          + " vectors for " + distinct.size() + " texts; embedAll must return one vector "
          + "per input, in input order");
    }
    final List<Annotation<float[]>> vectors = new ArrayList<>(annotations.size());
    for (int i = 0; i < annotations.size(); i++) {
      vectors.add(new Annotation<>(annotations.get(i).span(), embedded[route[i]]));
    }
    return document.with(layer, vectors);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The single-element set holding the source layer given at construction.</p>
   */
  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(source);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The single-element set holding {@link #layer()}.</p>
   */
  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(layer);
  }
}
