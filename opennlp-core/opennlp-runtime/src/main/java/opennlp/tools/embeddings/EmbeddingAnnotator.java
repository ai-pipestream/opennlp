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
import java.util.List;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;

/**
 * Adds dense vectors to the document graph: every annotation of a source layer is
 * embedded through a {@link TextEmbedder} and provided as a parallel layer of
 * {@code float[]} values on the same spans.
 *
 * <p>The provided layer's identifier is the source layer's identifier prefixed with
 * {@code "embeddings:"}, for example {@code embeddings:tokens} for a source layer with
 * the identifier {@code tokens}. Because the identifier is derived from the source, one
 * pipeline can carry token and sentence embeddings side by side by adding two instances
 * over different source layers; read each instance's provided layer through
 * {@link #layer()}.</p>
 *
 * <p>The embedder receives the covered text of each source annotation, taken from
 * {@link Document#text()} by the annotation's span; the annotation's stored value is
 * never consulted. Every resulting vector is anchored to the span it was computed from,
 * so the provided layer stays in original text coordinates.</p>
 *
 * <p>All fields of this class are immutable and {@link #annotate(Document)} keeps its
 * working state in local variables, so a single instance may serve concurrent pipelines
 * whenever the supplied {@link TextEmbedder} is safe for concurrent use.</p>
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
   *               sentence layer. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null}.
   */
  public EmbeddingAnnotator(TextEmbedder embedder, LayerKey<String> source) {
    if (embedder == null || source == null) {
      throw new IllegalArgumentException("embedder and source must not be null");
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
   * <p>A document without the source layer reads as an empty source layer, so the
   * returned document then carries the provided layer with no annotations. Every vector
   * is stored exactly as the embedder returned it; no dimension check is applied.</p>
   *
   * @param document The document to annotate. Must not be {@code null} and must not
   *                 already carry the provided layer.
   * @return A new {@link Document} with the vector layer added to the input layers.
   *         Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, if it
   *         already carries the provided layer, or if the embedder returns {@code null}
   *         for an annotation's covered text.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final CharSequence text = document.text();
    final List<Annotation<String>> annotations = document.get(source);
    final List<Annotation<float[]>> vectors = new ArrayList<>(annotations.size());
    for (final Annotation<String> annotation : annotations) {
      // Embed the covered text taken from the original document text by the span; the
      // annotation's stored value is deliberately not used, so the vector always
      // reflects the exact characters the span points at.
      vectors.add(new Annotation<>(annotation.span(), embedder.embed(
          text.subSequence(annotation.span().getStart(), annotation.span().getEnd()))));
    }
    return document.with(layer, vectors);
  }

  /**
   * @return The single-element set holding the source layer this annotator reads.
   *         Never {@code null}.
   */
  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(source);
  }

  /**
   * @return The single-element set holding the vector layer this annotator adds, equal
   *         to {@link #layer()}. Never {@code null}.
   */
  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(layer);
  }
}
