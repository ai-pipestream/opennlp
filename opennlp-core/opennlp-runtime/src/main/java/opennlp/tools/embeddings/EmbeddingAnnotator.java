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
 * <p>The provided layer's identifier is derived from the source layer, so one pipeline
 * can carry token and sentence embeddings side by side by adding two instances over
 * different source layers; read the layer through {@link #layer()}. The embedder sees
 * the covered text of each annotation, so spans stay in original text coordinates.</p>
 *
 * <p>The annotator holds no per-call state; it is as thread-safe as its embedder.</p>
 *
 * @since 3.0.0
 */
public class EmbeddingAnnotator implements DocumentAnnotator {

  private final TextEmbedder embedder;
  private final LayerKey<String> source;
  private final LayerKey<float[]> layer;

  /**
   * Initializes the annotator.
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
   *         source annotation's span. Never {@code null}.
   */
  public LayerKey<float[]> layer() {
    return layer;
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final CharSequence text = document.text();
    final List<Annotation<String>> annotations = document.get(source);
    final List<Annotation<float[]>> vectors = new ArrayList<>(annotations.size());
    for (final Annotation<String> annotation : annotations) {
      vectors.add(new Annotation<>(annotation.span(), embedder.embed(
          text.subSequence(annotation.span().getStart(), annotation.span().getEnd()))));
    }
    return document.with(layer, vectors);
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(source);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(layer);
  }
}
