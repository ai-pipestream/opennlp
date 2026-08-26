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

package opennlp.tools.document;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Writes the annotations of one layer that pass a predicate to a second layer. The
 * source layer stays untouched, so a pipeline keeps both the raw and the filtered view
 * and every consumer states which one it reads.
 *
 * <p>This annotator is safe for concurrent use when its predicate is.</p>
 *
 * @param <T> The value type of the filtered layer.
 *
 * @since 3.0.0
 */
public final class FilterAnnotator<T> implements DocumentAnnotator {

  /** The message prefix of every absent-required-layer rejection in this wrapper. */
  private static final String MISSING_LAYER = "document lacks the required layer ";

  private final LayerKey<T> source;
  private final LayerKey<T> target;
  private final Predicate<Annotation<T>> keep;

  /**
   * Initializes a {@link FilterAnnotator}.
   *
   * @param source The layer to read. Must not be {@code null}.
   * @param target The layer to write survivors to. Must not be {@code null} and must
   *               differ from {@code source}, because a document rejects a duplicate
   *               layer.
   * @param keep The predicate an annotation must pass to survive. Must not be
   *             {@code null}.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null} or the two
   *         layers are equal.
   */
  public FilterAnnotator(LayerKey<T> source, LayerKey<T> target,
      Predicate<Annotation<T>> keep) {
    if (source == null) {
      throw new IllegalArgumentException("source must not be null");
    }
    if (target == null) {
      throw new IllegalArgumentException("target must not be null");
    }
    if (source.equals(target)) {
      throw new IllegalArgumentException("target must differ from source");
    }
    if (keep == null) {
      throw new IllegalArgumentException("keep must not be null");
    }
    this.source = source;
    this.target = target;
    this.keep = keep;
  }

  /**
   * Reads the source layer and adds the target layer holding the annotations that pass
   * the predicate, in their source order. The source layer must be present, but it may
   * be empty: an empty source yields a present-but-empty target.
   *
   * @param document The document to annotate. Must not be {@code null} and must carry
   *                 the source layer.
   * @return A new {@link Document} with the target layer added. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null} or the
   *         source layer is absent.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    if (!document.layers().contains(source)) {
      throw new IllegalArgumentException(MISSING_LAYER + source);
    }
    final List<Annotation<T>> survivors = new ArrayList<>();
    for (final Annotation<T> annotation : document.get(source)) {
      if (keep.test(annotation)) {
        survivors.add(annotation);
      }
    }
    return document.with(target, survivors);
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(source);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(target);
  }
}
