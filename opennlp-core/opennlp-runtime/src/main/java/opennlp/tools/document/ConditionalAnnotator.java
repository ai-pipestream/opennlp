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

import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Runs a delegate annotator only when a condition over the document holds. When the
 * condition does not hold, the delegate's provided layers are still added, empty, so
 * the document satisfies every downstream {@link DocumentAnnotator#requires()} exactly
 * as if the delegate had run and found nothing.
 *
 * <p>This annotator is safe for concurrent use when its delegate and condition
 * are.</p>
 *
 * @since 3.0.0
 */
public final class ConditionalAnnotator implements DocumentAnnotator {

  /** The message prefix of every absent-required-layer rejection in this wrapper. */
  private static final String MISSING_LAYER = "document lacks the required layer ";

  private final Predicate<Document> condition;
  private final DocumentAnnotator delegate;

  /**
   * Initializes a {@link ConditionalAnnotator}.
   *
   * @param condition The condition deciding whether the delegate runs. Must not be
   *                  {@code null}.
   * @param delegate The annotator to run when the condition holds. Must not be
   *                 {@code null}.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null}.
   */
  public ConditionalAnnotator(Predicate<Document> condition, DocumentAnnotator delegate) {
    if (condition == null) {
      throw new IllegalArgumentException("condition must not be null");
    }
    if (delegate == null) {
      throw new IllegalArgumentException("delegate must not be null");
    }
    this.condition = condition;
    this.delegate = delegate;
  }

  /**
   * Tests the condition and either runs the delegate or adds every layer the delegate
   * provides as an empty layer. The delegate's required layers must be present in both
   * cases.
   *
   * @param document The document to annotate. Must not be {@code null} and must carry
   *                 every layer the delegate requires.
   * @return A new {@link Document} carrying the delegate's provided layers, populated
   *         when the condition held and empty otherwise. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null} or
   *         lacks a layer the delegate requires.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final Set<LayerKey<?>> present = document.layers();
    for (final LayerKey<?> required : delegate.requires()) {
      if (!present.contains(required)) {
        throw new IllegalArgumentException(MISSING_LAYER + required);
      }
    }
    if (condition.test(document)) {
      return delegate.annotate(document);
    }
    Document result = document;
    for (final LayerKey<?> provided : delegate.provides()) {
      result = withEmpty(result, provided);
    }
    return result;
  }

  /**
   * Adds one layer, present but empty.
   *
   * @param document The document.
   * @param layer The layer to add empty.
   * @param <T> The layer's value type.
   * @return The document with the layer present and empty.
   */
  private <T> Document withEmpty(Document document, LayerKey<T> layer) {
    return document.with(layer, List.of());
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return delegate.requires();
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return delegate.provides();
  }
}
