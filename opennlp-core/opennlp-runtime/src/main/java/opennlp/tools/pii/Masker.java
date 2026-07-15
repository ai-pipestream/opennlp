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

package opennlp.tools.pii;

import java.util.Collection;
import java.util.List;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.LayerKey;

/**
 * Redacts annotated spans from a document's text. Works with any span layer, not only
 * {@link PiiAnnotator#PII}: entities, glossary hits, or custom layers redact the same
 * way.
 *
 * <p>Masking is length preserving: every character inside a masked span is replaced by
 * the mask character and no character is inserted or removed, so the spans of every
 * other layer remain valid for the masked text.</p>
 *
 * @since 3.0.0
 */
public final class Masker {

  private Masker() {
  }

  /**
   * Masks the spans of one layer.
   *
   * @param document The document to redact. Must not be {@code null}.
   * @param layer The layer whose spans are masked. Must not be {@code null} and must be
   *              present on the document.
   * @param mask The replacement character.
   * @return The document text with every annotated span masked. Never {@code null};
   *         always the same length as the document text.
   * @throws IllegalArgumentException Thrown if {@code document} or {@code layer} is
   *         {@code null}, or the layer is not present on the document.
   */
  public static String mask(Document document, LayerKey<?> layer, char mask) {
    if (layer == null) {
      throw new IllegalArgumentException("layer must not be null");
    }
    return mask(document, List.of(layer), mask);
  }

  /**
   * Masks the spans of several layers at once.
   *
   * @param document The document to redact. Must not be {@code null}.
   * @param layers The layers whose spans are masked. Must not be {@code null} or empty,
   *               no layer may be {@code null}, and every layer must be present on the
   *               document.
   * @param mask The replacement character.
   * @return The document text with every annotated span masked. Never {@code null};
   *         always the same length as the document text.
   * @throws IllegalArgumentException Thrown if {@code document} or {@code layers} is
   *         {@code null}, {@code layers} is empty or contains {@code null}, or a layer
   *         is not present on the document.
   */
  public static String mask(Document document, Collection<LayerKey<?>> layers, char mask) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    if (layers == null || layers.isEmpty()) {
      throw new IllegalArgumentException("layers must not be null or empty");
    }
    final StringBuilder masked = new StringBuilder(document.text());
    for (final LayerKey<?> layer : layers) {
      if (layer == null) {
        throw new IllegalArgumentException("layers must not contain null");
      }
      if (!document.layers().contains(layer)) {
        throw new IllegalArgumentException("layer is not present on the document: " + layer);
      }
      for (final Annotation<?> annotation : document.get(layer)) {
        for (int i = annotation.span().getStart(); i < annotation.span().getEnd(); i++) {
          masked.setCharAt(i, mask);
        }
      }
    }
    return masked.toString();
  }
}
