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

import java.util.ArrayList;
import java.util.List;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;

/**
 * Reads the mentions of a document's {@link PiiAnnotator#PII} layer, with the checks every
 * document-taking method in this package promises.
 */
final class PiiLayer {

  private PiiLayer() {
    // This class holds one static reader only and is never instantiated.
  }

  /**
   * Reads the PII mentions of a document.
   *
   * @param document The document. Must not be {@code null} and must carry the layer.
   * @return The mentions in the order they are annotated. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null} or does
   *         not carry the PII layer.
   */
  static List<PiiMention> mentions(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    if (!document.layers().contains(PiiAnnotator.PII)) {
      throw new IllegalArgumentException("document does not carry the "
          + PiiAnnotator.PII.id() + " layer");
    }
    final List<Annotation<PiiMention>> annotations = document.get(PiiAnnotator.PII);
    final List<PiiMention> mentions = new ArrayList<>(annotations.size());
    for (final Annotation<PiiMention> annotation : annotations) {
      mentions.add(annotation.value());
    }
    return mentions;
  }
}
