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

package opennlp.tools.temporal;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;

/**
 * Adapts a {@link TemporalExtractor} to the document pipeline: scans the document text
 * and provides {@link #TEMPORALS}, one annotation per mention carrying its
 * {@link TemporalExpression}.
 *
 * <p>The extractor works on the raw text, so this annotator requires no other layer and
 * can run anywhere in a pipeline.</p>
 *
 * @since 3.0.0
 */
public class TemporalAnnotator implements DocumentAnnotator {

  /**
   * Temporal mentions; each annotation covers one mention and carries its normalized
   * {@link TemporalExpression}.
   */
  public static final LayerKey<TemporalExpression> TEMPORALS =
      Layers.key("temporals", TemporalExpression.class);

  private final TemporalExtractor extractor;

  /**
   * Initializes the adapter.
   *
   * @param extractor The extractor to delegate to. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code extractor} is {@code null}.
   */
  public TemporalAnnotator(TemporalExtractor extractor) {
    if (extractor == null) {
      throw new IllegalArgumentException("extractor must not be null");
    }
    this.extractor = extractor;
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final List<Annotation<TemporalExpression>> mentions = new ArrayList<>();
    for (final TemporalExpression expression : extractor.extract(document.text())) {
      mentions.add(new Annotation<>(expression.span(), expression));
    }
    return document.with(TEMPORALS, mentions);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(TEMPORALS);
  }
}
