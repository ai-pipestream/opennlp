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

package opennlp.tools.artifacts;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;

/**
 * Adapts an {@link ArtifactDetector} to the document pipeline: scans the document text
 * and provides {@link #ARTIFACTS}, one annotation per finding carrying its
 * {@link TextArtifact}.
 *
 * <p>The detector works on the raw text, so this annotator requires no other layer and
 * can run anywhere in a pipeline. Detection never modifies the text: downstream
 * consumers decide whether to mask, drop, or repair the flagged spans.</p>
 *
 * @since 3.0.0
 */
public class ArtifactAnnotator implements DocumentAnnotator {

  /**
   * Damaged or suspicious character sequences; each annotation covers one finding and
   * carries its {@link TextArtifact}.
   */
  public static final LayerKey<TextArtifact> ARTIFACTS =
      Layers.key("artifacts", TextArtifact.class);

  private final ArtifactDetector detector;

  /** Initializes the adapter with the built-in {@link CursorArtifactDetector}. */
  public ArtifactAnnotator() {
    this(new CursorArtifactDetector());
  }

  /**
   * Initializes the adapter.
   *
   * @param detector The detector to delegate to. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code detector} is {@code null}.
   */
  public ArtifactAnnotator(ArtifactDetector detector) {
    if (detector == null) {
      throw new IllegalArgumentException("detector must not be null");
    }
    this.detector = detector;
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final List<Annotation<TextArtifact>> found = new ArrayList<>();
    for (final TextArtifact artifact : detector.detect(document.text())) {
      found.add(new Annotation<>(artifact.span(), artifact));
    }
    return document.with(ARTIFACTS, found);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(ARTIFACTS);
  }
}
