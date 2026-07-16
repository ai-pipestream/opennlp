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

package opennlp.tools.glossary;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;

/**
 * Adapts a {@link GlossaryMatcher} to the document pipeline: scans the document text and
 * provides {@link #GLOSSARY}, one annotation per hit carrying its {@link GlossaryMatch}.
 *
 * <p>The matcher works on the raw text, so this annotator requires no other layer and
 * can run anywhere in a pipeline.</p>
 *
 * @since 3.0.0
 */
public class GlossaryAnnotator implements DocumentAnnotator {

  /**
   * Glossary hits; each annotation covers one hit and carries its {@link GlossaryMatch}.
   */
  public static final LayerKey<GlossaryMatch> GLOSSARY =
      LayerKey.of("glossary", GlossaryMatch.class);

  /** The matcher that produces the hits this annotator records as a layer. */
  private final GlossaryMatcher matcher;

  /**
   * Initializes the adapter.
   *
   * @param matcher The matcher to delegate to. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code matcher} is {@code null}.
   */
  public GlossaryAnnotator(GlossaryMatcher matcher) {
    if (matcher == null) {
      throw new IllegalArgumentException("matcher must not be null");
    }
    this.matcher = matcher;
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final List<Annotation<GlossaryMatch>> hits = new ArrayList<>();
    for (final GlossaryMatch match : matcher.match(document.text())) {
      hits.add(new Annotation<>(match.span(), match));
    }
    return document.with(GLOSSARY, hits);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(GLOSSARY);
  }
}
