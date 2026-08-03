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

import org.junit.jupiter.api.Test;

import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Runs the manual's predicate-annotator examples (docbkx {@code document.xml}, section
 * {@code tools.document.predicates}) verbatim over the chapter's token-length fixture:
 * every value the section states is asserted here.
 */
public class PredicateManualExampleTest {

  /** The chapter's custom layer, as its custom-annotator section declares it. */
  private static final LayerKey<Integer> TOKEN_LENGTHS =
      LayerKey.of("token-lengths", Integer.class);

  /** The chapter's example text, whose token lengths are 3, 3, 6, 2, 5. */
  private static final String TEXT = "The dog barks. It naps.";

  /**
   * A minimal producer of the chapter's token-length layer: space-delimited tokens,
   * one integer annotation per token.
   */
  private static final DocumentAnnotator LENGTHS = new DocumentAnnotator() {
    @Override
    public Document annotate(Document document) {
      final String text = document.text().toString();
      final List<Annotation<Integer>> lengths = new ArrayList<>();
      int start = -1;
      for (int i = 0; i <= text.length(); i++) {
        final boolean boundary = i == text.length() || text.charAt(i) == ' ';
        if (boundary && start >= 0) {
          lengths.add(new Annotation<>(new Span(start, i), i - start));
          start = -1;
        } else if (!boundary && start < 0) {
          start = i;
        }
      }
      return document.with(TOKEN_LENGTHS, lengths);
    }

    @Override
    public Set<LayerKey<?>> provides() {
      return Set.of(TOKEN_LENGTHS);
    }
  };

  /** The filter example: LONG_ONES holds 6 and 5, the source keeps all five. */
  @Test
  void testFilterExampleStatesTheSurvivors() {
    final LayerKey<Integer> longOnes = LayerKey.of("token-lengths-long", Integer.class);
    final DocumentAnnotator filter =
        new FilterAnnotator<>(TOKEN_LENGTHS, longOnes, a -> a.value() >= 5);
    final Document document = filter.annotate(LENGTHS.annotate(Document.of(TEXT)));
    assertEquals(List.of(6, 5),
        document.get(longOnes).stream().map(Annotation::value).toList());
    assertEquals(5, document.get(TOKEN_LENGTHS).size());
  }

  /** The conditional example: a short document passes with the layer empty. */
  @Test
  void testConditionalExampleKeepsTheContractOnShortDocuments() {
    final DocumentAnnotator guarded =
        new ConditionalAnnotator(d -> d.text().length() >= 20, LENGTHS);
    final Document ran = guarded.annotate(Document.of(TEXT));
    assertEquals(5, ran.get(TOKEN_LENGTHS).size());

    final Document skipped = guarded.annotate(Document.of("It naps."));
    assertTrue(skipped.layers().contains(TOKEN_LENGTHS));
    assertEquals(List.of(), skipped.get(TOKEN_LENGTHS));
  }
}
