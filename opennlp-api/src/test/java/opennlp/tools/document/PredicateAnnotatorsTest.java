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

import org.junit.jupiter.api.Test;

import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the two predicate-driven wrappers: the conditional keeps the layer contract of a
 * skipped delegate by providing its layers empty, and the filter writes survivors to a
 * distinct layer while the source stays untouched.
 */
public class PredicateAnnotatorsTest {

  private static final LayerKey<String> WORDS = Layers.key("test.words", String.class);
  private static final LayerKey<String> LONG_WORDS =
      Layers.key("test.words.long", String.class);

  /** A minimal producing annotator: every whitespace-free character run is a word. */
  private static final DocumentAnnotator PRODUCER = new DocumentAnnotator() {
    @Override
    public Document annotate(Document document) {
      final String text = document.text().toString();
      final java.util.ArrayList<Annotation<String>> words = new java.util.ArrayList<>();
      int start = -1;
      for (int i = 0; i <= text.length(); i++) {
        final boolean boundary = i == text.length() || text.charAt(i) == ' ';
        if (!boundary && start < 0) {
          start = i;
        } else if (boundary && start >= 0) {
          words.add(new Annotation<>(new Span(start, i), text.substring(start, i)));
          start = -1;
        }
      }
      return document.with(WORDS, words);
    }

    @Override
    public Set<LayerKey<?>> provides() {
      return Set.of(WORDS);
    }
  };

  @Test
  void testConditionRunsTheDelegate() {
    final Document document = new ConditionalAnnotator(d -> d.text().length() > 3,
        PRODUCER).annotate(Document.of("two words"));
    assertEquals(2, document.get(WORDS).size());
  }

  @Test
  void testSkippedDelegateStillProvidesItsLayersEmpty() {
    final ConditionalAnnotator guarded =
        new ConditionalAnnotator(d -> false, PRODUCER);
    final Document document = guarded.annotate(Document.of("two words"));
    assertTrue(document.layers().contains(WORDS),
        "a skipped delegate must still provide its layer so downstream requires hold");
    assertEquals(List.of(), document.get(WORDS));
    assertEquals(PRODUCER.provides(), guarded.provides());
    assertEquals(PRODUCER.requires(), guarded.requires());
  }

  @Test
  void testFilterWritesSurvivorsAndKeepsTheSource() {
    final Document produced = PRODUCER.annotate(Document.of("a lengthy pair of words"));
    final FilterAnnotator<String> filter = new FilterAnnotator<>(WORDS, LONG_WORDS,
        annotation -> annotation.value().length() > 4);
    final Document filtered = filter.annotate(produced);
    assertEquals(5, filtered.get(WORDS).size(), "the source layer stays untouched");
    assertEquals(List.of("lengthy", "words"),
        filtered.get(LONG_WORDS).stream().map(Annotation::value).toList());
    assertEquals(Set.of(WORDS), filter.requires());
    assertEquals(Set.of(LONG_WORDS), filter.provides());
  }

  @Test
  void testFilterOnAnEmptySourceProvidesAnEmptyTarget() {
    final Document produced = PRODUCER.annotate(Document.of(""));
    final Document filtered = new FilterAnnotator<>(WORDS, LONG_WORDS, a -> true)
        .annotate(produced);
    assertTrue(filtered.layers().contains(LONG_WORDS));
    assertEquals(List.of(), filtered.get(LONG_WORDS));
  }

  @Test
  void testRejectsContractViolations() {
    assertThrows(IllegalArgumentException.class,
        () -> new ConditionalAnnotator(null, PRODUCER));
    assertThrows(IllegalArgumentException.class,
        () -> new ConditionalAnnotator(d -> true, null));
    assertThrows(IllegalArgumentException.class,
        () -> new ConditionalAnnotator(d -> true, PRODUCER).annotate(null));
    assertThrows(IllegalArgumentException.class,
        () -> new FilterAnnotator<>(null, LONG_WORDS, a -> true));
    assertThrows(IllegalArgumentException.class,
        () -> new FilterAnnotator<>(WORDS, null, a -> true));
    assertThrows(IllegalArgumentException.class,
        () -> new FilterAnnotator<>(WORDS, WORDS, a -> true),
        "equal source and target must fail loud, a document rejects duplicate layers");
    assertThrows(IllegalArgumentException.class,
        () -> new FilterAnnotator<>(WORDS, LONG_WORDS, null));
    assertThrows(IllegalArgumentException.class,
        () -> new FilterAnnotator<>(WORDS, LONG_WORDS, a -> true).annotate(null));
  }
}
