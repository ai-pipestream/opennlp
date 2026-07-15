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

package opennlp.tools.coref;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;

public class CorefAnnotatorTest {

  /** Builds token annotations by locating each token left to right in the text. */
  private static List<Annotation<String>> tokens(String text, String... forms) {
    final List<Annotation<String>> annotations = new ArrayList<>(forms.length);
    int cursor = 0;
    for (final String form : forms) {
      final int start = text.indexOf(form, cursor);
      annotations.add(new Annotation<>(new Span(start, start + form.length()), form));
      cursor = start + form.length();
    }
    return annotations;
  }

  private static List<Annotation<String>> values(List<Annotation<String>> tokens,
      String... tags) {
    final List<Annotation<String>> annotations = new ArrayList<>(tags.length);
    for (int i = 0; i < tags.length; i++) {
      annotations.add(new Annotation<>(tokens.get(i).span(), tags[i]));
    }
    return annotations;
  }

  private static Document storyDocument() {
    final String text = "Mary Jones leads Acme Corp. She joined Acme in 2020. It thrived.";
    final List<Annotation<String>> toks = tokens(text,
        "Mary", "Jones", "leads", "Acme", "Corp", ".",
        "She", "joined", "Acme", "in", "2020", ".",
        "It", "thrived", ".");
    return Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 27), "s"),
            new Annotation<>(new Span(28, 52), "s"),
            new Annotation<>(new Span(53, 64), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "PROPN", "PROPN", "VERB", "PROPN", "PROPN", "PUNCT",
            "PRON", "VERB", "PROPN", "ADP", "NUM", "PUNCT",
            "PRON", "VERB", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 10), "person"),
            new Annotation<>(new Span(17, 26), "organization"),
            new Annotation<>(new Span(39, 43), "organization"),
            new Annotation<>(new Span(47, 51), "date")));
  }

  @Test
  void testSievesLinkNamesAndPronouns() {
    final Document document = new CorefAnnotator().annotate(storyDocument());

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(6, chains.size());
    final int[] chainIds = chains.stream().mapToInt(a -> a.value().chain()).toArray();
    Assertions.assertArrayEquals(new int[] {0, 1, 0, 1, 2, 1}, chainIds);
    Assertions.assertEquals(CorefMention.KIND_ENTITY, chains.get(0).value().kind());
    Assertions.assertEquals(CorefMention.KIND_PRONOUN, chains.get(2).value().kind());
    Assertions.assertEquals(0, chains.get(0).value().entity());
    Assertions.assertEquals(CorefMention.NO_ENTITY, chains.get(2).value().entity());
    Assertions.assertEquals("She", document.text().subSequence(
        chains.get(2).span().getStart(), chains.get(2).span().getEnd()).toString());
    Assertions.assertEquals("It", document.text().subSequence(
        chains.get(5).span().getStart(), chains.get(5).span().getEnd()).toString());
  }

  @Test
  void testSameTextDifferentTypeStaysApart() {
    final String text = "Paris met Paris.";
    final List<Annotation<String>> toks = tokens(text, "Paris", "met", "Paris", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 16), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks, "PROPN", "VERB", "PROPN", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 5), "person"),
            new Annotation<>(new Span(10, 15), "location"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(1, chains.get(1).value().chain());
  }

  @Test
  void testPronounWindowLimitsResolution() {
    final String text = "Acme Corp won. Time passed. Markets moved. It changed.";
    final List<Annotation<String>> toks = tokens(text,
        "Acme", "Corp", "won", ".", "Time", "passed", ".",
        "Markets", "moved", ".", "It", "changed", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 14), "s"),
            new Annotation<>(new Span(15, 27), "s"),
            new Annotation<>(new Span(28, 42), "s"),
            new Annotation<>(new Span(43, 54), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "PROPN", "PROPN", "VERB", "PUNCT", "NOUN", "VERB", "PUNCT",
            "NOUN", "VERB", "PUNCT", "PRON", "VERB", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 9), "organization"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(1, chains.get(1).value().chain());
  }

  @Test
  void testGenderedPronounSkipsNonPersonEntities() {
    final String text = "Acme Corp hired John. He started.";
    final List<Annotation<String>> toks = tokens(text,
        "Acme", "Corp", "hired", "John", ".", "He", "started", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 21), "s"),
            new Annotation<>(new Span(22, 33), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "NNP", "NNP", "VBD", "NNP", ".", "PRP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 9), "organization"),
            new Annotation<>(new Span(16, 20), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(3, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(1, chains.get(1).value().chain());
    Assertions.assertEquals(1, chains.get(2).value().chain());
    Assertions.assertEquals(CorefMention.KIND_PRONOUN, chains.get(2).value().kind());
  }

  @Test
  void testInvalidArguments() {
    final CorefAnnotator annotator = new CorefAnnotator();
    Assertions.assertThrows(IllegalArgumentException.class, () -> annotator.annotate(null));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CorefAnnotator(Set.of(), Set.of("organization")));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CorefAnnotator(null, Set.of("organization")));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CorefMention(-1, CorefMention.KIND_ENTITY, 0));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CorefMention(0, " ", 0));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CorefMention(0, CorefMention.KIND_ENTITY, -2));

    final String text = "a b";
    final List<Annotation<String>> toks = tokens(text, "a", "b");
    final Document misaligned = Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 3), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, List.of())
        .with(Layers.ENTITIES, List.of());
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> annotator.annotate(misaligned));
  }
}
