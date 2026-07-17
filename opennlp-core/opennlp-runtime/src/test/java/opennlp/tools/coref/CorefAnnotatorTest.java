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
import opennlp.tools.document.NameFinderAnnotator;
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

  /**
   * A pronoun opening the document has no preceding mention to link to, so it stays a
   * singleton and, being first in text order, opens chain zero.
   */
  @Test
  void testPronounAtDocumentStartStaysSingleton() {
    final String text = "He waved. John smiled.";
    final List<Annotation<String>> toks = tokens(text,
        "He", "waved", ".", "John", "smiled", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 9), "s"),
            new Annotation<>(new Span(10, 22), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks, "PRP", "VBD", ".", "NNP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(10, 14), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(new Span(0, 2), chains.get(0).span());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(CorefMention.KIND_PRONOUN, chains.get(0).value().kind());
    Assertions.assertEquals(CorefMention.NO_ENTITY, chains.get(0).value().entity());
    Assertions.assertEquals(1, chains.get(1).value().chain());
    Assertions.assertEquals(CorefMention.KIND_ENTITY, chains.get(1).value().kind());
  }

  /**
   * Two identical names of the same type merge into one chain through the exact match
   * sieve, even with several sentences between them; the pronoun window does not apply
   * to name matching.
   */
  @Test
  void testExactNameMatchLinksAcrossDistantSentences() {
    final String text = "Alice arrived. Rain fell. Wind blew. Alice left.";
    final List<Annotation<String>> toks = tokens(text,
        "Alice", "arrived", ".", "Rain", "fell", ".",
        "Wind", "blew", ".", "Alice", "left", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 14), "s"),
            new Annotation<>(new Span(15, 25), "s"),
            new Annotation<>(new Span(26, 36), "s"),
            new Annotation<>(new Span(37, 48), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "NNP", "VBD", ".", "NNP", "VBD", ".",
            "NNP", "VBD", ".", "NNP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 5), "person"),
            new Annotation<>(new Span(37, 42), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(0, chains.get(1).value().chain());
    Assertions.assertEquals(0, chains.get(0).value().entity());
    Assertions.assertEquals(1, chains.get(1).value().entity());
  }

  /**
   * Nested entity mentions are legal layer content: the containment sieve links the
   * inner mention to the outer one it prefixes, and the exact match sieve links the
   * later repetition, so all three mentions form one chain. Mentions sharing a start
   * offset keep their entity layer order.
   */
  @Test
  void testNestedEntityMentionsFormOneChain() {
    final String text = "New York City honored New York.";
    final List<Annotation<String>> toks = tokens(text,
        "New", "York", "City", "honored", "New", "York", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 31), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "NNP", "NNP", "NNP", "VBD", "NNP", "NNP", "."))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 13), "location"),
            new Annotation<>(new Span(0, 8), "location"),
            new Annotation<>(new Span(22, 30), "location"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(3, chains.size());
    Assertions.assertEquals(new Span(0, 13), chains.get(0).span());
    Assertions.assertEquals(new Span(0, 8), chains.get(1).span());
    Assertions.assertEquals(new Span(22, 30), chains.get(2).span());
    for (final Annotation<CorefMention> mention : chains) {
      Assertions.assertEquals(0, mention.value().chain());
    }
  }

  /**
   * The strict head match sieve links two orderings of one name that neither exact
   * match nor containment can relate: the heads match and neither mention carries a
   * content word the other chain lacks.
   */
  @Test
  void testStrictHeadMatchLinksReorderedName() {
    final String text = "The Florida Supreme Court ruled. The Supreme Court of Florida agreed.";
    final List<Annotation<String>> toks = tokens(text,
        "The", "Florida", "Supreme", "Court", "ruled", ".",
        "The", "Supreme", "Court", "of", "Florida", "agreed", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 32), "s"),
            new Annotation<>(new Span(33, 69), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "DT", "NNP", "NNP", "NNP", "VBD", ".",
            "DT", "NNP", "NNP", "IN", "NNP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(4, 25), "organization"),
            new Annotation<>(new Span(37, 61), "organization"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(0, chains.get(1).value().chain());
  }

  /**
   * A shared head word alone never links two names: word inclusion refuses the pair
   * because each mention carries a content word the other chain lacks.
   */
  @Test
  void testStrictHeadMatchRejectsSameHeadWithForeignWords() {
    final String text = "Stanford University welcomed guests. Harvard University declined.";
    final List<Annotation<String>> toks = tokens(text,
        "Stanford", "University", "welcomed", "guests", ".",
        "Harvard", "University", "declined", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 36), "s"),
            new Annotation<>(new Span(37, 65), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "NNP", "NNP", "VBD", "NNS", ".",
            "NNP", "NNP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 19), "organization"),
            new Annotation<>(new Span(37, 55), "organization"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(1, chains.get(1).value().chain());
  }

  /**
   * The strict head match sieve honors the type guard: a head match with word
   * inclusion still never links mentions whose known types differ.
   */
  @Test
  void testStrictHeadMatchHonorsTypeGuard() {
    final String text = "Acme Court expanded. Court is a town.";
    final List<Annotation<String>> toks = tokens(text,
        "Acme", "Court", "expanded", ".", "Court", "is", "a", "town", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 20), "s"),
            new Annotation<>(new Span(21, 37), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "NNP", "NNP", "VBD", ".", "NNP", "VBZ", "DT", "NN", "."))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 10), "organization"),
            new Annotation<>(new Span(21, 26), "location"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(1, chains.get(1).value().chain());
  }

  /**
   * Word inclusion works at chain level: once two orderings of a name share a chain, a
   * third variant whose words all appear in that chain joins it, even though no single
   * earlier mention contains them verbatim.
   */
  @Test
  void testStrictHeadMatchAccumulatesChainWords() {
    final String text = "The Florida Supreme Court ruled. The Supreme Court of Florida agreed."
        + " The Florida Court closed.";
    final List<Annotation<String>> toks = tokens(text,
        "The", "Florida", "Supreme", "Court", "ruled", ".",
        "The", "Supreme", "Court", "of", "Florida", "agreed", ".",
        "The", "Florida", "Court", "closed", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 32), "s"),
            new Annotation<>(new Span(33, 69), "s"),
            new Annotation<>(new Span(70, 95), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "DT", "NNP", "NNP", "NNP", "VBD", ".",
            "DT", "NNP", "NNP", "IN", "NNP", "VBD", ".",
            "DT", "NNP", "NNP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(4, 25), "organization"),
            new Annotation<>(new Span(37, 61), "organization"),
            new Annotation<>(new Span(74, 87), "organization"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(3, chains.size());
    for (final Annotation<CorefMention> mention : chains) {
      Assertions.assertEquals(0, mention.value().chain());
    }
  }

  /**
   * A neutral pronoun never resolves to a person entity, so {@code It} after a lone
   * person mention stays a singleton chain instead of linking to the wrong antecedent.
   */
  @Test
  void testNeutralPronounSkipsPersonEntities() {
    final String text = "John slept. It rained.";
    final List<Annotation<String>> toks = tokens(text,
        "John", "slept", ".", "It", "rained", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 11), "s"),
            new Annotation<>(new Span(12, 22), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks, "NNP", "VBD", ".", "PRP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 4), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(1, chains.get(1).value().chain());
    Assertions.assertEquals(CorefMention.KIND_PRONOUN, chains.get(1).value().kind());
  }

  /**
   * A plural pronoun accepts both entity type classes: {@code They} links to an
   * organization in one document and to a person in another.
   */
  @Test
  void testPluralPronounResolvesToEitherTypeClass() {
    final String orgText = "Acme grew. They hired.";
    final List<Annotation<String>> orgToks = tokens(orgText,
        "Acme", "grew", ".", "They", "hired", ".");
    final Document orgDocument = new CorefAnnotator().annotate(Document.of(orgText)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 10), "s"),
            new Annotation<>(new Span(11, 22), "s")))
        .with(Layers.TOKENS, orgToks)
        .with(Layers.POS_TAGS, values(orgToks, "NNP", "VBD", ".", "PRP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 4), "organization"))));
    final List<Annotation<CorefMention>> orgChains = orgDocument.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, orgChains.size());
    Assertions.assertEquals(0, orgChains.get(0).value().chain());
    Assertions.assertEquals(0, orgChains.get(1).value().chain());

    final String personText = "Smith won. They cheered.";
    final List<Annotation<String>> personToks = tokens(personText,
        "Smith", "won", ".", "They", "cheered", ".");
    final Document personDocument = new CorefAnnotator().annotate(Document.of(personText)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 10), "s"),
            new Annotation<>(new Span(11, 24), "s")))
        .with(Layers.TOKENS, personToks)
        .with(Layers.POS_TAGS, values(personToks, "NNP", "VBD", ".", "PRP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 5), "person"))));
    final List<Annotation<CorefMention>> personChains =
        personDocument.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, personChains.size());
    Assertions.assertEquals(0, personChains.get(0).value().chain());
    Assertions.assertEquals(0, personChains.get(1).value().chain());
  }

  /**
   * First and second person pronoun tokens never become mentions, even with a pronoun
   * POS tag, so only the entity mention appears in the chains layer.
   */
  @Test
  void testFirstAndSecondPersonPronounsAreNotMentions() {
    final String text = "John spoke. I listened. You nodded.";
    final List<Annotation<String>> toks = tokens(text,
        "John", "spoke", ".", "I", "listened", ".", "You", "nodded", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 11), "s"),
            new Annotation<>(new Span(12, 23), "s"),
            new Annotation<>(new Span(24, 35), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "NNP", "VBD", ".", "PRP", "VBD", ".", "PRP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 4), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(1, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(CorefMention.KIND_ENTITY, chains.get(0).value().kind());
  }

  /**
   * A pronoun-tagged token lying inside an entity span is not collected as a separate
   * mention, so an entity name starting with {@code It} yields one entity mention, and a
   * later free-standing {@code It} still resolves to it.
   */
  @Test
  void testPronounTokenInsideEntityIsNotAMention() {
    final String text = "It Corp grew. It hired.";
    final List<Annotation<String>> toks = tokens(text,
        "It", "Corp", "grew", ".", "It", "hired", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 13), "s"),
            new Annotation<>(new Span(14, 23), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks, "PRP", "NNP", "VBD", ".", "PRP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 7), "organization"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(new Span(0, 7), chains.get(0).span());
    Assertions.assertEquals(new Span(14, 16), chains.get(1).span());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(0, chains.get(1).value().chain());
  }

  /**
   * Entity type labels match case-insensitively on both sides: the default annotator
   * accepts an uppercase {@code PERSON} label, and an annotator configured with
   * uppercase labels accepts a mixed-case entity value.
   */
  @Test
  void testTypeLabelsMatchCaseInsensitively() {
    final String text = "Mary spoke. She left.";
    final List<Annotation<String>> toks = tokens(text,
        "Mary", "spoke", ".", "She", "left", ".");
    final Document upper = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 11), "s"),
            new Annotation<>(new Span(12, 21), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks, "NNP", "VBD", ".", "PRP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 4), "PERSON"))));
    final List<Annotation<CorefMention>> upperChains = upper.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, upperChains.size());
    Assertions.assertEquals(0, upperChains.get(0).value().chain());
    Assertions.assertEquals(0, upperChains.get(1).value().chain());

    final Document custom = new CorefAnnotator(Set.of("PER"), Set.of("ORG"))
        .annotate(Document.of(text)
            .with(Layers.SENTENCES, List.of(
                new Annotation<>(new Span(0, 11), "s"),
                new Annotation<>(new Span(12, 21), "s")))
            .with(Layers.TOKENS, toks)
            .with(Layers.POS_TAGS, values(toks, "NNP", "VBD", ".", "PRP", "VBD", "."))
            .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 4), "Per"))));
    final List<Annotation<CorefMention>> customChains = custom.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, customChains.size());
    Assertions.assertEquals(0, customChains.get(0).value().chain());
    Assertions.assertEquals(0, customChains.get(1).value().chain());
  }

  /**
   * In a document mixing typed and untyped entities, the typed rules are unchanged and
   * the untyped entities participate as unknown types: the neutral pronoun still skips
   * the person mention but links to the untyped {@code Acme}, and the gendered pronoun
   * links to the person mention as before.
   */
  @Test
  void testMixedTypedAndUntypedEntitiesResolve() {
    final String text = "John Smith joined Acme. Smith praised it. He stayed.";
    final List<Annotation<String>> toks = tokens(text,
        "John", "Smith", "joined", "Acme", ".",
        "Smith", "praised", "it", ".", "He", "stayed", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 23), "s"),
            new Annotation<>(new Span(24, 41), "s"),
            new Annotation<>(new Span(42, 52), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks,
            "NNP", "NNP", "VBD", "NNP", ".",
            "NNP", "VBD", "PRP", ".", "PRP", "VBD", "."))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 10), "person"),
            new Annotation<>(new Span(18, 22), NameFinderAnnotator.UNTYPED),
            new Annotation<>(new Span(24, 29), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(5, chains.size());
    Assertions.assertArrayEquals(new int[] {0, 1, 0, 1, 0},
        chains.stream().mapToInt(a -> a.value().chain()).toArray());
    Assertions.assertEquals(new Span(38, 40), chains.get(3).span());
    Assertions.assertEquals(CorefMention.KIND_PRONOUN, chains.get(3).value().kind());
    Assertions.assertEquals(new Span(42, 44), chains.get(4).span());
    Assertions.assertEquals(CorefMention.KIND_PRONOUN, chains.get(4).value().kind());
  }

  /**
   * An untyped entity is type-unknown, not type-mismatched: the exact match sieve links
   * it to a typed mention with identical text, unlike two differently typed mentions,
   * which stay apart.
   */
  @Test
  void testUntypedEntityMatchesTypedEntityWithSameText() {
    final String text = "Paris met Paris.";
    final List<Annotation<String>> toks = tokens(text, "Paris", "met", "Paris", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 16), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks, "PROPN", "VERB", "PROPN", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 5), "person"),
            new Annotation<>(new Span(10, 15), NameFinderAnnotator.UNTYPED))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(0, chains.get(1).value().chain());
  }

  /**
   * The mirror image of the case above: the untyped mention comes first and the typed
   * one second, exercising the other operand of the type guard and the candidate-side
   * scan of the exact match sieve.
   */
  @Test
  void testTypedEntityMatchesEarlierUntypedEntityWithSameText() {
    final String text = "Paris met Paris.";
    final List<Annotation<String>> toks = tokens(text, "Paris", "met", "Paris", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 16), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks, "PROPN", "VERB", "PROPN", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 5), NameFinderAnnotator.UNTYPED),
            new Annotation<>(new Span(10, 15), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(0, chains.get(0).value().chain());
    Assertions.assertEquals(0, chains.get(1).value().chain());
  }

  /**
   * An unknown-typed mention joins a chain but never bridges two known types: three
   * identical names typed person, untyped, and location resolve into a person chain
   * that absorbs the untyped mention, and a separate location chain. Without the
   * chain-level type, the untyped middle mention would link to the person and the
   * location would link to the untyped one, collapsing a person and a place into one
   * chain.
   */
  @Test
  void testUntypedMentionNeverBridgesDifferentlyTypedChains() {
    final String text = "Georgia met Georgia in Georgia.";
    final List<Annotation<String>> toks =
        tokens(text, "Georgia", "met", "Georgia", "in", "Georgia", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 31), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS,
            values(toks, "PROPN", "VERB", "PROPN", "ADP", "PROPN", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 7), "person"),
            new Annotation<>(new Span(12, 19), NameFinderAnnotator.UNTYPED),
            new Annotation<>(new Span(23, 30), "location"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(3, chains.size());
    Assertions.assertArrayEquals(new int[] {0, 0, 1},
        chains.stream().mapToInt(a -> a.value().chain()).toArray());
  }

  /**
   * The same three-way constellation with the known types swapped: whichever known
   * type absorbs the untyped mention, the chain invariant holds, and the person and
   * location mentions never share a chain.
   */
  @Test
  void testBridgeInvariantHoldsWithTypesInReverseOrder() {
    final String text = "Georgia met Georgia in Georgia.";
    final List<Annotation<String>> toks =
        tokens(text, "Georgia", "met", "Georgia", "in", "Georgia", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 31), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS,
            values(toks, "PROPN", "VERB", "PROPN", "ADP", "PROPN", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 7), "location"),
            new Annotation<>(new Span(12, 19), NameFinderAnnotator.UNTYPED),
            new Annotation<>(new Span(23, 30), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(3, chains.size());
    Assertions.assertArrayEquals(new int[] {0, 0, 1},
        chains.stream().mapToInt(a -> a.value().chain()).toArray());
  }

  /**
   * The containment sieve's cross-type path: a typed full name and an untyped
   * whitespace-delimited suffix of it link, in both orders of appearance, since an
   * unknown type never blocks a containment link.
   */
  @Test
  void testContainmentLinksTypedNameWithUntypedSuffix() {
    final String text = "Barack Obama spoke. Obama left.";
    final List<Annotation<String>> toks =
        tokens(text, "Barack", "Obama", "spoke", ".", "Obama", "left", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 19), "s"),
            new Annotation<>(new Span(20, 31), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS,
            values(toks, "PROPN", "PROPN", "VERB", "PUNCT", "PROPN", "VERB", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 12), "person"),
            new Annotation<>(new Span(20, 25), NameFinderAnnotator.UNTYPED))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(chains.get(0).value().chain(), chains.get(1).value().chain());
  }

  /**
   * The mirrored containment order: the untyped short form appears first and the
   * typed full name second, exercising the other containment operand.
   */
  @Test
  void testContainmentLinksUntypedPrefixWithLaterTypedName() {
    final String text = "Obama arrived. Barack Obama spoke.";
    final List<Annotation<String>> toks =
        tokens(text, "Obama", "arrived", ".", "Barack", "Obama", "spoke", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 14), "s"),
            new Annotation<>(new Span(15, 34), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS,
            values(toks, "PROPN", "VERB", "PUNCT", "PROPN", "PROPN", "VERB", "PUNCT"))
        .with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(0, 5), NameFinderAnnotator.UNTYPED),
            new Annotation<>(new Span(15, 27), "person"))));

    final List<Annotation<CorefMention>> chains = document.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(chains.get(0).value().chain(), chains.get(1).value().chain());
  }

  /**
   * A document without entities or third-person pronouns gets an empty chains layer
   * rather than an exception, and so does a document with no tokens at all.
   */
  @Test
  void testNoMentionsYieldsEmptyChainsLayer() {
    final String text = "Dogs bark.";
    final List<Annotation<String>> toks = tokens(text, "Dogs", "bark", ".");
    final Document document = new CorefAnnotator().annotate(Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 10), "s")))
        .with(Layers.TOKENS, toks)
        .with(Layers.POS_TAGS, values(toks, "NNS", "VBP", "."))
        .with(Layers.ENTITIES, List.of()));
    Assertions.assertTrue(document.get(CorefAnnotator.CHAINS).isEmpty());

    final Document empty = new CorefAnnotator().annotate(Document.of(""));
    Assertions.assertTrue(empty.get(CorefAnnotator.CHAINS).isEmpty());
  }
}
