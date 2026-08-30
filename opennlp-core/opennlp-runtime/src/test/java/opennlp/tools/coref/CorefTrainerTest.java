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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.ml.model.Event;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;

/**
 * Trains a ranking model on a small synthetic corpus where a neutral pronoun follows
 * an organization and a gendered pronoun follows a person, and checks the pairs, the
 * model round trip, and that the ranking annotator reproduces the pattern.
 */
public class CorefTrainerTest {

  /** Builds a two-sentence document: an entity, a verb, then a pronoun sentence. */
  private static Document document(String name, String type, String verb, String pronoun,
      int goldChainOfPronoun) {
    final String text = name + " " + verb + ". " + pronoun + " grew.";
    final List<Annotation<String>> tokens = new ArrayList<>();
    final List<Annotation<String>> tags = new ArrayList<>();
    int cursor = 0;
    for (final String[] token : new String[][] {{name, "NNP"}, {verb, "VBD"}, {".", "."},
        {pronoun, "PRP"}, {"grew", "VBD"}, {".", "."}}) {
      final int start = text.indexOf(token[0], cursor);
      final Span span = new Span(start, start + token[0].length());
      tokens.add(new Annotation<>(span, token[0]));
      tags.add(new Annotation<>(span, token[1]));
      cursor = span.getEnd();
    }
    final int firstEnd = text.indexOf('.') + 1;
    final List<Annotation<CorefMention>> gold = new ArrayList<>();
    gold.add(new Annotation<>(new Span(0, name.length()),
        new CorefMention(0, CorefMention.KIND_GOLD, CorefMention.NO_ENTITY)));
    gold.add(new Annotation<>(tokens.get(3).span(),
        new CorefMention(goldChainOfPronoun, CorefMention.KIND_GOLD, CorefMention.NO_ENTITY)));
    return Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, firstEnd), "s"),
            new Annotation<>(new Span(firstEnd + 1, text.length()), "s")))
        .with(Layers.TOKENS, tokens)
        .with(Layers.POS_TAGS, tags)
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, name.length()), type)))
        .with(CorefAnnotator.GOLD_CHAINS, gold);
  }

  /** A corpus where "It" always follows an organization and "He" always a person. */
  private static List<Document> corpus() {
    final List<Document> corpus = new ArrayList<>();
    final String[] organizations = {"Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay"};
    final String[] people = {"Kowalczyk", "Nowak", "Okafor", "Tanaka", "Ivanov", "Schmidt"};
    for (int i = 0; i < organizations.length; i++) {
      corpus.add(document(organizations[i], "organization", "expanded", "It", 0));
      corpus.add(document(people[i], "person", "arrived", "He", 0));
      // a pronoun of the other class stays apart: its gold chain is its own
      corpus.add(document(organizations[i], "organization", "expanded", "He", 1));
      corpus.add(document(people[i], "person", "arrived", "It", 1));
    }
    return corpus;
  }

  private static TrainingParameters parameters() {
    final TrainingParameters parameters = TrainingParameters.defaultParams();
    parameters.put(TrainingParameters.ITERATIONS_PARAM, 50);
    parameters.put(TrainingParameters.CUTOFF_PARAM, 1);
    return parameters;
  }

  @Test
  void testPairsLabelTheGoldAntecedentAndTheRest() {
    final List<Event> events = CorefTrainer.pairs(
        document("Acme", "organization", "expanded", "It", 0), new CorefAnnotator());
    Assertions.assertEquals(1, events.size());
    Assertions.assertEquals(SieveResolver.LINK, events.get(0).getOutcome());
    Assertions.assertTrue(List.of(events.get(0).getContext()).contains("kinds=pronoun>entity"));

    final List<Event> apart = CorefTrainer.pairs(
        document("Acme", "organization", "expanded", "He", 1), new CorefAnnotator());
    Assertions.assertEquals(1, apart.size());
    Assertions.assertEquals(SieveResolver.APART, apart.get(0).getOutcome());
  }

  @Test
  void testPairsRejectDocumentWithoutGoldChains() {
    final Document plain = document("Acme", "organization", "expanded", "It", 0);
    final Document withoutGold = Document.of(plain.text())
        .with(Layers.SENTENCES, plain.get(Layers.SENTENCES))
        .with(Layers.TOKENS, plain.get(Layers.TOKENS))
        .with(Layers.POS_TAGS, plain.get(Layers.POS_TAGS))
        .with(Layers.ENTITIES, plain.get(Layers.ENTITIES));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> CorefTrainer.pairs(withoutGold, new CorefAnnotator()));
  }

  @Test
  void testTrainedModelRoundTripsAndRanks() throws IOException {
    final CorefModel trained = CorefTrainer.train("eng",
        ObjectStreamUtils.createObjectStream(corpus()), parameters());
    Assertions.assertEquals("eng", trained.getLanguage());
    Assertions.assertEquals(2, trained.getPairModel().getNumOutcomes());

    final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    trained.serialize(bytes);
    final CorefModel model = new CorefModel(new ByteArrayInputStream(bytes.toByteArray()));

    // The synthetic corpus is balanced, so the pairs are calibrated around one half,
    // unlike a corpus where unlinked pairs dominate and the default floor applies.
    final CorefAnnotator annotator = new CorefAnnotator(Set.of("person"),
        Set.of("organization", "location"), model, 0.5);
    final Document linked = annotator.annotate(
        document("Cyberdyne", "organization", "expanded", "It", 0));
    final List<Annotation<CorefMention>> chains = linked.get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(2, chains.size());
    Assertions.assertEquals(chains.get(0).value().chain(), chains.get(1).value().chain());

    final Document apart = annotator.annotate(
        document("Cyberdyne", "organization", "expanded", "He", 1));
    final List<Annotation<CorefMention>> apartChains = apart.get(CorefAnnotator.CHAINS);
    Assertions.assertNotEquals(apartChains.get(0).value().chain(),
        apartChains.get(1).value().chain());
  }

  @Test
  void testTrainRejectsNullArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> CorefTrainer.train("eng", null, parameters()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> CorefTrainer.train("eng", ObjectStreamUtils.createObjectStream(corpus()), null));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> CorefTrainer.train("eng", ObjectStreamUtils.createObjectStream(corpus()),
            parameters(), null));
  }

  @Test
  void testAnnotatorRejectsNullModelAndBadThreshold() throws IOException {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CorefAnnotator((CorefModel) null));
    final CorefModel model = CorefTrainer.train("eng",
        ObjectStreamUtils.createObjectStream(corpus()), parameters());
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CorefAnnotator(Set.of("person"), Set.of("organization"), model, 1.5));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CorefModel("eng", null, null));
  }
}
