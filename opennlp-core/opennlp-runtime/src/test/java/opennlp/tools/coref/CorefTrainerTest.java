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

import opennlp.tools.chunker.ChunkerAnnotator;
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
  void testRankingModelRoundTripsAndLinksAgainstTheNewChainOption() throws IOException {
    final CorefModel trained = CorefTrainer.trainRanking("eng",
        ObjectStreamUtils.createObjectStream(corpus()), new CorefAnnotator());
    Assertions.assertTrue(trained.isRanking());
    final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    trained.serialize(bytes);
    final CorefModel model = new CorefModel(new ByteArrayInputStream(bytes.toByteArray()));
    Assertions.assertTrue(model.isRanking());

    final CorefAnnotator annotator = new CorefAnnotator(model);
    final List<Annotation<CorefMention>> linked = annotator.annotate(
        document("Cyberdyne", "organization", "expanded", "It", 0)).get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(linked.get(0).value().chain(), linked.get(1).value().chain());
    final List<Annotation<CorefMention>> apart = annotator.annotate(
        document("Cyberdyne", "organization", "expanded", "He", 1)).get(CorefAnnotator.CHAINS);
    Assertions.assertNotEquals(apart.get(0).value().chain(), apart.get(1).value().chain());
  }

  @Test
  void testPairwiseModelIsNotRanking() throws IOException {
    final CorefModel model = CorefTrainer.train("eng",
        ObjectStreamUtils.createObjectStream(corpus()), parameters());
    Assertions.assertFalse(model.isRanking());
  }

  @Test
  void testWordVectorsFeedSimilarityFeaturesForNominalAnaphors() {
    final WordVectors vectors = word -> switch (word) {
      case "acme", "firm" -> new float[] {1f, 0.1f};
      default -> null;
    };
    final CorefAnnotator annotator = new CorefAnnotator(Set.of("person"),
        Set.of("organization", "location"), null, CorefAnnotator.DEFAULT_THRESHOLD, vectors);
    final String text = "Acme expanded. The firm grew.";
    final List<Annotation<String>> tokens = new ArrayList<>();
    final List<Annotation<String>> tags = new ArrayList<>();
    int cursor = 0;
    for (final String[] token : new String[][] {{"Acme", "NNP"}, {"expanded", "VBD"},
        {".", "."}, {"The", "DT"}, {"firm", "NN"}, {"grew", "VBD"}, {".", "."}}) {
      final int start = text.indexOf(token[0], cursor);
      final Span span = new Span(start, start + token[0].length());
      tokens.add(new Annotation<>(span, token[0]));
      tags.add(new Annotation<>(span, token[1]));
      cursor = span.getEnd();
    }
    final Document document = Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 14), "s"),
            new Annotation<>(new Span(15, 29), "s")))
        .with(Layers.TOKENS, tokens)
        .with(Layers.POS_TAGS, tags)
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 4), "organization")))
        .with(ChunkerAnnotator.CHUNKS,
            List.of(new Annotation<>(new Span(0, 4), "NP"), new Annotation<>(new Span(15, 23), "NP")))
        .with(CorefAnnotator.GOLD_CHAINS, List.of(
            new Annotation<>(new Span(0, 4), new CorefMention(0, CorefMention.KIND_GOLD, -1)),
            new Annotation<>(new Span(15, 23), new CorefMention(0, CorefMention.KIND_GOLD, -1))));
    final List<Event> events = CorefTrainer.pairs(document, annotator);
    Assertions.assertEquals(1, events.size());
    final List<String> features = List.of(events.get(0).getContext());
    Assertions.assertTrue(features.contains("sim=same"), features.toString());
    Assertions.assertTrue(features.contains("clusterSim=same"), features.toString());
    // Without vectors the similarity features are absent.
    final List<String> plain = List.of(CorefTrainer.pairs(document, new CorefAnnotator())
        .get(0).getContext());
    Assertions.assertFalse(plain.stream().anyMatch(f -> f.startsWith("sim=")), plain.toString());
  }

  /** Builds the two-sentence "Acme expanded. The firm grew." document with chunks and gold. */
  private static Document firmDocument() {
    final String text = "Acme expanded. The firm grew.";
    final List<Annotation<String>> tokens = new ArrayList<>();
    final List<Annotation<String>> tags = new ArrayList<>();
    int cursor = 0;
    for (final String[] token : new String[][] {{"Acme", "NNP"}, {"expanded", "VBD"},
        {".", "."}, {"The", "DT"}, {"firm", "NN"}, {"grew", "VBD"}, {".", "."}}) {
      final int start = text.indexOf(token[0], cursor);
      final Span span = new Span(start, start + token[0].length());
      tokens.add(new Annotation<>(span, token[0]));
      tags.add(new Annotation<>(span, token[1]));
      cursor = span.getEnd();
    }
    return Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 14), "s"),
            new Annotation<>(new Span(15, 29), "s")))
        .with(Layers.TOKENS, tokens)
        .with(Layers.POS_TAGS, tags)
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 4), "organization")))
        .with(ChunkerAnnotator.CHUNKS,
            List.of(new Annotation<>(new Span(0, 4), "NP"), new Annotation<>(new Span(15, 23), "NP")))
        .with(CorefAnnotator.GOLD_CHAINS, List.of(
            new Annotation<>(new Span(0, 4), new CorefMention(0, CorefMention.KIND_GOLD, -1)),
            new Annotation<>(new Span(15, 23), new CorefMention(0, CorefMention.KIND_GOLD, -1))));
  }

  /**
   * A two-dimensional stand-in for a contextual encoder: organization names and the
   * neutral pronoun point one way, person names and the male pronoun the other.
   */
  private static final TokenVectors ENCODER = tokens -> {
    final float[][] vectors = new float[tokens.length][];
    for (int t = 0; t < tokens.length; t++) {
      final String word = tokens[t];
      if (word.equals("It") || word.equals("Acme") || word.equals("firm")
          || word.equals("Cyberdyne") || word.equals("Globex") || word.equals("Initech")
          || word.equals("Umbrella") || word.equals("Hooli") || word.equals("Vandelay")) {
        vectors[t] = new float[] {1f, 0f};
      } else if (word.equals("He") || Character.isUpperCase(word.charAt(0))) {
        vectors[t] = new float[] {0f, 1f};
      } else {
        vectors[t] = new float[] {0.5f, 0.5f};
      }
    }
    return vectors;
  };

  @Test
  void testTokenVectorsAddSpanFeaturesWithValues() {
    final CorefAnnotator annotator = new CorefAnnotator(Set.of("person"),
        Set.of("organization", "location"), null, CorefAnnotator.DEFAULT_THRESHOLD, null,
        ENCODER);
    final List<Event> events = CorefTrainer.pairs(firmDocument(), annotator);
    Assertions.assertEquals(1, events.size());
    final Event event = events.get(0);
    final List<String> names = List.of(event.getContext());
    final float[] values = event.getValues();
    Assertions.assertNotNull(values);
    Assertions.assertEquals(names.size(), values.length);
    // "The firm" averages {0, 1} and {1, 0} to {0.5, 0.5}; "Acme" is {1, 0}: the
    // cosine 0.707 falls in the eighth tenth, and the products and differences follow
    Assertions.assertTrue(names.contains("ctx=7"), names.toString());
    Assertions.assertTrue(names.contains("ctxCluster=7"), names.toString());
    Assertions.assertEquals(0.5f, values[names.indexOf("vp0")], 1e-6f);
    Assertions.assertEquals(0f, values[names.indexOf("vp1")], 1e-6f);
    Assertions.assertEquals(0.5f, values[names.indexOf("vd0")], 1e-6f);
    Assertions.assertEquals(0.5f, values[names.indexOf("vd1")], 1e-6f);
    Assertions.assertEquals(1f, values[names.indexOf("kinds=nominal>entity")], 1e-6f);
    // Without an encoder the features are binary and carry no values.
    Assertions.assertNull(CorefTrainer.pairs(firmDocument(), new CorefAnnotator())
        .get(0).getValues());
  }

  @Test
  void testRankingModelTrainsAndDecodesWithTokenVectors() throws IOException {
    final CorefAnnotator rules = new CorefAnnotator(Set.of("person"),
        Set.of("organization", "location"), null, CorefAnnotator.DEFAULT_THRESHOLD, null,
        ENCODER);
    final CorefModel trained = CorefTrainer.trainRanking("eng",
        ObjectStreamUtils.createObjectStream(corpus()), rules);
    final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    trained.serialize(bytes);
    final CorefModel model = new CorefModel(new ByteArrayInputStream(bytes.toByteArray()));
    Assertions.assertTrue(model.isRanking());
    Assertions.assertTrue(model.getPairModel().eval(new String[] {"vp0"}, new float[] {1f})
        [model.getPairModel().getIndex(SieveResolver.LINK)] > 0.5,
        "a shared direction should favor linking");

    final CorefAnnotator annotator = new CorefAnnotator(Set.of("person"),
        Set.of("organization", "location"), model, CorefAnnotator.DEFAULT_THRESHOLD, null,
        ENCODER);
    final List<Annotation<CorefMention>> linked = annotator.annotate(
        document("Cyberdyne", "organization", "expanded", "It", 0)).get(CorefAnnotator.CHAINS);
    Assertions.assertEquals(linked.get(0).value().chain(), linked.get(1).value().chain());
    final List<Annotation<CorefMention>> apart = annotator.annotate(
        document("Cyberdyne", "organization", "expanded", "He", 1)).get(CorefAnnotator.CHAINS);
    Assertions.assertNotEquals(apart.get(0).value().chain(), apart.get(1).value().chain());
  }

  @Test
  void testEncoderMustReturnOneVectorPerToken() {
    final TokenVectors short1 = tokens -> new float[][] {{1f}};
    final CorefAnnotator annotator = new CorefAnnotator(Set.of("person"),
        Set.of("organization", "location"), null, CorefAnnotator.DEFAULT_THRESHOLD, null, short1);
    Assertions.assertThrows(IllegalStateException.class,
        () -> annotator.annotate(firmDocument()));
    final TokenVectors none = tokens -> null;
    final CorefAnnotator nullAnnotator = new CorefAnnotator(Set.of("person"),
        Set.of("organization", "location"), null, CorefAnnotator.DEFAULT_THRESHOLD, null, none);
    Assertions.assertThrows(IllegalStateException.class,
        () -> nullAnnotator.annotate(firmDocument()));
  }

  /** Three sentences: an entity, a definite phrase, and a second definite phrase. */
  private static Document threeMentions(List<Annotation<CorefMention>> gold) {
    final String text = "Acme expanded. The firm grew. The company hired.";
    final List<Annotation<String>> tokens = new ArrayList<>();
    final List<Annotation<String>> tags = new ArrayList<>();
    int cursor = 0;
    for (final String[] token : new String[][] {{"Acme", "NNP"}, {"expanded", "VBD"},
        {".", "."}, {"The", "DT"}, {"firm", "NN"}, {"grew", "VBD"}, {".", "."},
        {"The", "DT"}, {"company", "NN"}, {"hired", "VBD"}, {".", "."}}) {
      final int start = text.indexOf(token[0], cursor);
      final Span span = new Span(start, start + token[0].length());
      tokens.add(new Annotation<>(span, token[0]));
      tags.add(new Annotation<>(span, token[1]));
      cursor = span.getEnd();
    }
    return Document.of(text)
        .with(Layers.SENTENCES, List.of(new Annotation<>(new Span(0, 14), "s"),
            new Annotation<>(new Span(15, 29), "s"), new Annotation<>(new Span(30, 48), "s")))
        .with(Layers.TOKENS, tokens)
        .with(Layers.POS_TAGS, tags)
        .with(Layers.ENTITIES, List.of(new Annotation<>(new Span(0, 4), "organization")))
        .with(ChunkerAnnotator.CHUNKS, List.of(new Annotation<>(new Span(0, 4), "NP"),
            new Annotation<>(new Span(15, 23), "NP"), new Annotation<>(new Span(30, 41), "NP")))
        .with(CorefAnnotator.GOLD_CHAINS, gold);
  }

  @Test
  void testGoldSingletonsMarkPartialAnnotationSoUnannotatedMentionsAreNotTaught() {
    // OntoNotes style: no singletons, so the unannotated "The company" is a true
    // non-mention or singleton and is taught to start a chain: 1 + 2 pairs.
    final Document complete = threeMentions(List.of(
        new Annotation<>(new Span(0, 4), new CorefMention(0, CorefMention.KIND_GOLD, -1)),
        new Annotation<>(new Span(15, 23), new CorefMention(0, CorefMention.KIND_GOLD, -1))));
    Assertions.assertEquals(3, CorefTrainer.pairs(complete, new CorefAnnotator()).size());
    // A corpus that annotates singletons annotates every mention of its scheme, so
    // "The company", absent from the gold layer, lies outside the scheme and is skipped.
    final Document partial = threeMentions(List.of(
        new Annotation<>(new Span(0, 4), new CorefMention(0, CorefMention.KIND_GOLD, -1)),
        new Annotation<>(new Span(15, 23), new CorefMention(1, CorefMention.KIND_GOLD, -1))));
    Assertions.assertEquals(1, CorefTrainer.pairs(partial, new CorefAnnotator()).size());
  }

  @Test
  void testRankingRejectsBadSettings() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> CorefTrainer.trainRanking("eng", ObjectStreamUtils.createObjectStream(corpus()),
            0, 0.1, 0.0, new CorefAnnotator()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> CorefTrainer.trainRanking("eng", ObjectStreamUtils.createObjectStream(corpus()),
            1, 0.0, 0.0, new CorefAnnotator()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> CorefTrainer.trainRanking("eng", ObjectStreamUtils.createObjectStream(corpus()),
            1, 0.1, -1.0, new CorefAnnotator()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> CorefTrainer.trainRanking("eng", null, new CorefAnnotator()));
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
