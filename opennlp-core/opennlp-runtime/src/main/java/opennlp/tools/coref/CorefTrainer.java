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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.ml.EventTrainer;
import opennlp.tools.ml.TrainerFactory;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.util.AbstractEventStream;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;

/**
 * Trains a {@link CorefModel} from documents that carry the input layers of
 * {@link CorefAnnotator} and a {@link CorefAnnotator#GOLD_CHAINS} layer.
 *
 * <p>Every document is processed the way the annotator will process it: mentions are
 * detected from its layers and the speaker and string sieves run. Then each remaining
 * anaphor,
 * in text order, yields one training pair per candidate antecedent in salience order:
 * the nearest preceding candidate that shares the anaphor's gold chain is a link, the
 * candidates between the two are apart, and when no candidate shares the chain every
 * candidate is apart. Gold links are applied as they are found, so cluster-level
 * features during training reflect the gold clustering reached so far, as the
 * predicted clustering will at annotation time. Gold chains are matched to detected
 * mentions by exact span; a gold mention the detector misses contributes nothing.</p>
 *
 * @since 3.0.0
 */
public final class CorefTrainer {

  /** Reads training pairs from documents. */
  private static final class PairEventStream extends AbstractEventStream<Document> {

    private final CorefAnnotator annotator;

    PairEventStream(ObjectStream<Document> documents, CorefAnnotator annotator) {
      super(documents);
      this.annotator = annotator;
    }

    @Override
    protected Iterator<Event> createEvents(Document document) {
      return pairs(document, annotator).iterator();
    }
  }

  private CorefTrainer() {
    // Not instantiated; this class provides the static train method only.
  }

  /**
   * Trains a model with the conventional entity types of {@link CorefAnnotator}.
   *
   * @param languageCode The ISO language code of the documents.
   * @param documents The training documents, each carrying the annotator's required
   *                  layers and {@link CorefAnnotator#GOLD_CHAINS}. Must not be
   *                  {@code null}.
   * @param parameters The training parameters. Must not be {@code null}.
   * @return The model. Never {@code null}.
   * @throws IOException Thrown if reading the documents fails.
   * @throws IllegalArgumentException Thrown if an argument is {@code null} or a
   *         document lacks a required layer.
   */
  public static CorefModel train(String languageCode, ObjectStream<Document> documents,
      TrainingParameters parameters) throws IOException {
    return train(languageCode, documents, parameters, new CorefAnnotator());
  }

  /**
   * Trains a model for the entity types and pronoun classes of an annotator.
   *
   * @param languageCode The ISO language code of the documents.
   * @param documents The training documents. Must not be {@code null}.
   * @param parameters The training parameters. Must not be {@code null}.
   * @param annotator The rule-based annotator whose mention detection and speaker
   *                  sieve the model is trained against. Must not be {@code null}.
   * @return The model. Never {@code null}.
   * @throws IOException Thrown if reading the documents fails.
   * @throws IllegalArgumentException Thrown if an argument is {@code null} or a
   *         document lacks a required layer.
   */
  public static CorefModel train(String languageCode, ObjectStream<Document> documents,
      TrainingParameters parameters, CorefAnnotator annotator) throws IOException {
    if (documents == null) {
      throw new IllegalArgumentException("documents must not be null");
    }
    if (parameters == null) {
      throw new IllegalArgumentException("parameters must not be null");
    }
    if (annotator == null) {
      throw new IllegalArgumentException("annotator must not be null");
    }
    final Map<String, String> manifestInfoEntries = new HashMap<>();
    final EventTrainer<TrainingParameters> trainer =
        TrainerFactory.getEventTrainer(parameters, manifestInfoEntries);
    final MaxentModel pairModel = trainer.train(new PairEventStream(documents, annotator));
    return new CorefModel(languageCode, pairModel, manifestInfoEntries);
  }

  /**
   * Builds the training pairs of one document.
   *
   * @param document The document with its gold chains.
   * @param annotator The annotator supplying mention detection.
   * @return The pair events in text order of their anaphors. Never {@code null}.
   * @throws IllegalArgumentException Thrown if the gold chains layer is absent.
   */
  static List<Event> pairs(Document document, CorefAnnotator annotator) {
    if (document == null || !document.layers().contains(CorefAnnotator.GOLD_CHAINS)) {
      throw new IllegalArgumentException(
          "document lacks the required layer " + CorefAnnotator.GOLD_CHAINS);
    }
    final SieveResolver resolver = annotator.resolver(document);
    final List<Event> events = new ArrayList<>();
    if (resolver == null) {
      return events;
    }
    resolver.resolvePrecise();
    final List<Mention> mentions = resolver.mentions();
    final Clusters clusters = resolver.clusters();
    final int[] gold = goldChains(document, mentions);
    final CorefContextGenerator features = new CorefContextGenerator(resolver);
    for (int j = 0; j < mentions.size(); j++) {
      if (clusters.find(j) != j || !resolver.rankable(j)) {
        continue;
      }
      final List<Integer> candidates = resolver.rankerCandidates(j);
      int antecedent = -1;
      if (gold[j] >= 0) {
        for (final int i : candidates) {
          if (gold[i] == gold[j] && i > antecedent) {
            antecedent = i;
          }
        }
      }
      for (final int i : candidates) {
        if (i < antecedent) {
          continue;
        }
        events.add(new Event(i == antecedent ? SieveResolver.LINK : SieveResolver.APART,
            features.features(i, j)));
      }
      if (antecedent >= 0) {
        clusters.union(antecedent, j);
      }
    }
    return events;
  }

  /** Maps each detected mention to its gold chain, or {@code -1} when it has none. */
  private static int[] goldChains(Document document, List<Mention> mentions) {
    final Map<Span, Integer> chainOfSpan = new HashMap<>();
    for (final Annotation<CorefMention> gold : document.get(CorefAnnotator.GOLD_CHAINS)) {
      chainOfSpan.put(new Span(gold.span().getStart(), gold.span().getEnd()),
          gold.value().chain());
    }
    final int[] chains = new int[mentions.size()];
    for (int m = 0; m < chains.length; m++) {
      final Span span = mentions.get(m).span();
      chains[m] = chainOfSpan.getOrDefault(new Span(span.getStart(), span.getEnd()), -1);
    }
    return chains;
  }
}
