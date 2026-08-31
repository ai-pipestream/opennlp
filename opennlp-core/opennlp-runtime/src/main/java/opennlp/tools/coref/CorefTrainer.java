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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.ml.EventTrainer;
import opennlp.tools.ml.TrainerFactory;
import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.Context;
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
 * candidate is apart, unless the chain started earlier in mentions the detector missed,
 * in which case the anaphor is left out as unlearnable. Gold links are applied as they
 * are found, so cluster-level
 * features during training reflect the gold clustering reached so far, as the
 * predicted clustering will at annotation time. Gold chains are matched to detected
 * mentions by exact span; a gold mention the detector misses contributes nothing.</p>
 *
 * <p>{@link #trainRanking} learns a mention ranker instead, after
 * <a href="https://aclanthology.org/D13-1203/">Durrett and Klein (EMNLP 2013)</a>: for
 * each anaphor the options are its candidates and the option of starting a new chain,
 * and the model maximizes the log probability of the gold options under a softmax over
 * all options, with any gold antecedent counting rather than only the nearest. The
 * learned weights are stored as a {@link GISModel} whose link probability is a
 * monotone function of the score, so the annotator compares candidates and the
 * new-chain option through the same model interface.</p>
 *
 * @since 3.0.0
 */
public final class CorefTrainer {

  /** The epochs of {@link #trainRanking(String, ObjectStream, CorefAnnotator)}. */
  public static final int DEFAULT_EPOCHS = 10;

  /** The AdaGrad step size of {@link #trainRanking(String, ObjectStream, CorefAnnotator)}. */
  public static final double DEFAULT_LEARNING_RATE = 0.05;

  /** The L2 weight of {@link #trainRanking(String, ObjectStream, CorefAnnotator)}. */
  public static final double DEFAULT_L2 = 0.001;

  /** The seed of the shuffle that orders anaphors within a ranking epoch. */
  private static final long SHUFFLE_SEED = 17L;

  /** The AdaGrad smoothing term. */
  private static final double ADAGRAD_EPSILON = 1e-8;

  /**
   * One anaphor's options for ranking: the binary feature ids per option, the last
   * being new chain, and with contextual vectors the anaphor's span vector and each
   * candidate's, from which the real-valued features derive as
   * {@link CorefContextGenerator.Features} describes.
   */
  private record RankingInstance(int[][] options, float[] anaphor, float[][] antecedents,
      boolean[] gold) {
  }

  /** The weights of the ranker: one per binary feature and three dense blocks. */
  private static final class Weights {
    final double[] sparse;
    final double[] product;
    final double[] difference;
    final double[] newChain;

    Weights(int features, int dimension) {
      sparse = new double[features];
      product = new double[dimension];
      difference = new double[dimension];
      newChain = new double[dimension];
    }
  }

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
   * Trains a ranking model with the settings chosen on the OntoGUM development split:
   * {@link #DEFAULT_EPOCHS}, {@link #DEFAULT_LEARNING_RATE}, and {@link #DEFAULT_L2}.
   *
   * @param languageCode The ISO language code of the documents.
   * @param documents The training documents. Must not be {@code null}.
   * @param annotator The rule-based annotator whose mention detection, speaker and
   *                  string sieves, and word vectors the model is trained against.
   *                  Must not be {@code null}.
   * @return The model. Never {@code null}.
   * @throws IOException Thrown if reading the documents fails.
   * @throws IllegalArgumentException Thrown if an argument is {@code null} or a
   *         document lacks a required layer.
   */
  public static CorefModel trainRanking(String languageCode, ObjectStream<Document> documents,
      CorefAnnotator annotator) throws IOException {
    return trainRanking(languageCode, documents, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
        DEFAULT_L2, annotator);
  }

  /**
   * Trains a ranking model.
   *
   * @param languageCode The ISO language code of the documents.
   * @param documents The training documents. Must not be {@code null}.
   * @param epochs How many passes over the anaphors to make. Must be positive.
   * @param learningRate The AdaGrad step size. Must be positive.
   * @param l2 The L2 regularization weight. Must not be negative.
   * @param annotator The rule-based annotator whose mention detection, speaker and
   *                  string sieves, and word vectors the model is trained against.
   *                  Must not be {@code null}.
   * @return The model. Never {@code null}.
   * @throws IOException Thrown if reading the documents fails.
   * @throws IllegalArgumentException Thrown if an argument is invalid or a document
   *         lacks a required layer.
   */
  public static CorefModel trainRanking(String languageCode, ObjectStream<Document> documents,
      int epochs, double learningRate, double l2, CorefAnnotator annotator)
      throws IOException {
    if (documents == null) {
      throw new IllegalArgumentException("documents must not be null");
    }
    if (annotator == null) {
      throw new IllegalArgumentException("annotator must not be null");
    }
    if (epochs <= 0) {
      throw new IllegalArgumentException("epochs must be positive: " + epochs);
    }
    if (!(learningRate > 0.0)) {
      throw new IllegalArgumentException("learningRate must be positive: " + learningRate);
    }
    if (!(l2 >= 0.0)) {
      throw new IllegalArgumentException("l2 must not be negative: " + l2);
    }
    final Map<String, Integer> featureIds = new HashMap<>();
    final List<RankingInstance> instances = new ArrayList<>();
    int dimension = 0;
    Document document;
    while ((document = documents.read()) != null) {
      for (final RankingInstance instance : rankingInstances(document, annotator, featureIds)) {
        instances.add(instance);
        if (instance.anaphor() != null) {
          dimension = instance.anaphor().length;
        }
      }
    }
    final Weights weights = new Weights(featureIds.size(), dimension);
    final Weights squares = new Weights(featureIds.size(), dimension);
    final double[] gradient = new double[featureIds.size()];
    final Random random = new Random(SHUFFLE_SEED);
    final List<RankingInstance> order = new ArrayList<>(instances);
    for (int epoch = 0; epoch < epochs; epoch++) {
      Collections.shuffle(order, random);
      for (final RankingInstance instance : order) {
        step(instance, weights, squares, gradient, learningRate, l2);
      }
    }
    final String[] predicates = new String[featureIds.size() + 3 * dimension];
    final double[] all = new double[predicates.length];
    for (final Map.Entry<String, Integer> feature : featureIds.entrySet()) {
      predicates[feature.getValue()] = feature.getKey();
      all[feature.getValue()] = weights.sparse[feature.getValue()];
    }
    for (int d = 0; d < dimension; d++) {
      final int p = featureIds.size() + 3 * d;
      predicates[p] = CorefContextGenerator.PRODUCT + d;
      all[p] = weights.product[d];
      predicates[p + 1] = CorefContextGenerator.DIFFERENCE + d;
      all[p + 1] = weights.difference[d];
      predicates[p + 2] = CorefContextGenerator.NEW_CHAIN + d;
      all[p + 2] = weights.newChain[d];
    }
    final Context[] parameters = new Context[all.length];
    for (int f = 0; f < all.length; f++) {
      parameters[f] = new Context(new int[] {0}, new double[] {all[f]});
    }
    final GISModel model = new GISModel(parameters, predicates,
        new String[] {SieveResolver.LINK, SieveResolver.APART});
    return new CorefModel(languageCode, model, true, new HashMap<>());
  }

  /**
   * Takes one AdaGrad step up the log probability of the gold options: the gradient is
   * the expected feature count under the gold options minus that under all options.
   */
  private static void step(RankingInstance instance, Weights weights, Weights squares,
      double[] gradient, double learningRate, double l2) {
    final int[][] options = instance.options();
    final float[] u = instance.anaphor();
    final double[] scores = new double[options.length];
    double max = Double.NEGATIVE_INFINITY;
    double goldMax = Double.NEGATIVE_INFINITY;
    for (int o = 0; o < options.length; o++) {
      double score = 0.0;
      for (final int f : options[o]) {
        score += weights.sparse[f];
      }
      if (u != null) {
        score += dense(weights, u, instance.antecedents()[o]);
      }
      scores[o] = score;
      max = Math.max(max, score);
      if (instance.gold()[o]) {
        goldMax = Math.max(goldMax, score);
      }
    }
    double all = 0.0;
    double gold = 0.0;
    for (int o = 0; o < options.length; o++) {
      all += Math.exp(scores[o] - max);
      if (instance.gold()[o]) {
        gold += Math.exp(scores[o] - goldMax);
      }
    }
    final List<Integer> touched = new ArrayList<>();
    final int dimension = u == null ? 0 : u.length;
    final double[] productGradient = new double[dimension];
    final double[] differenceGradient = new double[dimension];
    final double[] newChainGradient = new double[dimension];
    for (int o = 0; o < options.length; o++) {
      final double expected = Math.exp(scores[o] - max) / all;
      final double goldExpected = instance.gold()[o] ? Math.exp(scores[o] - goldMax) / gold : 0.0;
      final double delta = goldExpected - expected;
      if (delta == 0.0) {
        continue;
      }
      for (final int f : options[o]) {
        if (gradient[f] == 0.0) {
          touched.add(f);
        }
        gradient[f] += delta;
      }
      if (u != null) {
        final float[] v = instance.antecedents()[o];
        for (int d = 0; d < dimension; d++) {
          if (v != null) {
            productGradient[d] += delta * u[d] * v[d];
            differenceGradient[d] += delta * Math.abs(u[d] - v[d]);
          } else {
            newChainGradient[d] += delta * u[d];
          }
        }
      }
    }
    for (final int f : touched) {
      update(weights.sparse, squares.sparse, f, gradient[f], learningRate, l2);
      gradient[f] = 0.0;
    }
    for (int d = 0; d < dimension; d++) {
      update(weights.product, squares.product, d, productGradient[d], learningRate, l2);
      update(weights.difference, squares.difference, d, differenceGradient[d], learningRate, l2);
      update(weights.newChain, squares.newChain, d, newChainGradient[d], learningRate, l2);
    }
  }

  /** {@return the dense part of an option's score: a pair's, or the new-chain option's} */
  private static double dense(Weights weights, float[] u, float[] v) {
    double score = 0.0;
    for (int d = 0; d < u.length; d++) {
      if (v != null) {
        score += weights.product[d] * u[d] * v[d] + weights.difference[d] * Math.abs(u[d] - v[d]);
      } else {
        score += weights.newChain[d] * u[d];
      }
    }
    return score;
  }

  /** Applies one AdaGrad update to a weight from its gradient and L2 penalty. */
  private static void update(double[] weights, double[] squares, int f, double gradient,
      double learningRate, double l2) {
    final double g = gradient - l2 * weights[f];
    squares[f] += g * g;
    weights[f] += learningRate * g / (Math.sqrt(squares[f]) + ADAGRAD_EPSILON);
  }

  /**
   * Builds the ranking instances of one document: for each anaphor the feature ids of
   * every admissible candidate and of the new-chain option, with the gold options
   * marked, applying gold links as they are found.
   */
  private static List<RankingInstance> rankingInstances(Document document,
      CorefAnnotator annotator, Map<String, Integer> featureIds) {
    if (document == null || !document.layers().contains(CorefAnnotator.GOLD_CHAINS)) {
      throw new IllegalArgumentException(
          "document lacks the required layer " + CorefAnnotator.GOLD_CHAINS);
    }
    final List<RankingInstance> instances = new ArrayList<>();
    final SieveResolver resolver = annotator.resolver(document);
    if (resolver == null) {
      return instances;
    }
    resolver.resolvePrecise();
    final List<Mention> mentions = resolver.mentions();
    final Clusters clusters = resolver.clusters();
    final int[] gold = goldChains(document, mentions);
    final int[] chainStarts = chainStarts(document);
    final CorefContextGenerator features = new CorefContextGenerator(resolver);
    for (int j = 0; j < mentions.size(); j++) {
      if (clusters.find(j) != j || !resolver.rankable(j)) {
        continue;
      }
      final List<Integer> candidates = resolver.rankerCandidates(j);
      final int[][] options = new int[candidates.size() + 1][];
      final float[][] antecedents = new float[options.length][];
      final boolean[] goldOptions = new boolean[options.length];
      int nearest = -1;
      boolean anyGold = false;
      for (int o = 0; o < candidates.size(); o++) {
        final int i = candidates.get(o);
        final CorefContextGenerator.Features pair = features.features(i, j);
        options[o] = ids(pair.names(), featureIds);
        antecedents[o] = pair.antecedent();
        goldOptions[o] = gold[j] >= 0 && gold[i] == gold[j];
        anyGold |= goldOptions[o];
        if (goldOptions[o] && i > nearest) {
          nearest = i;
        }
      }
      if (!anyGold && unlearnable(mentions.get(j), gold[j], chainStarts)) {
        continue;
      }
      final CorefContextGenerator.Features newChain = features.newChainFeatures(j);
      options[candidates.size()] = ids(newChain.names(), featureIds);
      goldOptions[candidates.size()] = !anyGold;
      instances.add(new RankingInstance(options, newChain.anaphor(), antecedents, goldOptions));
      if (nearest >= 0) {
        clusters.union(nearest, j);
      }
    }
    return instances;
  }

  /** Maps feature strings to ids, minting ids for unseen features. */
  private static int[] ids(String[] features, Map<String, Integer> featureIds) {
    final int[] ids = new int[features.length];
    for (int f = 0; f < features.length; f++) {
      ids[f] = featureIds.computeIfAbsent(features[f], key -> featureIds.size());
    }
    return ids;
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
    final int[] chainStarts = chainStarts(document);
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
      if (antecedent < 0 && unlearnable(mentions.get(j), gold[j], chainStarts)) {
        continue;
      }
      for (final int i : candidates) {
        if (i < antecedent) {
          continue;
        }
        final CorefContextGenerator.Features pair = features.features(i, j);
        events.add(new Event(i == antecedent ? SieveResolver.LINK : SieveResolver.APART,
            features.names(pair), features.values(pair)));
      }
      if (antecedent >= 0) {
        clusters.union(antecedent, j);
      }
    }
    return events;
  }

  /**
   * Checks whether an anaphor's gold antecedents are all missing from the detected
   * mentions: its gold chain starts before it, yet no candidate shares the chain. Such
   * an anaphor is coreferent in truth, so teaching it as a first mention or as a
   * refusal to link would be wrong; it is left out of training.
   */
  private static boolean unlearnable(Mention anaphor, int goldChain, int[] chainStarts) {
    return goldChain >= 0 && chainStarts[goldChain] < anaphor.span().getStart();
  }

  /** {@return the start offset of the earliest gold mention of every chain} */
  private static int[] chainStarts(Document document) {
    int chains = 0;
    for (final Annotation<CorefMention> gold : document.get(CorefAnnotator.GOLD_CHAINS)) {
      chains = Math.max(chains, gold.value().chain() + 1);
    }
    final int[] starts = new int[chains];
    Arrays.fill(starts, Integer.MAX_VALUE);
    for (final Annotation<CorefMention> gold : document.get(CorefAnnotator.GOLD_CHAINS)) {
      starts[gold.value().chain()] = Math.min(starts[gold.value().chain()], gold.span().getStart());
    }
    return starts;
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
