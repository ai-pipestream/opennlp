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

package opennlp.tools.postag;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.StringUtil;

/**
 * Trains the {@link FeedforwardPOSModel} entirely in Java: one example per token with
 * gold tag history, minibatch AdaGrad over a softmax cross-entropy loss, cube
 * activation, and inverted dropout on the hidden layer. No external training framework
 * is involved, so the whole neural tagging tier, training and inference, is plain array
 * arithmetic inside the JVM.
 *
 * <p>Words and suffixes below their frequency cutoffs share learned unknown embeddings;
 * positions outside the sentence share a learned padding embedding. Training is
 * deterministic for a fixed {@link Settings#seed()}.</p>
 *
 * @since 3.0.0
 */
public final class FeedforwardPOSTrainer {

  private static final Logger logger = LoggerFactory.getLogger(FeedforwardPOSTrainer.class);

  private static final double ADAGRAD_EPSILON = 1e-6;

  private static final List<String> SHAPES = List.of(
      "*lower*", "*cap*", "*allcaps*", "*digit*", "*alnum*", "*other*");

  private FeedforwardPOSTrainer() {
    // static trainer only
  }

  /**
   * The training hyperparameters.
   *
   * @param embeddingSize The embedding dimensionality. Must be positive.
   * @param hiddenSize The hidden layer width. Must be positive.
   * @param epochs The number of passes over the examples. Must be positive.
   * @param batchSize The minibatch size. Must be positive.
   * @param learningRate The AdaGrad step size. Must be positive.
   * @param l2 The L2 penalty applied to the dense weights. Must not be negative.
   * @param dropout The hidden dropout probability. Must be in {@code [0, 1)}.
   * @param wordCutoff The minimum frequency for a word to get its own embedding. Must
   *                   not be negative.
   * @param suffixCutoff The minimum frequency for a suffix to get its own embedding.
   *                     Must not be negative.
   * @param seed The random seed making a run reproducible.
   */
  public record Settings(int embeddingSize, int hiddenSize, int epochs, int batchSize,
      double learningRate, double l2, double dropout, int wordCutoff, int suffixCutoff,
      long seed) {

    /**
     * Validates the hyperparameters.
     *
     * @throws IllegalArgumentException Thrown if a value is outside its documented
     *         range.
     */
    public Settings {
      if (embeddingSize <= 0 || hiddenSize <= 0 || epochs <= 0 || batchSize <= 0) {
        throw new IllegalArgumentException("sizes, epochs and batch must be positive");
      }
      if (learningRate <= 0.0 || l2 < 0.0) {
        throw new IllegalArgumentException("learningRate must be positive, l2 not negative");
      }
      if (!(dropout >= 0.0 && dropout < 1.0)) {
        throw new IllegalArgumentException("dropout must be in [0, 1): " + dropout);
      }
      if (wordCutoff < 0 || suffixCutoff < 0) {
        throw new IllegalArgumentException("cutoffs must not be negative");
      }
    }

    /**
     * @return The default hyperparameters. Never {@code null}.
     */
    public static Settings defaults() {
      return new Settings(50, 200, 10, 256, 0.02, 1e-8, 0.5, 2, 2, 17L);
    }
  }

  /**
   * Trains a model from POS samples.
   *
   * @param samples The training samples. Must not be {@code null}.
   * @param settings The hyperparameters. Must not be {@code null}.
   * @return A trained {@link FeedforwardPOSModel}. Never {@code null}.
   * @throws IOException Thrown if reading the samples fails.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null} or the
   *         samples contain no token.
   */
  public static FeedforwardPOSModel train(ObjectStream<POSSample> samples,
      Settings settings) throws IOException {
    if (samples == null || settings == null) {
      throw new IllegalArgumentException("samples and settings must not be null");
    }
    final List<POSSample> corpus = new ArrayList<>();
    POSSample sample;
    while ((sample = samples.read()) != null) {
      if (sample.getSentence().length > 0) {
        corpus.add(sample);
      }
    }
    if (corpus.isEmpty()) {
      throw new IllegalArgumentException("no trainable examples in the samples");
    }
    final FeedforwardPOSModel model = initialize(corpus, settings);

    final Map<String, Integer> outputIds = new HashMap<>();
    final String[] tags = model.tags();
    for (int i = 0; i < tags.length; i++) {
      outputIds.put(tags[i], i);
    }
    final List<int[]> featureList = new ArrayList<>();
    final List<Integer> goldList = new ArrayList<>();
    for (final POSSample s : corpus) {
      final String[] sentence = s.getSentence();
      final String[] gold = s.getTags();
      for (int i = 0; i < sentence.length; i++) {
        featureList.add(model.featureIds(FeedforwardPOSContext.extract(sentence, i,
            i > 0 ? gold[i - 1] : null, i > 1 ? gold[i - 2] : null)));
        goldList.add(outputIds.get(gold[i]));
      }
    }
    optimize(model, featureList, goldList, settings);
    return model;
  }

  /** Builds the vocabularies and randomly initialized weights. */
  private static FeedforwardPOSModel initialize(List<POSSample> corpus,
      Settings settings) {
    final Map<String, Integer> wordCounts = new LinkedHashMap<>();
    final Map<String, Integer> suffixCounts = new LinkedHashMap<>();
    final Map<String, Integer> tagSet = new LinkedHashMap<>();
    for (final POSSample s : corpus) {
      for (final String token : s.getSentence()) {
        final String word = FeedforwardPOSModel.normalize(token);
        wordCounts.merge(word, 1, Integer::sum);
        final String lowered = StringUtil.toLowerCase(token);
        suffixCounts.merge(FeedforwardPOSContext.suffix(lowered, 2), 1, Integer::sum);
        suffixCounts.merge(FeedforwardPOSContext.suffix(lowered, 3), 1, Integer::sum);
      }
      for (final String tag : s.getTags()) {
        tagSet.putIfAbsent(tag, 0);
      }
    }

    int row = 0;
    final Map<String, Integer> wordIds = new LinkedHashMap<>();
    for (final String special : List.of(FeedforwardPOSModel.UNKNOWN,
        FeedforwardPOSModel.ABSENT)) {
      wordIds.put(special, row++);
    }
    for (final Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
      if (entry.getValue() >= settings.wordCutoff() && !wordIds.containsKey(entry.getKey())) {
        wordIds.put(entry.getKey(), row++);
      }
    }
    final Map<String, Integer> suffixIds = new LinkedHashMap<>();
    for (final String special : List.of(FeedforwardPOSModel.UNKNOWN,
        FeedforwardPOSModel.ABSENT)) {
      suffixIds.put(special, row++);
    }
    for (final Map.Entry<String, Integer> entry : suffixCounts.entrySet()) {
      if (entry.getValue() >= settings.suffixCutoff()
          && !suffixIds.containsKey(entry.getKey())) {
        suffixIds.put(entry.getKey(), row++);
      }
    }
    final Map<String, Integer> shapeIds = new LinkedHashMap<>();
    for (final String special : List.of(FeedforwardPOSModel.UNKNOWN,
        FeedforwardPOSModel.ABSENT)) {
      shapeIds.put(special, row++);
    }
    for (final String shape : SHAPES) {
      shapeIds.put(shape, row++);
    }
    final Map<String, Integer> tagIds = new LinkedHashMap<>();
    for (final String special : List.of(FeedforwardPOSModel.UNKNOWN,
        FeedforwardPOSModel.ABSENT)) {
      tagIds.put(special, row++);
    }
    final String[] tags = new String[tagSet.size()];
    int tagIndex = 0;
    for (final String tag : tagSet.keySet()) {
      tags[tagIndex++] = tag;
      tagIds.put(tag, row++);
    }

    final Random random = new Random(settings.seed());
    final int inputSize = FeedforwardPOSContext.SLOTS * settings.embeddingSize();
    final float[][] embeddings = uniform(random, row, settings.embeddingSize(), 0.01);
    final float[][] hiddenWeights = uniform(random, settings.hiddenSize(), inputSize,
        Math.sqrt(6.0 / (inputSize + settings.hiddenSize())));
    final float[][] outputWeights = uniform(random, tags.length, settings.hiddenSize(),
        Math.sqrt(6.0 / (settings.hiddenSize() + tags.length)));
    return new FeedforwardPOSModel(wordIds, suffixIds, shapeIds, tagIds, tags,
        settings.embeddingSize(), embeddings, hiddenWeights,
        new float[settings.hiddenSize()], outputWeights, new float[tags.length]);
  }

  /** Minibatch AdaGrad over softmax cross-entropy with cube activation and dropout. */
  private static void optimize(FeedforwardPOSModel model, List<int[]> featureList,
      List<Integer> goldList, Settings settings) {
    final int exampleCount = featureList.size();
    final int[][] features = featureList.toArray(new int[0][]);
    final int[] gold = new int[exampleCount];
    for (int i = 0; i < exampleCount; i++) {
      gold[i] = goldList.get(i);
    }

    final float[][] embeddings = model.embeddings();
    final float[][] hiddenWeights = model.hiddenWeights();
    final float[] hiddenBias = model.hiddenBias();
    final float[][] outputWeights = model.outputWeights();
    final float[] outputBias = model.outputBias();
    final int embeddingSize = settings.embeddingSize();
    final int hiddenSize = settings.hiddenSize();
    final int outputSize = outputBias.length;
    final int inputSize = features[0].length * embeddingSize;

    final double[][] embeddingAccumulator = new double[embeddings.length][embeddingSize];
    final double[][] hiddenAccumulator = new double[hiddenSize][inputSize];
    final double[] hiddenBiasAccumulator = new double[hiddenSize];
    final double[][] outputAccumulator = new double[outputSize][hiddenSize];
    final double[] outputBiasAccumulator = new double[outputSize];

    final double[][] hiddenGradient = new double[hiddenSize][inputSize];
    final double[] hiddenBiasGradient = new double[hiddenSize];
    final double[][] outputGradient = new double[outputSize][hiddenSize];
    final double[] outputBiasGradient = new double[outputSize];
    final Map<Integer, double[]> embeddingGradients = new HashMap<>();

    final Random random = new Random(settings.seed());
    final int[] order = new int[exampleCount];
    for (int i = 0; i < exampleCount; i++) {
      order[i] = i;
    }

    final double keep = 1.0 - settings.dropout();
    final double[] x = new double[inputSize];
    final double[] pre = new double[hiddenSize];
    final double[] hidden = new double[hiddenSize];
    final boolean[] mask = new boolean[hiddenSize];
    final double[] probabilities = new double[outputSize];
    final double[] hiddenDelta = new double[hiddenSize];
    final double[] inputDelta = new double[inputSize];

    for (int epoch = 1; epoch <= settings.epochs(); epoch++) {
      final long epochStart = System.currentTimeMillis();
      shuffle(order, random);
      double loss = 0.0;
      for (int batchStart = 0; batchStart < exampleCount;
          batchStart += settings.batchSize()) {
        final int batchEnd = Math.min(batchStart + settings.batchSize(), exampleCount);
        final int batch = batchEnd - batchStart;
        zero(hiddenGradient);
        Arrays.fill(hiddenBiasGradient, 0.0);
        zero(outputGradient);
        Arrays.fill(outputBiasGradient, 0.0);
        embeddingGradients.clear();

        for (int b = batchStart; b < batchEnd; b++) {
          final int[] feats = features[order[b]];
          final int goldTag = gold[order[b]];
          for (int f = 0; f < feats.length; f++) {
            final float[] embedding = embeddings[feats[f]];
            final int offset = f * embeddingSize;
            for (int d = 0; d < embeddingSize; d++) {
              x[offset + d] = embedding[d];
            }
          }
          for (int j = 0; j < hiddenSize; j++) {
            mask[j] = random.nextDouble() < keep;
            if (!mask[j]) {
              pre[j] = 0.0;
              hidden[j] = 0.0;
              continue;
            }
            final float[] weightRow = hiddenWeights[j];
            double sum = hiddenBias[j];
            for (int k = 0; k < inputSize; k++) {
              sum += weightRow[k] * x[k];
            }
            pre[j] = sum;
            hidden[j] = sum * sum * sum / keep;
          }
          double max = Double.NEGATIVE_INFINITY;
          for (int o = 0; o < outputSize; o++) {
            final float[] weightRow = outputWeights[o];
            double sum = outputBias[o];
            for (int j = 0; j < hiddenSize; j++) {
              sum += weightRow[j] * hidden[j];
            }
            probabilities[o] = sum;
            max = Math.max(max, sum);
          }
          double normalizer = 0.0;
          for (int o = 0; o < outputSize; o++) {
            probabilities[o] = Math.exp(probabilities[o] - max);
            normalizer += probabilities[o];
          }
          for (int o = 0; o < outputSize; o++) {
            probabilities[o] /= normalizer;
          }
          loss -= Math.log(Math.max(probabilities[goldTag], 1e-12));

          Arrays.fill(hiddenDelta, 0.0);
          Arrays.fill(inputDelta, 0.0);
          for (int o = 0; o < outputSize; o++) {
            final double delta = probabilities[o] - (o == goldTag ? 1.0 : 0.0);
            outputBiasGradient[o] += delta;
            final double[] gradientRow = outputGradient[o];
            final float[] weightRow = outputWeights[o];
            for (int j = 0; j < hiddenSize; j++) {
              gradientRow[j] += delta * hidden[j];
              hiddenDelta[j] += delta * weightRow[j];
            }
          }
          for (int j = 0; j < hiddenSize; j++) {
            if (!mask[j]) {
              continue;
            }
            final double preDelta = hiddenDelta[j] * 3.0 * pre[j] * pre[j] / keep;
            hiddenBiasGradient[j] += preDelta;
            final double[] gradientRow = hiddenGradient[j];
            final float[] weightRow = hiddenWeights[j];
            for (int k = 0; k < inputSize; k++) {
              gradientRow[k] += preDelta * x[k];
              inputDelta[k] += preDelta * weightRow[k];
            }
          }
          for (int f = 0; f < feats.length; f++) {
            final double[] embeddingGradient = embeddingGradients
                .computeIfAbsent(feats[f], key -> new double[embeddingSize]);
            final int offset = f * embeddingSize;
            for (int d = 0; d < embeddingSize; d++) {
              embeddingGradient[d] += inputDelta[offset + d];
            }
          }
        }

        update(hiddenWeights, hiddenGradient, hiddenAccumulator, batch, settings);
        updateVector(hiddenBias, hiddenBiasGradient, hiddenBiasAccumulator, batch, settings);
        update(outputWeights, outputGradient, outputAccumulator, batch, settings);
        updateVector(outputBias, outputBiasGradient, outputBiasAccumulator, batch, settings);
        for (final Map.Entry<Integer, double[]> entry : embeddingGradients.entrySet()) {
          final float[] embeddingRow = embeddings[entry.getKey()];
          final double[] accumulatorRow = embeddingAccumulator[entry.getKey()];
          final double[] gradientRow = entry.getValue();
          for (int d = 0; d < embeddingSize; d++) {
            final double gradient = gradientRow[d] / batch;
            accumulatorRow[d] += gradient * gradient;
            embeddingRow[d] -= settings.learningRate() * gradient
                / (Math.sqrt(accumulatorRow[d]) + ADAGRAD_EPSILON);
          }
        }
      }
      logger.info("tagger epoch {}: loss {} in {} ms", epoch, loss / exampleCount,
          System.currentTimeMillis() - epochStart);
    }
  }

  private static void update(float[][] weights, double[][] gradients,
      double[][] accumulators, int batch, Settings settings) {
    for (int r = 0; r < weights.length; r++) {
      final float[] weightRow = weights[r];
      final double[] gradientRow = gradients[r];
      final double[] accumulatorRow = accumulators[r];
      for (int c = 0; c < weightRow.length; c++) {
        final double gradient = gradientRow[c] / batch + settings.l2() * weightRow[c];
        accumulatorRow[c] += gradient * gradient;
        weightRow[c] -= settings.learningRate() * gradient
            / (Math.sqrt(accumulatorRow[c]) + ADAGRAD_EPSILON);
      }
    }
  }

  private static void updateVector(float[] weights, double[] gradients,
      double[] accumulators, int batch, Settings settings) {
    for (int i = 0; i < weights.length; i++) {
      final double gradient = gradients[i] / batch;
      accumulators[i] += gradient * gradient;
      weights[i] -= settings.learningRate() * gradient
          / (Math.sqrt(accumulators[i]) + ADAGRAD_EPSILON);
    }
  }

  private static float[][] uniform(Random random, int rows, int columns, double scale) {
    final float[][] matrix = new float[rows][columns];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        matrix[r][c] = (float) ((random.nextDouble() * 2.0 - 1.0) * scale);
      }
    }
    return matrix;
  }

  private static void zero(double[][] matrix) {
    for (final double[] row : matrix) {
      Arrays.fill(row, 0.0);
    }
  }

  private static void shuffle(int[] order, Random random) {
    for (int i = order.length - 1; i > 0; i--) {
      final int j = random.nextInt(i + 1);
      final int swap = order[i];
      order[i] = order[j];
      order[j] = swap;
    }
  }
}
