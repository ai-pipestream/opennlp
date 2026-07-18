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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.util.ObjectStream;

/**
 * Trains the bidirectional LSTM POS tagger: character-level BiLSTM word
 * representations concatenated with learned word embeddings (and an optional frozen
 * pretrained-vector table), a word-level BiLSTM sentence encoder, and a linear
 * softmax tagger, all fit by cross-entropy with backpropagation through time, Adam,
 * global-norm gradient clipping, and input dropout on the word representations.
 *
 * <p>Everything is hand-rolled {@code double} arithmetic with no native runtime; two
 * separately seeded random streams (one for initialization, one for shuffling and
 * dropout masks) make a run with one settings object fully deterministic.</p>
 *
 * @see BilstmPOSModel
 * @see BilstmPOSTagger
 * @since 3.0.0
 */
public final class BilstmPOSTrainer {

  private static final Logger logger = LoggerFactory.getLogger(BilstmPOSTrainer.class);

  private BilstmPOSTrainer() {
  }

  /**
   * The hyperparameters of one training run.
   *
   * @param wordEmbeddingSize Dimension of the learned word embeddings.
   * @param charEmbeddingSize Dimension of the character embeddings.
   * @param charHiddenSize Hidden units per direction of the character BiLSTM.
   * @param hiddenSize Hidden units per direction of the sentence BiLSTM.
   * @param epochs Training epochs over the corpus.
   * @param batchSize Sentences per gradient update.
   * @param learningRate Adam step size.
   * @param clipNorm Global gradient norm clipping bound.
   * @param dropout Drop probability on word representations during training.
   * @param wordCutoff Minimum training count for a word-vocabulary entry.
   * @param maxWordLength Maximum characters of a token fed to the character BiLSTM.
   * @param seed The seed of both random streams.
   */
  public record Settings(int wordEmbeddingSize, int charEmbeddingSize, int charHiddenSize,
      int hiddenSize, int epochs, int batchSize, double learningRate, double clipNorm,
      double dropout, int wordCutoff, int maxWordLength, long seed) {

    /**
     * Validates the hyperparameters.
     *
     * @throws IllegalArgumentException Thrown if a value is out of range.
     */
    public Settings {
      if (wordEmbeddingSize <= 0 || charEmbeddingSize <= 0 || charHiddenSize <= 0
          || hiddenSize <= 0) {
        throw new IllegalArgumentException("sizes must be positive");
      }
      if (epochs <= 0 || batchSize <= 0) {
        throw new IllegalArgumentException("epochs and batchSize must be positive");
      }
      if (learningRate <= 0.0d || clipNorm <= 0.0d) {
        throw new IllegalArgumentException("learningRate and clipNorm must be positive");
      }
      if (dropout < 0.0d || dropout >= 1.0d) {
        throw new IllegalArgumentException("dropout must be in [0, 1)");
      }
      if (wordCutoff <= 0 || maxWordLength <= 0) {
        throw new IllegalArgumentException("wordCutoff and maxWordLength must be positive");
      }
    }

    /**
     * @return The default hyperparameters. Never {@code null}.
     */
    public static Settings defaults() {
      return new Settings(100, 25, 50, 128, 20, 16, 1e-3d, 5.0d, 0.33d, 2, 40, 17L);
    }
  }

  /**
   * Trains a model from POS samples without pretrained vectors.
   *
   * @param samples The training samples. Must not be {@code null}.
   * @param settings The hyperparameters. Must not be {@code null}.
   * @return A trained {@link BilstmPOSModel}. Never {@code null}.
   * @throws IOException Thrown if reading the samples fails.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null} or the
   *         samples contain no token.
   */
  public static BilstmPOSModel train(ObjectStream<POSSample> samples, Settings settings)
      throws IOException {
    return trainWith(samples, settings, null, null);
  }

  /**
   * Trains a model from POS samples with a frozen pretrained-vector table beside the
   * learned word embedding. For every distinct normalized (lowercased) training word
   * the function is asked once for a vector; {@code null} means the word has none.
   * The returned vectors must all share one length, they are never updated by
   * training, and the collected slice is stored inside the model, so tagging later
   * needs no embedding component.
   *
   * @param samples The training samples. Must not be {@code null}.
   * @param settings The hyperparameters. Must not be {@code null}.
   * @param wordVectors The word vector source consulted at training time. Must not be
   *                    {@code null}; must return vectors of one consistent positive
   *                    length and a vector for at least one training word.
   * @return A trained {@link BilstmPOSModel} carrying the vector table. Never
   *         {@code null}.
   * @throws IOException Thrown if reading the samples fails.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null}, the
   *         samples contain no token, or {@code wordVectors} violates its contract.
   */
  public static BilstmPOSModel train(ObjectStream<POSSample> samples, Settings settings,
      Function<CharSequence, float[]> wordVectors) throws IOException {
    if (wordVectors == null) {
      throw new IllegalArgumentException("wordVectors must not be null");
    }
    return trainWith(samples, settings, wordVectors, null);
  }

  /**
   * Trains a model from POS samples with pretrained vectors, storing vectors for the
   * words of an additional lexicon besides the training words, widening tagging-time
   * coverage to words never seen in training.
   *
   * @param samples The training samples. Must not be {@code null}.
   * @param settings The hyperparameters. Must not be {@code null}.
   * @param wordVectors The word vector source consulted at training time. Must not be
   *                    {@code null}; must return vectors of one consistent positive
   *                    length and a vector for at least one training word.
   * @param lexicon Additional words to store vectors for, normalized like the training
   *                words. Must not be {@code null} or contain {@code null}; words the
   *                source has no vector for are skipped.
   * @return A trained {@link BilstmPOSModel} carrying the vector table. Never
   *         {@code null}.
   * @throws IOException Thrown if reading the samples fails.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null}, the
   *         samples contain no token, or {@code wordVectors} violates its contract.
   */
  public static BilstmPOSModel train(ObjectStream<POSSample> samples, Settings settings,
      Function<CharSequence, float[]> wordVectors,
      Iterable<? extends CharSequence> lexicon) throws IOException {
    if (wordVectors == null) {
      throw new IllegalArgumentException("wordVectors must not be null");
    }
    if (lexicon == null) {
      throw new IllegalArgumentException("lexicon must not be null");
    }
    return trainWith(samples, settings, wordVectors, lexicon);
  }

  private static BilstmPOSModel trainWith(ObjectStream<POSSample> samples,
      Settings settings, Function<CharSequence, float[]> wordVectors,
      Iterable<? extends CharSequence> lexicon) throws IOException {
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
      throw new IllegalArgumentException("samples contain no token");
    }

    final TrainingContext context = TrainingContext.build(corpus, settings, wordVectors,
        lexicon);
    final Random trainRandom = new Random(settings.seed());
    final int[] order = new int[corpus.size()];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    int timestep = 0;
    for (int epoch = 1; epoch <= settings.epochs(); epoch++) {
      shuffle(order, trainRandom);
      double loss = 0.0d;
      int tokens = 0;
      for (int start = 0; start < order.length; start += settings.batchSize()) {
        final int end = Math.min(order.length, start + settings.batchSize());
        for (int i = start; i < end; i++) {
          final POSSample sentence = corpus.get(order[i]);
          loss += context.sentenceGradients(sentence, trainRandom);
          tokens += sentence.getSentence().length;
        }
        timestep++;
        final double norm = context.adam.globalNorm();
        if (norm > settings.clipNorm()) {
          context.adam.scaleGradients(settings.clipNorm() / norm);
        }
        context.adam.step(settings.learningRate(), timestep);
        context.adam.zero();
      }
      logger.info("epoch {}/{} loss {} over {} tokens", epoch, settings.epochs(),
          loss / tokens, tokens);
    }
    return context.toModel();
  }

  private static void shuffle(int[] order, Random random) {
    for (int i = order.length - 1; i > 0; i--) {
      final int j = random.nextInt(i + 1);
      final int tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
  }

  /**
   * The mutable state of one training run: vocabularies, parameters, optimizer, and
   * the per-sentence forward/backward wiring. Single-threaded by design. Package
   * visible for white-box gradient checks of the wiring.
   */
  static final class TrainingContext {

    private final Settings settings;
    private final LinkedHashMap<String, Integer> words;
    private final LinkedHashMap<String, Integer> chars;
    private final String[] tags;
    private final Map<String, Integer> tagIds;
    private final double[][] wordEmbeddings;
    private final double[][] charEmbeddings;
    private final LstmLayer charForward;
    private final LstmLayer charBackward;
    private final LstmLayer wordForward;
    private final LstmLayer wordBackward;
    private final double[][] outputWeights;
    private final double[] outputBias;
    private final LinkedHashMap<String, Integer> pretrainedIds;
    private final float[][] pretrainedVectors;
    private final int pretrainedSize;
    private final int inputSize;
    final AdamOptimizer adam;
    private final LstmLayer.Gradients charForwardGrads;
    private final LstmLayer.Gradients charBackwardGrads;
    private final LstmLayer.Gradients wordForwardGrads;
    private final LstmLayer.Gradients wordBackwardGrads;
    private final double[][] wordEmbeddingGrads;
    private final double[][] charEmbeddingGrads;
    private final double[][] outputWeightGrads;
    private final double[] outputBiasGrads;

    private TrainingContext(Settings settings, LinkedHashMap<String, Integer> words,
        LinkedHashMap<String, Integer> chars, String[] tags,
        Map<String, Integer> tagIds, double[][] wordEmbeddings,
        double[][] charEmbeddings, LstmLayer charForward, LstmLayer charBackward,
        LstmLayer wordForward, LstmLayer wordBackward, double[][] outputWeights,
        double[] outputBias, LinkedHashMap<String, Integer> pretrainedIds,
        float[][] pretrainedVectors, int inputSize, AdamOptimizer adam) {
      this.settings = settings;
      this.words = words;
      this.chars = chars;
      this.tags = tags;
      this.tagIds = tagIds;
      this.wordEmbeddings = wordEmbeddings;
      this.charEmbeddings = charEmbeddings;
      this.charForward = charForward;
      this.charBackward = charBackward;
      this.wordForward = wordForward;
      this.wordBackward = wordBackward;
      this.outputWeights = outputWeights;
      this.outputBias = outputBias;
      this.pretrainedIds = pretrainedIds;
      this.pretrainedVectors = pretrainedVectors;
      this.pretrainedSize = pretrainedVectors != null ? pretrainedVectors[0].length : 0;
      this.inputSize = inputSize;
      this.adam = adam;
      wordEmbeddingGrads = adam.gradient(0);
      charEmbeddingGrads = adam.gradient(1);
      charForwardGrads = LstmLayer.Gradients.wrap(adam.gradient(2), adam.gradient(3),
          adam.gradient(4)[0]);
      charBackwardGrads = LstmLayer.Gradients.wrap(adam.gradient(5), adam.gradient(6),
          adam.gradient(7)[0]);
      wordForwardGrads = LstmLayer.Gradients.wrap(adam.gradient(8), adam.gradient(9),
          adam.gradient(10)[0]);
      wordBackwardGrads = LstmLayer.Gradients.wrap(adam.gradient(11), adam.gradient(12),
          adam.gradient(13)[0]);
      outputWeightGrads = adam.gradient(14);
      outputBiasGrads = adam.gradient(15)[0];
    }

    /** @return The live word embeddings, exposed for white-box gradient checks. */
    double[][] testingWordEmbeddings() {
      return wordEmbeddings;
    }

    /** @return The live char embeddings, exposed for white-box gradient checks. */
    double[][] testingCharEmbeddings() {
      return charEmbeddings;
    }

    /** @return The live output weights, exposed for white-box gradient checks. */
    double[][] testingOutputWeights() {
      return outputWeights;
    }

    /** @return The live output bias, exposed for white-box gradient checks. */
    double[] testingOutputBias() {
      return outputBias;
    }

    /** @return The live forward sentence layer, exposed for white-box checks. */
    LstmLayer testingWordForward() {
      return wordForward;
    }

    static TrainingContext build(List<POSSample> corpus, Settings settings,
        Function<CharSequence, float[]> wordVectors,
        Iterable<? extends CharSequence> lexicon) {
      final Map<String, Integer> wordCounts = new LinkedHashMap<>();
      final Map<String, Integer> charCounts = new LinkedHashMap<>();
      final TreeSet<String> tagSet = new TreeSet<>();
      for (final POSSample sample : corpus) {
        final String[] sentence = sample.getSentence();
        for (final String token : sentence) {
          wordCounts.merge(BilstmPOSModel.normalize(token), 1, Integer::sum);
          for (int i = 0; i < token.length(); i++) {
            charCounts.merge(String.valueOf(token.charAt(i)), 1, Integer::sum);
          }
        }
        for (final String tag : sample.getTags()) {
          tagSet.add(tag);
        }
      }

      final LinkedHashMap<String, Integer> words = new LinkedHashMap<>();
      words.put(BilstmPOSModel.UNKNOWN, 0);
      for (final Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
        if (entry.getValue() >= settings.wordCutoff()) {
          words.put(entry.getKey(), words.size());
        }
      }
      final LinkedHashMap<String, Integer> chars = new LinkedHashMap<>();
      chars.put(BilstmPOSModel.UNKNOWN, 0);
      for (final Map.Entry<String, Integer> entry : charCounts.entrySet()) {
        chars.put(entry.getKey(), chars.size());
      }
      final String[] tags = tagSet.toArray(new String[0]);
      final Map<String, Integer> tagIds = new LinkedHashMap<>();
      for (int i = 0; i < tags.length; i++) {
        tagIds.put(tags[i], i);
      }

      final Random initRandom = new Random(settings.seed());
      final double[][] wordEmbeddings =
          randomMatrix(words.size(), settings.wordEmbeddingSize(), 0.1d, initRandom);
      final double[][] charEmbeddings =
          randomMatrix(chars.size(), settings.charEmbeddingSize(), 0.1d, initRandom);
      final LstmLayer charForward = new LstmLayer(settings.charEmbeddingSize(),
          settings.charHiddenSize(), initRandom);
      final LstmLayer charBackward = new LstmLayer(settings.charEmbeddingSize(),
          settings.charHiddenSize(), initRandom);

      final LinkedHashMap<String, Integer> pretrainedIds;
      final float[][] pretrainedVectors;
      if (wordVectors != null) {
        pretrainedIds = new LinkedHashMap<>();
        final List<float[]> collected = new ArrayList<>();
        collectVectors(wordCounts.keySet(), wordVectors, pretrainedIds, collected);
        if (lexicon != null) {
          final List<String> normalized = new ArrayList<>();
          for (final CharSequence word : lexicon) {
            if (word == null) {
              throw new IllegalArgumentException("lexicon must not contain null");
            }
            normalized.add(BilstmPOSModel.normalize(word.toString()));
          }
          collectVectors(normalized, wordVectors, pretrainedIds, collected);
        }
        if (collected.isEmpty()) {
          throw new IllegalArgumentException(
              "wordVectors returned no vector for any word");
        }
        pretrainedVectors = collected.toArray(new float[0][]);
      }
      else {
        pretrainedIds = null;
        pretrainedVectors = null;
      }
      final int pretrainedSize = pretrainedVectors != null ? pretrainedVectors[0].length : 0;

      final int inputSize = settings.wordEmbeddingSize() + 2 * settings.charHiddenSize()
          + pretrainedSize;
      final LstmLayer wordForward =
          new LstmLayer(inputSize, settings.hiddenSize(), initRandom);
      final LstmLayer wordBackward =
          new LstmLayer(inputSize, settings.hiddenSize(), initRandom);
      final double[][] outputWeights = new double[tags.length][2 * settings.hiddenSize()];
      final double outputLimit =
          Math.sqrt(6.0d / (tags.length + 2.0d * settings.hiddenSize()));
      for (int o = 0; o < tags.length; o++) {
        for (int j = 0; j < outputWeights[o].length; j++) {
          outputWeights[o][j] = (initRandom.nextDouble() * 2.0d - 1.0d) * outputLimit;
        }
      }
      final double[] outputBias = new double[tags.length];

      final AdamOptimizer adam = new AdamOptimizer();
      // registration order, relied on by white-box tests: 0 word embeddings,
      // 1 char embeddings, 2-4 char forward w/u/b, 5-7 char backward w/u/b,
      // 8-10 word forward w/u/b, 11-13 word backward w/u/b, 14 output weights,
      // 15 output bias
      adam.register(wordEmbeddings);
      adam.register(charEmbeddings);
      adam.register(charForward.w());
      adam.register(charForward.u());
      adam.register(charForward.b());
      adam.register(charBackward.w());
      adam.register(charBackward.u());
      adam.register(charBackward.b());
      adam.register(wordForward.w());
      adam.register(wordForward.u());
      adam.register(wordForward.b());
      adam.register(wordBackward.w());
      adam.register(wordBackward.u());
      adam.register(wordBackward.b());
      adam.register(outputWeights);
      adam.register(outputBias);

      return new TrainingContext(settings, words, chars, tags, tagIds, wordEmbeddings,
          charEmbeddings, charForward, charBackward, wordForward, wordBackward,
          outputWeights, outputBias, pretrainedIds, pretrainedVectors, inputSize, adam);
    }

    private static void collectVectors(Iterable<String> candidates,
        Function<CharSequence, float[]> wordVectors,
        LinkedHashMap<String, Integer> ids, List<float[]> collected) {
      for (final String word : candidates) {
        if (ids.containsKey(word)) {
          continue;
        }
        final float[] vector = wordVectors.apply(word);
        if (vector != null) {
          if (!collected.isEmpty() && vector.length != collected.get(0).length) {
            throw new IllegalArgumentException(
                "wordVectors must return vectors of one consistent length");
          }
          if (vector.length == 0) {
            throw new IllegalArgumentException(
                "wordVectors must return vectors of positive length");
          }
          ids.put(word, ids.size());
          collected.add(vector.clone());
        }
      }
    }

    private static double[][] randomMatrix(int rows, int cols, double limit,
        Random random) {
      final double[][] matrix = new double[rows][cols];
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          matrix[r][c] = (random.nextDouble() * 2.0d - 1.0d) * limit;
        }
      }
      return matrix;
    }

    /**
     * Runs the full forward and backward pass of one sentence, accumulating every
     * parameter gradient into the optimizer's mirrors.
     *
     * @param sample The sentence with gold tags.
     * @param random The dropout-mask random stream.
     * @return The summed cross-entropy loss of the sentence.
     */
    double sentenceGradients(POSSample sample, Random random) {
      final String[] sentence = sample.getSentence();
      final String[] goldTags = sample.getTags();
      final int steps = sentence.length;
      final int hidden = settings.hiddenSize();
      final int charHidden = settings.charHiddenSize();
      final int wordSize = settings.wordEmbeddingSize();
      final double keepProbability = 1.0d - settings.dropout();

      final int[][] charIdsPerToken = new int[steps][];
      final double[][][] charXsPerToken = new double[steps][][];
      final LstmLayer.ForwardCache[] charForwardCaches = new LstmLayer.ForwardCache[steps];
      final LstmLayer.ForwardCache[] charBackwardCaches = new LstmLayer.ForwardCache[steps];
      final double[][] xs = new double[steps][inputSize];
      final double[][] masks = new double[steps][inputSize];

      for (int t = 0; t < steps; t++) {
        final String token = sentence[t];
        final int[] ids = charIds(token);
        charIdsPerToken[t] = ids;
        final double[] x = xs[t];
        final double[] wordEmbedding = wordEmbeddings[wordId(token)];
        System.arraycopy(wordEmbedding, 0, x, 0, wordSize);
        if (ids.length > 0) {
          final double[][] charXs = new double[ids.length][];
          for (int i = 0; i < ids.length; i++) {
            charXs[i] = charEmbeddings[ids[i]];
          }
          charXsPerToken[t] = charXs;
          final LstmLayer.ForwardCache fwdCache =
              LstmLayer.ForwardCache.of(ids.length, charHidden);
          final double[][] hFwd = charForward.run(charXs, fwdCache);
          charForwardCaches[t] = fwdCache;
          System.arraycopy(hFwd[ids.length - 1], 0, x, wordSize, charHidden);
          final double[][] reversed = new double[ids.length][];
          for (int i = 0; i < ids.length; i++) {
            reversed[i] = charXs[ids.length - 1 - i];
          }
          final LstmLayer.ForwardCache bwdCache =
              LstmLayer.ForwardCache.of(ids.length, charHidden);
          final double[][] hBwd = charBackward.run(reversed, bwdCache);
          charBackwardCaches[t] = bwdCache;
          System.arraycopy(hBwd[ids.length - 1], 0, x, wordSize + charHidden, charHidden);
        }
        final float[] pretrained = pretrainedVector(token);
        if (pretrained != null) {
          final int offset = wordSize + 2 * charHidden;
          for (int i = 0; i < pretrainedSize; i++) {
            x[offset + i] = pretrained[i];
          }
        }
        final double[] mask = masks[t];
        for (int i = 0; i < inputSize; i++) {
          mask[i] = random.nextDouble() < keepProbability ? 1.0d / keepProbability : 0.0d;
          x[i] *= mask[i];
        }
      }

      final LstmLayer.ForwardCache wordForwardCache =
          LstmLayer.ForwardCache.of(steps, hidden);
      final double[][] hFwd = wordForward.run(xs, wordForwardCache);
      final double[][] reversedXs = new double[steps][];
      for (int t = 0; t < steps; t++) {
        reversedXs[t] = xs[steps - 1 - t];
      }
      final LstmLayer.ForwardCache wordBackwardCache =
          LstmLayer.ForwardCache.of(steps, hidden);
      final double[][] hBackwardReversed = wordBackward.run(reversedXs, wordBackwardCache);

      final double[][] dHFwd = new double[steps][hidden];
      final double[][] dHBwd = new double[steps][hidden];
      double loss = 0.0d;
      for (int t = 0; t < steps; t++) {
        final double[] hB = hBackwardReversed[steps - 1 - t];
        final double[] logits = new double[tags.length];
        double max = Double.NEGATIVE_INFINITY;
        for (int o = 0; o < tags.length; o++) {
          final double[] row = outputWeights[o];
          double sum = outputBias[o];
          for (int j = 0; j < hidden; j++) {
            sum += row[j] * hFwd[t][j] + row[hidden + j] * hB[j];
          }
          logits[o] = sum;
          max = Math.max(max, sum);
        }
        double total = 0.0d;
        for (int o = 0; o < tags.length; o++) {
          logits[o] = Math.exp(logits[o] - max);
          total += logits[o];
        }
        final int gold = tagIds.get(goldTags[t]);
        // per-token cross-entropy: at this point logits[gold] holds exp(score - max)
        loss += Math.log(total) - Math.log(logits[gold]);
        for (int o = 0; o < tags.length; o++) {
          logits[o] /= total;
        }
        for (int o = 0; o < tags.length; o++) {
          final double gradient = logits[o] - (o == gold ? 1.0d : 0.0d);
          outputBiasGrads[o] += gradient;
          final double[] gradRow = outputWeightGrads[o];
          final double[] weightRow = outputWeights[o];
          for (int j = 0; j < hidden; j++) {
            gradRow[j] += gradient * hFwd[t][j];
            gradRow[hidden + j] += gradient * hB[j];
            dHFwd[t][j] += weightRow[j] * gradient;
            dHBwd[t][j] += weightRow[hidden + j] * gradient;
          }
        }
      }

      final double[][] dXsFwd = new double[steps][inputSize];
      wordForward.backward(xs, wordForwardCache, dHFwd, dXsFwd, wordForwardGrads);
      final double[][] dHBwdReversed = new double[steps][hidden];
      for (int t = 0; t < steps; t++) {
        dHBwdReversed[t] = dHBwd[steps - 1 - t];
      }
      final double[][] dXsBwdReversed = new double[steps][inputSize];
      wordBackward.backward(reversedXs, wordBackwardCache, dHBwdReversed,
          dXsBwdReversed, wordBackwardGrads);

      for (int t = 0; t < steps; t++) {
        final double[] mask = masks[t];
        final double[] dx = new double[inputSize];
        final double[] dxBwd = dXsBwdReversed[steps - 1 - t];
        for (int i = 0; i < inputSize; i++) {
          dx[i] = (dXsFwd[t][i] + dxBwd[i]) * mask[i];
        }
        final int wordRow = wordId(sentence[t]);
        for (int i = 0; i < wordSize; i++) {
          wordEmbeddingGrads[wordRow][i] += dx[i];
        }
        final int[] ids = charIdsPerToken[t];
        if (ids.length > 0) {
          final double[][] dHChar = new double[ids.length][charHidden];
          System.arraycopy(dx, wordSize, dHChar[ids.length - 1], 0, charHidden);
          final double[][] dCharXs = new double[ids.length][settings.charEmbeddingSize()];
          charForward.backward(charXsPerToken[t], charForwardCaches[t], dHChar, dCharXs,
              charForwardGrads);
          for (int i = 0; i < ids.length; i++) {
            for (int c = 0; c < settings.charEmbeddingSize(); c++) {
              charEmbeddingGrads[ids[i]][c] += dCharXs[i][c];
            }
          }
          final double[][] dHCharBwd = new double[ids.length][charHidden];
          System.arraycopy(dx, wordSize + charHidden, dHCharBwd[ids.length - 1], 0,
              charHidden);
          final double[][] reversed = new double[ids.length][];
          for (int i = 0; i < ids.length; i++) {
            reversed[i] = charXsPerToken[t][ids.length - 1 - i];
          }
          final double[][] dCharXsBwd = new double[ids.length][settings.charEmbeddingSize()];
          charBackward.backward(reversed, charBackwardCaches[t], dHCharBwd, dCharXsBwd,
              charBackwardGrads);
          for (int i = 0; i < ids.length; i++) {
            final double[] dOriginal = dCharXsBwd[ids.length - 1 - i];
            for (int c = 0; c < settings.charEmbeddingSize(); c++) {
              charEmbeddingGrads[ids[i]][c] += dOriginal[c];
            }
          }
        }
      }
      return loss;
    }

    private int wordId(String token) {
      return words.getOrDefault(BilstmPOSModel.normalize(token), 0);
    }

    private int[] charIds(String token) {
      final int length = Math.min(token.length(), settings.maxWordLength());
      final int[] ids = new int[length];
      for (int i = 0; i < length; i++) {
        ids[i] = chars.getOrDefault(String.valueOf(token.charAt(i)), 0);
      }
      return ids;
    }

    private float[] pretrainedVector(String token) {
      if (pretrainedIds == null) {
        return null;
      }
      final Integer row = pretrainedIds.get(BilstmPOSModel.normalize(token));
      return row != null ? pretrainedVectors[row] : null;
    }

    private BilstmPOSModel toModel() {
      return new BilstmPOSModel(words, chars, tags, wordEmbeddings, charEmbeddings,
          charForward, charBackward, wordForward, wordBackward, outputWeights,
          outputBias, settings.maxWordLength(), pretrainedIds, pretrainedVectors);
    }
  }
}
