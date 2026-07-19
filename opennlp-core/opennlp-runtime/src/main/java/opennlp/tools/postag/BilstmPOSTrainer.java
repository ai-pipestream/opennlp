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
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
   * @param threads Worker threads for sentence-parallel batches. Dropout masks are
   *                seeded per sentence and worker gradients are reduced in a fixed
   *                order, so results are deterministic per (seed, threads) and
   *                agree across thread counts up to floating-point noise from JIT
   *                multiply-add contraction.
   * @param wordDropout Probability of replacing a training token's word embedding
   *                    with the unknown row, forcing the character and pretrained
   *                    paths to carry out-of-vocabulary words, which is exactly the
   *                    tagging-time condition they face.
   * @param learningRateHalfLife Epochs between learning-rate halvings; zero disables
   *                             decay.
   * @param crf Whether to score tag sequences with a linear-chain CRF (Viterbi
   *            decoding) instead of independent per-position softmax.
   * @param encoderLayers Stacked BiLSTM layers in the sentence encoder, 1 or 2.
   * @param pretrainedDropout Probability of zeroing a training token's whole
   *                          pretrained-vector block, training the character path for
   *                          the tokens that resolve no vector at tagging time; word
   *                          dropout alone keeps the pretrained block visible.
   * @param encoderDropout Drop probability on the encoder outputs: between the
   *                       stacked layers and on the states feeding the tag scorer.
   * @param auxLossWeight Weight of the auxiliary tagging losses when training
   *                      multi-task; zero disables the auxiliary heads.
   * @param pretrainedTuning Learning-rate multiplier for the frozen pretrained
   *                         vector table; zero keeps it frozen, a positive value
   *                         fine-tunes it at that fraction of the base rate.
   */
  public record Settings(int wordEmbeddingSize, int charEmbeddingSize, int charHiddenSize,
      int hiddenSize, int epochs, int batchSize, double learningRate, double clipNorm,
      double dropout, int wordCutoff, int maxWordLength, long seed, int threads,
      double wordDropout, int learningRateHalfLife, boolean crf, int encoderLayers,
      double pretrainedDropout, double encoderDropout, double auxLossWeight,
      double pretrainedTuning) {

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
      if (threads <= 0) {
        throw new IllegalArgumentException("threads must be positive");
      }
      if (wordDropout < 0.0d || wordDropout >= 1.0d) {
        throw new IllegalArgumentException("wordDropout must be in [0, 1)");
      }
      if (learningRateHalfLife < 0) {
        throw new IllegalArgumentException("learningRateHalfLife must not be negative");
      }
      if (encoderLayers < 1 || encoderLayers > 2) {
        throw new IllegalArgumentException("encoderLayers must be 1 or 2");
      }
      if (pretrainedDropout < 0.0d || pretrainedDropout >= 1.0d) {
        throw new IllegalArgumentException("pretrainedDropout must be in [0, 1)");
      }
      if (encoderDropout < 0.0d || encoderDropout >= 1.0d) {
        throw new IllegalArgumentException("encoderDropout must be in [0, 1)");
      }
      if (auxLossWeight < 0.0d) {
        throw new IllegalArgumentException("auxLossWeight must not be negative");
      }
      if (pretrainedTuning < 0.0d) {
        throw new IllegalArgumentException("pretrainedTuning must not be negative");
      }
    }

    /**
     * @return The default hyperparameters, with one worker per available processor.
     *         Never {@code null}.
     */
    public static Settings defaults() {
      return new Settings(100, 25, 50, 128, 20, 16, 1e-3d, 5.0d, 0.33d, 2, 40, 17L,
          Runtime.getRuntime().availableProcessors(), 0.0d, 0, false, 1, 0.0d, 0.0d,
          1.0d, 0.0d);
    }
  }

  /**
   * One multi-task training sentence: tokens with their UPOS tags and optional
   * auxiliary taggings, each auxiliary label the composite string of its column.
   * Auxiliary heads exist only at training time as regularizers of the shared
   * encoder; the built model tags UPOS exactly as a single-task model does.
   *
   * @param tokens The tokens. Must not be {@code null} or empty.
   * @param upos The universal POS tags, aligned with {@code tokens}. Must not be
   *             {@code null}.
   * @param xpos The treebank-specific tags, aligned with {@code tokens}, or
   *             {@code null} when the sentence carries none.
   * @param feats The morphological feature strings, aligned with {@code tokens}, or
   *              {@code null} when the sentence carries none.
   */
  public record MultiTaskSample(String[] tokens, String[] upos, String[] xpos,
      String[] feats) {

    /**
     * Validates the alignment of the taggings.
     *
     * @throws IllegalArgumentException Thrown if {@code tokens} or {@code upos} is
     *         {@code null} or empty, or a tagging's length differs from the tokens.
     */
    public MultiTaskSample {
      if (tokens == null || tokens.length == 0 || upos == null) {
        throw new IllegalArgumentException("tokens and upos must not be null or empty");
      }
      if (upos.length != tokens.length
          || (xpos != null && xpos.length != tokens.length)
          || (feats != null && feats.length != tokens.length)) {
        throw new IllegalArgumentException("taggings must align with the tokens");
      }
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

  /**
   * Trains a model multi-task: the shared encoder is additionally fit to predict the
   * auxiliary taggings the samples carry, weighted by
   * {@link Settings#auxLossWeight()}. The auxiliary heads are dropped after training;
   * the returned model is a plain UPOS tagger.
   *
   * @param samples The multi-task training samples. Must not be {@code null}.
   * @param settings The hyperparameters. Must not be {@code null}.
   * @param wordVectors The word vector source consulted at training time, or
   *                    {@code null} to train without pretrained vectors; the same
   *                    contract as {@link #train(ObjectStream, Settings, Function)}.
   * @param lexicon Additional words to store vectors for, or {@code null} for none;
   *                ignored when {@code wordVectors} is {@code null}.
   * @return A trained {@link BilstmPOSModel}. Never {@code null}.
   * @throws IOException Thrown if reading the samples fails.
   * @throws IllegalArgumentException Thrown if samples or settings are {@code null},
   *         the samples contain no token, or {@code wordVectors} violates its
   *         contract.
   */
  public static BilstmPOSModel trainMultiTask(ObjectStream<MultiTaskSample> samples,
      Settings settings, Function<CharSequence, float[]> wordVectors,
      Iterable<? extends CharSequence> lexicon) throws IOException {
    if (samples == null || settings == null) {
      throw new IllegalArgumentException("samples and settings must not be null");
    }
    final List<MultiTaskSample> corpus = new ArrayList<>();
    MultiTaskSample sample;
    while ((sample = samples.read()) != null) {
      corpus.add(sample);
    }
    return trainCorpus(corpus, settings, wordVectors, lexicon);
  }

  private static BilstmPOSModel trainWith(ObjectStream<POSSample> samples,
      Settings settings, Function<CharSequence, float[]> wordVectors,
      Iterable<? extends CharSequence> lexicon) throws IOException {
    if (samples == null || settings == null) {
      throw new IllegalArgumentException("samples and settings must not be null");
    }
    final List<MultiTaskSample> corpus = new ArrayList<>();
    POSSample sample;
    while ((sample = samples.read()) != null) {
      if (sample.getSentence().length > 0) {
        corpus.add(new MultiTaskSample(sample.getSentence(), sample.getTags(), null,
            null));
      }
    }
    return trainCorpus(corpus, settings, wordVectors, lexicon);
  }

  private static BilstmPOSModel trainCorpus(List<MultiTaskSample> corpus,
      Settings settings, Function<CharSequence, float[]> wordVectors,
      Iterable<? extends CharSequence> lexicon) {
    if (corpus.isEmpty()) {
      throw new IllegalArgumentException("samples contain no token");
    }

    final TrainingContext context = TrainingContext.build(corpus, settings, wordVectors,
        lexicon);
    final Random shuffleRandom = new Random(settings.seed());
    final int[] order = new int[corpus.size()];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    final int threads = settings.threads();
    final List<TrainingContext.Worker> workers = new ArrayList<>();
    for (int w = 0; w < threads; w++) {
      workers.add(context.newWorker());
    }
    final ExecutorService pool =
        threads > 1 ? Executors.newFixedThreadPool(threads) : null;
    try {
      int timestep = 0;
      for (int epoch = 1; epoch <= settings.epochs(); epoch++) {
        final double learningRate = settings.learningRateHalfLife() > 0
            ? settings.learningRate()
                * Math.pow(0.5d, (epoch - 1) / (double) settings.learningRateHalfLife())
            : settings.learningRate();
        shuffle(order, shuffleRandom);
        double loss = 0.0d;
        int tokens = 0;
        final int currentEpoch = epoch;
        for (int start = 0; start < order.length; start += settings.batchSize()) {
          final int end = Math.min(order.length, start + settings.batchSize());
          for (int i = start; i < end; i++) {
            tokens += corpus.get(order[i]).tokens().length;
          }
          if (pool == null) {
            final TrainingContext.Worker worker = workers.get(0);
            for (int i = start; i < end; i++) {
              loss += context.sentenceGradients(corpus.get(order[i]),
                  maskRandom(settings.seed(), currentEpoch, i), worker);
            }
          }
          else {
            final int batchStart = start;
            final List<Callable<Double>> tasks = new ArrayList<>();
            for (int w = 0; w < threads; w++) {
              final int workerIndex = w;
              tasks.add(() -> {
                double workerLoss = 0.0d;
                final TrainingContext.Worker worker = workers.get(workerIndex);
                for (int i = batchStart + workerIndex; i < end; i += threads) {
                  workerLoss += context.sentenceGradients(corpus.get(order[i]),
                      maskRandom(settings.seed(), currentEpoch, i), worker);
                }
                return workerLoss;
              });
            }
            try {
              for (final Future<Double> result : pool.invokeAll(tasks)) {
                loss += result.get();
              }
            }
            catch (InterruptedException e) {
              Thread.currentThread().interrupt();
              throw new IllegalStateException("training interrupted", e);
            }
            catch (ExecutionException e) {
              throw new IllegalStateException("training worker failed", e.getCause());
            }
          }
          for (final TrainingContext.Worker worker : workers) {
            context.absorb(worker);
          }
          timestep++;
          final double norm = context.adam.globalNorm();
          if (norm > settings.clipNorm()) {
            context.adam.scaleGradients(settings.clipNorm() / norm);
          }
          context.adam.step(learningRate, timestep);
          context.adam.zero();
        }
        logger.info("epoch {}/{} loss {} over {} tokens (lr {})", epoch, settings.epochs(),
            loss / tokens, tokens, learningRate);
      }
    }
    finally {
      if (pool != null) {
        pool.shutdownNow();
      }
    }
    return context.toModel();
  }

  private static Random maskRandom(long seed, int epoch, int position) {
    return new Random(seed + epoch * 1000003L + position);
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
    private final LstmLayer wordForward2;
    private final LstmLayer wordBackward2;
    private final double[][] outputWeights;
    private final double[] outputBias;
    private final double[][] transitionWeights;
    private final double[] startWeights;
    private final double[] endWeights;
    private final CrfScorer crfScorer;
    private final LinkedHashMap<String, Integer> pretrainedIds;
    private final float[][] pretrainedVectors;
    private final double[][] pretrainedTrainable;
    private final int pretrainedSize;
    private final int inputSize;
    final AdamOptimizer adam;

    private final String[] xposTags;
    private final Map<String, Integer> xposIds;
    private final double[][] xposWeights;
    private final double[] xposBias;
    private final String[] featsTags;
    private final Map<String, Integer> featsIds;
    private final double[][] featsWeights;
    private final double[] featsBias;

    private TrainingContext(Settings settings, LinkedHashMap<String, Integer> words,
        LinkedHashMap<String, Integer> chars, String[] tags,
        Map<String, Integer> tagIds, double[][] wordEmbeddings,
        double[][] charEmbeddings, LstmLayer charForward, LstmLayer charBackward,
        LstmLayer wordForward, LstmLayer wordBackward, LstmLayer wordForward2,
        LstmLayer wordBackward2, double[][] outputWeights,
        double[] outputBias, double[][] transitionWeights, double[] startWeights,
        double[] endWeights, LinkedHashMap<String, Integer> pretrainedIds,
        float[][] pretrainedVectors, double[][] pretrainedTrainable, int inputSize,
        AdamOptimizer adam, String[] xposTags, Map<String, Integer> xposIds,
        double[][] xposWeights, double[] xposBias, String[] featsTags,
        Map<String, Integer> featsIds, double[][] featsWeights, double[] featsBias) {
      this.xposTags = xposTags;
      this.xposIds = xposIds;
      this.xposWeights = xposWeights;
      this.xposBias = xposBias;
      this.featsTags = featsTags;
      this.featsIds = featsIds;
      this.featsWeights = featsWeights;
      this.featsBias = featsBias;
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
      this.wordForward2 = wordForward2;
      this.wordBackward2 = wordBackward2;
      this.outputWeights = outputWeights;
      this.outputBias = outputBias;
      this.transitionWeights = transitionWeights;
      this.startWeights = startWeights;
      this.endWeights = endWeights;
      this.crfScorer = settings.crf() ? new CrfScorer(tags.length) : null;
      this.pretrainedIds = pretrainedIds;
      this.pretrainedVectors = pretrainedVectors;
      this.pretrainedTrainable = pretrainedTrainable;
      this.pretrainedSize = pretrainedVectors != null ? pretrainedVectors[0].length : 0;
      this.inputSize = inputSize;
      this.adam = adam;
      int index = 16;
      crfIndex = settings.crf() ? index : -1;
      if (settings.crf()) {
        index += 3;
      }
      layer2Index = settings.encoderLayers() == 2 ? index : -1;
      if (settings.encoderLayers() == 2) {
        index += 6;
      }
      xposIndex = xposWeights != null ? index : -1;
      if (xposWeights != null) {
        index += 2;
      }
      featsIndex = featsWeights != null ? index : -1;
      if (featsWeights != null) {
        index += 2;
      }
      tuningIndex = pretrainedTrainable != null ? index : -1;
    }

    private final int crfIndex;
    private final int layer2Index;
    private final int xposIndex;
    private final int featsIndex;
    private final int tuningIndex;

    /**
     * Allocates a worker with its own gradient buffers, for sentence-parallel batch
     * processing.
     *
     * @return A fresh worker. Never {@code null}.
     */
    Worker newWorker() {
      return new Worker(adam.newGradientBuffers(tuningIndex), crfIndex, layer2Index,
          xposIndex, featsIndex, tuningIndex);
    }

    /**
     * Reduces one worker's gradients into the optimizer and resets the worker for
     * the next batch; the sparse fine-tuning rows go through
     * {@link AdamOptimizer#absorbSparse}.
     *
     * @param worker The worker to reduce. Must not be {@code null}.
     */
    void absorb(Worker worker) {
      adam.absorb(worker.buffers());
      if (tuningIndex >= 0) {
        adam.absorbSparse(tuningIndex, worker.pretrainedGrads);
        worker.pretrainedGrads.clear();
      }
      AdamOptimizer.zero(worker.buffers());
    }

    /**
     * Resets a worker's gradient storage, dense buffers and sparse rows alike.
     *
     * @param worker The worker to reset. Must not be {@code null}.
     */
    void resetWorker(Worker worker) {
      AdamOptimizer.zero(worker.buffers());
      if (worker.pretrainedGrads != null) {
        worker.pretrainedGrads.clear();
      }
    }

    /**
     * Materializes a worker's sparse fine-tuning gradients as a dense table, with
     * untouched rows left at zero. Exposed for gradient checks.
     *
     * @param worker The worker holding the sparse rows. Must not be {@code null}.
     * @return The dense table gradient, shaped like the vector table.
     */
    double[][] denseTuningGradient(Worker worker) {
      final double[][] dense = new double[pretrainedTrainable.length][pretrainedSize];
      for (final Map.Entry<Integer, double[]> entry : worker.pretrainedGrads.entrySet()) {
        System.arraycopy(entry.getValue(), 0, dense[entry.getKey()], 0, pretrainedSize);
      }
      return dense;
    }

    /**
     * One parallel worker's gradient storage. A sentence's backward pass accumulates
     * into the worker's own buffers, never into shared state, so workers are safe to
     * run concurrently; the training loop reduces them into the optimizer in a fixed
     * order, which keeps parallel training deterministic per (seed, threads).
     */
    static final class Worker {

      private final List<double[][]> buffers;
      private final LstmLayer.Gradients charForwardGrads;
      private final LstmLayer.Gradients charBackwardGrads;
      private final LstmLayer.Gradients wordForwardGrads;
      private final LstmLayer.Gradients wordBackwardGrads;
      private final double[][] wordEmbeddingGrads;
      private final double[][] charEmbeddingGrads;
      private final double[][] outputWeightGrads;
      private final double[] outputBiasGrads;
      private final double[][] transitionGrads;
      private final double[] startGrads;
      private final double[] endGrads;
      private final LstmLayer.Gradients wordForward2Grads;
      private final LstmLayer.Gradients wordBackward2Grads;
      private final double[][] xposWeightGrads;
      private final double[] xposBiasGrads;
      private final double[][] featsWeightGrads;
      private final double[] featsBiasGrads;
      private final SortedMap<Integer, double[]> pretrainedGrads;

      private Worker(List<double[][]> buffers, int crfIndex, int layer2Index,
          int xposIndex, int featsIndex, int tuningIndex) {
        this.buffers = buffers;
        wordEmbeddingGrads = buffers.get(0);
        charEmbeddingGrads = buffers.get(1);
        charForwardGrads = LstmLayer.Gradients.wrap(buffers.get(2), buffers.get(3),
            buffers.get(4)[0]);
        charBackwardGrads = LstmLayer.Gradients.wrap(buffers.get(5), buffers.get(6),
            buffers.get(7)[0]);
        wordForwardGrads = LstmLayer.Gradients.wrap(buffers.get(8), buffers.get(9),
            buffers.get(10)[0]);
        wordBackwardGrads = LstmLayer.Gradients.wrap(buffers.get(11), buffers.get(12),
            buffers.get(13)[0]);
        outputWeightGrads = buffers.get(14);
        outputBiasGrads = buffers.get(15)[0];
        transitionGrads = crfIndex >= 0 ? buffers.get(crfIndex) : null;
        startGrads = crfIndex >= 0 ? buffers.get(crfIndex + 1)[0] : null;
        endGrads = crfIndex >= 0 ? buffers.get(crfIndex + 2)[0] : null;
        wordForward2Grads = layer2Index >= 0 ? LstmLayer.Gradients.wrap(
            buffers.get(layer2Index), buffers.get(layer2Index + 1),
            buffers.get(layer2Index + 2)[0]) : null;
        wordBackward2Grads = layer2Index >= 0 ? LstmLayer.Gradients.wrap(
            buffers.get(layer2Index + 3), buffers.get(layer2Index + 4),
            buffers.get(layer2Index + 5)[0]) : null;
        xposWeightGrads = xposIndex >= 0 ? buffers.get(xposIndex) : null;
        xposBiasGrads = xposIndex >= 0 ? buffers.get(xposIndex + 1)[0] : null;
        featsWeightGrads = featsIndex >= 0 ? buffers.get(featsIndex) : null;
        featsBiasGrads = featsIndex >= 0 ? buffers.get(featsIndex + 1)[0] : null;
        pretrainedGrads = tuningIndex >= 0 ? new TreeMap<>() : null;
      }

      /**
       * @return The gradient buffers in registration order. Never {@code null}.
       */
      List<double[][]> buffers() {
        return buffers;
      }
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

    /** @return The live CRF transitions, exposed for white-box gradient checks. */
    double[][] testingTransitionWeights() {
      return transitionWeights;
    }

    /** @return The live CRF start scores, exposed for white-box gradient checks. */
    double[] testingStartWeights() {
      return startWeights;
    }

    /** @return The live CRF end scores, exposed for white-box gradient checks. */
    double[] testingEndWeights() {
      return endWeights;
    }

    /** @return The live fine-tuning vector table, exposed for white-box checks. */
    double[][] testingPretrainedTrainable() {
      return pretrainedTrainable;
    }

    /** @return The live forward sentence layer, exposed for white-box checks. */
    LstmLayer testingWordForward() {
      return wordForward;
    }

    /** @return The live second forward sentence layer, exposed for white-box checks. */
    LstmLayer testingWordForward2() {
      return wordForward2;
    }

    /** @return The live backward sentence layer, exposed for white-box checks. */
    LstmLayer testingWordBackward() {
      return wordBackward;
    }

    /** @return The live xpos head weights, exposed for white-box gradient checks. */
    double[][] testingXposWeights() {
      return xposWeights;
    }

    /** @return The live xpos head bias, exposed for white-box gradient checks. */
    double[] testingXposBias() {
      return xposBias;
    }

    /** @return The live feats head weights, exposed for white-box gradient checks. */
    double[][] testingFeatsWeights() {
      return featsWeights;
    }

    /** @return The live forward char layer, exposed for white-box checks. */
    LstmLayer testingCharForward() {
      return charForward;
    }

    /** @return The live backward char layer, exposed for white-box checks. */
    LstmLayer testingCharBackward() {
      return charBackward;
    }

    static TrainingContext build(List<MultiTaskSample> corpus, Settings settings,
        Function<CharSequence, float[]> wordVectors,
        Iterable<? extends CharSequence> lexicon) {
      final Map<String, Integer> wordCounts = new LinkedHashMap<>();
      final Map<String, Integer> charCounts = new LinkedHashMap<>();
      final TreeSet<String> tagSet = new TreeSet<>();
      final TreeSet<String> xposSet = new TreeSet<>();
      final TreeSet<String> featsSet = new TreeSet<>();
      for (final MultiTaskSample sample : corpus) {
        final String[] sentence = sample.tokens();
        for (final String token : sentence) {
          wordCounts.merge(BilstmPOSModel.normalize(token), 1, Integer::sum);
          for (int i = 0; i < token.length(); i++) {
            charCounts.merge(String.valueOf(token.charAt(i)), 1, Integer::sum);
          }
        }
        for (final String tag : sample.upos()) {
          tagSet.add(tag);
        }
        if (settings.auxLossWeight() > 0.0d) {
          if (sample.xpos() != null) {
            for (final String tag : sample.xpos()) {
              xposSet.add(tag);
            }
          }
          if (sample.feats() != null) {
            for (final String tag : sample.feats()) {
              featsSet.add(tag);
            }
          }
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
      final String[] xposTags = xposSet.toArray(new String[0]);
      final Map<String, Integer> xposIds = new LinkedHashMap<>();
      for (int i = 0; i < xposTags.length; i++) {
        xposIds.put(xposTags[i], i);
      }
      final String[] featsTags = featsSet.toArray(new String[0]);
      final Map<String, Integer> featsIds = new LinkedHashMap<>();
      for (int i = 0; i < featsTags.length; i++) {
        featsIds.put(featsTags[i], i);
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
      final double[][] pretrainedTrainable;
      if (pretrainedVectors != null && settings.pretrainedTuning() > 0.0d) {
        pretrainedTrainable = new double[pretrainedVectors.length][];
        for (int r = 0; r < pretrainedVectors.length; r++) {
          final float[] row = pretrainedVectors[r];
          pretrainedTrainable[r] = new double[row.length];
          for (int i = 0; i < row.length; i++) {
            pretrainedTrainable[r][i] = row[i];
          }
        }
      }
      else {
        pretrainedTrainable = null;
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

      final double[][] transitionWeights;
      final double[] startWeights;
      final double[] endWeights;
      if (settings.crf()) {
        transitionWeights = randomMatrix(tags.length, tags.length, 0.1d, initRandom);
        startWeights = new double[tags.length];
        endWeights = new double[tags.length];
      }
      else {
        transitionWeights = null;
        startWeights = null;
        endWeights = null;
      }
      final double[][] xposWeights;
      final double[] xposBias;
      if (xposTags.length > 0) {
        xposWeights = xavierOutput(xposTags.length, 2 * settings.hiddenSize(),
            initRandom);
        xposBias = new double[xposTags.length];
      }
      else {
        xposWeights = null;
        xposBias = null;
      }
      final double[][] featsWeights;
      final double[] featsBias;
      if (featsTags.length > 0) {
        featsWeights = xavierOutput(featsTags.length, 2 * settings.hiddenSize(),
            initRandom);
        featsBias = new double[featsTags.length];
      }
      else {
        featsWeights = null;
        featsBias = null;
      }
      final LstmLayer wordForward2;
      final LstmLayer wordBackward2;
      if (settings.encoderLayers() == 2) {
        wordForward2 = new LstmLayer(2 * settings.hiddenSize(), settings.hiddenSize(),
            initRandom);
        wordBackward2 = new LstmLayer(2 * settings.hiddenSize(), settings.hiddenSize(),
            initRandom);
      }
      else {
        wordForward2 = null;
        wordBackward2 = null;
      }

      final AdamOptimizer adam = new AdamOptimizer();
      // registration order, relied on by white-box tests: 0 word embeddings,
      // 1 char embeddings, 2-4 char forward w/u/b, 5-7 char backward w/u/b,
      // 8-10 word forward w/u/b, 11-13 word backward w/u/b, 14 output weights,
      // 15 output bias, then CRF transitions/start/end (crf only), then
      // second-layer forward/backward w/u/b (encoderLayers 2 only), then the
      // auxiliary xpos weights/bias and feats weights/bias (multi-task only)
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
      if (settings.crf()) {
        adam.register(transitionWeights);
        adam.register(startWeights);
        adam.register(endWeights);
      }
      if (settings.encoderLayers() == 2) {
        adam.register(wordForward2.w());
        adam.register(wordForward2.u());
        adam.register(wordForward2.b());
        adam.register(wordBackward2.w());
        adam.register(wordBackward2.u());
        adam.register(wordBackward2.b());
      }
      if (xposWeights != null) {
        adam.register(xposWeights);
        adam.register(xposBias);
      }
      if (featsWeights != null) {
        adam.register(featsWeights);
        adam.register(featsBias);
      }
      if (pretrainedTrainable != null) {
        adam.register(pretrainedTrainable, settings.pretrainedTuning());
      }

      return new TrainingContext(settings, words, chars, tags, tagIds, wordEmbeddings,
          charEmbeddings, charForward, charBackward, wordForward, wordBackward,
          wordForward2, wordBackward2, outputWeights, outputBias, transitionWeights,
          startWeights, endWeights, pretrainedIds, pretrainedVectors,
          pretrainedTrainable, inputSize, adam, xposTags, xposIds, xposWeights,
          xposBias, featsTags, featsIds, featsWeights, featsBias);
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

    /**
     * Initializes an output-layer weight matrix with the same Xavier bound the UPOS
     * scorer uses.
     *
     * @param rows The label count.
     * @param cols The encoder state width.
     * @param random The initialization stream.
     * @return The initialized matrix. Never {@code null}.
     */
    private static double[][] xavierOutput(int rows, int cols, Random random) {
      final double limit = Math.sqrt(6.0d / (rows + (double) cols));
      final double[][] matrix = new double[rows][cols];
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          matrix[r][c] = (random.nextDouble() * 2.0d - 1.0d) * limit;
        }
      }
      return matrix;
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
     * parameter gradient into the worker's buffers.
     *
     * @param sample The sentence with gold tags.
     * @param random The dropout-mask random stream, seeded per sentence by the caller
     *               so parallel training is deterministic per (seed, threads).
     * @param worker The gradient-storage owner. Must not be {@code null}.
     * @return The summed cross-entropy loss of the sentence.
     */
    double sentenceGradients(MultiTaskSample sample, Random random, Worker worker) {
      final double[][] wordEmbeddingGrads = worker.wordEmbeddingGrads;
      final double[][] charEmbeddingGrads = worker.charEmbeddingGrads;
      final double[][] outputWeightGrads = worker.outputWeightGrads;
      final double[] outputBiasGrads = worker.outputBiasGrads;
      final LstmLayer.Gradients charForwardGrads = worker.charForwardGrads;
      final LstmLayer.Gradients charBackwardGrads = worker.charBackwardGrads;
      final LstmLayer.Gradients wordForwardGrads = worker.wordForwardGrads;
      final LstmLayer.Gradients wordBackwardGrads = worker.wordBackwardGrads;
      final String[] sentence = sample.tokens();
      final String[] goldTags = sample.upos();
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
      final int[] wordRows = new int[steps];
      final int[] pretrainedRows = new int[steps];
      final boolean[] blockDropped = new boolean[steps];

      for (int t = 0; t < steps; t++) {
        pretrainedRows[t] = -1;
        final String token = sentence[t];
        final int[] ids = charIds(token);
        charIdsPerToken[t] = ids;
        int wordRow = wordId(token);
        if (settings.wordDropout() > 0.0d
            && random.nextDouble() < settings.wordDropout()) {
          wordRow = 0;
        }
        wordRows[t] = wordRow;
        final double[] x = xs[t];
        final double[] wordEmbedding = wordEmbeddings[wordRow];
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
        if (pretrainedIds != null) {
          final Integer row = pretrainedIds.get(BilstmPOSModel.normalize(token));
          if (row != null) {
            pretrainedRows[t] = row;
            // whole-block dropout trains the character path for the tokens that
            // resolve no vector at tagging time; drawn only when enabled, so runs
            // without it keep their exact mask stream
            if (settings.pretrainedDropout() > 0.0d
                && random.nextDouble() < settings.pretrainedDropout()) {
              blockDropped[t] = true;
            }
            else {
              final int offset = wordSize + 2 * charHidden;
              if (pretrainedTrainable != null) {
                System.arraycopy(pretrainedTrainable[row], 0, x, offset, pretrainedSize);
              }
              else {
                final float[] pretrained = pretrainedVectors[row];
                for (int i = 0; i < pretrainedSize; i++) {
                  x[offset + i] = pretrained[i];
                }
              }
            }
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

      final double[][] states = new double[steps][2 * hidden];
      for (int t = 0; t < steps; t++) {
        System.arraycopy(hFwd[t], 0, states[t], 0, hidden);
        System.arraycopy(hBackwardReversed[steps - 1 - t], 0, states[t], hidden, hidden);
      }
      // inverted dropout on the encoder outputs: between stacked layers and on the
      // states feeding the scorers; masks drawn only when enabled, so runs without
      // it keep their exact mask stream
      final double encoderKeep = 1.0d - settings.encoderDropout();
      double[][] topStates = states;
      LstmLayer.ForwardCache layer2ForwardCache = null;
      LstmLayer.ForwardCache layer2BackwardCache = null;
      double[][] reversedStates = null;
      double[][] stateMasks = null;
      if (wordForward2 != null) {
        if (settings.encoderDropout() > 0.0d) {
          stateMasks = encoderMasks(steps, 2 * hidden, encoderKeep, random);
          applyMasks(states, stateMasks);
        }
        layer2ForwardCache = LstmLayer.ForwardCache.of(steps, hidden);
        final double[][] hFwd2 = wordForward2.run(states, layer2ForwardCache);
        reversedStates = new double[steps][];
        for (int t = 0; t < steps; t++) {
          reversedStates[t] = states[steps - 1 - t];
        }
        layer2BackwardCache = LstmLayer.ForwardCache.of(steps, hidden);
        final double[][] hBwd2Reversed = wordBackward2.run(reversedStates,
            layer2BackwardCache);
        topStates = new double[steps][2 * hidden];
        for (int t = 0; t < steps; t++) {
          System.arraycopy(hFwd2[t], 0, topStates[t], 0, hidden);
          System.arraycopy(hBwd2Reversed[steps - 1 - t], 0, topStates[t], hidden,
              hidden);
        }
      }
      double[][] topMasks = null;
      if (settings.encoderDropout() > 0.0d) {
        topMasks = encoderMasks(steps, 2 * hidden, encoderKeep, random);
        applyMasks(topStates, topMasks);
      }

      final double[][] emissions = new double[steps][tags.length];
      for (int t = 0; t < steps; t++) {
        for (int o = 0; o < tags.length; o++) {
          final double[] row = outputWeights[o];
          double sum = outputBias[o];
          for (int j = 0; j < 2 * hidden; j++) {
            sum += row[j] * topStates[t][j];
          }
          emissions[t][o] = sum;
        }
      }
      final int[] goldIds = new int[steps];
      for (int t = 0; t < steps; t++) {
        goldIds[t] = tagIds.get(goldTags[t]);
      }
      final double[][] emissionGrads = new double[steps][tags.length];
      double loss;
      if (crfScorer != null) {
        loss = crfScorer.lossAndGradients(emissions, goldIds, transitionWeights,
            startWeights, endWeights, emissionGrads, worker.transitionGrads,
            worker.startGrads, worker.endGrads);
      }
      else {
        loss = 0.0d;
        for (int t = 0; t < steps; t++) {
          double max = Double.NEGATIVE_INFINITY;
          for (int o = 0; o < tags.length; o++) {
            max = Math.max(max, emissions[t][o]);
          }
          double total = 0.0d;
          for (int o = 0; o < tags.length; o++) {
            emissionGrads[t][o] = Math.exp(emissions[t][o] - max);
            total += emissionGrads[t][o];
          }
          loss += Math.log(total) - Math.log(emissionGrads[t][goldIds[t]]);
          for (int o = 0; o < tags.length; o++) {
            emissionGrads[t][o] = emissionGrads[t][o] / total
                - (o == goldIds[t] ? 1.0d : 0.0d);
          }
        }
      }
      final double[][] dTop = new double[steps][2 * hidden];
      for (int t = 0; t < steps; t++) {
        for (int o = 0; o < tags.length; o++) {
          final double gradient = emissionGrads[t][o];
          outputBiasGrads[o] += gradient;
          final double[] gradRow = outputWeightGrads[o];
          final double[] weightRow = outputWeights[o];
          for (int j = 0; j < 2 * hidden; j++) {
            gradRow[j] += gradient * topStates[t][j];
            dTop[t][j] += weightRow[j] * gradient;
          }
        }
      }

      if (xposWeights != null && sample.xpos() != null) {
        loss += auxiliaryLoss(sample.xpos(), xposIds, xposWeights, xposBias,
            worker.xposWeightGrads, worker.xposBiasGrads, topStates, dTop, hidden);
      }
      if (featsWeights != null && sample.feats() != null) {
        loss += auxiliaryLoss(sample.feats(), featsIds, featsWeights, featsBias,
            worker.featsWeightGrads, worker.featsBiasGrads, topStates, dTop, hidden);
      }

      if (topMasks != null) {
        applyMasks(dTop, topMasks);
      }

      double[][] dStates = dTop;
      if (wordForward2 != null) {
        final double[][] dTopFwd = new double[steps][hidden];
        final double[][] dTopBwd = new double[steps][hidden];
        for (int t = 0; t < steps; t++) {
          System.arraycopy(dTop[t], 0, dTopFwd[t], 0, hidden);
          System.arraycopy(dTop[t], hidden, dTopBwd[t], 0, hidden);
        }
        final double[][] dXs2Fwd = new double[steps][2 * hidden];
        wordForward2.backward(states, layer2ForwardCache, dTopFwd, dXs2Fwd,
            worker.wordForward2Grads);
        final double[][] dTopBwdReversed = new double[steps][hidden];
        for (int t = 0; t < steps; t++) {
          dTopBwdReversed[t] = dTopBwd[steps - 1 - t];
        }
        final double[][] dXs2BwdReversed = new double[steps][2 * hidden];
        wordBackward2.backward(reversedStates, layer2BackwardCache, dTopBwdReversed,
            dXs2BwdReversed, worker.wordBackward2Grads);
        dStates = new double[steps][2 * hidden];
        for (int t = 0; t < steps; t++) {
          final double[] dBwd = dXs2BwdReversed[steps - 1 - t];
          for (int i = 0; i < 2 * hidden; i++) {
            dStates[t][i] = dXs2Fwd[t][i] + dBwd[i];
          }
        }
        if (stateMasks != null) {
          applyMasks(dStates, stateMasks);
        }
      }
      final double[][] dHFwd = new double[steps][hidden];
      final double[][] dHBwd = new double[steps][hidden];
      for (int t = 0; t < steps; t++) {
        System.arraycopy(dStates[t], 0, dHFwd[t], 0, hidden);
        System.arraycopy(dStates[t], hidden, dHBwd[t], 0, hidden);
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
        final int wordRow = wordRows[t];
        for (int i = 0; i < wordSize; i++) {
          wordEmbeddingGrads[wordRow][i] += dx[i];
        }
        if (pretrainedRows[t] >= 0 && !blockDropped[t] && worker.pretrainedGrads != null) {
          final int offset = wordSize + 2 * charHidden;
          final double[] rowGrads = worker.pretrainedGrads.computeIfAbsent(
              pretrainedRows[t], row -> new double[pretrainedSize]);
          for (int i = 0; i < pretrainedSize; i++) {
            rowGrads[i] += dx[offset + i];
          }
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

    /**
     * Draws one inverted-dropout mask per timestep over an encoder state.
     *
     * @param steps The sentence length.
     * @param width The state width.
     * @param keep The keep probability.
     * @param random The mask stream.
     * @return The masks, entries {@code 0} or {@code 1/keep}. Never {@code null}.
     */
    private static double[][] encoderMasks(int steps, int width, double keep,
        Random random) {
      final double[][] masks = new double[steps][width];
      for (int t = 0; t < steps; t++) {
        for (int i = 0; i < width; i++) {
          masks[t][i] = random.nextDouble() < keep ? 1.0d / keep : 0.0d;
        }
      }
      return masks;
    }

    /**
     * Multiplies values elementwise by masks, in place; used identically on the
     * forward states and on their incoming gradients.
     *
     * @param values The values, mutated in place.
     * @param masks The masks, same shape.
     */
    private static void applyMasks(double[][] values, double[][] masks) {
      for (int t = 0; t < values.length; t++) {
        for (int i = 0; i < values[t].length; i++) {
          values[t][i] *= masks[t][i];
        }
      }
    }

    /**
     * Scores one auxiliary tagging over the shared encoder states with a softmax
     * cross-entropy head, accumulating the head's own gradients and adding the
     * encoder's share into {@code dTop}, both scaled by the auxiliary loss weight.
     *
     * @param gold The gold auxiliary labels.
     * @param ids The label vocabulary.
     * @param weights The head weights.
     * @param bias The head bias.
     * @param weightGrads The head weight gradient buffer.
     * @param biasGrads The head bias gradient buffer.
     * @param topStates The shared encoder states.
     * @param dTop The encoder-state gradient accumulator.
     * @param hidden The encoder hidden size per direction.
     * @return The weighted auxiliary loss of the sentence.
     */
    private double auxiliaryLoss(String[] gold, Map<String, Integer> ids,
        double[][] weights, double[] bias, double[][] weightGrads, double[] biasGrads,
        double[][] topStates, double[][] dTop, int hidden) {
      final double weight = settings.auxLossWeight();
      final int labels = weights.length;
      final double[] scores = new double[labels];
      double loss = 0.0d;
      for (int t = 0; t < gold.length; t++) {
        final Integer goldId = ids.get(gold[t]);
        if (goldId == null) {
          // a label outside the training vocabulary carries no signal
          continue;
        }
        double max = Double.NEGATIVE_INFINITY;
        for (int o = 0; o < labels; o++) {
          final double[] row = weights[o];
          double sum = bias[o];
          for (int j = 0; j < 2 * hidden; j++) {
            sum += row[j] * topStates[t][j];
          }
          scores[o] = sum;
          max = Math.max(max, sum);
        }
        double total = 0.0d;
        for (int o = 0; o < labels; o++) {
          scores[o] = Math.exp(scores[o] - max);
          total += scores[o];
        }
        loss += weight * (Math.log(total) - Math.log(scores[goldId]));
        for (int o = 0; o < labels; o++) {
          final double gradient =
              weight * (scores[o] / total - (o == goldId ? 1.0d : 0.0d));
          biasGrads[o] += gradient;
          final double[] gradRow = weightGrads[o];
          final double[] weightRow = weights[o];
          for (int j = 0; j < 2 * hidden; j++) {
            gradRow[j] += gradient * topStates[t][j];
            dTop[t][j] += weightRow[j] * gradient;
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

    private BilstmPOSModel toModel() {
      final float[][] vectors;
      if (pretrainedTrainable != null) {
        vectors = new float[pretrainedTrainable.length][];
        for (int r = 0; r < pretrainedTrainable.length; r++) {
          final double[] row = pretrainedTrainable[r];
          vectors[r] = new float[row.length];
          for (int i = 0; i < row.length; i++) {
            vectors[r][i] = (float) row[i];
          }
        }
      }
      else {
        vectors = pretrainedVectors;
      }
      return new BilstmPOSModel(words, chars, tags, wordEmbeddings, charEmbeddings,
          charForward, charBackward, wordForward, wordBackward, wordForward2,
          wordBackward2, outputWeights, outputBias, settings.maxWordLength(),
          pretrainedIds, vectors, transitionWeights, startWeights,
          endWeights);
    }
  }
}
