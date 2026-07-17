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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;

import opennlp.tools.util.StringUtil;

/**
 * The weights of the feedforward tagger: embeddings for words, suffixes, word shapes,
 * and previous tags, one hidden layer with cube activation, and a tag output layer,
 * stored in a plain versioned binary format with no serialization framework involved.
 *
 * <p>This is the pure-Java neural tier for tagging: the network is executed with
 * ordinary array arithmetic, so tagging needs no native runtime. Unknown words and
 * suffixes fall back to learned unknown symbols; words are matched case-insensitively,
 * with capitalization carried by the shape features instead. Instances are immutable
 * and safe to share between threads.</p>
 *
 * @see FeedforwardPOSTagger
 * @see FeedforwardPOSTrainer
 * @since 3.0.0
 */
public class FeedforwardPOSModel {

  private static final String MAGIC = "ONLP-FFPT-1";

  static final String UNKNOWN = "*UNK*";
  static final String ABSENT = "*NULL*";

  /** The lazy scoring cache; {@code null} until {@link #enableScoringCache()}. */
  private volatile ContributionCache cache;

  private final Map<String, Integer> wordIds;
  private final Map<String, Integer> suffixIds;
  private final Map<String, Integer> shapeIds;
  private final Map<String, Integer> tagIds;
  private final String[] tags;

  private final int embeddingSize;
  private final float[][] embeddings;
  private final float[][] hiddenWeights;
  private final float[] hiddenBias;
  private final float[][] outputWeights;
  private final float[] outputBias;

  /**
   * Initializes a model from its vocabularies and weight arrays. The trainer is the
   * only caller; the arrays are taken over without copying and must not be mutated
   * afterwards except by the trainer that created them.
   *
   * @param wordIds The word symbol to embedding row mapping.
   * @param suffixIds The suffix symbol to embedding row mapping.
   * @param shapeIds The shape symbol to embedding row mapping.
   * @param tagIds The tag symbol to embedding row mapping.
   * @param tags The tag inventory by output index.
   * @param embeddingSize The embedding dimensionality.
   * @param embeddings The embedding matrix, one row per symbol.
   * @param hiddenWeights The hidden layer weight matrix.
   * @param hiddenBias The hidden layer bias vector.
   * @param outputWeights The output layer weight matrix.
   * @param outputBias The output layer bias vector.
   */
  FeedforwardPOSModel(Map<String, Integer> wordIds, Map<String, Integer> suffixIds,
      Map<String, Integer> shapeIds, Map<String, Integer> tagIds, String[] tags,
      int embeddingSize, float[][] embeddings, float[][] hiddenWeights,
      float[] hiddenBias, float[][] outputWeights, float[] outputBias) {
    this.wordIds = wordIds;
    this.suffixIds = suffixIds;
    this.shapeIds = shapeIds;
    this.tagIds = tagIds;
    this.tags = tags;
    this.embeddingSize = embeddingSize;
    this.embeddings = embeddings;
    this.hiddenWeights = hiddenWeights;
    this.hiddenBias = hiddenBias;
    this.outputWeights = outputWeights;
    this.outputBias = outputBias;
  }

  /**
   * Scores every tag for a position described by embedding row indices.
   *
   * @param features The embedding rows of the position, as produced by
   *                 {@link #featureIds(String[])}. Must not be {@code null}.
   * @return One unnormalized score per tag, indexed like {@link #tags()}. Never
   *         {@code null}.
   */
  public double[] score(int[] features) {
    final int hidden = hiddenBias.length;
    final double[] h = new double[hidden];
    for (int j = 0; j < hidden; j++) {
      h[j] = hiddenBias[j];
    }
    final ContributionCache cache = this.cache;
    for (int f = 0; f < features.length; f++) {
      final int row = features[f];
      final float[] contribution = cache == null ? null : cache.contribution(this, f, row);
      if (contribution != null) {
        for (int j = 0; j < hidden; j++) {
          h[j] += contribution[j];
        }
      } else {
        final float[] embedding = embeddings[row];
        final int offset = f * embeddingSize;
        for (int j = 0; j < hidden; j++) {
          final float[] weights = hiddenWeights[j];
          double sum = 0.0;
          for (int d = 0; d < embeddingSize; d++) {
            sum += weights[offset + d] * embedding[d];
          }
          h[j] += sum;
        }
      }
    }
    for (int j = 0; j < hidden; j++) {
      h[j] = h[j] * h[j] * h[j];
    }
    final double[] scores = new double[tags.length];
    for (int o = 0; o < scores.length; o++) {
      final float[] row = outputWeights[o];
      double sum = outputBias[o];
      for (int j = 0; j < hidden; j++) {
        sum += row[j] * h[j];
      }
      scores[o] = sum;
    }
    return scores;
  }

  /**
   * Turns on the scoring cache: the hidden-layer contribution of a (template slot,
   * embedding row) pair is a fixed vector for a frozen model, so it is computed once
   * on first sight and afterwards added instead of being re-derived from the
   * embedding on every token. Suffix, shape, and tag rows, whose inventories are
   * small, are fully cached almost immediately; word rows follow their frequency,
   * which is the adaptive form of the precomputation described for this architecture
   * by Chen and Manning (2014).
   *
   * <p>Cached contributions are rounded to floats once, so scores may differ from the
   * uncached path in the last bits; tag decisions are unaffected at any realistic
   * margin. The cache is bounded, safe for concurrent readers, and only valid on a
   * model whose weights no longer change: the trainer works on models it never
   * exposes until frozen.</p>
   */
  void enableScoringCache() {
    if (cache == null) {
      cache = new ContributionCache(FeedforwardPOSContext.SLOTS, embeddings.length);
    }
  }

  /**
   * The bounded lazy contribution cache behind {@link #enableScoringCache()}: one
   * slot per (template slot, embedding row) pair, filled on first use. Filling is
   * idempotent, so concurrent readers may compute a contribution twice but never see
   * a partial one, and a shared budget bounds the total memory; pairs beyond the
   * budget simply keep the direct path.
   */
  private static final class ContributionCache {

    /** The most (slot, row) pairs the cache will hold; typical models stay far below
     * the cap because the non-word inventories are small and word usage is
     * Zipf-shaped. */
    private static final int MAX_PAIRS = 65536;

    private final AtomicReferenceArray<float[]>[] bySlot;
    private final AtomicInteger remaining = new AtomicInteger(MAX_PAIRS);

    @SuppressWarnings("unchecked")
    private ContributionCache(int slots, int rows) {
      bySlot = new AtomicReferenceArray[slots];
      for (int f = 0; f < slots; f++) {
        bySlot[f] = new AtomicReferenceArray<>(rows);
      }
    }

    /**
     * Returns the cached hidden-layer contribution of one pair, computing and
     * publishing it on first sight while the budget lasts.
     *
     * @param model The frozen model the contributions derive from.
     * @param slot The template slot.
     * @param row The embedding row at that slot.
     * @return The contribution vector, or {@code null} when the budget is spent and
     *         the pair is not cached.
     */
    private float[] contribution(FeedforwardPOSModel model, int slot, int row) {
      final AtomicReferenceArray<float[]> slots = bySlot[slot];
      float[] contribution = slots.get(row);
      if (contribution != null) {
        return contribution;
      }
      if (remaining.get() <= 0) {
        return null;
      }
      final int hidden = model.hiddenBias.length;
      final float[] embedding = model.embeddings[row];
      final int offset = slot * model.embeddingSize;
      contribution = new float[hidden];
      for (int j = 0; j < hidden; j++) {
        final float[] weights = model.hiddenWeights[j];
        double sum = 0.0;
        for (int d = 0; d < model.embeddingSize; d++) {
          sum += weights[offset + d] * embedding[d];
        }
        contribution[j] = (float) sum;
      }
      if (slots.compareAndSet(row, null, contribution)) {
        remaining.decrementAndGet();
      } else {
        contribution = slots.get(row);
      }
      return contribution;
    }
  }

  /**
   * Maps the symbolic features of the tagger template onto embedding rows.
   *
   * @param symbols The symbolic features. Must not be {@code null}.
   * @return The embedding row per feature. Never {@code null}.
   */
  public int[] featureIds(String[] symbols) {
    final int[] ids = new int[symbols.length];
    int slot = 0;
    for (int i = 0; i < FeedforwardPOSContext.WORD_SLOTS; i++, slot++) {
      ids[slot] = lookup(wordIds, normalize(symbols[slot]));
    }
    for (int i = 0; i < FeedforwardPOSContext.SUFFIX_SLOTS; i++, slot++) {
      ids[slot] = lookup(suffixIds, symbols[slot]);
    }
    for (int i = 0; i < FeedforwardPOSContext.SHAPE_SLOTS; i++, slot++) {
      ids[slot] = lookup(shapeIds, symbols[slot]);
    }
    for (int i = 0; i < FeedforwardPOSContext.TAG_SLOTS; i++, slot++) {
      ids[slot] = lookup(tagIds, symbols[slot]);
    }
    return ids;
  }

  /**
   * @return The tag inventory by output index. Never {@code null}.
   */
  public String[] tags() {
    return tags.clone();
  }

  /**
   * Lowercases a word symbol for the case-insensitive vocabulary lookup. Special
   * symbols starting with {@code *} and absent positions pass through unchanged.
   *
   * @param word The word symbol, or {@code null} for an absent position.
   * @return The normalized symbol, or {@code null} if {@code word} was {@code null}.
   */
  static String normalize(String word) {
    if (word == null) {
      return null;
    }
    return word.startsWith("*") ? word : StringUtil.toLowerCase(word);
  }

  /**
   * Resolves a symbol to its embedding row. An absent position maps to the padding
   * symbol and a symbol outside the vocabulary maps to the unknown symbol, so the
   * lookup always succeeds.
   *
   * @param ids The vocabulary to look the symbol up in.
   * @param symbol The symbol, or {@code null} for an absent position.
   * @return The embedding row index.
   */
  private static int lookup(Map<String, Integer> ids, String symbol) {
    Integer id = ids.get(symbol == null ? ABSENT : symbol);
    if (id == null) {
      id = ids.get(UNKNOWN);
    }
    return id;
  }

  /**
   * Writes the model in the versioned binary format.
   *
   * @param out The stream to write to. Must not be {@code null}. Not closed.
   * @throws IOException Thrown if writing fails.
   */
  public void serialize(OutputStream out) throws IOException {
    if (out == null) {
      throw new IllegalArgumentException("out must not be null");
    }
    final DataOutputStream data = new DataOutputStream(new BufferedOutputStream(out));
    data.writeUTF(MAGIC);
    writeVocabulary(data, wordIds);
    writeVocabulary(data, suffixIds);
    writeVocabulary(data, shapeIds);
    writeVocabulary(data, tagIds);
    data.writeInt(tags.length);
    for (final String tag : tags) {
      data.writeUTF(tag);
    }
    data.writeInt(embeddingSize);
    writeMatrix(data, embeddings);
    writeMatrix(data, hiddenWeights);
    writeVector(data, hiddenBias);
    writeMatrix(data, outputWeights);
    writeVector(data, outputBias);
    data.flush();
  }

  /**
   * Loads a model from the versioned binary format.
   *
   * @param in The stream to read from. Must not be {@code null}. Not closed.
   * @return The loaded model. Never {@code null}.
   * @throws IOException Thrown if reading fails or the content is not this format.
   */
  public static FeedforwardPOSModel load(InputStream in) throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("in must not be null");
    }
    final DataInputStream data = new DataInputStream(new BufferedInputStream(in));
    final String magic = data.readUTF();
    if (!MAGIC.equals(magic)) {
      throw new IOException("not a feedforward tagger model: " + magic);
    }
    final Map<String, Integer> wordIds = readVocabulary(data);
    final Map<String, Integer> suffixIds = readVocabulary(data);
    final Map<String, Integer> shapeIds = readVocabulary(data);
    final Map<String, Integer> tagIds = readVocabulary(data);
    final String[] tags = new String[data.readInt()];
    for (int i = 0; i < tags.length; i++) {
      tags[i] = data.readUTF();
    }
    final int embeddingSize = data.readInt();
    return new FeedforwardPOSModel(wordIds, suffixIds, shapeIds, tagIds, tags,
        embeddingSize, readMatrix(data), readMatrix(data), readVector(data),
        readMatrix(data), readVector(data));
  }

  /**
   * Loads a model from a file.
   *
   * @param path The file to read. Must not be {@code null}.
   * @return The loaded model. Never {@code null}.
   * @throws IOException Thrown if reading fails or the content is not this format.
   */
  public static FeedforwardPOSModel load(Path path) throws IOException {
    if (path == null) {
      throw new IllegalArgumentException("path must not be null");
    }
    try (InputStream in = Files.newInputStream(path)) {
      return load(in);
    }
  }

  /** @return The embedding matrix, exposed to the trainer for in-place updates. */
  float[][] embeddings() {
    return embeddings;
  }

  /** @return The hidden layer weights, exposed to the trainer for in-place updates. */
  float[][] hiddenWeights() {
    return hiddenWeights;
  }

  /** @return The hidden layer bias, exposed to the trainer for in-place updates. */
  float[] hiddenBias() {
    return hiddenBias;
  }

  /** @return The output layer weights, exposed to the trainer for in-place updates. */
  float[][] outputWeights() {
    return outputWeights;
  }

  /** @return The output layer bias, exposed to the trainer for in-place updates. */
  float[] outputBias() {
    return outputBias;
  }

  /** @return The word symbol to embedding row mapping. */
  Map<String, Integer> wordIds() {
    return wordIds;
  }

  /**
   * Writes a vocabulary as its size followed by every symbol and row index pair.
   *
   * @param data The stream to write to.
   * @param ids The vocabulary to write.
   * @throws IOException Thrown if writing fails.
   */
  private static void writeVocabulary(DataOutputStream data, Map<String, Integer> ids)
      throws IOException {
    data.writeInt(ids.size());
    for (final Map.Entry<String, Integer> entry : ids.entrySet()) {
      data.writeUTF(entry.getKey());
      data.writeInt(entry.getValue());
    }
  }

  /**
   * Reads a vocabulary written by {@link #writeVocabulary}, preserving entry order.
   *
   * @param data The stream to read from.
   * @return The restored vocabulary. Never {@code null}.
   * @throws IOException Thrown if reading fails.
   */
  private static Map<String, Integer> readVocabulary(DataInputStream data)
      throws IOException {
    final int size = data.readInt();
    final Map<String, Integer> ids = new LinkedHashMap<>(size * 2);
    for (int i = 0; i < size; i++) {
      final String key = data.readUTF();
      ids.put(key, data.readInt());
    }
    return ids;
  }

  /**
   * Writes a matrix as its row and column counts followed by the values in row order.
   *
   * @param data The stream to write to.
   * @param matrix The matrix to write.
   * @throws IOException Thrown if writing fails.
   */
  private static void writeMatrix(DataOutputStream data, float[][] matrix)
      throws IOException {
    data.writeInt(matrix.length);
    data.writeInt(matrix.length == 0 ? 0 : matrix[0].length);
    for (final float[] row : matrix) {
      for (final float value : row) {
        data.writeFloat(value);
      }
    }
  }

  /**
   * Reads a matrix written by {@link #writeMatrix}.
   *
   * @param data The stream to read from.
   * @return The restored matrix. Never {@code null}.
   * @throws IOException Thrown if reading fails.
   */
  private static float[][] readMatrix(DataInputStream data) throws IOException {
    final int rows = data.readInt();
    final int columns = data.readInt();
    final float[][] matrix = new float[rows][columns];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        matrix[r][c] = data.readFloat();
      }
    }
    return matrix;
  }

  /**
   * Writes a vector as its length followed by the values.
   *
   * @param data The stream to write to.
   * @param vector The vector to write.
   * @throws IOException Thrown if writing fails.
   */
  private static void writeVector(DataOutputStream data, float[] vector)
      throws IOException {
    data.writeInt(vector.length);
    for (final float value : vector) {
      data.writeFloat(value);
    }
  }

  /**
   * Reads a vector written by {@link #writeVector}.
   *
   * @param data The stream to read from.
   * @return The restored vector. Never {@code null}.
   * @throws IOException Thrown if reading fails.
   */
  private static float[] readVector(DataInputStream data) throws IOException {
    final float[] vector = new float[data.readInt()];
    for (int i = 0; i < vector.length; i++) {
      vector[i] = data.readFloat();
    }
    return vector;
  }
}
