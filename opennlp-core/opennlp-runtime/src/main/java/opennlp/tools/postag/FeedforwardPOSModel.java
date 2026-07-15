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
      final float[] row = hiddenWeights[j];
      double sum = hiddenBias[j];
      for (int f = 0; f < features.length; f++) {
        final float[] embedding = embeddings[features[f]];
        final int offset = f * embeddingSize;
        for (int d = 0; d < embeddingSize; d++) {
          sum += row[offset + d] * embedding[d];
        }
      }
      h[j] = sum * sum * sum;
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

  /** Lowercases a word symbol; special symbols and absences pass through. */
  static String normalize(String word) {
    if (word == null) {
      return null;
    }
    return word.startsWith("*") ? word : StringUtil.toLowerCase(word);
  }

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

  float[][] embeddings() {
    return embeddings;
  }

  float[][] hiddenWeights() {
    return hiddenWeights;
  }

  float[] hiddenBias() {
    return hiddenBias;
  }

  float[][] outputWeights() {
    return outputWeights;
  }

  float[] outputBias() {
    return outputBias;
  }

  Map<String, Integer> wordIds() {
    return wordIds;
  }

  private static void writeVocabulary(DataOutputStream data, Map<String, Integer> ids)
      throws IOException {
    data.writeInt(ids.size());
    for (final Map.Entry<String, Integer> entry : ids.entrySet()) {
      data.writeUTF(entry.getKey());
      data.writeInt(entry.getValue());
    }
  }

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

  private static void writeVector(DataOutputStream data, float[] vector)
      throws IOException {
    data.writeInt(vector.length);
    for (final float value : vector) {
      data.writeFloat(value);
    }
  }

  private static float[] readVector(DataInputStream data) throws IOException {
    final float[] vector = new float[data.readInt()];
    for (int i = 0; i < vector.length; i++) {
      vector[i] = data.readFloat();
    }
    return vector;
  }
}
