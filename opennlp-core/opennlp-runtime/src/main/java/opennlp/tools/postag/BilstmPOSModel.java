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
import java.util.Locale;
import java.util.Map;

/**
 * The trained parameters of the bidirectional LSTM POS tagger: word and character
 * embeddings, a character-level BiLSTM that builds word representations, a word-level
 * BiLSTM that encodes the sentence, a linear tag scorer, and an optional frozen
 * pretrained-vector table concatenated into the word representation.
 *
 * <p>Row zero of the word and character vocabularies is the unknown entry, so any
 * token can be encoded. A model is immutable once built and carries no per-call state,
 * which makes inference thread-safe; {@link #enableRepresentationCache()} may be
 * called by consumers that only read the model to skip recomputing word
 * representations of repeated tokens.</p>
 *
 * <p>The serialized form is a self-describing versioned binary with magic
 * {@code ONLP-BLPT-1}, following the same layout conventions as
 * {@link FeedforwardPOSModel}.</p>
 *
 * @see BilstmPOSTrainer
 * @see BilstmPOSTagger
 * @since 3.0.0
 */
public class BilstmPOSModel {

  static final String MAGIC = "ONLP-BLPT-1";
  static final String UNKNOWN = "*UNK*";

  private final LinkedHashMap<String, Integer> words;
  private final LinkedHashMap<String, Integer> chars;
  private final String[] tags;
  private final double[][] wordEmbeddings;
  private final double[][] charEmbeddings;
  private final LstmLayer charForward;
  private final LstmLayer charBackward;
  private final LstmLayer wordForward;
  private final LstmLayer wordBackward;
  private final double[][] outputWeights;
  private final double[] outputBias;
  private final int maxWordLength;
  private final int pretrainedSize;
  private final LinkedHashMap<String, Integer> pretrainedIds;
  private final float[][] pretrainedVectors;

  private volatile Map<String, double[]> representationCache;

  BilstmPOSModel(LinkedHashMap<String, Integer> words, LinkedHashMap<String, Integer> chars,
      String[] tags, double[][] wordEmbeddings, double[][] charEmbeddings,
      LstmLayer charForward, LstmLayer charBackward, LstmLayer wordForward,
      LstmLayer wordBackward, double[][] outputWeights, double[] outputBias,
      int maxWordLength, LinkedHashMap<String, Integer> pretrainedIds,
      float[][] pretrainedVectors) {
    this.words = words;
    this.chars = chars;
    this.tags = tags;
    this.wordEmbeddings = wordEmbeddings;
    this.charEmbeddings = charEmbeddings;
    this.charForward = charForward;
    this.charBackward = charBackward;
    this.wordForward = wordForward;
    this.wordBackward = wordBackward;
    this.outputWeights = outputWeights;
    this.outputBias = outputBias;
    this.maxWordLength = maxWordLength;
    this.pretrainedSize = pretrainedVectors != null ? pretrainedVectors[0].length : 0;
    this.pretrainedIds = pretrainedIds;
    this.pretrainedVectors = pretrainedVectors;
  }

  /**
   * @return The tag inventory, in model order. A defensive copy, never {@code null}.
   */
  public String[] tags() {
    return tags.clone();
  }

  /**
   * @return The number of hidden units per direction of the sentence BiLSTM.
   */
  int hiddenSize() {
    return wordForward.hiddenSize();
  }

  /**
   * @return The dimension of a word representation: learned word embedding plus both
   *         character BiLSTM directions plus the frozen pretrained vector when present.
   */
  int representationSize() {
    return wordEmbeddings[0].length + 2 * charForward.hiddenSize() + pretrainedSize;
  }

  /**
   * @return The maximum number of characters of a token fed to the character BiLSTM.
   */
  public int maxWordLength() {
    return maxWordLength;
  }

  /**
   * Allows a consumer that only reads this model to memoize word representations per
   * token string. A word representation is a pure function of the token, so the cache
   * is sound on a frozen model and invalid during training; trainers must never call
   * this.
   */
  public void enableRepresentationCache() {
    representationCache = new java.util.concurrent.ConcurrentHashMap<>();
  }

  /**
   * Normalizes a token for vocabulary lookup by lowercasing it; the built-in symbols
   * are returned unchanged.
   *
   * @param token The token. Must not be {@code null}.
   * @return The lookup form. Never {@code null}.
   */
  static String normalize(String token) {
    return token.startsWith("*") ? token : token.toLowerCase(Locale.ROOT);
  }

  /**
   * Maps a token to its word-vocabulary row, unknown words to row zero.
   *
   * @param token The token. Must not be {@code null}.
   * @return The embedding row.
   */
  int wordId(String token) {
    return words.getOrDefault(normalize(token), 0);
  }

  /**
   * Maps a token to its character-vocabulary rows, capped at {@link #maxWordLength()}
   * leading characters, unknown characters to row zero.
   *
   * @param token The token. Must not be {@code null}.
   * @return One row per kept character, empty for an empty token. Never {@code null}.
   */
  int[] charIds(String token) {
    final int length = Math.min(token.length(), maxWordLength);
    final int[] ids = new int[length];
    for (int i = 0; i < length; i++) {
      ids[i] = chars.getOrDefault(String.valueOf(token.charAt(i)), 0);
    }
    return ids;
  }

  /**
   * Looks up the frozen pretrained vector for a token.
   *
   * @param token The token. Must not be {@code null}.
   * @return The stored vector, or {@code null} when the model has no table or the
   *         token is not in it.
   */
  float[] pretrainedVector(String token) {
    if (pretrainedIds == null) {
      return null;
    }
    final Integer row = pretrainedIds.get(normalize(token));
    return row != null ? pretrainedVectors[row] : null;
  }

  /**
   * Builds the word representation of one token: learned word embedding, both
   * character BiLSTM final states, and the frozen pretrained vector (zeros when the
   * token has none). Allocates fresh result arrays, so concurrent calls are safe on a
   * model that is not being mutated.
   *
   * @param token The token. Must not be {@code null}.
   * @return The representation, length {@link #representationSize()}. Never
   *         {@code null}.
   */
  double[] wordRepresentation(String token) {
    if (representationCache != null) {
      final double[] cached = representationCache.get(token);
      if (cached != null) {
        return cached.clone();
      }
    }
    final double[] representation = computeRepresentation(token);
    if (representationCache != null && representationCache.size() < 100_000) {
      representationCache.putIfAbsent(token, representation.clone());
    }
    return representation;
  }

  private double[] computeRepresentation(String token) {
    final double[] wordEmbedding = wordEmbeddings[wordId(token)];
    final int charHidden = charForward.hiddenSize();
    final double[] representation =
        new double[wordEmbedding.length + 2 * charHidden + pretrainedSize];
    System.arraycopy(wordEmbedding, 0, representation, 0, wordEmbedding.length);
    final int[] ids = charIds(token);
    if (ids.length > 0) {
      final double[][] charXs = new double[ids.length][];
      for (int i = 0; i < ids.length; i++) {
        charXs[i] = charEmbeddings[ids[i]];
      }
      final LstmLayer.ForwardCache fwdCache =
          LstmLayer.ForwardCache.of(ids.length, charHidden);
      final double[][] hFwd = charForward.run(charXs, fwdCache);
      System.arraycopy(hFwd[ids.length - 1], 0, representation, wordEmbedding.length,
          charHidden);
      final double[][] reversed = new double[ids.length][];
      for (int i = 0; i < ids.length; i++) {
        reversed[i] = charXs[ids.length - 1 - i];
      }
      final LstmLayer.ForwardCache bwdCache =
          LstmLayer.ForwardCache.of(ids.length, charHidden);
      final double[][] hBwd = charBackward.run(reversed, bwdCache);
      System.arraycopy(hBwd[ids.length - 1], 0, representation,
          wordEmbedding.length + charHidden, charHidden);
    }
    final float[] pretrained = pretrainedVector(token);
    if (pretrained != null) {
      final int offset = wordEmbedding.length + 2 * charHidden;
      for (int i = 0; i < pretrainedSize; i++) {
        representation[offset + i] = pretrained[i];
      }
    }
    return representation;
  }

  /**
   * Scores every tag at every position of a sentence: encodes each token, runs the
   * sentence BiLSTM in both directions, and applies the linear tagger to the
   * concatenated states.
   *
   * @param tokens The sentence. Must not be {@code null} or empty.
   * @return The unnormalized tag scores, {@code [tokens.length][tags.length]}.
   *         Never {@code null}.
   */
  public double[][] score(String[] tokens) {
    if (tokens == null || tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be null or empty");
    }
    final int steps = tokens.length;
    final double[][] xs = new double[steps][];
    for (int t = 0; t < steps; t++) {
      xs[t] = wordRepresentation(tokens[t]);
    }
    final int hidden = wordForward.hiddenSize();
    final LstmLayer.ForwardCache fwdCache = LstmLayer.ForwardCache.of(steps, hidden);
    final double[][] hFwd = wordForward.run(xs, fwdCache);
    final double[][] reversed = new double[steps][];
    for (int t = 0; t < steps; t++) {
      reversed[t] = xs[steps - 1 - t];
    }
    final LstmLayer.ForwardCache bwdCache = LstmLayer.ForwardCache.of(steps, hidden);
    final double[][] hBwdRev = wordBackward.run(reversed, bwdCache);
    final double[][] scores = new double[steps][tags.length];
    for (int t = 0; t < steps; t++) {
      final double[] hF = hFwd[t];
      final double[] hB = hBwdRev[steps - 1 - t];
      for (int o = 0; o < tags.length; o++) {
        final double[] row = outputWeights[o];
        double sum = outputBias[o];
        for (int j = 0; j < hidden; j++) {
          sum += row[j] * hF[j] + row[hidden + j] * hB[j];
        }
        scores[t][o] = sum;
      }
    }
    return scores;
  }

  /**
   * Serializes this model in the versioned {@code ONLP-BLPT-1} binary format.
   *
   * @param out The stream to write to; not closed. Must not be {@code null}.
   * @throws IOException Thrown if writing fails.
   */
  public void serialize(OutputStream out) throws IOException {
    final DataOutputStream data =
        new DataOutputStream(new BufferedOutputStream(out));
    data.writeUTF(MAGIC);
    writeVocabulary(data, words);
    writeVocabulary(data, chars);
    data.writeInt(tags.length);
    for (final String tag : tags) {
      data.writeUTF(tag);
    }
    writeMatrix(data, wordEmbeddings);
    writeMatrix(data, charEmbeddings);
    writeLstm(data, charForward);
    writeLstm(data, charBackward);
    writeLstm(data, wordForward);
    writeLstm(data, wordBackward);
    writeMatrix(data, outputWeights);
    writeVector(data, outputBias);
    data.writeInt(maxWordLength);
    data.writeBoolean(pretrainedIds != null);
    if (pretrainedIds != null) {
      writeVocabulary(data, pretrainedIds);
      data.writeInt(pretrainedSize);
      data.writeInt(pretrainedVectors.length);
      for (final float[] row : pretrainedVectors) {
        for (final float value : row) {
          data.writeFloat(value);
        }
      }
    }
    data.flush();
  }

  /**
   * Serializes this model to a file, replacing any existing content.
   *
   * @param file The target path. Must not be {@code null}.
   * @throws IOException Thrown if writing fails.
   */
  public void serialize(Path file) throws IOException {
    try (OutputStream out = Files.newOutputStream(file)) {
      serialize(out);
    }
  }

  /**
   * Loads a model from the versioned binary format.
   *
   * @param in The stream to read from; not closed. Must not be {@code null}.
   * @return The loaded model. Never {@code null}.
   * @throws IOException Thrown if reading fails or the content is not an
   *         {@code ONLP-BLPT-1} model.
   */
  public static BilstmPOSModel load(InputStream in) throws IOException {
    final DataInputStream data = new DataInputStream(new BufferedInputStream(in));
    final String magic = data.readUTF();
    if (!MAGIC.equals(magic)) {
      throw new IOException("not an ONLP-BLPT-1 model: " + magic);
    }
    final LinkedHashMap<String, Integer> words = readVocabulary(data);
    final LinkedHashMap<String, Integer> chars = readVocabulary(data);
    final int tagCount = data.readInt();
    final String[] tags = new String[tagCount];
    for (int i = 0; i < tagCount; i++) {
      tags[i] = data.readUTF();
    }
    final double[][] wordEmbeddings = readMatrix(data);
    final double[][] charEmbeddings = readMatrix(data);
    final LstmLayer charForward = readLstm(data);
    final LstmLayer charBackward = readLstm(data);
    final LstmLayer wordForward = readLstm(data);
    final LstmLayer wordBackward = readLstm(data);
    final double[][] outputWeights = readMatrix(data);
    final double[] outputBias = readVector(data);
    final int maxWordLength = data.readInt();
    final boolean hasPretrained = data.readBoolean();
    LinkedHashMap<String, Integer> pretrainedIds = null;
    float[][] pretrainedVectors = null;
    if (hasPretrained) {
      pretrainedIds = readVocabulary(data);
      final int dimension = data.readInt();
      final int rows = data.readInt();
      pretrainedVectors = new float[rows][dimension];
      for (int r = 0; r < rows; r++) {
        for (int i = 0; i < dimension; i++) {
          pretrainedVectors[r][i] = data.readFloat();
        }
      }
    }
    return new BilstmPOSModel(words, chars, tags, wordEmbeddings, charEmbeddings,
        charForward, charBackward, wordForward, wordBackward, outputWeights,
        outputBias, maxWordLength, pretrainedIds, pretrainedVectors);
  }

  /**
   * Loads a model from a file.
   *
   * @param file The model file. Must not be {@code null}.
   * @return The loaded model. Never {@code null}.
   * @throws IOException Thrown if reading fails or the content is not an
   *         {@code ONLP-BLPT-1} model.
   */
  public static BilstmPOSModel load(Path file) throws IOException {
    try (InputStream in = Files.newInputStream(file)) {
      return load(in);
    }
  }

  private static void writeLstm(DataOutputStream data, LstmLayer layer)
      throws IOException {
    data.writeInt(layer.inputSize());
    data.writeInt(layer.hiddenSize());
    writeMatrix(data, layer.w());
    writeMatrix(data, layer.u());
    writeVector(data, layer.b());
  }

  private static LstmLayer readLstm(DataInputStream data) throws IOException {
    final int inputSize = data.readInt();
    final int hiddenSize = data.readInt();
    final double[][] w = readMatrix(data);
    final double[][] u = readMatrix(data);
    final double[] b = readVector(data);
    return LstmLayer.ofWeights(inputSize, hiddenSize, w, u, b);
  }

  private static void writeVocabulary(DataOutputStream data,
      LinkedHashMap<String, Integer> vocabulary) throws IOException {
    data.writeInt(vocabulary.size());
    for (final Map.Entry<String, Integer> entry : vocabulary.entrySet()) {
      data.writeUTF(entry.getKey());
      data.writeInt(entry.getValue());
    }
  }

  private static LinkedHashMap<String, Integer> readVocabulary(DataInputStream data)
      throws IOException {
    final int size = data.readInt();
    final LinkedHashMap<String, Integer> vocabulary = new LinkedHashMap<>();
    for (int i = 0; i < size; i++) {
      vocabulary.put(data.readUTF(), data.readInt());
    }
    return vocabulary;
  }

  private static void writeMatrix(DataOutputStream data, double[][] matrix)
      throws IOException {
    data.writeInt(matrix.length);
    data.writeInt(matrix[0].length);
    for (final double[] row : matrix) {
      for (final double value : row) {
        data.writeDouble(value);
      }
    }
  }

  private static double[][] readMatrix(DataInputStream data) throws IOException {
    final int rows = data.readInt();
    final int cols = data.readInt();
    final double[][] matrix = new double[rows][cols];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        matrix[r][c] = data.readDouble();
      }
    }
    return matrix;
  }

  private static void writeVector(DataOutputStream data, double[] vector)
      throws IOException {
    data.writeInt(vector.length);
    for (final double value : vector) {
      data.writeDouble(value);
    }
  }

  private static double[] readVector(DataInputStream data) throws IOException {
    final double[] vector = new double[data.readInt()];
    for (int i = 0; i < vector.length; i++) {
      vector[i] = data.readDouble();
    }
    return vector;
  }
}
