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

package opennlp.dl.vectors;

import java.io.File;
import java.io.IOException;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import opennlp.dl.AbstractDL;
import opennlp.dl.InferenceOptions;
import opennlp.tools.coref.TokenVectors;

/**
 * Produces one contextual vector per word of a sentence from a BERT-style encoder
 * exported to ONNX, such as a sentence-transformers model whose first output is the
 * last hidden state. Each word is wordpiece-tokenized on its own, so its pieces are
 * known, and the word's vector is the mean of its pieces' hidden states; a word that
 * yields no piece gets a zero vector. Words are packed into windows of at most
 * {@value #MAX_PIECES} pieces including the classification and separator tokens, each
 * window encoded on its own, so long sentences never exceed the encoder's input length.
 *
 * <p>Instances are thread-safe once constructed and until {@link #close()} is called.</p>
 *
 * @since 3.0.0
 */
public class TokenVectorsDL extends AbstractDL implements TokenVectors {

  /** The most pieces one encoder input holds, the BERT position limit. */
  public static final int MAX_PIECES = 512;

  /** The pieces a window spends on the classification and separator tokens. */
  private static final int SPECIAL_PIECES = 2;

  /** The classification and separator token ids, in that order. */
  private final long[] specials;

  /**
   * Initializes the encoder for an uncased model.
   *
   * @param model The ONNX model file. Must not be {@code null}.
   * @param vocabulary The vocabulary file. Must not be {@code null}.
   * @throws OrtException Thrown if the model cannot be loaded.
   * @throws IOException Thrown if the model or vocabulary cannot be read.
   */
  public TokenVectorsDL(final File model, final File vocabulary)
      throws OrtException, IOException {
    this(model, vocabulary, true);
  }

  /**
   * Initializes the encoder.
   *
   * @param model The ONNX model file. Must not be {@code null}.
   * @param vocabulary The vocabulary file. Must not be {@code null}.
   * @param lowerCase {@code true} for an uncased model, {@code false} for a cased one.
   * @throws OrtException Thrown if the model cannot be loaded.
   * @throws IOException Thrown if the model or vocabulary cannot be read.
   */
  public TokenVectorsDL(final File model, final File vocabulary, final boolean lowerCase)
      throws OrtException, IOException {
    super(model, vocabulary, new OrtSession.SessionOptions(), lowerCase);
    this.specials = specials();
  }

  /**
   * Initializes the encoder with inference options, for GPU execution.
   *
   * @param model The ONNX model file. Must not be {@code null}.
   * @param vocabulary The vocabulary file. Must not be {@code null}.
   * @param lowerCase {@code true} for an uncased model, {@code false} for a cased one.
   * @param options The inference options. Must not be {@code null}.
   * @throws OrtException Thrown if the model cannot be loaded.
   * @throws IOException Thrown if the model or vocabulary cannot be read.
   */
  public TokenVectorsDL(final File model, final File vocabulary, final boolean lowerCase,
      final InferenceOptions options) throws OrtException, IOException {
    super(model, vocabulary, sessionOptions(options), lowerCase);
    this.specials = specials();
  }

  /**
   * Initializes an encoder over shared state, for tests that exercise the tokenization
   * and windowing without a model.
   *
   * @param env The ONNX environment, or {@code null}.
   * @param session The ONNX session, or {@code null}.
   * @param vocab The vocabulary map.
   * @param lowerCase {@code true} for an uncased model.
   */
  protected TokenVectorsDL(final OrtEnvironment env, final OrtSession session,
      final Map<String, Integer> vocab, final boolean lowerCase) {
    super(env, session, vocab, lowerCase);
    this.specials = specials();
  }

  /** {@return the ids of the classification and separator tokens the tokenizer adds} */
  private long[] specials() {
    final String[] empty = tokenizer.tokenize("");
    return new long[] {id(empty[0]), id(empty[empty.length - 1])};
  }

  /**
   * {@inheritDoc} Each word's vector is the mean of its wordpiece hidden states.
   *
   * @throws IllegalArgumentException Thrown if {@code tokens} is {@code null} or empty.
   * @throws IllegalStateException Thrown if inference fails.
   */
  @Override
  public float[][] vectors(final String[] tokens) {
    if (tokens == null || tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be null or empty");
    }
    final List<long[]> pieces = new ArrayList<>(tokens.length);
    for (final String token : tokens) {
      pieces.add(pieceIds(token));
    }
    final float[][] vectors = new float[tokens.length][];
    for (final int[] window : windows(pieces, MAX_PIECES)) {
      encode(pieces, window[0], window[1], vectors);
    }
    return vectors;
  }

  /**
   * Splits a word into wordpiece ids without the surrounding special tokens.
   *
   * @param word The word.
   * @return Its piece ids, possibly empty. Never {@code null}.
   */
  long[] pieceIds(final String word) {
    final String[] pieces = tokenizer.tokenize(word);
    final int count = Math.max(0, pieces.length - SPECIAL_PIECES);
    final long[] ids = new long[count];
    for (int p = 0; p < count; p++) {
      ids[p] = id(pieces[p + 1]);
    }
    return ids;
  }

  /**
   * Packs consecutive words into windows whose pieces, plus the special tokens, fit
   * the limit. A single word longer than the limit forms a window of its own and is
   * truncated when encoded.
   *
   * @param pieces The piece ids of each word.
   * @param maxPieces The most pieces a window may hold.
   * @return The {@code [first word, last word exclusive]} bounds of each window.
   */
  static List<int[]> windows(final List<long[]> pieces, final int maxPieces) {
    final List<int[]> windows = new ArrayList<>();
    final int capacity = maxPieces - SPECIAL_PIECES;
    int start = 0;
    int used = 0;
    for (int w = 0; w < pieces.size(); w++) {
      final int count = pieces.get(w).length;
      if (w > start && used + count > capacity) {
        windows.add(new int[] {start, w});
        start = w;
        used = 0;
      }
      used += count;
    }
    windows.add(new int[] {start, pieces.size()});
    return windows;
  }

  /** Encodes the words of one window and writes their mean piece vectors. */
  private void encode(final List<long[]> pieces, final int from, final int to,
      final float[][] vectors) {
    final int capacity = MAX_PIECES - SPECIAL_PIECES;
    final List<Long> input = new ArrayList<>();
    input.add(specials[0]);
    final int[] firstPiece = new int[to - from];
    final int[] pieceCount = new int[to - from];
    for (int w = from; w < to; w++) {
      firstPiece[w - from] = input.size();
      for (final long piece : pieces.get(w)) {
        if (input.size() - 1 < capacity) {
          input.add(piece);
          pieceCount[w - from]++;
        }
      }
    }
    input.add(specials[1]);
    final long[] ids = new long[input.size()];
    final long[] ones = new long[ids.length];
    final long[] zeros = new long[ids.length];
    for (int p = 0; p < ids.length; p++) {
      ids[p] = input.get(p);
      ones[p] = 1L;
    }
    final float[][] hidden = run(ids, ones, zeros);
    for (int w = from; w < to; w++) {
      final int count = pieceCount[w - from];
      final float[] mean = new float[hidden[0].length];
      for (int p = 0; p < count; p++) {
        final float[] piece = hidden[firstPiece[w - from] + p];
        for (int d = 0; d < mean.length; d++) {
          mean[d] += piece[d];
        }
      }
      if (count > 1) {
        for (int d = 0; d < mean.length; d++) {
          mean[d] /= count;
        }
      }
      vectors[w] = mean;
    }
  }

  /** Runs the encoder over one input and returns the hidden state per position. */
  private float[][] run(final long[] ids, final long[] mask, final long[] types) {
    final Map<String, OnnxTensor> inputs = new HashMap<>();
    try {
      final long[] shape = {1, ids.length};
      inputs.put(INPUT_IDS, OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape));
      inputs.put(ATTENTION_MASK, OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape));
      inputs.put(TOKEN_TYPE_IDS, OnnxTensor.createTensor(env, LongBuffer.wrap(types), shape));
      try (OrtSession.Result result = session.run(inputs)) {
        final float[][][] hidden = (float[][][]) result.get(0).getValue();
        return hidden[0];
      }
    } catch (OrtException e) {
      throw new IllegalStateException("token encoding failed", e);
    } finally {
      inputs.values().forEach(OnnxTensor::close);
    }
  }

  /**
   * {@return the vocabulary id of a piece}
   *
   * @throws IllegalStateException Thrown if the tokenizer emitted a piece the
   *         vocabulary lacks.
   */
  private long id(final String piece) {
    final Integer id = vocab.get(piece);
    if (id == null) {
      throw new IllegalStateException("piece is not in the vocabulary: " + piece);
    }
    return id;
  }
}
