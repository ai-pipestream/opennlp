/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The model, the vocabulary and the fixed input set shared by the phase B measurement classes,
 * so the benchmark and the batch-plan report are provably reading the same bytes.
 *
 * <p>The inputs are the first {@link #POOL_SIZE} sentences of a Leipzig corpus file, taken in
 * file order with the leading line number and its tab removed. A call of size {@code n} is the
 * first {@code n} of those sentences, so every bar at a given call size embeds exactly the same
 * texts in exactly the same order and the selection is reproducible from the file alone.</p>
 */
final class BatchBenchmarkInputs {

  /** The number of sentences read from the head of the corpus, the largest call size measured. */
  static final int POOL_SIZE = 128;

  /** The call sizes measured. */
  static final int[] CALL_SIZES = {1, 8, 32, 64, 128};

  /** The tolerance the padded bars must match the per-input loop within. */
  static final float TOLERANCE = 1.0e-5f;

  private static final String MODEL_PROPERTY = "opennlp.bench.model";
  private static final String VOCABULARY_PROPERTY = "opennlp.bench.vocabulary";
  private static final String CORPUS_PROPERTY = "opennlp.bench.corpus";

  private static final String DEFAULT_MODEL =
      "/mnt/nas-corpus/opennlp-data/onnx/sentence-transformers/model.onnx";
  private static final String DEFAULT_VOCABULARY =
      "/mnt/nas-corpus/opennlp-data/onnx/sentence-transformers/vocab.txt";
  private static final String DEFAULT_CORPUS =
      "/mnt/nas-corpus/opennlp-data/leipzig/eng_news_2010_300K-sentences.txt";

  private BatchBenchmarkInputs() {
  }

  /** {@return the ONNX model file} */
  static File model() {
    return new File(System.getProperty(MODEL_PROPERTY, DEFAULT_MODEL));
  }

  /** {@return the vocabulary file of the model} */
  static File vocabulary() {
    return new File(System.getProperty(VOCABULARY_PROPERTY, DEFAULT_VOCABULARY));
  }

  /** {@return the corpus file the input set is read from} */
  static File corpus() {
    return new File(System.getProperty(CORPUS_PROPERTY, DEFAULT_CORPUS));
  }

  /**
   * {@return the fixed input set, {@link #POOL_SIZE} sentences in corpus order}
   *
   * <p>Each line of the corpus is a line number, a tab and the sentence. The number and the tab
   * are dropped and the sentence is kept exactly as it is otherwise. Lines whose sentence part
   * is blank are skipped so a call never contains an empty input by accident.</p>
   *
   * @throws IOException Thrown if the corpus cannot be read.
   */
  static List<String> pool() throws IOException {
    final List<String> sentences = new ArrayList<>(POOL_SIZE);
    for (final String line : Files.readAllLines(Path.of(corpus().getPath()),
        StandardCharsets.UTF_8)) {
      final int tab = line.indexOf('\t');
      final String sentence = (tab < 0 ? line : line.substring(tab + 1)).strip();
      if (!sentence.isEmpty()) {
        sentences.add(sentence);
        if (sentences.size() == POOL_SIZE) {
          return sentences;
        }
      }
    }
    throw new IOException("The corpus holds fewer than " + POOL_SIZE + " usable sentences: "
        + corpus());
  }

  /**
   * {@return the inputs of one call, the first {@code callSize} sentences of the pool}
   *
   * @param pool The input pool.
   * @param callSize The number of inputs per call.
   */
  static List<String> call(final List<String> pool, final int callSize) {
    return List.copyOf(pool.subList(0, callSize));
  }

  /**
   * {@return the largest absolute difference between two sets of vectors}
   *
   * @param expected The vectors of the per-input loop.
   * @param actual The vectors of the batched call.
   */
  static float deviation(final float[][] expected, final float[][] actual) {
    if (expected.length != actual.length) {
      throw new IllegalStateException("Vector counts differ: " + expected.length + " against "
          + actual.length);
    }
    float worst = 0;
    for (int i = 0; i < expected.length; i++) {
      if (expected[i].length != actual[i].length) {
        throw new IllegalStateException("Vector " + i + " has length " + actual[i].length
            + " but the loop produced " + expected[i].length);
      }
      for (int d = 0; d < expected[i].length; d++) {
        worst = Math.max(worst, Math.abs(expected[i][d] - actual[i][d]));
      }
    }
    return worst;
  }

}
