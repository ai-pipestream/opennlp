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

package opennlp.dl;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OrtException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import opennlp.dl.vectors.PaddingStrategy;
import opennlp.dl.vectors.Pooling;
import opennlp.dl.vectors.SentenceVectorsDL;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * That {@link SentenceVectorsDL} really does read its output through a pinned output tensor, and
 * that doing so changes none of its vectors.
 *
 * <p>This test lives in {@code opennlp.dl} rather than beside the other embedder tests because the
 * evidence it needs is package private to {@code opennlp.dl}: the arena count and the direct byte
 * total of the {@link OnnxInference} the embedder holds. Reading a component's internals is how a
 * claim about which read path a caller takes becomes falsifiable rather than a comment.</p>
 *
 * <p>Against the deterministic {@code tiny-vectors.onnx} graph, which computes
 * {@code output[b][t] = float(input_ids[b][t]) * W} with {@code W = [0.5, -1, 2]}, and
 * {@code tiny-pooled.onnx}, which sums a row's token vectors inside the graph and so has the rank 2
 * output the embedder also has to pin.</p>
 */
class SentenceVectorsDLPinnedOutputTest {

  /** The hidden size of both tiny graphs. */
  private static final int HIDDEN = 3;

  private static final int FLOAT_BYTES = 4;

  /** Reaches the {@link OnnxInference} of a component, which {@link AbstractDL} keeps protected. */
  private static final class Probing extends SentenceVectorsDL {

    private Probing(final File model, final File vocabulary, final Pooling pooling,
        final boolean normalize, final PaddingStrategy padding) throws OrtException, IOException {
      super(model, vocabulary, true, pooling, normalize, SentenceVectorsDL.DEFAULT_MAX_LENGTH,
          padding);
    }

    private OnnxInference inference() {
      return inference;
    }
  }

  // Copied out of the classpath rather than resolved in place, for the same reason as in
  // SentenceVectorsDLEmbedderTest: the resource may live inside a test-jar.
  private static File model(final Path dir, final String name) throws IOException {
    final Path file = dir.resolve(name);
    try (InputStream is = Objects.requireNonNull(SentenceVectorsDLPinnedOutputTest.class
        .getResourceAsStream("/opennlp/dl/vectors/" + name))) {
      Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
    }
    return file.toFile();
  }

  private static File vocab(final Path dir) throws IOException {
    final Path file = dir.resolve("vocab.txt");
    Files.write(file, List.of("[PAD]", "unused1", "[UNK]", "[SEP]", "hello", "world",
        "unused2", "[CLS]"));
    return file.toFile();
  }

  private static Probing tokenOutput(final Path dir, final PaddingStrategy padding)
      throws Exception {
    return new Probing(model(dir, "tiny-vectors.onnx"), vocab(dir), Pooling.MEAN, false, padding);
  }

  private static Probing pooledOutput(final Path dir, final PaddingStrategy padding)
      throws Exception {
    return new Probing(model(dir, "tiny-pooled.onnx"), vocab(dir), Pooling.MEAN, false, padding);
  }

  /** The weights both tiny graphs multiply a token id by. */
  private static final float[] W = {0.5f, -1f, 2f};

  /**
   * Inputs whose tokenized ids are known from the vocabulary of {@link #vocab(Path)}:
   * {@code [PAD]=0 [UNK]=2 [SEP]=3 hello=4 world=5 [CLS]=7}, wrapped as
   * {@code [CLS] ... [SEP]}.
   */
  private static final List<String> KNOWN_TEXTS = List.of(
      "x",                  // 7 2 3,     3 ids, sum 12,   mean 4
      "hello hello",        // 7 4 4 3,   4 ids, sum 18,   mean 4.5
      "hello hello world",  // 7 4 4 5 3, 5 ids, sum 23,   mean 4.6
      "hello",              // 7 4 3,     3 ids, sum 14,   mean 14/3
      "hello world",        // 7 4 5 3,   4 ids, sum 19,   mean 4.75
      "world",              // 7 5 3,     3 ids, sum 15,   mean 5
      "");                  // 7 3,       2 ids, sum 10,   mean 5

  private static final float[] KNOWN_MEANS = {4f, 4.5f, 4.6f, 14 / 3f, 4.75f, 5f, 5f};

  private static final float[] KNOWN_SUMS = {12f, 18f, 23f, 14f, 19f, 15f, 10f};

  private static float[] scale(final float factor) {
    return new float[] {W[0] * factor, W[1] * factor, W[2] * factor};
  }

  /** Inputs of several tokenized lengths, so one batch of them pads most rows. */
  private static List<String> texts() {
    return List.of("x", "hello hello", "hello hello world", "hello", "hello world", "world",
        "hello world hello world hello", "");
  }

  // ---------------------------------------------------------------------------------------------
  // The embedder pins
  // ---------------------------------------------------------------------------------------------

  @Test
  void testTheEmbedderReadsARankThreeOutputThroughAPinnedTensor(@TempDir final Path dir)
      throws Exception {
    try (Probing embedder = tokenOutput(dir, PaddingStrategy.LONGEST)) {
      assertEquals(0, embedder.inference().pinnedOutputArenas(),
          "construction should not have pinned anything for a model declaring its hidden size");

      embedder.embedAll(texts());

      assertEquals(1, embedder.inference().pinnedOutputArenas(),
          "the embedder did not pin its output");
      // LONGEST runs the whole call as one inference, 8 rows at the longest tokenized length, which
      // is 7 for "hello world hello world hello" plus [CLS] and [SEP]. That is the arena size.
      assertEquals((long) 8 * 7 * HIDDEN * FLOAT_BYTES, embedder.inference().pinnedOutputBytes());
    }
  }

  @Test
  void testTheEmbedderReadsARankTwoOutputThroughAPinnedTensor(@TempDir final Path dir)
      throws Exception {
    try (Probing embedder = pooledOutput(dir, PaddingStrategy.LONGEST)) {
      embedder.embedAll(texts());

      assertEquals(1, embedder.inference().pinnedOutputArenas());
      // An in-graph pooled output is {rows, hidden}, so the width of the run does not enter it.
      assertEquals((long) 8 * HIDDEN * FLOAT_BYTES, embedder.inference().pinnedOutputBytes());
    }
  }

  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testEveryPaddingStrategyPins(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    try (Probing embedder = tokenOutput(dir, padding)) {
      embedder.embedAll(texts());

      assertEquals(1, embedder.inference().pinnedOutputArenas(),
          padding + " did not pin its output");
      assertTrue(embedder.inference().pinnedOutputBytes() > 0);
    }
  }

  @Test
  void testTheArenaGrowsToTheLargestInferenceOfACallAndNoFurther(@TempDir final Path dir)
      throws Exception {
    try (Probing embedder = tokenOutput(dir, PaddingStrategy.EXACT_LENGTH)) {
      // EXACT_LENGTH runs one inference per distinct tokenized length, so the arena ends at the
      // largest of them rather than at the sum, which is the whole point of reusing it.
      embedder.embedAll(texts());
      final long afterOneCall = embedder.inference().pinnedOutputBytes();

      embedder.embedAll(texts());

      assertEquals(afterOneCall, embedder.inference().pinnedOutputBytes(),
          "a second identical call should reuse the arena, not grow it");
      assertEquals(1, embedder.inference().pinnedOutputArenas());
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Pinning changes no vector
  // ---------------------------------------------------------------------------------------------

  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testEveryVectorOfAPinnedBatchIsTheValueTheGraphRequires(final PaddingStrategy padding,
      @TempDir final Path dir) throws Exception {
    // Absolute values, not agreement between two calls of this class. A read path that returned
    // zeroes, or the same numbers for every input, would satisfy any self-consistency check while
    // being entirely wrong, so the expectation here is computed from the graph and the vocabulary:
    // tiny-vectors.onnx gives token t of row b the vector float(id) * W, and mean pooling over a
    // row's unpadded positions therefore gives the mean of the row's ids times W.
    try (Probing embedder = tokenOutput(dir, padding)) {
      final float[][] vectors = embedder.embedAll(KNOWN_TEXTS);

      assertEquals(KNOWN_MEANS.length, vectors.length);
      for (int t = 0; t < KNOWN_MEANS.length; t++) {
        assertArrayEquals(scale(KNOWN_MEANS[t]), vectors[t], 1e-6f,
            padding + " vector of " + KNOWN_TEXTS.get(t));
      }
    }
  }

  @Test
  void testEveryVectorOfAPinnedRankTwoBatchIsTheValueTheGraphRequires(@TempDir final Path dir)
      throws Exception {
    // The same check for the in-graph pooled output, where the graph sums a row's token vectors, so
    // an input of n tokens gives the sum of its ids times W. The vocabulary here pads at id 0 and
    // this graph ignores the attention mask, so padding adds nothing to the sum.
    try (Probing embedder = pooledOutput(dir, PaddingStrategy.LONGEST)) {
      final float[][] vectors = embedder.embedAll(KNOWN_TEXTS);

      for (int t = 0; t < KNOWN_SUMS.length; t++) {
        assertArrayEquals(scale(KNOWN_SUMS[t]), vectors[t], 1e-6f,
            "vector of " + KNOWN_TEXTS.get(t));
      }
    }
  }

  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testAPinnedBatchGivesEveryRowTheVectorItGetsOnItsOwn(final PaddingStrategy padding,
      @TempDir final Path dir) throws Exception {
    try (Probing embedder = tokenOutput(dir, padding)) {
      final float[][] batched = embedder.embedAll(texts());

      for (int t = 0; t < texts().size(); t++) {
        assertArrayEquals(embedder.embed(texts().get(t)), batched[t], 1e-6f,
            padding + " row " + t + " of a batch differs from the same input on its own");
      }
    }
  }

  @Test
  void testASecondCallDoesNotSeeTheFirstCallsValues(@TempDir final Path dir) throws Exception {
    // The two calls are chosen so that the second is narrower and shorter than the first, which is
    // the case where the reused arena still holds the first call's numbers past the second call's
    // slice. If the read walked the arena rather than the output, the second call's vectors would
    // pick up "hello world" repeated many times rather than a single token.
    try (Probing embedder = tokenOutput(dir, PaddingStrategy.LONGEST)) {
      final List<String> wide = List.of(
          "hello world hello world hello world hello world hello world",
          "hello world hello world hello world");
      final List<String> narrow = List.of("x");

      embedder.embedAll(wide);
      final float[][] second = embedder.embedAll(narrow);
      final float[] alone;
      try (Probing fresh = tokenOutput(dir, PaddingStrategy.LONGEST)) {
        alone = fresh.embed("x");
      }

      assertArrayEquals(alone, second[0], 0f,
          "the narrow call read values the wide call left in the arena");
    }
  }

  @Test
  void testTenCallsInARowAllGiveTheSameVectors(@TempDir final Path dir) throws Exception {
    try (Probing embedder = tokenOutput(dir, PaddingStrategy.LONGEST)) {
      final float[][] first = embedder.embedAll(texts());

      for (int call = 0; call < 10; call++) {
        final float[][] again = embedder.embedAll(texts());
        for (int row = 0; row < first.length; row++) {
          assertArrayEquals(first[row], again[row], 0f, "call " + call + " row " + row);
        }
      }
    }
  }

  @Test
  void testCallsOfAlternatingShapesDoNotContaminateEachOther(@TempDir final Path dir)
      throws Exception {
    try (Probing embedder = tokenOutput(dir, PaddingStrategy.LONGEST)) {
      final List<String> wide = List.of("hello world hello world hello", "hello");
      final List<String> narrow = List.of("x", "world");
      final float[][] wideFirst = embedder.embedAll(wide);
      final float[][] narrowFirst = embedder.embedAll(narrow);

      for (int round = 0; round < 6; round++) {
        assertArrayEquals(wideFirst[0], embedder.embedAll(wide)[0], 0f, "wide, round " + round);
        assertArrayEquals(narrowFirst[0], embedder.embedAll(narrow)[0], 0f,
            "narrow, round " + round);
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // close()
  // ---------------------------------------------------------------------------------------------

  @Test
  void testClosingTheEmbedderReleasesTheDirectOutputMemory(@TempDir final Path dir)
      throws Exception {
    final Probing embedder = tokenOutput(dir, PaddingStrategy.LONGEST);
    embedder.embedAll(texts());
    assertTrue(embedder.inference().pinnedOutputBytes() > 0);

    embedder.close();

    assertEquals(0L, embedder.inference().pinnedOutputBytes(),
        "AbstractDL.close() did not close the inference");
    assertEquals(0, embedder.inference().pinnedOutputArenas());
  }

  @Test
  void testClosingTheEmbedderTwiceIsSafe(@TempDir final Path dir) throws Exception {
    final Probing embedder = tokenOutput(dir, PaddingStrategy.LONGEST);
    embedder.embedAll(texts());

    embedder.close();
    embedder.close();

    assertEquals(0L, embedder.inference().pinnedOutputBytes());
  }

  @Test
  void testEmbeddingAfterCloseFails(@TempDir final Path dir) throws Exception {
    final Probing embedder = tokenOutput(dir, PaddingStrategy.LONGEST);
    embedder.embedAll(texts());
    embedder.close();

    assertThrows(RuntimeException.class, () -> embedder.embedAll(texts()));
  }

  // ---------------------------------------------------------------------------------------------
  // Concurrency
  // ---------------------------------------------------------------------------------------------

  @Test
  void testSixThreadsEmbeddingDifferentTextsEachGetTheirOwnVectors(@TempDir final Path dir)
      throws Exception {
    // Each thread embeds a batch no other thread embeds, and checks every component against the
    // vector the same text gets on its own. Constant outputs could not tell a shared buffer from a
    // confined one, so no two threads share an input here.
    final int threads = 6;
    try (Probing embedder = tokenOutput(dir, PaddingStrategy.LONGEST)) {
      final List<List<String>> batches = new ArrayList<>();
      final List<float[][]> expected = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final List<String> batch = new ArrayList<>();
        for (int i = 0; i <= t; i++) {
          final StringBuilder text = new StringBuilder("hello");
          for (int w = 0; w <= t + i; w++) {
            text.append(w % 2 == 0 ? " world" : " hello");
          }
          batch.add(text.toString());
        }
        batches.add(batch);
        // The reference is taken single threaded, before any thread starts, so it cannot itself be
        // contaminated by the concurrent run.
        expected.add(embedder.embedAll(batch));
      }

      final ExecutorService pool = Executors.newFixedThreadPool(threads);
      try {
        final List<Callable<Boolean>> work = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
          final List<String> batch = batches.get(t);
          final float[][] reference = expected.get(t);
          work.add(() -> {
            for (int repeat = 0; repeat < 12; repeat++) {
              final float[][] actual = embedder.embedAll(batch);
              for (int row = 0; row < reference.length; row++) {
                for (int d = 0; d < reference[row].length; d++) {
                  if (reference[row][d] != actual[row][d]) {
                    throw new AssertionError("batch of " + batch.size() + " row " + row
                        + " component " + d + " read " + actual[row][d] + " but should be "
                        + reference[row][d]);
                  }
                }
              }
            }
            return Boolean.TRUE;
          });
        }
        for (final Future<Boolean> result : pool.invokeAll(work)) {
          assertTrue(result.get(180, TimeUnit.SECONDS));
        }
      } finally {
        pool.shutdownNow();
        assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS));
      }

      // One arena per worker thread plus the one the reference run made on this thread, which is
      // what confinement means: no two of them are the same buffer.
      assertEquals(threads + 1, embedder.inference().pinnedOutputArenas());
    }
  }
}
