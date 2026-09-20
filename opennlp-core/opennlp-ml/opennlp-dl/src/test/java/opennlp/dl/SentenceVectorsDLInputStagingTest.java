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
import java.nio.LongBuffer;
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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * That {@link SentenceVectorsDL} really does write its rows straight into the reusable direct memory
 * of its {@link OnnxInference}, and that doing so changes none of its vectors and lets no inference
 * read the rows of the one before it.
 *
 * <p>This test lives in {@code opennlp.dl} rather than beside the other embedder tests, and for the
 * same reason {@link SentenceVectorsDLPinnedOutputTest} does: the evidence it needs, the arena count,
 * the direct byte total and the staged buffers of the {@link OnnxInference} the embedder holds, is
 * package private to {@code opennlp.dl}.</p>
 *
 * <p>Against the deterministic {@code tiny-vectors.onnx} graph, which computes
 * {@code output[b][t] = float(input_ids[b][t]) * W} with {@code W = [0.5, -1, 2]}, so that a mean
 * pooled row is its own token ids' mean times {@code W}. A row that read another row's tokens, or the
 * tokens of a previous call, therefore pools to a number this test can name.</p>
 */
class SentenceVectorsDLInputStagingTest {

  private static final int LONG_BYTES = 8;

  /** The inputs the embedder stages, which is all three a sentence-transformers model declares. */
  private static final int INPUTS = 3;

  /** Reaches the {@link OnnxInference} of a component, which {@link AbstractDL} keeps protected. */
  private static final class Probing extends SentenceVectorsDL {

    private Probing(final File model, final File vocabulary, final PaddingStrategy padding)
        throws OrtException, IOException {
      super(model, vocabulary, true, Pooling.MEAN, false, SentenceVectorsDL.DEFAULT_MAX_LENGTH,
          padding);
    }

    private OnnxInference inference() {
      return inference;
    }
  }

  // Copied out of the classpath rather than resolved in place, for the same reason as in
  // SentenceVectorsDLEmbedderTest: the resource may live inside a test-jar.
  private static File model(final Path dir) throws IOException {
    final Path file = dir.resolve("tiny-vectors.onnx");
    try (InputStream is = Objects.requireNonNull(SentenceVectorsDLInputStagingTest.class
        .getResourceAsStream("/opennlp/dl/vectors/tiny-vectors.onnx"))) {
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

  private static Probing embedder(final Path dir, final PaddingStrategy padding) throws Exception {
    return new Probing(model(dir), vocab(dir), padding);
  }

  /** The weights the graph multiplies a token id by. */
  private static final float[] W = {0.5f, -1f, 2f};

  /**
   * Inputs whose tokenized ids follow from the vocabulary of {@link #vocab(Path)}:
   * {@code [PAD]=0 [UNK]=2 [SEP]=3 hello=4 world=5 [CLS]=7}, wrapped as {@code [CLS] ... [SEP]}.
   */
  private static final String THREE = "x";                    // 7 2 3,     mean 4
  private static final String FOUR = "hello hello";           // 7 4 4 3,   mean 4.5
  private static final String FIVE = "hello hello world";     // 7 4 4 5 3, mean 4.6
  private static final String THREE_OTHER = "world";          // 7 5 3,     mean 5
  private static final String FOUR_OTHER = "hello world";     // 7 4 5 3,   mean 4.75
  private static final String TWO = "";                       // 7 3,       mean 5

  private static float[] scale(final float mean) {
    return new float[] {W[0] * mean, W[1] * mean, W[2] * mean};
  }

  // ---------------------------------------------------------------------------------------------
  // The vectors do not change
  // ---------------------------------------------------------------------------------------------

  @ParameterizedTest(name = "{0}")
  @EnumSource(PaddingStrategy.class)
  void testEveryRowOfAMixedCallPoolsItsOwnTokensUnderEveryStrategy(final PaddingStrategy padding,
      @TempDir final Path dir) throws Exception {
    // One call of six inputs of four distinct tokenized lengths, which EXACT_LENGTH turns into four
    // inferences of recurring extents, LONGEST into one padded inference, and MAX_LENGTH into one
    // inference 512 wide. Every row's vector is its own mean under all three.
    try (Probing embedder = embedder(dir, padding)) {
      final float[][] vectors = embedder.embedAll(
          List.of(THREE, FIVE, FOUR, THREE_OTHER, TWO, FOUR_OTHER));

      assertArrayEquals(scale(4f), vectors[0], 1e-5f);
      assertArrayEquals(scale(4.6f), vectors[1], 1e-5f);
      assertArrayEquals(scale(4.5f), vectors[2], 1e-5f);
      assertArrayEquals(scale(5f), vectors[3], 1e-5f);
      assertArrayEquals(scale(5f), vectors[4], 1e-5f);
      assertArrayEquals(scale(4.75f), vectors[5], 1e-5f);
    }
  }

  @ParameterizedTest(name = "{0}")
  @EnumSource(PaddingStrategy.class)
  void testABatchedRowEqualsTheSameTextEmbeddedOnItsOwn(final PaddingStrategy padding,
      @TempDir final Path dir) throws Exception {
    try (Probing embedder = embedder(dir, padding)) {
      final List<String> texts = List.of(THREE, FIVE, FOUR, THREE_OTHER, TWO, FOUR_OTHER);
      final float[][] batched = embedder.embedAll(texts);

      for (int i = 0; i < texts.size(); i++) {
        assertArrayEquals(embedder.embed(texts.get(i)), batched[i], 1e-5f, texts.get(i));
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Stale rows
  // ---------------------------------------------------------------------------------------------

  @Test
  void testACallRepeatingAnExtentWithOtherContentPoolsTheNewContent(@TempDir final Path dir)
      throws Exception {
    // Both calls run one inference of the same extent, a single row of three tokens, so the second
    // reuses the buffers of the first without re-cutting or reallocating them. Its answer must be its
    // own: "x" pools to 4 and "world" to 5, which differ in every component.
    try (Probing embedder = embedder(dir, PaddingStrategy.EXACT_LENGTH)) {
      assertArrayEquals(scale(4f), embedder.embed(THREE), 1e-5f);
      assertArrayEquals(scale(5f), embedder.embed(THREE_OTHER), 1e-5f);
      assertArrayEquals(scale(4f), embedder.embed(THREE), 1e-5f);
    }
  }

  @Test
  void testAnExtentRecurringAfterAWiderOneStillPoolsItsOwnRows(@TempDir final Path dir)
      throws Exception {
    // A three token call, then a five token one that grows the arena and re-cuts its slots, then the
    // three token extent again with different content.
    try (Probing embedder = embedder(dir, PaddingStrategy.EXACT_LENGTH)) {
      assertArrayEquals(scale(4f), embedder.embed(THREE), 1e-5f);
      assertArrayEquals(scale(4.6f), embedder.embed(FIVE), 1e-5f);
      assertArrayEquals(scale(5f), embedder.embed(THREE_OTHER), 1e-5f);
      assertArrayEquals(scale(4f), embedder.embed(THREE), 1e-5f);
    }
  }

  @Test
  void testAShorterCallAfterALongerOneReadsNoneOfTheLongerCallsTail(@TempDir final Path dir)
      throws Exception {
    // Padding to the longest member makes the first call's inference 5 positions wide and the second
    // call's 2 positions wide, so the arena still holds the first call's tokens past the second call's
    // extent. The second call's vectors must show no trace of them.
    try (Probing embedder = embedder(dir, PaddingStrategy.LONGEST)) {
      embedder.embedAll(List.of(FIVE, FOUR, THREE));

      final float[][] shorter = embedder.embedAll(List.of(TWO, TWO));

      assertArrayEquals(scale(5f), shorter[0], 1e-5f);
      assertArrayEquals(scale(5f), shorter[1], 1e-5f);
    }
  }

  @Test
  void testTheTokenTypeIdsOfARecurringExtentAreWrittenAgainRatherThanAssumed(
      @TempDir final Path dir) throws Exception {
    // token_type_ids is 0 for every row of every batch this component stages, which makes it the one
    // input a shortcut would look safe on. The vectors alone cannot show whether it was rewritten, so
    // the buffer is read: every position of it is 0 after a call whose rows are of two lengths, which
    // is only guaranteed if the padded tail was written rather than inherited.
    try (Probing embedder = embedder(dir, PaddingStrategy.LONGEST)) {
      embedder.embedAll(List.of(FIVE, THREE));

      final LongBuffer types =
          embedder.inference().inputStagingBuffer(AbstractDL.TOKEN_TYPE_IDS);
      assertNotNull(types);
      assertEquals(10, types.capacity(), "two rows padded to five positions");
      for (int i = 0; i < types.capacity(); i++) {
        assertEquals(0L, types.get(i), "position " + i + " of token_type_ids");
      }
      // And the attention mask of the padded row is 0 past its own three tokens, written not left.
      // PaddingStrategy.LONGEST orders a batch by ascending tokenized length, so the three token row
      // is staged first and the five token one second.
      final LongBuffer mask = embedder.inference().inputStagingBuffer(AbstractDL.ATTENTION_MASK);
      for (int i = 0; i < 3; i++) {
        assertEquals(1L, mask.get(i), "position " + i + " of the three token row");
      }
      for (int i = 3; i < 5; i++) {
        assertEquals(0L, mask.get(i), "padded position " + i + " of the three token row");
      }
      for (int i = 5; i < 10; i++) {
        assertEquals(1L, mask.get(i), "position " + i + " of the five token row");
      }
    }
  }

  @Test
  void testEveryInputOfARecurringExtentIsRewrittenOverValuesThatWouldBeVisible(
      @TempDir final Path dir) throws Exception {
    // The previous test can only see zeroes, and a skipped write to an input this component always
    // fills with zeroes leaves zeroes behind, so it proves nothing on its own. This one writes
    // something else into all three buffers between two inferences of the same extent and then
    // asserts they come back as the inference requires. A write this component skips because it
    // believes the buffer already holds the right values now leaves that poison in place, which the
    // vectors show and these reads name.
    try (Probing embedder = embedder(dir, PaddingStrategy.LONGEST)) {
      final float[][] before = embedder.embedAll(List.of(FIVE, THREE));

      poison(embedder, AbstractDL.INPUT_IDS);
      poison(embedder, AbstractDL.ATTENTION_MASK);
      poison(embedder, AbstractDL.TOKEN_TYPE_IDS);

      final float[][] after = embedder.embedAll(List.of(FIVE, THREE));

      assertArrayEquals(before[0], after[0], 0f, "the five token row after the buffers were poisoned");
      assertArrayEquals(before[1], after[1], 0f, "the three token row after the buffers were poisoned");
      for (int i = 0; i < 10; i++) {
        assertEquals(0L, embedder.inference().inputStagingBuffer(AbstractDL.TOKEN_TYPE_IDS).get(i),
            "position " + i + " of token_type_ids kept the poison rather than being rewritten");
      }
      // Three real positions then two padded ones, then five real ones, in ascending length order.
      final LongBuffer mask = embedder.inference().inputStagingBuffer(AbstractDL.ATTENTION_MASK);
      assertEquals(0L, mask.get(3), "the padded attention mask kept the poison");
      assertEquals(0L, mask.get(4), "the padded attention mask kept the poison");
    }
  }

  /** Writes a value into every position of one staged input that no inference here ever writes. */
  private static void poison(final Probing embedder, final String input) {
    final LongBuffer buffer = embedder.inference().inputStagingBuffer(input);
    assertNotNull(buffer, input + " was never staged, so there is nothing to poison");
    for (int i = 0; i < buffer.capacity(); i++) {
      buffer.put(i, 6L);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // The memory is staged into, direct, reused and bounded
  // ---------------------------------------------------------------------------------------------

  @Test
  void testTheEmbedderStagesItsRowsIntoDirectMemoryItDoesNotAllocatePerInference(
      @TempDir final Path dir) throws Exception {
    try (Probing embedder = embedder(dir, PaddingStrategy.EXACT_LENGTH)) {
      assertEquals(0, embedder.inference().inputStagingArenas(),
          "a constructed embedder has run once to discover its hidden size, on the calling thread");
      embedder.embed(FOUR);

      assertEquals(1, embedder.inference().inputStagingArenas());
      final LongBuffer ids = embedder.inference().inputStagingBuffer(AbstractDL.INPUT_IDS);
      assertNotNull(ids, "the ids of an inference were not staged into memory this class owns");
      assertTrue(ids.isDirect(), "a heap buffer would reintroduce ONNX Runtime's own copy");
      assertEquals(4, ids.capacity(), "one row of four tokens");
      assertEquals(INPUTS * 4L * LONG_BYTES, embedder.inference().inputStagingBytes());

      // A second inference of the same extent reuses the very same memory.
      embedder.embed(FOUR_OTHER);
      assertSame(ids, embedder.inference().inputStagingBuffer(AbstractDL.INPUT_IDS));
      assertEquals(INPUTS * 4L * LONG_BYTES, embedder.inference().inputStagingBytes());
    }
  }

  @Test
  void testACallOfManyLengthsGrowsOneArenaToItsWidestInferenceAndNoFurther(@TempDir final Path dir)
      throws Exception {
    try (Probing embedder = embedder(dir, PaddingStrategy.EXACT_LENGTH)) {
      // Four distinct tokenized lengths, so four inferences: one row of 5 positions, one of 2, two
      // of 4 and two of 3, in call order. The widest of them is the pair of four token rows at 8
      // token positions, and one arena grown to that once is what the whole call holds.
      embedder.embedAll(List.of(FIVE, TWO, FOUR, THREE, FOUR_OTHER, THREE_OTHER));

      assertEquals(1, embedder.inference().inputStagingArenas());
      assertEquals(INPUTS * 8L * LONG_BYTES, embedder.inference().inputStagingBytes(),
          "the arena holds the widest inference of the call, not the sum of them");
    }
  }

  @Test
  void testTheStagingMemoryOfEveryThreadIsReleasedOnClose(@TempDir final Path dir) throws Exception {
    final Probing embedder = embedder(dir, PaddingStrategy.EXACT_LENGTH);
    try {
      final int threads = 4;
      onThreads(threads, thread -> embedder.embed(FOUR));
      assertEquals(threads, embedder.inference().inputStagingArenas());
      assertTrue(embedder.inference().inputStagingBytes() > 0);
    } finally {
      embedder.close();
    }

    assertEquals(0, embedder.inference().inputStagingArenas());
    assertEquals(0L, embedder.inference().inputStagingBytes());
    assertThrows(RuntimeException.class, () -> embedder.embed(FOUR));
  }

  // ---------------------------------------------------------------------------------------------
  // Concurrency
  // ---------------------------------------------------------------------------------------------

  @Test
  void testEightThreadsEmbeddingDifferentTextsOfOneLengthEachGetTheirOwn(@TempDir final Path dir)
      throws Exception {
    // Both texts tokenize to three positions, so both threads run the same extent at the same time.
    // A shared staging buffer would let one thread pool the other's tokens, and 4 against 5 differs
    // in every component of the vector.
    try (Probing embedder = embedder(dir, PaddingStrategy.EXACT_LENGTH)) {
      final int threads = 8;
      onThreads(threads, thread -> {
        final boolean even = thread % 2 == 0;
        final String text = even ? THREE : THREE_OTHER;
        final float[] expected = scale(even ? 4f : 5f);
        for (int repeat = 0; repeat < 50; repeat++) {
          assertArrayEquals(expected, embedder.embed(text), 1e-5f, text);
        }
        return null;
      });

      assertEquals(threads, embedder.inference().inputStagingArenas(),
          "one arena per thread, not one shared between them");
    }
  }

  @Test
  void testThreadsEmbeddingCallsOfDifferentShapesDoNotDisturbEachOther(@TempDir final Path dir)
      throws Exception {
    try (Probing embedder = embedder(dir, PaddingStrategy.LONGEST)) {
      final List<String> texts = List.of(THREE, FIVE, FOUR, THREE_OTHER, TWO, FOUR_OTHER);
      final float[][] expected = embedder.embedAll(texts);
      final int threads = 6;

      onThreads(threads, thread -> {
        for (int repeat = 0; repeat < 30; repeat++) {
          // A different slice of the same inputs per thread, so the extents differ as well as the
          // content, and each thread checks its own rows against the single threaded answer.
          final int from = thread % texts.size();
          final List<String> slice = texts.subList(from, texts.size());
          final float[][] vectors = embedder.embedAll(slice);
          for (int i = 0; i < slice.size(); i++) {
            assertArrayEquals(expected[from + i], vectors[i], 1e-5f, slice.get(i));
          }
        }
        return null;
      });
    }
  }

  /** Runs {@code work} once on each of {@code threads} threads and rethrows whatever any threw. */
  private static void onThreads(final int threads, final Work work) throws Exception {
    final ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      final List<Future<Object>> results = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final int thread = t;
        final Callable<Object> task = () -> work.run(thread);
        results.add(pool.submit(task));
      }
      for (final Future<Object> result : results) {
        result.get(60, TimeUnit.SECONDS);
      }
    } finally {
      pool.shutdownNow();
      pool.awaitTermination(30, TimeUnit.SECONDS);
    }
  }

  /** One thread's share of a concurrency test. */
  private interface Work {
    Object run(int thread) throws Exception;
  }
}
