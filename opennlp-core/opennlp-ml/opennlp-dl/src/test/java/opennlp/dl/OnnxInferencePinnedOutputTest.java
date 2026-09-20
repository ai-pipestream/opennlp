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

import java.io.IOException;
import java.io.InputStream;
import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The output read path of {@link OnnxInference}: the pinned output tensor, the flat buffer read it
 * falls back to, the reuse of the pinned tensor across runs, the bound on the direct memory the
 * reusable buffers hold, and the release of all of it.
 *
 * <p>Against the same deterministic {@code tiny-pooled.onnx} graph
 * {@link OnnxInferenceTest} uses, where
 * {@code token_embeddings[b][t][d] = float(input_ids[b][t]) * W[d]} of shape
 * <code>{batch, tokens, 3}</code> and {@code sentence_embedding[b][d]} is the sum of a row's token
 * embeddings, with {@code W = [0.5, -1, 2]}. Every output value therefore names the vocabulary id
 * that produced it, which is what makes stale data from a previous run, a row read at the wrong
 * offset and a value read from the wrong thread all visible rather than merely plausible.</p>
 *
 * <p>A test that asserts a shape mismatch is rejected makes ONNX Runtime print its own error to
 * standard error before returning the failure. That output is expected.</p>
 */
class OnnxInferencePinnedOutputTest {

  /** The weights the graph multiplies each token id by. */
  private static final float[] W = {0.5f, -1f, 2f};

  /** The hidden size of the graph, the last dimension of both of its outputs. */
  private static final int HIDDEN = 3;

  private static final String TOKENS_OUTPUT = "token_embeddings";
  private static final String POOLED_OUTPUT = "sentence_embedding";

  private OrtEnvironment env;
  private OrtSession session;
  private OnnxInference inference;

  @BeforeEach
  void openSession(@TempDir final Path dir) throws IOException, OrtException {
    final Path model = dir.resolve("tiny-pooled.onnx");
    try (InputStream is = Objects.requireNonNull(OnnxInferencePinnedOutputTest.class
        .getResourceAsStream("/opennlp/dl/vectors/tiny-pooled.onnx"))) {
      Files.copy(is, model, StandardCopyOption.REPLACE_EXISTING);
    }
    env = OrtEnvironment.getEnvironment();
    session = env.createSession(model.toString(), new OrtSession.SessionOptions());
    inference = new OnnxInference(env, session);
  }

  @AfterEach
  void closeSession() throws OrtException {
    if (inference != null) {
      inference.close();
    }
    if (session != null) {
      session.close();
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------------------------

  /** {@return ids {@code 1..rows*width}, so no two positions of a run hold the same value} */
  private static long[] distinctIds(final int rows, final int width) {
    final long[] ids = new long[rows * width];
    for (int i = 0; i < ids.length; i++) {
      ids[i] = i + 1;
    }
    return ids;
  }

  private static long[] ones(final int elements) {
    final long[] mask = new long[elements];
    Arrays.fill(mask, 1L);
    return mask;
  }

  /** A reader that copies the whole output out flat, so two read paths can be compared directly. */
  private static OnnxInference.OutputReader<float[]> flat() {
    return (values, shape) -> {
      final float[] copy = new float[values.remaining()];
      for (int i = 0; i < copy.length; i++) {
        copy[i] = values.get(i);
      }
      return copy;
    };
  }

  private float[] unpinned(final int rows, final int width, final long[] ids, final String output)
      throws OrtException {
    return inference.run(new long[] {rows, width}, ids, ones(ids.length), new long[ids.length],
        output, flat());
  }

  private float[] pinned(final int rows, final int width, final long[] ids, final String output,
      final long[] outputShape) throws OrtException {
    return inference.run(new long[] {rows, width}, ids, ones(ids.length), new long[ids.length],
        output, outputShape, flat());
  }

  // ---------------------------------------------------------------------------------------------
  // The pinned path agrees with the unpinned one
  // ---------------------------------------------------------------------------------------------

  @ParameterizedTest(name = "rank 3 output of a {0} by {1} run")
  @CsvSource({"1, 1", "1, 12", "12, 1", "3, 4", "16, 9", "64, 7"})
  void testPinnedAndUnpinnedReadsAgreeOnARankThreeOutput(final int rows, final int width)
      throws OrtException {
    final long[] ids = distinctIds(rows, width);

    final float[] byCopy = unpinned(rows, width, ids, TOKENS_OUTPUT);
    final float[] byPin = pinned(rows, width, ids, TOKENS_OUTPUT,
        new long[] {rows, width, HIDDEN});

    assertEquals(rows * width * HIDDEN, byPin.length);
    // Not just equal to each other: equal to what the graph has to produce, so a pinned read that
    // silently returned the unpinned read's buffer, or zeroes, would not pass.
    for (int i = 0; i < ids.length; i++) {
      for (int d = 0; d < HIDDEN; d++) {
        assertEquals(W[d] * ids[i], byPin[i * HIDDEN + d], "position " + i + " component " + d);
      }
    }
    assertArrayEquals(byCopy, byPin);
  }

  @ParameterizedTest(name = "rank 2 output of a {0} by {1} run")
  @CsvSource({"1, 1", "1, 12", "12, 1", "5, 6"})
  void testPinnedAndUnpinnedReadsAgreeOnARankTwoOutput(final int rows, final int width)
      throws OrtException {
    final long[] ids = distinctIds(rows, width);

    final float[] byCopy = unpinned(rows, width, ids, POOLED_OUTPUT);
    final float[] byPin = pinned(rows, width, ids, POOLED_OUTPUT, new long[] {rows, HIDDEN});

    assertEquals(rows * HIDDEN, byPin.length);
    for (int r = 0; r < rows; r++) {
      long sum = 0;
      for (int t = 0; t < width; t++) {
        sum += ids[r * width + t];
      }
      for (int d = 0; d < HIDDEN; d++) {
        assertEquals(W[d] * sum, byPin[r * HIDDEN + d], "row " + r + " component " + d);
      }
    }
    assertArrayEquals(byCopy, byPin);
  }

  @Test
  void testRowsOfVeryDifferentWidthsInOnePinnedBatchReadBackAtTheirOwnOffsets()
      throws OrtException {
    // What a batch padded to its longest member looks like: one row of 11 real positions, one of 1,
    // one of 6, all staged at width 11 with a pad id of 99 in the tails. A pinned read has to give
    // each row its own stride, and a row's real prefix has to be what the row gets on its own.
    final int width = 11;
    final long pad = 99;
    final long[][] rows = {
        {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}, {31}, {41, 42, 43, 44, 45, 46}};
    final long[] ids = new long[rows.length * width];
    Arrays.fill(ids, pad);
    for (int r = 0; r < rows.length; r++) {
      System.arraycopy(rows[r], 0, ids, r * width, rows[r].length);
    }

    final float[] batched = pinned(rows.length, width, ids, TOKENS_OUTPUT,
        new long[] {rows.length, width, HIDDEN});

    for (int r = 0; r < rows.length; r++) {
      final float[] alone = pinned(1, rows[r].length, rows[r], TOKENS_OUTPUT,
          new long[] {1, rows[r].length, HIDDEN});
      for (int t = 0; t < rows[r].length; t++) {
        for (int d = 0; d < HIDDEN; d++) {
          assertEquals(alone[t * HIDDEN + d], batched[(r * width + t) * HIDDEN + d],
              "row " + r + " position " + t + " component " + d);
        }
      }
      // And the padded tail is the pad id's embedding, which is what proves the offsets are right
      // rather than the rows happening to line up.
      for (int t = rows[r].length; t < width; t++) {
        assertEquals(W[0] * pad, batched[(r * width + t) * HIDDEN], "row " + r + " pad at " + t);
      }
    }
  }

  @Test
  void testOnnxRuntimeWritesIntoTheBufferThisClassOwnsRatherThanCopyingOut() throws OrtException {
    // The copy-free claim, asserted rather than assumed. A pinned tensor is backed by a buffer the
    // caller allocated, which is the case ONNX Runtime documents getBufferRef as answering with a
    // non-empty Optional, unlike a tensor it allocated itself. Poisoning that buffer and then
    // running again shows whether the run wrote into it: if the results were copied out of memory
    // ONNX Runtime allocated, the poison would survive.
    final long[] ids = distinctIds(3, 4);
    pinned(3, 4, ids, TOKENS_OUTPUT, new long[] {3, 4, HIDDEN});
    final OnnxTensor tensor = inference.pinnedOutputTensor();
    assertNotNull(tensor);
    final Optional<Buffer> backing = tensor.getBufferRef();
    assertTrue(backing.isPresent(), "a pinned tensor should be backed by the caller's buffer");
    final FloatBuffer poisoned = (FloatBuffer) backing.get();
    assertTrue(poisoned.isDirect(), "only a direct buffer can be written into without a copy");
    final int elements = 3 * 4 * HIDDEN;
    assertEquals(elements, poisoned.capacity());
    for (int i = 0; i < elements; i++) {
      poisoned.put(i, -999f);
    }

    final float[] read = pinned(3, 4, ids, TOKENS_OUTPUT, new long[] {3, 4, HIDDEN});

    for (int i = 0; i < elements; i++) {
      assertEquals(W[i % HIDDEN] * ids[i / HIDDEN], read[i], "position " + i);
      assertEquals(read[i], poisoned.get(i),
          "position " + i + " of the buffer this class owns was not written by the run");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // A shape mismatch fails loudly
  // ---------------------------------------------------------------------------------------------

  @ParameterizedTest(name = "pinning {0} for a 2 by 3 run")
  @CsvSource({"2, 2, 3", "2, 5, 3", "1, 3, 3", "3, 3, 3", "2, 3, 4", "2, 3, 2"})
  void testAPinnedOutputOfTheWrongShapeFailsTheRun(final int rows, final int width,
      final int hidden) {
    final long[] ids = distinctIds(2, 3);

    final OrtException e = assertThrows(OrtException.class,
        () -> pinned(2, 3, ids, TOKENS_OUTPUT, new long[] {rows, width, hidden}));

    // Not merely "it threw": ONNX Runtime names the shape it was handed, which is what makes this a
    // rejection of the shape rather than an unrelated failure.
    assertTrue(e.getMessage().contains("token_embeddings"), e.getMessage());
    assertTrue(e.getMessage().contains(rows + "," + width + "," + hidden)
        || e.getMessage().contains("Got: " + hidden), e.getMessage());
  }

  @Test
  void testAPinnedOutputOfTheWrongRankFailsTheRun() {
    final long[] ids = distinctIds(2, 3);

    final OrtException e = assertThrows(OrtException.class,
        () -> pinned(2, 3, ids, TOKENS_OUTPUT, new long[] {2, 3}));

    assertTrue(e.getMessage().contains("rank"), e.getMessage());
  }

  @Test
  void testAFailedPinnedRunLeavesTheInferenceUsable() throws OrtException {
    // The rejection happens with a tensor already rebuilt at the bad shape, so the recovery path is
    // the one worth asserting: the next run of the good shape has to rebuild the tensor again and
    // answer exactly as it did before. A failure that left the tensor at the bad shape, released the
    // arena, or lost the accounting would show up here rather than in review.
    final long[] good = distinctIds(2, 3);
    final float[] before = pinned(2, 3, good, TOKENS_OUTPUT, new long[] {2, 3, HIDDEN});
    final long bytesBefore = inference.pinnedOutputBytes();

    assertThrows(OrtException.class,
        () -> pinned(2, 3, good, TOKENS_OUTPUT, new long[] {2, 2, HIDDEN}));

    assertArrayEquals(before, pinned(2, 3, good, TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}));
    assertEquals(bytesBefore, inference.pinnedOutputBytes());
    assertEquals(1, inference.pinnedOutputArenas());
  }

  @ParameterizedTest(name = "an output shape holding {0}")
  @CsvSource({"0", "-1"})
  void testANonPositiveOutputDimensionIsRejectedBeforeTheRun(final int dimension) {
    final long[] ids = distinctIds(2, 3);

    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
        () -> pinned(2, 3, ids, TOKENS_OUTPUT, new long[] {2, dimension, HIDDEN}));

    assertTrue(e.getMessage().contains("positive"), e.getMessage());
    assertEquals(0, inference.pinnedOutputArenas(), "nothing was allocated for a rejected shape");
  }

  @Test
  void testPinningWithoutNamingTheOutputIsRejected() {
    final long[] ids = distinctIds(2, 3);

    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
        () -> inference.run(new long[] {2, 3}, ids, ones(6), new long[6], null,
            new long[] {2, 3, HIDDEN}, flat()));

    assertTrue(e.getMessage().contains("name"), e.getMessage());
  }

  // ---------------------------------------------------------------------------------------------
  // Reuse across runs
  // ---------------------------------------------------------------------------------------------

  @Test
  void testTheSameShapeRunTwiceReusesTheSameTensorAndTheSameMemory() throws OrtException {
    pinned(4, 5, distinctIds(4, 5), TOKENS_OUTPUT, new long[] {4, 5, HIDDEN});
    final OnnxTensor first = inference.pinnedOutputTensor();
    final long bytes = inference.pinnedOutputBytes();
    assertNotNull(first);
    assertEquals(4L * 5 * HIDDEN * 4, bytes);

    pinned(4, 5, distinctIds(4, 5), TOKENS_OUTPUT, new long[] {4, 5, HIDDEN});

    assertSame(first, inference.pinnedOutputTensor(), "a repeated shape allocates nothing");
    assertEquals(bytes, inference.pinnedOutputBytes());
    assertEquals(1, inference.pinnedOutputArenas());
  }

  @Test
  void testASmallerShapeReusesTheArenaAndALargerOneGrowsItOnce() throws OrtException {
    pinned(4, 5, distinctIds(4, 5), TOKENS_OUTPUT, new long[] {4, 5, HIDDEN});
    final long big = inference.pinnedOutputBytes();

    pinned(2, 3, distinctIds(2, 3), TOKENS_OUTPUT, new long[] {2, 3, HIDDEN});
    assertEquals(big, inference.pinnedOutputBytes(), "a smaller output does not shrink the arena");

    pinned(8, 5, distinctIds(8, 5), TOKENS_OUTPUT, new long[] {8, 5, HIDDEN});
    assertEquals(8L * 5 * HIDDEN * 4, inference.pinnedOutputBytes(),
        "a larger output grows the arena to exactly what it needs");
    assertEquals(1, inference.pinnedOutputArenas(), "growing does not add an arena");
  }

  @Test
  void testALaterRunOfTheSameShapeDoesNotSeeTheEarlierRunsValues() throws OrtException {
    // The input pattern is the point: the first run's ids are all large and the second run's are all
    // small, and none of the second run's values can be produced by any of the first run's ids. A
    // buffer reused without being fully overwritten would show the large values.
    final long[] large = new long[20];
    Arrays.fill(large, 9000L);
    final long[] small = new long[20];
    Arrays.fill(small, 1L);

    final float[] firstRun = pinned(4, 5, large, TOKENS_OUTPUT, new long[] {4, 5, HIDDEN});
    for (final float value : firstRun) {
      assertTrue(Math.abs(value) >= 4500f, "the first run should be large everywhere: " + value);
    }

    final float[] secondRun = pinned(4, 5, small, TOKENS_OUTPUT, new long[] {4, 5, HIDDEN});

    assertEquals(20 * HIDDEN, secondRun.length);
    for (int i = 0; i < secondRun.length; i++) {
      assertEquals(W[i % HIDDEN], secondRun[i], "position " + i + " kept a previous run's value");
    }
  }

  @Test
  void testALaterNarrowerRunSeesOnlyItsOwnPartOfTheArena() throws OrtException {
    // The arena keeps the wide run's values past the narrow run's slice. The narrow run's reader
    // must be handed only its own elements, or the caller would pool a previous run's numbers into
    // the tail of a row.
    final long[] large = new long[40];
    Arrays.fill(large, 9000L);
    pinned(8, 5, large, TOKENS_OUTPUT, new long[] {8, 5, HIDDEN});

    final long[] small = new long[6];
    Arrays.fill(small, 2L);
    final float[] narrow = pinned(2, 3, small, TOKENS_OUTPUT, new long[] {2, 3, HIDDEN});

    assertEquals(2 * 3 * HIDDEN, narrow.length, "the reader saw more than its own output");
    for (int i = 0; i < narrow.length; i++) {
      assertEquals(2f * W[i % HIDDEN], narrow[i], "position " + i);
    }
  }

  @Test
  void testTenRunsOfTheSameShapeAllGiveTheSameAnswer() throws OrtException {
    final long[] ids = distinctIds(6, 4);
    final float[] expected = pinned(6, 4, ids, TOKENS_OUTPUT, new long[] {6, 4, HIDDEN});

    for (int run = 0; run < 10; run++) {
      assertArrayEquals(expected, pinned(6, 4, ids, TOKENS_OUTPUT, new long[] {6, 4, HIDDEN}),
          "run " + run);
    }
    assertEquals(1, inference.pinnedOutputArenas());
    assertEquals(6L * 4 * HIDDEN * 4, inference.pinnedOutputBytes());
  }

  // ---------------------------------------------------------------------------------------------
  // The unpinned path allocates nothing reusable
  // ---------------------------------------------------------------------------------------------

  @Test
  void testAnUnpinnedRunHoldsNoDirectOutputMemory() throws OrtException {
    unpinned(4, 5, distinctIds(4, 5), TOKENS_OUTPUT);

    assertEquals(0, inference.pinnedOutputArenas());
    assertEquals(0L, inference.pinnedOutputBytes());
    assertNull(inference.pinnedOutputTensor());
  }

  @Test
  void testTheObjectReturningRunHoldsNoDirectOutputMemory() throws OrtException {
    // The path a name finder and a document categorizer take: they name no output shape, so they
    // pin nothing and there is nothing of theirs to release.
    inference.run(new long[] {1, 3}, distinctIds(1, 3), ones(3), new long[3]);

    assertEquals(0, inference.pinnedOutputArenas());
    assertEquals(0L, inference.pinnedOutputBytes());
  }

  @Test
  void testTheObjectReturningRunShapesTheSameValuesTheFlatReadGives() throws OrtException {
    final long[] ids = distinctIds(3, 4);

    final float[][][] nested = (float[][][]) inference.run(new long[] {3, 4}, ids, ones(12),
        new long[12], TOKENS_OUTPUT);
    final float[] pinnedFlat = pinned(3, 4, ids, TOKENS_OUTPUT, new long[] {3, 4, HIDDEN});

    for (int r = 0; r < 3; r++) {
      for (int t = 0; t < 4; t++) {
        for (int d = 0; d < HIDDEN; d++) {
          assertEquals(pinnedFlat[((r * 4) + t) * HIDDEN + d], nested[r][t][d],
              "row " + r + " position " + t + " component " + d);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // close()
  // ---------------------------------------------------------------------------------------------

  @Test
  void testCloseReleasesTheTensorAndTheAccounting() throws OrtException {
    pinned(4, 5, distinctIds(4, 5), TOKENS_OUTPUT, new long[] {4, 5, HIDDEN});
    final OnnxTensor tensor = inference.pinnedOutputTensor();
    assertNotNull(tensor);
    assertFalse(tensor.isClosed());
    assertTrue(inference.pinnedOutputBytes() > 0);

    inference.close();

    assertTrue(tensor.isClosed(), "the pinned output tensor was not released");
    assertEquals(0L, inference.pinnedOutputBytes());
    assertEquals(0, inference.pinnedOutputArenas());
  }

  @Test
  void testCloseTwiceIsSafe() throws OrtException {
    pinned(2, 3, distinctIds(2, 3), TOKENS_OUTPUT, new long[] {2, 3, HIDDEN});

    inference.close();
    inference.close();

    assertEquals(0L, inference.pinnedOutputBytes());
  }

  @Test
  void testCloseOnAnInferenceThatNeverPinnedAnythingIsSafe() {
    inference.close();

    assertEquals(0L, inference.pinnedOutputBytes());
  }

  @Test
  void testAPinnedRunAfterCloseFailsClearly() throws OrtException {
    pinned(2, 3, distinctIds(2, 3), TOKENS_OUTPUT, new long[] {2, 3, HIDDEN});
    inference.close();

    final IllegalStateException e = assertThrows(IllegalStateException.class,
        () -> pinned(2, 3, distinctIds(2, 3), TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}));

    assertTrue(e.getMessage().contains("closed"), e.getMessage());
  }

  @Test
  void testAnUnpinnedRunAfterCloseFailsClearlyToo() {
    inference.close();

    final IllegalStateException e = assertThrows(IllegalStateException.class,
        () -> unpinned(2, 3, distinctIds(2, 3), TOKENS_OUTPUT));

    assertTrue(e.getMessage().contains("closed"), e.getMessage());
  }

  @Test
  void testAnObjectReturningRunAfterCloseFailsClearlyToo() {
    inference.close();

    assertThrows(IllegalStateException.class,
        () -> inference.run(new long[] {1, 3}, distinctIds(1, 3), ones(3), new long[3]));
  }

  // ---------------------------------------------------------------------------------------------
  // The direct memory bound
  // ---------------------------------------------------------------------------------------------

  @Test
  void testAnOutputAtTheBoundIsPinnedAndOneBeyondItIsRefused() throws OrtException {
    // The bound is exactly the bytes a 2 by 3 rank 3 output needs, so the first run sits on it.
    final long bound = 2L * 3 * HIDDEN * 4;
    try (OnnxInference bounded = new OnnxInference(env, session, bound)) {
      final float[] atTheBound = bounded.run(new long[] {2, 3}, distinctIds(2, 3), ones(6),
          new long[6], TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}, flat());
      assertEquals(2 * 3 * HIDDEN, atTheBound.length);
      assertEquals(bound, bounded.pinnedOutputBytes());

      // One float more than the arena holds, so growing it would pass the bound by 4 bytes.
      final IllegalStateException e = assertThrows(IllegalStateException.class,
          () -> bounded.run(new long[] {2, 4}, distinctIds(2, 4), ones(8), new long[8],
              TOKENS_OUTPUT, new long[] {2, 4, HIDDEN}, flat()));

      assertTrue(e.getMessage().contains(String.valueOf(bound)), e.getMessage());
      assertTrue(e.getMessage().contains("direct memory"), e.getMessage());
      assertEquals(bound, bounded.pinnedOutputBytes(), "a refused run charges nothing");
    }
  }

  @Test
  void testAnInferenceStillWorksAtTheBoundAfterARefusal() throws OrtException {
    final long bound = 2L * 3 * HIDDEN * 4;
    try (OnnxInference bounded = new OnnxInference(env, session, bound)) {
      final float[] first = bounded.run(new long[] {2, 3}, distinctIds(2, 3), ones(6), new long[6],
          TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}, flat());
      assertThrows(IllegalStateException.class,
          () -> bounded.run(new long[] {4, 4}, distinctIds(4, 4), ones(16), new long[16],
              TOKENS_OUTPUT, new long[] {4, 4, HIDDEN}, flat()));

      // The refusal left the arena intact, so the shape that fitted still fits and still answers.
      assertArrayEquals(first, bounded.run(new long[] {2, 3}, distinctIds(2, 3), ones(6),
          new long[6], TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}, flat()));
    }
  }

  @Test
  void testARefusedRunCanStillBeReadUnpinned() throws OrtException {
    final long bound = 2L * 3 * HIDDEN * 4;
    try (OnnxInference bounded = new OnnxInference(env, session, bound)) {
      final long[] ids = distinctIds(4, 4);
      assertThrows(IllegalStateException.class,
          () -> bounded.run(new long[] {4, 4}, ids, ones(16), new long[16], TOKENS_OUTPUT,
              new long[] {4, 4, HIDDEN}, flat()));

      final float[] flatRead = bounded.run(new long[] {4, 4}, ids, ones(16), new long[16],
          TOKENS_OUTPUT, flat());

      assertEquals(4 * 4 * HIDDEN, flatRead.length);
      assertEquals(0L, bounded.pinnedOutputBytes());
    }
  }

  @Test
  void testTheBoundCoversEveryThreadTogetherNotEachThreadSeparately()
      throws Exception {
    // Two threads, an arena each, and a bound that only one of them fits in. The second has to be
    // refused, which is what makes the bound a property of the instance rather than of a thread.
    final long bound = 2L * 3 * HIDDEN * 4;
    final ExecutorService pool = Executors.newFixedThreadPool(2);
    try (OnnxInference bounded = new OnnxInference(env, session, bound)) {
      final List<Future<Object>> results = new ArrayList<>();
      for (int t = 0; t < 2; t++) {
        results.add(pool.submit(() -> {
          try {
            return bounded.run(new long[] {2, 3}, distinctIds(2, 3), ones(6), new long[6],
                TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}, flat());
          } catch (IllegalStateException e) {
            return e;
          }
        }));
      }
      int refused = 0;
      int served = 0;
      for (final Future<Object> result : results) {
        if (result.get(30, TimeUnit.SECONDS) instanceof IllegalStateException) {
          refused++;
        } else {
          served++;
        }
      }
      assertEquals(1, served, "exactly one thread should fit in a one-arena bound");
      assertEquals(1, refused, "the other thread should be refused");
      assertEquals(bound, bounded.pinnedOutputBytes());
    } finally {
      pool.shutdownNow();
    }
  }

  @Test
  void testANonPositiveBoundIsRejected() {
    assertThrows(IllegalArgumentException.class, () -> new OnnxInference(env, session, 0L));
    assertThrows(IllegalArgumentException.class, () -> new OnnxInference(env, session, -1L));
  }

  @Test
  void testTheDefaultBoundAdmitsFourOfTheLargestBatchTheTokenPositionBoundPermits() {
    // The two limits must not contradict each other: the caller-side bound of 16384 token positions
    // with a hidden size of 1024 is the widest output a batch can ask for, and the default is four
    // of those. Asserted rather than left in prose, so a change to either number breaks here.
    final long widestArena = 16384L * 1024 * 4;
    assertEquals(4 * widestArena, OnnxInference.DEFAULT_MAX_PINNED_OUTPUT_BYTES);
  }

  // ---------------------------------------------------------------------------------------------
  // Concurrency
  // ---------------------------------------------------------------------------------------------

  @Test
  void testEightThreadsPinningTheSameShapeEachGetTheirOwnAnswers() throws Exception {
    // Every thread runs a different batch, so a shared buffer would hand one thread another's
    // numbers. Nothing here is constant across threads: the ids differ, so every single output value
    // differs, and each thread checks every value of its own output against what the graph must
    // produce for its own ids.
    final int threads = 8;
    final int rows = 6;
    final int width = 5;
    final ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      final List<Callable<Boolean>> work = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final long base = (t + 1) * 1000L;
        work.add(() -> {
          for (int repeat = 0; repeat < 20; repeat++) {
            final long[] ids = new long[rows * width];
            for (int i = 0; i < ids.length; i++) {
              ids[i] = base + i;
            }
            final float[] out = inference.run(new long[] {rows, width}, ids, ones(ids.length),
                new long[ids.length], TOKENS_OUTPUT, new long[] {rows, width, HIDDEN}, flat());
            for (int i = 0; i < ids.length; i++) {
              for (int d = 0; d < HIDDEN; d++) {
                if (out[i * HIDDEN + d] != W[d] * ids[i]) {
                  throw new AssertionError("thread with base " + base + " repeat " + repeat
                      + " position " + i + " component " + d + " read " + out[i * HIDDEN + d]
                      + " but its own ids require " + W[d] * ids[i]);
                }
              }
            }
          }
          return Boolean.TRUE;
        });
      }
      for (final Future<Boolean> result : pool.invokeAll(work)) {
        assertTrue(result.get(120, TimeUnit.SECONDS));
      }
    } finally {
      pool.shutdownNow();
      assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS));
    }

    assertEquals(threads, inference.pinnedOutputArenas(), "one arena per thread, not one shared");
    assertEquals((long) threads * rows * width * HIDDEN * 4, inference.pinnedOutputBytes());
  }

  @Test
  void testThreadsPinningDifferentShapesDoNotDisturbEachOther() throws Exception {
    final int threads = 4;
    final ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      final List<Callable<Boolean>> work = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final int width = 2 + t * 3;
        final int rows = 1 + t;
        final long base = (t + 1) * 500L;
        work.add(() -> {
          for (int repeat = 0; repeat < 15; repeat++) {
            final long[] ids = new long[rows * width];
            for (int i = 0; i < ids.length; i++) {
              ids[i] = base + i;
            }
            final float[] out = inference.run(new long[] {rows, width}, ids, ones(ids.length),
                new long[ids.length], POOLED_OUTPUT, new long[] {rows, HIDDEN}, flat());
            for (int r = 0; r < rows; r++) {
              long sum = 0;
              for (int c = 0; c < width; c++) {
                sum += ids[r * width + c];
              }
              for (int d = 0; d < HIDDEN; d++) {
                if (out[r * HIDDEN + d] != W[d] * sum) {
                  throw new AssertionError("thread " + base + " row " + r + " component " + d);
                }
              }
            }
          }
          return Boolean.TRUE;
        });
      }
      for (final Future<Boolean> result : pool.invokeAll(work)) {
        assertTrue(result.get(120, TimeUnit.SECONDS));
      }
    } finally {
      pool.shutdownNow();
      assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS));
    }
    assertEquals(threads, inference.pinnedOutputArenas());
  }

  // ---------------------------------------------------------------------------------------------
  // The reader contract
  // ---------------------------------------------------------------------------------------------

  @Test
  void testTheReaderIsHandedTheOutputShapeAndExactlyItsElements() throws OrtException {
    final long[] seen = new long[3];
    final int[] remaining = new int[1];

    inference.run(new long[] {5, 7}, distinctIds(5, 7), ones(35), new long[35], TOKENS_OUTPUT,
        new long[] {5, 7, HIDDEN}, (values, shape) -> {
          System.arraycopy(shape, 0, seen, 0, shape.length);
          remaining[0] = values.remaining();
          return null;
        });

    assertArrayEquals(new long[] {5, 7, HIDDEN}, seen);
    assertEquals(5 * 7 * HIDDEN, remaining[0]);
  }

  @Test
  void testTheReaderIsHandedTheShapeTheRunReportsOnTheUnpinnedPathToo() throws OrtException {
    final long[] seen = new long[3];

    inference.run(new long[] {5, 7}, distinctIds(5, 7), ones(35), new long[35], TOKENS_OUTPUT,
        (values, shape) -> {
          System.arraycopy(shape, 0, seen, 0, shape.length);
          return null;
        });

    assertArrayEquals(new long[] {5, 7, HIDDEN}, seen);
  }

  @Test
  void testAReaderThatRejectsTheShapeSurfacesItsOwnFailure() {
    final OrtException e = assertThrows(OrtException.class,
        () -> inference.run(new long[] {2, 3}, distinctIds(2, 3), ones(6), new long[6],
            TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}, (values, shape) -> {
              throw new OrtException("the reader wanted a different shape");
            }));

    assertEquals("the reader wanted a different shape", e.getMessage());
  }

  @Test
  void testANullReaderIsRejected() {
    assertThrows(IllegalArgumentException.class,
        () -> inference.run(new long[] {2, 3}, distinctIds(2, 3), ones(6), new long[6],
            TOKENS_OUTPUT, (OnnxInference.OutputReader<float[]>) null));
  }

  @Test
  void testAnOutputNameTheModelDoesNotHaveIsReportedOnTheReaderPath() {
    final OrtException e = assertThrows(OrtException.class,
        () -> unpinned(2, 3, distinctIds(2, 3), "pooler_output"));

    assertEquals("The model returned no output named pooler_output", e.getMessage());
  }

  @Test
  void testEveryInputTensorIsClosedOnAPinnedRunAndThePinnedOutputIsNot() throws OrtException {
    final HashMap<String, OnnxTensor> staged = new HashMap<>();

    inference.run(staged, new long[] {2, 3}, distinctIds(2, 3), ones(6), new long[6],
        TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}, flat());

    assertEquals(3, staged.size());
    for (final OnnxTensor input : staged.values()) {
      assertTrue(input.isClosed(), "an input tensor of a pinned run was not closed");
    }
    final OnnxTensor pinnedOutput = inference.pinnedOutputTensor();
    assertNotNull(pinnedOutput);
    assertFalse(pinnedOutput.isClosed(), "the pinned output must survive the run to be reused");
  }

  @Test
  void testEveryInputTensorIsClosedWhenAPinnedRunFails() {
    final HashMap<String, OnnxTensor> staged = new HashMap<>();

    assertThrows(OrtException.class, () -> inference.run(staged, new long[] {2, 3},
        distinctIds(2, 3), ones(6), new long[6], TOKENS_OUTPUT, new long[] {2, 2, HIDDEN}, flat()));

    assertEquals(3, staged.size());
    for (final OnnxTensor input : staged.values()) {
      assertTrue(input.isClosed(), "an input tensor of a failed pinned run was not closed");
    }
  }

  @Test
  void testTheReaderMayReadTheBufferInAnyOrderWithoutDisturbingTheNextRun() throws OrtException {
    final long[] ids = distinctIds(3, 4);
    final float[] forwards = pinned(3, 4, ids, TOKENS_OUTPUT, new long[] {3, 4, HIDDEN});

    final float[] backwards = inference.run(new long[] {3, 4}, ids, ones(12), new long[12],
        TOKENS_OUTPUT, new long[] {3, 4, HIDDEN}, (values, shape) -> {
          final float[] copy = new float[values.remaining()];
          for (int i = copy.length - 1; i >= 0; i--) {
            copy[i] = values.get(i);
          }
          return copy;
        });

    assertArrayEquals(forwards, backwards);
    // And the run after a reader that walked the buffer backwards is still correct, which is what
    // the absolute-get contract of the reader buys.
    assertArrayEquals(forwards, pinned(3, 4, ids, TOKENS_OUTPUT, new long[] {3, 4, HIDDEN}));
  }
}
