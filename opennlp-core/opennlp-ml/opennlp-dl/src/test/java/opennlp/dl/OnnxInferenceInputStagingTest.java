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
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The input staging path of {@link OnnxInference}: that the memory the three inputs of a run reach
 * ONNX Runtime in is direct and belongs to this class, that it is reused across runs rather than
 * allocated per run, that no run can read a position a previous run wrote, that the direct memory it
 * holds is bounded and released, and that none of it is shared between threads.
 *
 * <p>Against the same deterministic {@code tiny-pooled.onnx} graph {@link OnnxInferenceTest} uses,
 * where {@code token_embeddings[b][t][d] = float(input_ids[b][t]) * W[d]} of shape
 * <code>{batch, tokens, 3}</code>, with {@code W = [0.5, -1, 2]}. Every output value therefore names
 * the token id that produced it, which is what turns stale data from a previous run into something a
 * test can see rather than something it has to take on trust: a position the current run did not
 * write shows up as the previous run's id, not as a plausible number.</p>
 *
 * <p>A test that asserts a shape mismatch is rejected makes ONNX Runtime print its own error to
 * standard error before returning the failure. That output is expected.</p>
 */
class OnnxInferenceInputStagingTest {

  /** The weights the graph multiplies each token id by. */
  private static final float[] W = {0.5f, -1f, 2f};

  /** The hidden size of the graph, the last dimension of its token output. */
  private static final int HIDDEN = 3;

  private static final String TOKENS_OUTPUT = "token_embeddings";

  /** The inputs one run stages, and the bytes one of their positions takes. */
  private static final int INPUTS = 3;

  private static final int LONG_BYTES = 8;

  private OrtEnvironment env;
  private OrtSession session;
  private OnnxInference inference;

  @BeforeEach
  void openSession(@TempDir final Path dir) throws IOException, OrtException {
    final Path model = dir.resolve("tiny-pooled.onnx");
    try (InputStream is = Objects.requireNonNull(OnnxInferenceInputStagingTest.class
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

  /** {@return {@code rows * width} ids counting up from {@code first}, so every position differs} */
  private static long[] idsFrom(final int rows, final int width, final long first) {
    final long[] ids = new long[rows * width];
    for (int i = 0; i < ids.length; i++) {
      ids[i] = first + i;
    }
    return ids;
  }

  private static long[] ones(final int elements) {
    final long[] mask = new long[elements];
    Arrays.fill(mask, 1L);
    return mask;
  }

  /** A reader that copies the whole output out flat. */
  private static OnnxInference.OutputReader<float[]> flat() {
    return (values, shape) -> {
      final float[] copy = new float[values.remaining()];
      for (int i = 0; i < copy.length; i++) {
        copy[i] = values.get(i);
      }
      return copy;
    };
  }

  /** Runs the {@code long[]} path, reading the token output through one flat copy. */
  private float[] fromArrays(final int rows, final int width, final long[] ids) throws OrtException {
    return inference.run(new long[] {rows, width}, ids, ones(ids.length), new long[ids.length],
        TOKENS_OUTPUT, flat());
  }

  /** Runs the {@link OnnxInference.InputStager} path over the same ids. */
  private float[] fromStager(final int rows, final int width, final long[] ids) throws OrtException {
    return inference.run(new long[] {rows, width}, stagerOf(ids), TOKENS_OUTPUT, null, flat());
  }

  /** A stager that writes {@code ids}, a mask of ones and zero token types, all in full. */
  private static OnnxInference.InputStager stagerOf(final long[] ids) {
    return (idsSlot, maskSlot, typesSlot) -> {
      for (final long id : ids) {
        idsSlot.put(id);
        maskSlot.put(1L);
        typesSlot.put(0L);
      }
    };
  }

  /**
   * Asserts that a flat token output holds exactly what the graph computes for {@code ids}, so a
   * position carrying a previous run's id fails here and names the id it carried.
   */
  private static void assertEmbeds(final long[] ids, final float[] output) {
    assertEquals(ids.length * HIDDEN, output.length);
    for (int i = 0; i < ids.length; i++) {
      for (int d = 0; d < HIDDEN; d++) {
        assertEquals(W[d] * ids[i], output[i * HIDDEN + d],
            "position " + i + " component " + d + " should embed id " + ids[i] + " but embeds "
                + output[i * HIDDEN + d] / W[d]);
      }
    }
  }

  /** {@return the direct bytes an arena holds once it has been grown to {@code elements}} */
  private static long arenaBytes(final int elements) {
    return (long) elements * INPUTS * LONG_BYTES;
  }

  // ---------------------------------------------------------------------------------------------
  // The staged values are the values the run asked for, at every extent
  // ---------------------------------------------------------------------------------------------

  @ParameterizedTest(name = "a {0} by {1} run out of arrays")
  @CsvSource({"1, 1", "1, 12", "12, 1", "3, 4", "16, 9", "64, 7", "128, 56"})
  void testEveryPositionOfEveryExtentEmbedsItsOwnId(final int rows, final int width)
      throws OrtException {
    final long[] ids = idsFrom(rows, width, 1);

    assertEmbeds(ids, fromArrays(rows, width, ids));
  }

  @ParameterizedTest(name = "a {0} by {1} run out of a stager")
  @CsvSource({"1, 1", "1, 12", "12, 1", "3, 4", "16, 9", "64, 7", "128, 56"})
  void testTheStagerPathEmbedsTheSameValuesAsTheArrayPath(final int rows, final int width)
      throws OrtException {
    final long[] ids = idsFrom(rows, width, 1);

    final float[] byArrays = fromArrays(rows, width, ids);
    final float[] byStager = fromStager(rows, width, ids);

    assertEmbeds(ids, byStager);
    assertArrayEquals(byArrays, byStager);
  }

  @Test
  void testASequenceOfMixedExtentsEachAnswersItsOwnShape() throws OrtException {
    // Widths up and down and row counts up and down, which is what a call fragmented by tokenized
    // length produces, and the arena is re-cut for every one of them.
    final int[][] extents = {{1, 3}, {4, 7}, {2, 2}, {9, 1}, {1, 20}, {3, 5}, {1, 3}, {6, 6}};

    for (final int[] extent : extents) {
      final long[] ids = idsFrom(extent[0], extent[1], 100L * extent[0] + extent[1]);
      assertEmbeds(ids, fromArrays(extent[0], extent[1], ids));
      assertEmbeds(ids, fromStager(extent[0], extent[1], ids));
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Stale data: no run reads a position a previous run wrote
  // ---------------------------------------------------------------------------------------------

  @Test
  void testTheSameExtentRunTwiceWithDifferentContentAnswersTheSecondContent() throws OrtException {
    // The second run repeats the extent, which is the case the reuse is for and therefore the case a
    // partial overwrite would hide in: nothing is re-cut and nothing is reallocated between them.
    final long[] first = idsFrom(4, 5, 1000);
    final long[] second = idsFrom(4, 5, 1);

    assertEmbeds(first, fromArrays(4, 5, first));
    assertEmbeds(second, fromArrays(4, 5, second));
    // And the same from a stager, whose write is the caller's rather than a bulk copy of an array.
    assertEmbeds(first, fromStager(4, 5, first));
    assertEmbeds(second, fromStager(4, 5, second));
  }

  @Test
  void testAnExtentRecurringAfterADifferentOneAnswersItsOwnContent() throws OrtException {
    // A, then B, then A again with different content. The intervening B re-cuts the slots, so the
    // second A exercises the path where the extent is restored rather than merely repeated.
    final long[] firstA = idsFrom(3, 4, 500);
    final long[] b = idsFrom(8, 9, 7000);
    final long[] secondA = idsFrom(3, 4, 1);

    assertEmbeds(firstA, fromArrays(3, 4, firstA));
    assertEmbeds(b, fromArrays(8, 9, b));
    assertEmbeds(secondA, fromArrays(3, 4, secondA));
  }

  @Test
  void testANarrowerRunAfterAWiderOneReadsNoneOfTheWiderRunsTail() throws OrtException {
    // The arena only grows, so after the wide run its memory still holds the wide run's ids past the
    // narrow run's extent. The narrow run must not reach them: every value it embeds is its own.
    final long[] wide = idsFrom(8, 8, 900);
    final long[] narrow = idsFrom(2, 3, 1);

    assertEmbeds(wide, fromArrays(8, 8, wide));
    final float[] narrowOutput = fromArrays(2, 3, narrow);

    assertEmbeds(narrow, narrowOutput);
    // Said the other way round, so the failure names the defect rather than an index. The two runs
    // use disjoint id ranges, 900 upwards against 1 to 6, so no value of the narrow run may embed an
    // id of the wide one.
    for (final float value : narrowOutput) {
      for (final long wideId : wide) {
        for (int d = 0; d < HIDDEN; d++) {
          assertTrue(value != W[d] * wideId,
              "a value of the narrow run embeds " + wideId + ", which only the wider run staged");
        }
      }
    }
  }

  @Test
  void testAStagerThatLeavesAnInputShortFailsRatherThanLettingTheRunReadThePreviousOne() {
    final OnnxInference.InputStager shortOnTypes = (ids, mask, types) -> {
      // Writes every position of the ids and of the mask, and stops one short on the token types,
      // which is the shape of the mistake a caller makes when it believes an input already holds the
      // right values because the last run left them there.
      while (ids.hasRemaining()) {
        ids.put(1L);
        mask.put(1L);
      }
      while (types.remaining() > 1) {
        types.put(0L);
      }
    };

    final IllegalStateException e = assertThrows(IllegalStateException.class, () ->
        inference.run(new long[] {2, 3}, shortOnTypes, TOKENS_OUTPUT, null, flat()));

    assertTrue(e.getMessage().contains(AbstractDL.TOKEN_TYPE_IDS), e.getMessage());
    assertTrue(e.getMessage().contains("5 of the 6 values"), e.getMessage());
    assertTrue(e.getMessage().contains("previous run"), e.getMessage());
  }

  @Test
  void testAStagerThatWritesNothingAtAllFails() {
    final OnnxInference.InputStager writesNothing = (ids, mask, types) -> {
      // Deliberately empty: a stager that believes all three inputs already hold the right values.
    };

    final IllegalStateException e = assertThrows(IllegalStateException.class, () ->
        inference.run(new long[] {2, 3}, writesNothing, TOKENS_OUTPUT, null, flat()));

    assertTrue(e.getMessage().contains(AbstractDL.INPUT_IDS), e.getMessage());
    assertTrue(e.getMessage().contains("0 of the 6 values"), e.getMessage());
  }

  @Test
  void testAStagerUsingTheAbsolutePutIsRejectedRatherThanTrusted() {
    // An absolute put writes the value but does not move the position, so this stager has in fact
    // filled every slot. It is still refused, because the position is the only evidence available
    // that a slot was covered and a stager that does not leave it is indistinguishable from one that
    // skipped the input. The contract says relative puts, and the check holds it to that.
    final OnnxInference.InputStager absolute = (ids, mask, types) -> {
      for (int i = 0; i < 6; i++) {
        ids.put(i, 1L);
        mask.put(i, 1L);
        types.put(i, 0L);
      }
    };

    assertThrows(IllegalStateException.class, () ->
        inference.run(new long[] {2, 3}, absolute, TOKENS_OUTPUT, null, flat()));
  }

  @Test
  void testAnArrayShorterThanTheShapeFailsAndStagesNoTensorForThatInput() {
    // A short array would fill part of a reused slot and leave the previous run's values in the
    // rest, so it is refused rather than padded or truncated. The ids tensor that was created before
    // the mask failed is still in the map and still closed by the run.
    final Map<String, OnnxTensor> staged = new HashMap<>();

    final OrtException e = assertThrows(OrtException.class, () ->
        inference.run(staged, new long[] {2, 3}, idsFrom(2, 3, 1), ones(5), new long[6],
            TOKENS_OUTPUT, null, flat()));

    assertTrue(e.getMessage().contains(AbstractDL.ATTENTION_MASK), e.getMessage());
    assertTrue(e.getMessage().contains("5 values"), e.getMessage());
    assertEquals(1, staged.size());
    assertTrue(staged.get(AbstractDL.INPUT_IDS).isClosed());
  }

  @Test
  void testAnArrayLongerThanTheShapeIsRefusedToo() {
    final OrtException e = assertThrows(OrtException.class, () ->
        fromArrays(2, 3, idsFrom(2, 4, 1)));

    assertTrue(e.getMessage().contains("8 values"), e.getMessage());
  }

  @Test
  void testARunAfterAFailedStagingStillAnswersCorrectly() throws OrtException {
    final long[] ids = idsFrom(2, 3, 1);
    assertThrows(OrtException.class, () -> inference.run(new long[] {2, 3}, ids, ones(5),
        new long[6], TOKENS_OUTPUT, flat()));

    // The failed run left the ids in the slot and nothing else, and the next run overwrites all of
    // it, so the arena is not left in a state that poisons what follows.
    assertEmbeds(ids, fromArrays(2, 3, ids));
  }

  // ---------------------------------------------------------------------------------------------
  // The memory is direct, it is this class's own, and it is reused
  // ---------------------------------------------------------------------------------------------

  @Test
  void testTheBufferAnInputReachesOnnxRuntimeInIsTheDirectMemoryThisClassOwns() throws OrtException {
    // The whole point of the staging: ONNX Runtime wraps a direct buffer where it lies, and answers a
    // heap buffer with an allocateDirect plus a full copy. Which of the two happened is readable off
    // the tensor, through getBufferRef, and the test runs inside the output reader because that is
    // where the input tensors are still open. A sentinel written through the tensor's own reference
    // has to appear in the buffer this class staged into, which it can only do if the two are one
    // allocation.
    final Map<String, OnnxTensor> staged = new HashMap<>();
    final long sentinel = 424242L;

    inference.run(staged, new long[] {2, 3}, idsFrom(2, 3, 1), ones(6), new long[6], TOKENS_OUTPUT,
        null, (values, shape) -> {
          for (final String name : List.of(AbstractDL.INPUT_IDS, AbstractDL.ATTENTION_MASK,
              AbstractDL.TOKEN_TYPE_IDS)) {
            final LongBuffer ours = inference.inputStagingBuffer(name);
            assertNotNull(ours, name + " was not staged into memory this class owns");
            assertTrue(ours.isDirect(), name + " was staged into a heap buffer, which ONNX Runtime"
                + " answers with an allocateDirect plus a full copy");
            final Optional<Buffer> ref = staged.get(name).getBufferRef();
            assertTrue(ref.isPresent(), name + " is not a buffer backed tensor");
            assertTrue(ref.get().isDirect(), name + " reached ONNX Runtime in a heap buffer");
            ((LongBuffer) ref.get()).put(0, sentinel);
            assertEquals(sentinel, ours.get(0), "the buffer ONNX Runtime holds for " + name
                + " is not the buffer this class staged into, so it was copied");
          }
          return null;
        });
  }

  @Test
  void testTheSameStagingMemoryServesEveryRunOfOneThread() throws OrtException {
    fromArrays(4, 5, idsFrom(4, 5, 1));
    final LongBuffer first = inference.inputStagingBuffer(AbstractDL.INPUT_IDS);
    final long afterFirst = inference.inputStagingBytes();

    fromArrays(4, 5, idsFrom(4, 5, 100));

    // The same view object, which is only possible if nothing was reallocated and nothing re-cut.
    assertSame(first, inference.inputStagingBuffer(AbstractDL.INPUT_IDS));
    assertEquals(afterFirst, inference.inputStagingBytes());
    assertEquals(1, inference.inputStagingArenas());
  }

  @Test
  void testAnArenaHoldsAllThreeInputsAndIsChargedForAllThree() throws OrtException {
    fromArrays(4, 5, idsFrom(4, 5, 1));

    assertEquals(arenaBytes(20), inference.inputStagingBytes());
    assertEquals(1, inference.inputStagingArenas());
    for (final String name : List.of(AbstractDL.INPUT_IDS, AbstractDL.ATTENTION_MASK,
        AbstractDL.TOKEN_TYPE_IDS)) {
      assertEquals(20, inference.inputStagingBuffer(name).capacity(), name);
    }
  }

  @Test
  void testAnArenaGrowsForAWiderRunAndIsNotShrunkByANarrowerOne() throws OrtException {
    fromArrays(2, 3, idsFrom(2, 3, 1));
    assertEquals(arenaBytes(6), inference.inputStagingBytes());

    fromArrays(5, 8, idsFrom(5, 8, 1));
    assertEquals(arenaBytes(40), inference.inputStagingBytes(), "the arena grew to the wider run");

    fromArrays(1, 2, idsFrom(1, 2, 1));
    assertEquals(arenaBytes(40), inference.inputStagingBytes(),
        "a narrower run does not shrink the arena");
    assertEquals(1, inference.inputStagingArenas(), "growing does not add an arena");
  }

  @Test
  void testNoStagingMemoryIsHeldBeforeTheFirstRun() {
    assertEquals(0L, inference.inputStagingBytes());
    assertEquals(0, inference.inputStagingArenas());
    assertNull(inference.inputStagingBuffer(AbstractDL.INPUT_IDS));
  }

  @Test
  void testAnInputTheModelDoesNotDeclareStagesNoTensorAndLeavesItsSlotUnread() throws OrtException {
    // What a model taking input_ids alone does. The arena still holds three slots, because they are
    // one allocation, but only the one that was written is handed to a tensor.
    final Map<String, OnnxTensor> staged = new HashMap<>();

    inference.run(staged, new long[] {1, 2}, new long[] {7, 4}, null, null, null);

    assertEquals(1, staged.size());
    assertEquals(arenaBytes(2), inference.inputStagingBytes());
  }

  // ---------------------------------------------------------------------------------------------
  // Release
  // ---------------------------------------------------------------------------------------------

  @Test
  void testCloseReleasesTheStagingMemoryOfEveryThread() throws Exception {
    final int threads = 4;
    runOnThreads(threads, thread -> fromArrays(3, 4, idsFrom(3, 4, 1000L * thread)));

    assertEquals(threads, inference.inputStagingArenas());
    assertEquals(threads * arenaBytes(12), inference.inputStagingBytes());

    inference.close();

    assertEquals(0, inference.inputStagingArenas());
    assertEquals(0L, inference.inputStagingBytes());
  }

  @Test
  void testASecondCloseIsANoOp() throws OrtException {
    fromArrays(2, 3, idsFrom(2, 3, 1));
    inference.close();
    inference.close();

    assertEquals(0L, inference.inputStagingBytes());
    assertEquals(0, inference.inputStagingArenas());
  }

  @Test
  void testCloseWithoutAnyRunReleasesNothingAndFailsNothing() {
    inference.close();

    assertEquals(0L, inference.inputStagingBytes());
    assertEquals(0, inference.inputStagingArenas());
  }

  @Test
  void testAStagerRunAfterCloseFailsClearly() {
    inference.close();

    final IllegalStateException e = assertThrows(IllegalStateException.class, () ->
        fromStager(2, 3, idsFrom(2, 3, 1)));

    assertTrue(e.getMessage().contains("closed"), e.getMessage());
    assertTrue(e.getMessage().contains("input"), e.getMessage());
  }

  @Test
  void testAnArrayRunAfterCloseFailsClearlyForTheSameThreadThatRanBefore() throws OrtException {
    fromArrays(2, 3, idsFrom(2, 3, 1));
    inference.close();

    final IllegalStateException e = assertThrows(IllegalStateException.class, () ->
        fromArrays(2, 3, idsFrom(2, 3, 1)));

    assertTrue(e.getMessage().contains("closed"), e.getMessage());
  }

  @Test
  void testEveryInputTensorOfAStagerRunIsClosedByTheRun() throws OrtException {
    final Map<String, OnnxTensor> staged = new HashMap<>();

    inference.run(staged, new long[] {2, 3}, stagerOf(idsFrom(2, 3, 1)), TOKENS_OUTPUT, null,
        flat());

    assertEquals(3, staged.size());
    for (final Map.Entry<String, OnnxTensor> input : staged.entrySet()) {
      assertTrue(input.getValue().isClosed(), input.getKey() + " was not closed");
    }
    // And the memory behind them is still held, because that is what the next run reuses.
    assertEquals(arenaBytes(6), inference.inputStagingBytes());
  }

  @Test
  void testEveryInputTensorIsClosedWhenAStagerRunFailsInTheSession() {
    final Map<String, OnnxTensor> staged = new HashMap<>();

    // input_ids of rank 1, which the graph rejects inside the run rather than at tensor creation.
    assertThrows(OrtException.class, () -> inference.run(staged, new long[] {6},
        stagerOf(idsFrom(1, 6, 1)), TOKENS_OUTPUT, null, flat()));

    assertEquals(3, staged.size());
    for (final Map.Entry<String, OnnxTensor> input : staged.entrySet()) {
      assertTrue(input.getValue().isClosed(), input.getKey() + " was not closed");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Argument checks
  // ---------------------------------------------------------------------------------------------

  @Test
  void testANullStagerIsRejected() {
    assertThrows(IllegalArgumentException.class, () -> inference.run(new long[] {2, 3},
        (OnnxInference.InputStager) null, TOKENS_OUTPUT, null, flat()));
  }

  @Test
  void testANullReaderIsRejectedOnTheStagerPath() {
    assertThrows(IllegalArgumentException.class, () -> inference.run(new long[] {2, 3},
        stagerOf(idsFrom(2, 3, 1)), TOKENS_OUTPUT, null, null));
  }

  @Test
  void testAPinnedOutputWithoutANameIsRejectedOnTheStagerPath() {
    assertThrows(IllegalArgumentException.class, () -> inference.run(new long[] {2, 3},
        stagerOf(idsFrom(2, 3, 1)), null, new long[] {2, 3, HIDDEN}, flat()));
  }

  @Test
  void testAShapeDescribingNoElementsIsRejected() {
    assertThrows(OrtException.class, () -> inference.run(new long[] {0, 3}, new long[0],
        null, null, TOKENS_OUTPUT, flat()));
    assertThrows(OrtException.class, () -> inference.run(new long[0], new long[0], null, null,
        TOKENS_OUTPUT, flat()));
    assertEquals(0L, inference.inputStagingBytes(), "a rejected shape allocates nothing");
  }

  @Test
  void testAStagerRunCanPinItsOutputToo() throws OrtException {
    final long[] ids = idsFrom(3, 4, 1);

    final float[] pinned = inference.run(new long[] {3, 4}, stagerOf(ids), TOKENS_OUTPUT,
        new long[] {3, 4, HIDDEN}, flat());

    assertEmbeds(ids, pinned);
    assertArrayEquals(fromStager(3, 4, ids), pinned);
    // Both kinds of reusable direct memory are held, each counted on its own.
    assertEquals(arenaBytes(12), inference.inputStagingBytes());
    assertEquals(3L * 4 * HIDDEN * 4, inference.pinnedOutputBytes());
  }

  // ---------------------------------------------------------------------------------------------
  // The bound on reusable input memory
  // ---------------------------------------------------------------------------------------------

  @Test
  void testStagingAtTheBoundIsServedAndOneElementBeyondItIsRefused() throws OrtException {
    // The bound is exactly the three slots a 2 by 3 run needs, so the first run sits on it.
    final long bound = arenaBytes(6);
    try (OnnxInference bounded = new OnnxInference(env, session,
        OnnxInference.DEFAULT_MAX_PINNED_OUTPUT_BYTES, bound)) {
      final long[] ids = idsFrom(2, 3, 1);
      final float[] atTheBound = bounded.run(new long[] {2, 3}, ids, ones(6), new long[6],
          TOKENS_OUTPUT, flat());
      assertEmbeds(ids, atTheBound);
      assertEquals(bound, bounded.inputStagingBytes());

      final IllegalStateException e = assertThrows(IllegalStateException.class, () ->
          bounded.run(new long[] {2, 4}, idsFrom(2, 4, 1), ones(8), new long[8], TOKENS_OUTPUT,
              flat()));

      assertTrue(e.getMessage().contains(String.valueOf(bound)), e.getMessage());
      assertTrue(e.getMessage().contains("direct memory"), e.getMessage());
      assertEquals(bound, bounded.inputStagingBytes(), "a refused run charges nothing");
    }
  }

  @Test
  void testAnInferenceStillWorksAtTheInputBoundAfterARefusal() throws OrtException {
    final long bound = arenaBytes(6);
    try (OnnxInference bounded = new OnnxInference(env, session,
        OnnxInference.DEFAULT_MAX_PINNED_OUTPUT_BYTES, bound)) {
      final long[] ids = idsFrom(2, 3, 1);
      final float[] first = bounded.run(new long[] {2, 3}, ids, ones(6), new long[6], TOKENS_OUTPUT,
          flat());
      assertThrows(IllegalStateException.class, () -> bounded.run(new long[] {4, 4},
          idsFrom(4, 4, 1), ones(16), new long[16], TOKENS_OUTPUT, flat()));

      // The refusal left the arena intact, so the extent that fitted still fits and still answers.
      assertArrayEquals(first, bounded.run(new long[] {2, 3}, ids, ones(6), new long[6],
          TOKENS_OUTPUT, flat()));
    }
  }

  @Test
  void testTheInputBoundCoversEveryThreadTogetherNotEachThreadSeparately() throws Exception {
    // Two threads, an arena each, and a bound that only one of them fits in, which is what makes the
    // bound a property of the instance rather than of a thread.
    final long bound = arenaBytes(6);
    final ExecutorService pool = Executors.newFixedThreadPool(2);
    try (OnnxInference bounded = new OnnxInference(env, session,
        OnnxInference.DEFAULT_MAX_PINNED_OUTPUT_BYTES, bound)) {
      final List<Future<Object>> results = new ArrayList<>();
      for (int t = 0; t < 2; t++) {
        results.add(pool.submit(() -> {
          try {
            return bounded.run(new long[] {2, 3}, idsFrom(2, 3, 1), ones(6), new long[6],
                TOKENS_OUTPUT, flat());
          } catch (final IllegalStateException e) {
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
      assertEquals(bound, bounded.inputStagingBytes());
    } finally {
      pool.shutdownNow();
    }
  }

  @Test
  void testANonPositiveInputBoundIsRejected() {
    assertThrows(IllegalArgumentException.class, () -> new OnnxInference(env, session,
        OnnxInference.DEFAULT_MAX_PINNED_OUTPUT_BYTES, 0L));
    assertThrows(IllegalArgumentException.class, () -> new OnnxInference(env, session,
        OnnxInference.DEFAULT_MAX_PINNED_OUTPUT_BYTES, -1L));
  }

  @Test
  void testTheInputBoundAndTheOutputBoundDoNotChargeEachOther() throws OrtException {
    // A run whose pinned output is exactly the output bound is served, although its inputs need
    // direct memory of their own. That is the reason the two are counted separately rather than out
    // of one number: one shared allowance would refuse this run, and the bound on output memory would
    // no longer mean what it says.
    final long outputBound = 2L * 3 * HIDDEN * 4;
    try (OnnxInference bounded = new OnnxInference(env, session, outputBound, arenaBytes(6))) {
      final long[] ids = idsFrom(2, 3, 1);

      final float[] output = bounded.run(new long[] {2, 3}, ids, ones(6), new long[6],
          TOKENS_OUTPUT, new long[] {2, 3, HIDDEN}, flat());

      assertEmbeds(ids, output);
      assertEquals(outputBound, bounded.pinnedOutputBytes());
      assertEquals(arenaBytes(6), bounded.inputStagingBytes());
    }
  }

  @Test
  void testTheTwoDirectMemoryBoundsAreReconciledRatherThanChosenIndependently() {
    // The input bound is the output bound divided by the ratio of the two per-token-position costs at
    // a hidden size of 384, the sentence-transformers MiniLM family: 4 bytes per float times 384
    // against 8 bytes per long times 3 inputs. At that hidden size the two bounds therefore admit the
    // same number of threads at the widest batch, and at a larger hidden size the input bound admits
    // more, which is the safe direction. Asserted rather than left in prose, so a change to either
    // number breaks here.
    final long outputBytesPerPosition = 4L * 384;
    final long inputBytesPerPosition = (long) INPUTS * LONG_BYTES;
    final long ratio = outputBytesPerPosition / inputBytesPerPosition;

    assertEquals(64, ratio);
    assertEquals(OnnxInference.DEFAULT_MAX_PINNED_OUTPUT_BYTES / ratio,
        OnnxInference.DEFAULT_MAX_INPUT_STAGING_BYTES);
    // And the input bound admits at least as many arenas at the caller-side token position bound of
    // 16384 as the output bound admits of its own widest arena, which is four.
    assertTrue(OnnxInference.DEFAULT_MAX_INPUT_STAGING_BYTES / arenaBytes(16384) >= 4,
        "the input bound admits fewer threads at the widest batch than the output bound does");
  }

  // ---------------------------------------------------------------------------------------------
  // Concurrency
  // ---------------------------------------------------------------------------------------------

  @Test
  void testEightThreadsStagingTheSameExtentEachEmbedTheirOwnIds() throws Exception {
    // Every thread stages a different set of ids at the same extent, so a shared staging buffer would
    // hand one thread another's tokens and every single output value would be wrong. Each thread
    // checks every value of its own output against what the graph must produce for its own ids.
    final int threads = 8;
    final List<Object> answers = runOnThreads(threads, thread -> {
      for (int repeat = 0; repeat < 20; repeat++) {
        final long[] ids = idsFrom(4, 5, 1000L * thread + 1);
        assertEmbeds(ids, fromArrays(4, 5, ids));
        assertEmbeds(ids, fromStager(4, 5, ids));
      }
      return null;
    });

    assertEquals(threads, answers.size());
    assertEquals(threads, inference.inputStagingArenas(),
        "one arena per thread, not one shared between them");
    assertEquals(threads * arenaBytes(20), inference.inputStagingBytes());
  }

  @Test
  void testThreadsAtDifferentExtentsDoNotReCutEachOthersSlots() throws Exception {
    // Extents differ per thread as well as content, so a shared arena would be re-cut under a thread
    // mid-run and the run would read the wrong number of values as well as the wrong ones.
    final int threads = 6;
    runOnThreads(threads, thread -> {
      final int width = 2 + thread;
      for (int repeat = 0; repeat < 20; repeat++) {
        final long[] ids = idsFrom(thread + 1, width, 100L * thread + 1);
        assertEmbeds(ids, fromArrays(thread + 1, width, ids));
      }
      return null;
    });

    assertEquals(threads, inference.inputStagingArenas());
  }

  /**
   * Runs {@code work} once on each of {@code threads} threads, handing each its own index, and fails
   * the calling test with whatever any of them threw.
   */
  private List<Object> runOnThreads(final int threads, final Work work) throws Exception {
    final ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      final List<Future<Object>> results = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final int thread = t;
        final Callable<Object> task = () -> work.run(thread);
        results.add(pool.submit(task));
      }
      final List<Object> answers = new ArrayList<>();
      for (final Future<Object> result : results) {
        answers.add(String.valueOf(result.get(60, TimeUnit.SECONDS)));
      }
      return answers;
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
