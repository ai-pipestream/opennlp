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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link OnnxInference} against the deterministic {@code tiny-pooled.onnx} graph, which every
 * component of this package now depends on for its tensors, its run and the release of its native
 * handles.
 *
 * <p>The graph declares the three BERT-style inputs and two float outputs, in this order:
 * {@code token_embeddings[b][t][d] = float(input_ids[b][t]) * W[d]} of shape
 * <code>{batch, tokens, 3}</code>, and {@code sentence_embedding[b][d]}, the sum of a row's token
 * embeddings, of shape <code>{batch, 3}</code>, with {@code W = [0.5, -1, 2]}. So the value of
 * every output position is the vocabulary id that produced it times a known factor, which makes
 * both the rank of the output and the row-major layout of the staged tensors directly readable, and
 * it makes the rank 2 output reachable only by name because it is not the first output.</p>
 */
class OnnxInferenceTest {

  /** The weights the graph multiplies each token id by. */
  private static final float[] W = {0.5f, -1f, 2f};

  private OrtEnvironment env;
  private OrtSession session;
  private OnnxInference inference;

  /** {@return the token embedding the graph produces for one vocabulary id} */
  private static float[] embedding(final long id) {
    return new float[] {W[0] * id, W[1] * id, W[2] * id};
  }

  /** {@return an attention mask of ones as long as the tensor the run stages} */
  private static long[] ones(final int elements) {
    final long[] mask = new long[elements];
    for (int i = 0; i < elements; i++) {
      mask[i] = 1;
    }
    return mask;
  }

  // Copied out of the classpath rather than resolved in place, for the same reason as in
  // SentenceVectorsDLEmbedderTest: the resource may live inside a test-jar.
  @BeforeEach
  void openSession(@TempDir final Path dir) throws IOException, OrtException {
    final Path model = dir.resolve("tiny-pooled.onnx");
    try (InputStream is = Objects.requireNonNull(OnnxInferenceTest.class
        .getResourceAsStream("/opennlp/dl/vectors/tiny-pooled.onnx"))) {
      Files.copy(is, model, StandardCopyOption.REPLACE_EXISTING);
    }
    env = OrtEnvironment.getEnvironment();
    session = env.createSession(model.toString(), new OrtSession.SessionOptions());
    inference = new OnnxInference(env, session);
  }

  @AfterEach
  void closeSession() throws OrtException {
    if (session != null) {
      session.close();
    }
  }

  @Test
  void testBatchOfOneRowReadsTheFirstOutput() throws OrtException {
    final long[] ids = {7, 4, 3};

    final Object value = inference.run(new long[] {1, 3}, ids, ones(3), new long[3]);

    final float[][][] tokens = assertInstanceOf(float[][][].class, value);
    assertEquals(1, tokens.length);
    assertEquals(3, tokens[0].length);
    for (int t = 0; t < ids.length; t++) {
      assertArrayEquals(embedding(ids[t]), tokens[0][t]);
    }
  }

  @Test
  void testBatchOfManyRowsReturnsOneRowPerInputInOrder() throws OrtException {
    final long[][] rows = {{7, 4}, {3, 5}, {2, 6}};
    final long[] ids = {7, 4, 3, 5, 2, 6};

    final Object value = inference.run(new long[] {3, 2}, ids, ones(6), new long[6]);

    final float[][][] tokens = assertInstanceOf(float[][][].class, value);
    assertEquals(rows.length, tokens.length);
    for (int b = 0; b < rows.length; b++) {
      assertEquals(2, tokens[b].length);
      for (int t = 0; t < rows[b].length; t++) {
        assertArrayEquals(embedding(rows[b][t]), tokens[b][t], "row " + b + " token " + t);
      }
    }
  }

  @Test
  void testRowsOfDifferingWidthsRunAtTheWidthTheCallerPadsThemTo() throws OrtException {
    // The caller owns padding: it picks the width and writes the pad id into the tail of a short
    // row. Staging must place row r at offset r * width and leave the other rows alone, so the
    // padded row reads back its own ids followed by the pad id, and a row's own positions are the
    // ones it gets in a batch of itself.
    final long padId = 1;
    final long[] ids = {7, 4, 3, 7, 5, padId};

    final float[][][] batched = assertInstanceOf(float[][][].class,
        inference.run(new long[] {2, 3}, ids, ones(6), new long[6]));
    final float[][][] alone = assertInstanceOf(float[][][].class,
        inference.run(new long[] {1, 2}, new long[] {7, 5}, ones(2), new long[2]));

    assertArrayEquals(embedding(3), batched[0][2]);
    assertArrayEquals(embedding(padId), batched[1][2], "the padded tail of the short row");
    assertArrayEquals(alone[0][0], batched[1][0]);
    assertArrayEquals(alone[0][1], batched[1][1]);
  }

  @Test
  void testNamedOutputOfRankTwoIsReadBackByName() throws OrtException {
    // sentence_embedding is the second output, so a run that reads it proves the name is used
    // rather than the position, and it is the rank 2 output the embedder also has to handle.
    final long[] ids = {7, 4, 3};

    final Object value = inference.run(new long[] {1, 3}, ids, ones(3), new long[3],
        "sentence_embedding");

    final float[][] pooled = assertInstanceOf(float[][].class, value);
    assertEquals(1, pooled.length);
    assertArrayEquals(embedding(7 + 4 + 3), pooled[0]);
  }

  @Test
  void testNamedOutputOfRankThreeIsReadBackByName() throws OrtException {
    final Object value = inference.run(new long[] {1, 2}, new long[] {7, 4}, ones(2), new long[2],
        "token_embeddings");

    final float[][][] tokens = assertInstanceOf(float[][][].class, value);
    assertArrayEquals(embedding(7), tokens[0][0]);
    assertArrayEquals(embedding(4), tokens[0][1]);
  }

  @Test
  void testOutputNameTheModelDoesNotHaveIsReported() {
    final OrtException e = assertThrows(OrtException.class, () ->
        inference.run(new long[] {1, 2}, new long[] {7, 3}, ones(2), new long[2], "pooler_output"));

    assertEquals("The model returned no output named pooler_output", e.getMessage());
  }

  @Test
  void testInputsTheModelDoesNotDeclareAreLeftOut() throws OrtException {
    // What a name finder or a categorizer does with a model that takes input_ids alone: no array
    // means no tensor and no input of that name, rather than a tensor of zeroes.
    final Map<String, OnnxTensor> staged = new HashMap<>();

    final Object value = inference.run(staged, new long[] {1, 2}, new long[] {7, 4}, null, null,
        null);

    final float[][][] tokens = assertInstanceOf(float[][][].class, value);
    assertArrayEquals(embedding(7), tokens[0][0]);
    assertEquals(Set.of(AbstractDL.INPUT_IDS), staged.keySet());
    assertAllClosed(staged);
  }

  @Test
  void testEveryTensorIsClosedOnTheNormalPath() throws OrtException {
    // The map is the seam: the run stages into it and closes what it staged, so a test that keeps
    // the reference can assert the release instead of assuming it.
    final Map<String, OnnxTensor> staged = new HashMap<>();

    final Object value = inference.run(staged, new long[] {1, 2}, new long[] {7, 3}, ones(2),
        new long[2], null);

    assertInstanceOf(float[][][].class, value);
    assertEquals(Set.of(AbstractDL.INPUT_IDS, AbstractDL.ATTENTION_MASK,
        AbstractDL.TOKEN_TYPE_IDS), staged.keySet());
    assertAllClosed(staged);
  }

  @Test
  void testEveryTensorIsClosedWhenTheRunFails() {
    // A deliberately wrong input, so the failure is the model's and arrives every time: the graph
    // takes input_ids of rank 2 and this run hands it a rank 1 tensor, which ONNX Runtime rejects
    // inside the run rather than while the tensor is being created. Every tensor that reached the
    // failing run must still be released.
    final Map<String, OnnxTensor> staged = new HashMap<>();

    final OrtException e = assertThrows(OrtException.class, () ->
        inference.run(staged, new long[] {2}, new long[] {7, 3}, ones(2), new long[2], null));

    assertEquals(Set.of(AbstractDL.INPUT_IDS, AbstractDL.ATTENTION_MASK,
        AbstractDL.TOKEN_TYPE_IDS), staged.keySet(), e.getMessage());
    assertAllClosed(staged);
  }

  @Test
  void testEveryTensorIsClosedWhenStagingFailsPartWayThrough() {
    // A mask holding fewer elements than the shape describes fails while the inputs are still
    // being built, after the ids tensor was created. That one has to be released too.
    final Map<String, OnnxTensor> staged = new HashMap<>();

    assertThrows(OrtException.class, () ->
        inference.run(staged, new long[] {1, 2}, new long[] {7, 3}, ones(1), null, null));

    assertEquals(Set.of(AbstractDL.INPUT_IDS), staged.keySet());
    assertAllClosed(staged);
  }

  private static void assertAllClosed(final Map<String, OnnxTensor> staged) {
    assertTrue(staged.size() > 0, "nothing was staged, so the release is not being observed");
    for (final Map.Entry<String, OnnxTensor> input : staged.entrySet()) {
      assertTrue(input.getValue().isClosed(), input.getKey() + " was not closed");
    }
  }
}
