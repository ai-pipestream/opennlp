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

import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import opennlp.tools.commons.Internal;
import opennlp.tools.commons.ThreadSafe;

/**
 * The one ONNX Runtime interaction of this package: it stages the token ids, the attention mask and
 * the token type ids of a batch into tensors, runs the session on them, reads one output back, and
 * releases every native handle it created.
 *
 * <p>Every component here used to write that sequence out for itself, once per model type, which
 * left three copies of the same tensor plumbing to keep in step with each other. They now differ
 * only in what is genuinely theirs: which inputs their model declares, which output they read, and
 * what they do with the numbers that come back.</p>
 *
 * <h2>Shapes</h2>
 *
 * <p>There is one code path and it is the batch path. A run takes a {@code shape} of
 * <code>{rows, width}</code> and three flat arrays of {@code rows * width} elements each, laid out
 * row after row. A single input is the batch of one row that a shape of <code>{1, width}</code>
 * describes, not a separate case. Rows of differing token counts are the caller's problem: the
 * caller decides the width of the run and pads its short rows into the flat arrays before handing
 * them over, because what a padded position must hold is a property of the model and of the
 * vocabulary, and the choice of width is a policy this class has no business knowing. Nothing here
 * knows about padding strategies, pooling, scoring or spans.</p>
 *
 * <h2>Ownership</h2>
 *
 * <p>This class owns every native handle it creates and closes all of them before returning,
 * whether the run succeeded, failed in the middle of staging the inputs, failed inside
 * {@link OrtSession#run(Map)}, or failed while reading the output. The caller owns nothing
 * afterwards and has nothing to close.</p>
 *
 * <p>What the caller receives is the plain Java value of one output, which
 * {@link OnnxValue#getValue()} has already copied out of native memory, so it stays valid after the
 * result and its tensors are closed. The value is returned as {@code Object} and the caller
 * dispatches on its shape, because a rank 3 output arrives as {@code float[][][]}, a rank 2 output
 * as {@code float[][]}, and some models return {@code float[]}; how an unexpected shape is reported
 * is each caller's own contract and stays with the caller.</p>
 *
 * <p>The {@link OrtEnvironment} and the {@link OrtSession} are <b>not</b> owned here. They belong to
 * the {@link AbstractDL} that built them, which closes the session and deliberately leaves the
 * process-wide environment alone. An instance of this class holds no state of its own in this
 * release, so it needs no {@code close()}; the first pooled or pinned native state added to it
 * changes that, and the owner then has to close it from {@link AbstractDL#close()}.</p>
 *
 * <p>Neither the environment nor the session is checked for {@code null}, on purpose: the test seam
 * of {@link AbstractDL#AbstractDL(OrtEnvironment, OrtSession, Map, boolean)} constructs components
 * without a model precisely so that an inference failure can be exercised, and a run on such a
 * component has always failed with a {@link NullPointerException} that its caller wraps in its own
 * loud failure. Rejecting {@code null} here would turn that into a construction failure instead.</p>
 *
 * <h2>Where the buffer and output work goes</h2>
 *
 * <p>This class is deliberately still the naive implementation, so that the cost of what it does
 * today is measurable before any of it changes. {@link #tensor(long[], long[])} is the one place a
 * tensor is created: {@link LongBuffer#wrap(long[])} means ONNX Runtime allocates a direct buffer
 * and copies the row data into it on every call, so a pool of reused direct buffers, keyed by
 * element count and confined to the calling thread, replaces the body of that method and nothing
 * else. {@link #run(Map, long[], long[], long[], long[], String)} is the one place a session is
 * run and an output is read: pinned outputs go into the {@code run} call there, and a flat
 * buffer read of the output replaces the {@link OnnxValue#getValue()} in
 * {@link #value(OrtSession.Result, String)}, which today builds every nested array of a
 * <code>{rows, width, hidden}</code> output whether the caller reads it or not. Those three
 * methods are the whole surface that work touches.</p>
 *
 * <h2>Thread safety and language level</h2>
 *
 * <p>Instances are immutable and hold nothing per call, so one instance serves any number of
 * threads, which is what {@link ThreadSafe} on the components using it promises. ONNX Runtime
 * allows concurrent {@link OrtSession#run(Map)} calls on one session. Note that a pool of reused
 * buffers is not thread safe and that adding one means confining it per thread rather than sharing
 * it.</p>
 *
 * <p>This class stays on Java 11 era APIs and uses no records, so that it remains reachable from an
 * Android runtime once the module baseline allows one. {@code HashMap.newHashMap(int)}, which the
 * code moved here used, is Java 19, so a plain sized {@link HashMap} stands in its place.</p>
 */
@Internal
@ThreadSafe
public final class OnnxInference {

  /**
   * Enough capacity for the three inputs a BERT-style encoder declares without a resize. Written as
   * a sized {@link HashMap} rather than {@code HashMap.newHashMap(3)} for the language level reason
   * given in the class documentation.
   */
  private static final int INPUT_CAPACITY = 4;

  private final OrtEnvironment env;

  private final OrtSession session;

  /**
   * Binds the environment and session every run of this instance uses.
   *
   * @param env The ONNX Runtime environment tensors are created in. Not owned, not closed here,
   *     and not checked for {@code null} for the reason given in the class documentation.
   * @param session The session to run. Not owned and not closed here.
   */
  public OnnxInference(final OrtEnvironment env, final OrtSession session) {
    this.env = env;
    this.session = session;
  }

  /**
   * Runs one batch and reads the first output the model declares, which is how a model with a
   * single output is read.
   *
   * @param shape The tensor shape <code>{rows, width}</code> of the run. Must not be {@code null}.
   * @param ids The token ids of every row, row after row, {@code rows * width} of them.
   * @param mask The attention mask in the same layout, or {@code null} for a model that does not
   *     declare {@code attention_mask}.
   * @param types The token type ids in the same layout, or {@code null} for a model that does not
   *     declare {@code token_type_ids}.
   * @return The value of the first output, already copied out of native memory.
   *
   * @throws OrtException Thrown if a tensor cannot be created or the session cannot be run.
   */
  public Object run(final long[] shape, final long[] ids, final long[] mask, final long[] types)
      throws OrtException {
    return run(new HashMap<>(INPUT_CAPACITY), shape, ids, mask, types, null);
  }

  /**
   * Runs one batch and reads the named output.
   *
   * @param shape The tensor shape <code>{rows, width}</code> of the run. Must not be {@code null}.
   * @param ids The token ids of every row, row after row, {@code rows * width} of them.
   * @param mask The attention mask in the same layout, or {@code null} for a model that does not
   *     declare {@code attention_mask}.
   * @param types The token type ids in the same layout, or {@code null} for a model that does not
   *     declare {@code token_type_ids}.
   * @param outputName The name of the output to read. Must not be {@code null}.
   * @return The value of that output, already copied out of native memory.
   *
   * @throws OrtException Thrown if a tensor cannot be created, if the session cannot be run, or if
   *     the model declares no output of that name.
   */
  public Object run(final long[] shape, final long[] ids, final long[] mask, final long[] types,
      final String outputName) throws OrtException {
    return run(new HashMap<>(INPUT_CAPACITY), shape, ids, mask, types, outputName);
  }

  /**
   * The implementation of both public runs, and the seam a test reaches through to observe the
   * tensors a run created.
   *
   * <p>The tensors are put into the map the caller passes and every one of them is closed before
   * this method returns, on every path out of it including an {@link Error}. A test that wants to
   * assert that release passes a map of its own, keeps the reference, and reads
   * {@link OnnxValue#isClosed()} off the tensors afterwards; production callers pass a fresh map
   * and never see it again.</p>
   *
   * @param inputs An empty, mutable map to stage the tensors into. Must not be {@code null}.
   * @param shape The tensor shape <code>{rows, width}</code> of the run.
   * @param ids The token ids of every row, row after row.
   * @param mask The attention mask, or {@code null} to leave the input out.
   * @param types The token type ids, or {@code null} to leave the input out.
   * @param outputName The name of the output to read, or {@code null} for the first output.
   * @return The value of the selected output.
   *
   * @throws OrtException Thrown if a tensor cannot be created, if the session cannot be run, or if
   *     the model declares no output named {@code outputName}.
   */
  Object run(final Map<String, OnnxTensor> inputs, final long[] shape, final long[] ids,
      final long[] mask, final long[] types, final String outputName) throws OrtException {
    try {
      inputs.put(AbstractDL.INPUT_IDS, tensor(ids, shape));
      if (mask != null) {
        inputs.put(AbstractDL.ATTENTION_MASK, tensor(mask, shape));
      }
      if (types != null) {
        inputs.put(AbstractDL.TOKEN_TYPE_IDS, tensor(types, shape));
      }
      try (OrtSession.Result result = session.run(inputs)) {
        return value(result, outputName);
      }
    } finally {
      // Closes what was staged whatever went wrong, including a failure part way through staging,
      // where the map holds the tensors created before the one that failed.
      for (final OnnxTensor input : inputs.values()) {
        input.close();
      }
    }
  }

  /**
   * Creates one input tensor. The single place this package turns a row array into native memory,
   * and the place a pool of reused direct buffers belongs.
   *
   * @param data The values, row after row, {@code shape[0] * shape[1]} of them.
   * @param shape The tensor shape.
   * @return A new tensor the caller closes.
   *
   * @throws OrtException Thrown if ONNX Runtime cannot create the tensor, which is what happens
   *     when {@code data} does not hold as many elements as {@code shape} describes.
   */
  private OnnxTensor tensor(final long[] data, final long[] shape) throws OrtException {
    return OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape);
  }

  /**
   * Reads one output of a result out of native memory. The single place an output is extracted, and
   * the place a flat buffer read belongs.
   *
   * @param result The result of a run, still open.
   * @param outputName The name of the output to read, or {@code null} for the first output.
   * @return The value of that output, copied into Java arrays and therefore valid after
   *     {@code result} is closed.
   *
   * @throws OrtException Thrown if the model declares no output named {@code outputName}, or if the
   *     value cannot be read.
   */
  private Object value(final OrtSession.Result result, final String outputName)
      throws OrtException {
    final OnnxValue output = outputName == null ? result.get(0)
        : result.get(outputName).orElseThrow(() -> new OrtException(
            "The model returned no output named " + outputName));
    return output.getValue();
  }
}
