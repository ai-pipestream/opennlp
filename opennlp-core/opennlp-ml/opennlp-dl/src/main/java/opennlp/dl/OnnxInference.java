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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

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
 * <h2>How an output is read</h2>
 *
 * <p>There are two read paths and the caller picks between them by whether it can name the shape of
 * the output before the run.</p>
 *
 * <p><b>A pinned output</b>, which {@link #run(long[], long[], long[], long[], String, long[],
 * OutputReader)} takes when its {@code outputShape} is not {@code null}. This class allocates the
 * output tensor itself over a <b>direct</b> {@link FloatBuffer} and passes it as a pinned output to
 * {@link OrtSession#run(Map, Set, Map)}, so ONNX Runtime writes the results straight into memory
 * Java can already see. Nothing is copied on the way out: the {@link OutputReader} reads the same
 * buffer the kernels wrote. Pinned outputs are explicitly not owned by the
 * {@link OrtSession.Result} and are not closed with it, which is what lets one of them be reused
 * across runs.</p>
 *
 * <p><b>A flat buffer read</b> otherwise, which every other overload takes: the run goes through
 * {@link OrtSession#run(Map)} and the output is read with {@link OnnxTensor#getFloatBuffer()}, one
 * flat copy of the tensor. That is the fallback for an output whose shape is not derivable before
 * the run, and for a model whose output is not a float tensor at all.</p>
 *
 * <p>Neither path builds the nested arrays that {@link OnnxValue#getValue()} builds, which for a
 * <code>{rows, width, hidden}</code> output means {@code rows * width} arrays of
 * {@code float[hidden]} plus {@code rows} arrays of {@code float[width][]}, whether the caller reads
 * them or not. At <code>{128, 56, 384}</code> that is 7168 array objects and roughly 11 MB per call,
 * and on a batch padded to its longest member 54 percent of those rows are never read, because a
 * pooled sentence vector reads only each row's own unpadded prefix. ONNX Runtime's own
 * {@link OnnxTensor} documentation says to use the buffer extractors rather than
 * {@code getValue()} above rank 2. The {@link OutputReader} is handed the flat values and the
 * output shape and indexes what it needs.</p>
 *
 * <p>The {@code Object} returning overloads remain for a caller that wants the plain Java value and
 * dispatches on its rank, which is what a name finder and a document categorizer do: their output
 * is one row wide, they read all of it, and the count of labels or of categories their model
 * declares is something they validate rather than assume, so there is nothing for them to pin. Those
 * overloads now shape their nested arrays out of the same flat read; {@code getValue()} is reached
 * only for an output that is not a float tensor, where a flat float read has no meaning.</p>
 *
 * <h2>How the inputs are staged</h2>
 *
 * <p>ONNX Runtime branches on whether the buffer behind an input is direct. A direct buffer is
 * wrapped where it lies and the tensor keeps a reference to it, so nothing is copied. A heap buffer
 * takes the other branch, which is one {@code ByteBuffer.allocateDirect} plus a full element copy
 * per tensor per run. {@link LongBuffer#wrap(long[])} produces a heap buffer, so staging the three
 * inputs of a BERT-style encoder used to cost three direct allocations and three full copies per
 * inference, and a call that fragments into 41 inferences paid that 41 times over.</p>
 *
 * <p>So the buffers the inputs are staged into belong to this class and are direct. Each thread that
 * runs keeps one grow-only direct arena holding three equal slots, one per input, and the tensors of
 * a run are created over exact slices of those slots. ONNX Runtime then reads the very memory this
 * class wrote and allocates nothing of its own for an input.</p>
 *
 * <p>There are two ways to fill them, and a caller picks by whether it already holds its rows as
 * flat arrays:</p>
 *
 * <ul>
 *   <li>The {@code long[]} overloads copy each array into its slot. That is what a name finder and a
 *   document categorizer want: their rows are the arrays a {@link Tokens} already holds, so there is
 *   nothing to build and the copy is from an array that exists either way.</li>
 *   <li>An {@link InputStager} writes the three slots directly, which is what a caller composing a
 *   batch out of several rows wants: the flat {@code rows * width} arrays it would otherwise build
 *   to hand over, one per input per batch, are not built at all.</li>
 * </ul>
 *
 * <h2>Stale input data, and why a partial write cannot happen quietly</h2>
 *
 * <p>A reused buffer that is not fully overwritten makes an inference read the previous run's tokens
 * and return a plausible wrong answer, with nothing failing. That is a worse defect than the
 * allocation this reuse removes, so neither filling path can leave a position unwritten:</p>
 *
 * <ul>
 *   <li>On the {@code long[]} path the array has to hold exactly as many values as the shape
 *   describes, checked here rather than left to ONNX Runtime, and every one of them is written. An
 *   array of the wrong length fails the run with an {@link OrtException} and stages no tensor for
 *   that input.</li>
 *   <li>On the {@link InputStager} path the stager is handed each slot positioned at {@code 0} and
 *   limited to the element count, and each slot's position is checked against that count once the
 *   stager returns. A stager that writes a row short, or that skips an input because it believes it
 *   already holds the right values, fails the run with an {@link IllegalStateException} naming the
 *   input and the shortfall. Believing an input is already correct is exactly how stale data gets
 *   read, so it is not a belief this class accepts.</li>
 * </ul>
 *
 * <p>A shrinking extent leaves the previous run's values in the arena past the new slice, and they
 * are unreachable: the slice a tensor is created over is limited to the element count of its run, and
 * ONNX Runtime is handed that count as the byte size of the buffer, so it reads no position the
 * current run did not write.</p>
 *
 * <h2>Pinned output lifecycle, and the direct memory bound</h2>
 *
 * <p>Allocating a pinned output per run would trade one copy for one {@code allocateDirect}, which
 * is a malloc plus a zeroing, so pinned outputs are reused. Each thread that runs a pinned output
 * keeps one grow-only direct arena and one tensor over an exact slice of it:</p>
 *
 * <ul>
 *   <li>The arena only ever grows. A run needing more floats than the arena holds replaces it; a
 *   run needing fewer reuses it.</li>
 *   <li>The tensor is rebuilt when the shape changes, because ONNX Runtime requires the buffer
 *   behind a tensor to hold exactly as many elements as the shape describes. A run repeating the
 *   previous shape, which is what padding a batch to a common width produces, reuses the tensor as
 *   it stands and allocates nothing at all.</li>
 *   <li>{@link #close()} closes every tensor of every thread and releases the accounting. The
 *   arenas themselves are ordinary direct buffers and are reclaimed when the {@code OnnxInference}
 *   becomes unreachable.</li>
 * </ul>
 *
 * <p>Direct memory is bounded separately from the heap and is not reclaimed by an ordinary
 * collection promptly, so the total is capped. {@link #DEFAULT_MAX_PINNED_OUTPUT_BYTES} is
 * {@value #DEFAULT_MAX_PINNED_OUTPUT_BYTES} bytes across every thread of one instance, and a run
 * that would take the total past it fails with an {@link IllegalStateException} naming the bound
 * rather than growing quietly. The number follows from the caller-side bound that already exists:
 * {@code SentenceVectorsDL.MAX_BATCH_TOKEN_POSITIONS} caps one inference at 16384 token positions,
 * so the widest pinned output a batch can ask for is 16384 positions times a hidden size, and at a
 * hidden size of 1024, the largest in common use, that is 64 MiB. The default therefore admits four
 * threads each holding the largest arena that bound permits. The two limits do not contradict:
 * the token position bound decides how large one arena can become, and this one decides how many
 * of them may exist at once.</p>
 *
 * <p>The input arenas hold direct memory out of the same finite native budget, so their total is
 * bounded too, by {@link #DEFAULT_MAX_INPUT_STAGING_BYTES}. They are counted and bounded
 * <b>separately</b> rather than out of one shared number, for two reasons. The bound on output
 * memory means what it says: a run whose pinned output is exactly the bound is served, which it could
 * not be if the inputs of that same run had to come out of the same allowance and were charged first.
 * And the two footprints differ by orders of magnitude for the same batch: an input arena is
 * {@code 3 * 8} bytes per token position, where an output arena is {@code 4 * hidden} bytes, so one
 * number governing both would be the output number with a rounding error added to it.</p>
 *
 * <p>They are nonetheless reconciled rather than chosen independently, and the reconciliation is
 * asserted in a test rather than left in this prose. The input bound is the output bound divided by
 * {@code 64}, and {@code 64} is the ratio of the two per-position costs at a hidden size of
 * {@code 384}, which the sentence-transformers MiniLM family has: {@code 4 * 384} output bytes per
 * position against {@code 3 * 8} input bytes per position. So at that hidden size the input bound
 * admits exactly as many threads at the widest batch as the output bound admits, and at any larger
 * hidden size it admits more, which is the safe direction. Changing either number means revisiting
 * the other.</p>
 *
 * <p>A thread that runs and then dies keeps its arenas accounted until {@link #close()}, because the
 * tensor over the output arena is held for release and the input arena is held for the same reason.
 * Pools of long lived threads, which is what a server does, are unaffected; a caller that creates and
 * discards threads per request will see the budget fill.</p>
 *
 * <h2>Ownership</h2>
 *
 * <p>This class owns every native handle it creates. Input tensors are closed before the run
 * returns, whether the run succeeded, failed in the middle of staging the inputs, failed inside
 * {@link OrtSession#run(Map)}, or failed while reading the output. A pinned output tensor outlives
 * the run on purpose and is closed by {@link #close()}. The caller owns nothing and has nothing to
 * close.</p>
 *
 * <p>An input tensor being closed per run is deliberate and is what separates the two halves of the
 * reuse here: the direct <b>memory</b> an input is staged into is reused across runs and released by
 * {@link #close()}, while the {@link OnnxTensor} over it is a handle on that memory for the length of
 * one run and is released with the run. ONNX Runtime does document reusing a buffer backed input
 * tensor across runs as well, through {@link OnnxTensor#getBufferRef()}, which would save the handle
 * pair as well as the memory; that is not done here, because an input tensor outliving its run is a
 * contract this package tests for in five places and the saving is a pair of handle operations
 * against the allocation and the copy that reusing the memory already removes.</p>
 *
 * <p>What an {@link OutputReader} receives is valid only for the duration of the
 * {@link OutputReader#read(FloatBuffer, long[])} call. On the pinned path the next run of the same
 * thread overwrites it, and on the unpinned path it is a copy this class made for that one call and
 * has no reason to outlive it. A reader therefore takes what it needs out of the buffer rather than
 * keeping the buffer. The {@code Object} returning overloads hand back arrays that are already
 * copies and stay valid afterwards.</p>
 *
 * <p>The {@link OrtEnvironment} and the {@link OrtSession} are <b>not</b> owned here. They belong to
 * the {@link AbstractDL} that built them, which closes the session and deliberately leaves the
 * process-wide environment alone.</p>
 *
 * <p>Neither the environment nor the session is checked for {@code null}, on purpose: the test seam
 * of {@link AbstractDL#AbstractDL(OrtEnvironment, OrtSession, Map, boolean)} constructs components
 * without a model precisely so that an inference failure can be exercised, and a run on such a
 * component has always failed with a {@link NullPointerException} that its caller wraps in its own
 * loud failure. Rejecting {@code null} here would turn that into a construction failure instead.</p>
 *
 * <h2>Thread safety and language level</h2>
 *
 * <p>One instance serves any number of threads, which is what {@link ThreadSafe} on the components
 * using it promises, and ONNX Runtime allows concurrent {@link OrtSession#run(Map)} calls on one
 * session. The reusable input arenas and the reusable pinned outputs are the only mutable state here
 * and each is confined to the thread that created it, so no two threads ever touch one arena: a
 * thread stages its own inputs into its own direct memory and reads its own output out of its own. In
 * particular two threads running the same shape at the same time cannot see each other's tokens, which
 * a shared pool keyed by shape would have allowed. What crosses threads is the registry of arenas,
 * which exists so {@link #close()} can release them and is a concurrent set, and the byte totals,
 * which are {@link AtomicLong}s. {@link #close()} must not race a run, which is the contract the
 * components using this class already document for their own {@code close()}.</p>
 *
 * <p>This class stays on Java 11 era APIs and uses no records, so that it remains reachable from an
 * Android runtime once the module baseline allows one. {@code HashMap.newHashMap(int)}, which the
 * code moved here used, is Java 19, so a plain sized {@link HashMap} stands in its place, and
 * {@code ByteBuffer.slice(int, int)} is Java 13, so the older {@code duplicate} then
 * {@code position} then {@code limit} then {@code slice} sequence stands in for it.</p>
 *
 * <p>That is a property a comment cannot keep, so it is checked: {@code AndroidReachabilityTest}
 * compiles this source with {@code --release 11} and fails the build if an edit reaches for anything
 * newer. The README of this module says what the check covers, what it does not, and what an actual
 * Android build would need beyond it.</p>
 */
@Internal
@ThreadSafe
public final class OnnxInference implements AutoCloseable {

  /**
   * Reads one output of a run out of a flat buffer of its values.
   *
   * <p>Not a {@code java.util.function} type because it has to be allowed to throw
   * {@link OrtException}, and not a record or a sealed hierarchy for the language level reason given
   * in the class documentation.</p>
   *
   * @param <T> What the reader makes of the output.
   */
  @Internal
  public interface OutputReader<T> {

    /**
     * Reads the output.
     *
     * @param values The values of the output, row major, {@code values.remaining()} of them
     *     starting at index {@code 0}. Valid only for the duration of this call, so a reader takes
     *     what it needs rather than keeping the buffer. Read it with the absolute
     *     {@link FloatBuffer#get(int)}, since the position of the buffer is not this reader's to
     *     move.
     * @param shape The shape of the output, outermost dimension first.
     * @return Whatever the caller wanted made of the output.
     *
     * @throws OrtException Thrown if the output does not have the shape the reader requires.
     */
    T read(FloatBuffer values, long[] shape) throws OrtException;
  }

  /**
   * Writes the three inputs of one run straight into the direct memory they are staged in, so that a
   * caller composing a batch out of several rows never builds the flat {@code rows * width} array
   * that handing the inputs over as {@code long[]} would need.
   *
   * <p>Not a {@code java.util.function} type because it takes three arguments and returns nothing,
   * and not a record or a sealed hierarchy for the language level reason given in the class
   * documentation.</p>
   */
  @Internal
  public interface InputStager {

    /**
     * Writes every position of all three inputs.
     *
     * <p>Each buffer arrives positioned at {@code 0} and limited to the element count of the run's
     * shape, and must be filled with the <b>relative</b> {@link LongBuffer#put(long)} and
     * {@link LongBuffer#put(long[], int, int)} so that its position records what was written. The
     * buffers are direct memory this class reuses across runs and they hold the previous run's
     * values, so a position this stager does not write is a position an inference reads stale data
     * out of. That is checked rather than trusted: a stager that leaves any buffer short of its
     * limit fails the run.</p>
     *
     * @param ids The token ids of every row, row after row, the run's element count of them.
     * @param mask The attention mask in the same layout.
     * @param types The token type ids in the same layout.
     */
    void stage(LongBuffer ids, LongBuffer mask, LongBuffer types);
  }

  /**
   * Enough capacity for the three inputs a BERT-style encoder declares without a resize. Written as
   * a sized {@link HashMap} rather than {@code HashMap.newHashMap(3)} for the language level reason
   * given in the class documentation.
   */
  private static final int INPUT_CAPACITY = 4;

  private static final int BYTES_PER_FLOAT = 4;

  private static final int BYTES_PER_LONG = 8;

  /** The inputs a BERT-style encoder declares, which is how many slots an input arena holds. */
  private static final int INPUT_SLOTS = 3;

  private static final int IDS_SLOT = 0;

  private static final int MASK_SLOT = 1;

  private static final int TYPES_SLOT = 2;

  /**
   * The total direct memory one instance holds in reusable pinned output arenas, 256 MiB, which is
   * four arenas at the largest size the caller-side token position bound permits. The reasoning is
   * in the class documentation, under the pinned output lifecycle.
   */
  public static final long DEFAULT_MAX_PINNED_OUTPUT_BYTES = 256L * 1024 * 1024;

  /**
   * The total direct memory one instance holds in reusable input staging arenas, 4 MiB, which is
   * {@link #DEFAULT_MAX_PINNED_OUTPUT_BYTES} divided by the ratio of the two per-token-position
   * costs at a hidden size of 384. The reasoning, and why the two bounds are counted separately
   * rather than out of one number, is in the class documentation under the direct memory bound.
   */
  public static final long DEFAULT_MAX_INPUT_STAGING_BYTES = DEFAULT_MAX_PINNED_OUTPUT_BYTES / 64;

  /** What a run on a closed instance, or on a released arena of one, reports. */
  private static final String CLOSED =
      "This inference is closed; its native input and output buffers have been released.";

  private final OrtEnvironment env;

  private final OrtSession session;

  private final long maxPinnedOutputBytes;

  private final long maxInputStagingBytes;

  /** The direct bytes currently held in input staging arenas, across every thread. */
  private final AtomicLong inputStagingBytes = new AtomicLong();

  /**
   * Every input arena any thread created, so {@link #close()} can release them. Written only by the
   * owning thread when it creates its arena, and read only by {@link #close()}.
   */
  private final Set<InputStaging> stagingAreas = ConcurrentHashMap.newKeySet();

  /** The input arena of the calling thread, which no other thread touches. */
  private final ThreadLocal<InputStaging> stagingArea = new ThreadLocal<>();

  /** The direct bytes currently held in pinned output arenas, across every thread. */
  private final AtomicLong pinnedOutputBytes = new AtomicLong();

  /**
   * Every arena any thread created, so {@link #close()} can release the tensors over them. Written
   * only by the owning thread when it creates its arena, and read only by {@link #close()}.
   */
  private final Set<PinnedOutput> arenas = ConcurrentHashMap.newKeySet();

  /** The arena of the calling thread, which no other thread touches. */
  private final ThreadLocal<PinnedOutput> arena = new ThreadLocal<>();

  private final AtomicBoolean closed = new AtomicBoolean();

  /**
   * Binds the environment and session every run of this instance uses, with the default bound on
   * reusable direct output memory.
   *
   * @param env The ONNX Runtime environment tensors are created in. Not owned, not closed here,
   *     and not checked for {@code null} for the reason given in the class documentation.
   * @param session The session to run. Not owned and not closed here.
   */
  public OnnxInference(final OrtEnvironment env, final OrtSession session) {
    this(env, session, DEFAULT_MAX_PINNED_OUTPUT_BYTES, DEFAULT_MAX_INPUT_STAGING_BYTES);
  }

  /**
   * Binds the environment and session with an explicit bound on reusable direct output memory. The
   * seam a test reaches through to drive that bound without allocating 256 MiB to do it.
   *
   * @param env The ONNX Runtime environment tensors are created in.
   * @param session The session to run.
   * @param maxPinnedOutputBytes The total direct bytes this instance may hold in reusable pinned
   *     output arenas, across every thread. Must be positive.
   */
  OnnxInference(final OrtEnvironment env, final OrtSession session,
      final long maxPinnedOutputBytes) {
    this(env, session, maxPinnedOutputBytes, DEFAULT_MAX_INPUT_STAGING_BYTES);
  }

  /**
   * Binds the environment and session with an explicit bound on each of the two kinds of reusable
   * direct memory. The seam a test reaches through to drive either bound without allocating its
   * default to do it.
   *
   * @param env The ONNX Runtime environment tensors are created in.
   * @param session The session to run.
   * @param maxPinnedOutputBytes The total direct bytes this instance may hold in reusable pinned
   *     output arenas, across every thread. Must be positive.
   * @param maxInputStagingBytes The total direct bytes this instance may hold in reusable input
   *     staging arenas, across every thread. Must be positive.
   */
  OnnxInference(final OrtEnvironment env, final OrtSession session,
      final long maxPinnedOutputBytes, final long maxInputStagingBytes) {
    if (maxPinnedOutputBytes <= 0) {
      throw new IllegalArgumentException(
          "maxPinnedOutputBytes must be positive but was " + maxPinnedOutputBytes);
    }
    if (maxInputStagingBytes <= 0) {
      throw new IllegalArgumentException(
          "maxInputStagingBytes must be positive but was " + maxInputStagingBytes);
    }
    this.env = env;
    this.session = session;
    this.maxPinnedOutputBytes = maxPinnedOutputBytes;
    this.maxInputStagingBytes = maxInputStagingBytes;
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
   * Runs one batch and lets {@code reader} read the named output out of a flat buffer of its values.
   *
   * <p>Equivalent to {@link #run(long[], long[], long[], long[], String, long[], OutputReader)} with
   * no output shape, so the output is read through one flat copy rather than pinned.</p>
   *
   * @param <T> What the reader makes of the output.
   * @param shape The tensor shape <code>{rows, width}</code> of the run. Must not be {@code null}.
   * @param ids The token ids of every row, row after row.
   * @param mask The attention mask in the same layout, or {@code null} to leave the input out.
   * @param types The token type ids in the same layout, or {@code null} to leave the input out.
   * @param outputName The name of the output to read, or {@code null} for the first output.
   * @param reader The reader of the output. Must not be {@code null}.
   * @return What the reader returned.
   *
   * @throws OrtException Thrown if a tensor cannot be created, if the session cannot be run, if the
   *     model declares no output of that name, if the output is not a float tensor, or if the
   *     reader rejects the shape.
   */
  public <T> T run(final long[] shape, final long[] ids, final long[] mask, final long[] types,
      final String outputName, final OutputReader<T> reader) throws OrtException {
    return run(shape, ids, mask, types, outputName, null, reader);
  }

  /**
   * Runs one batch and lets {@code reader} read the named output, pinning the output tensor when the
   * caller can name its shape.
   *
   * <p>With an {@code outputShape}, this instance allocates the output tensor over a reused direct
   * buffer and ONNX Runtime writes the results into it, so the reader reads the very memory the
   * kernels wrote and nothing is copied. The shape has to be exactly right: ONNX Runtime compares it
   * against the shape the graph computes for the run and fails the run with an
   * {@link OrtException} on any difference, in the rank or in any dimension, rather than truncating
   * or writing past the buffer. Without an {@code outputShape}, the output is read through one flat
   * copy instead.</p>
   *
   * <p>On a GPU execution provider the values of an output depend on the width of the tensor it was
   * produced in, by around {@code 1.2e-4} absolute in a measured comparison of a padded batch
   * against the same rows run at their own lengths, because cuBLAS picks its blocking from the
   * shape and floating point addition does not associate. The worst measured deviation was
   * {@code 5.15e-7} in one minus cosine similarity. On the CPU provider the padded and unpadded
   * paths agree exactly. Anyone caching or persisting embeddings has to key the cache by the batch
   * shape as well, or accept that a vector is only reproducible to that tolerance. Pinning does not
   * introduce this and does not change it: the effect is the tensor width, not the read path.</p>
   *
   * @param <T> What the reader makes of the output.
   * @param shape The tensor shape <code>{rows, width}</code> of the run. Must not be {@code null}.
   * @param ids The token ids of every row, row after row.
   * @param mask The attention mask in the same layout, or {@code null} to leave the input out.
   * @param types The token type ids in the same layout, or {@code null} to leave the input out.
   * @param outputName The name of the output to read. Must not be {@code null} when
   *     {@code outputShape} is given, because a pinned output is pinned by name.
   * @param outputShape The shape of the output, to pin it at, or {@code null} to read it through one
   *     flat copy instead. Every dimension must be positive and the element count must fit an
   *     {@code int}.
   * @param reader The reader of the output. Must not be {@code null}.
   * @return What the reader returned.
   *
   * @throws OrtException Thrown if a tensor cannot be created, if the session cannot be run, if the
   *     model declares no output of that name, if {@code outputShape} is not the shape the graph
   *     computes, if the output is not a float tensor, or if the reader rejects the shape.
   * @throws IllegalStateException Thrown if this instance is closed, or if the run would take the
   *     reusable direct output memory of this instance past its bound.
   * @throws IllegalArgumentException Thrown if {@code reader} is {@code null}, if
   *     {@code outputShape} is given without an {@code outputName}, or if {@code outputShape} is
   *     empty, holds a dimension that is not positive, or describes more elements than an
   *     {@code int} can count.
   */
  public <T> T run(final long[] shape, final long[] ids, final long[] mask, final long[] types,
      final String outputName, final long[] outputShape, final OutputReader<T> reader)
      throws OrtException {
    if (reader == null) {
      throw new IllegalArgumentException("The reader must not be null.");
    }
    if (outputShape != null && outputName == null) {
      throw new IllegalArgumentException("A pinned output needs the name of the output to pin.");
    }
    return run(new HashMap<>(INPUT_CAPACITY), shape, ids, mask, types, outputName, outputShape,
        reader);
  }

  /**
   * Runs one batch whose three inputs {@code stager} writes into this instance's own direct memory,
   * and lets {@code reader} read the named output, pinning the output tensor when the caller can name
   * its shape.
   *
   * <p>This is the overload for a caller that composes a batch out of several rows. Handing the
   * inputs over as {@code long[]} would mean building three flat arrays of {@code rows * width}
   * elements first, which at a batch of 128 rows padded to 56 positions is around 170 KB of them per
   * inference, and then copying all three into the buffers the tensors are made over. A stager writes
   * those buffers itself, so neither the arrays nor the copy out of them exists. Everything else,
   * including the pinned output, works exactly as
   * {@link #run(long[], long[], long[], long[], String, long[], OutputReader)} describes.</p>
   *
   * <p>All three inputs are staged, because a caller that shapes its own batch is a caller whose
   * model declares all three. A model declaring fewer takes the {@code long[]} overloads, which leave
   * an input out when its array is {@code null}.</p>
   *
   * @param <T> What the reader makes of the output.
   * @param shape The tensor shape <code>{rows, width}</code> of the run. Must not be {@code null}.
   * @param stager The writer of the three inputs. Must not be {@code null}, and must write every
   *     position of every buffer it is handed.
   * @param outputName The name of the output to read. Must not be {@code null} when
   *     {@code outputShape} is given, because a pinned output is pinned by name.
   * @param outputShape The shape of the output, to pin it at, or {@code null} to read it through one
   *     flat copy instead.
   * @param reader The reader of the output. Must not be {@code null}.
   * @return What the reader returned.
   *
   * @throws OrtException Thrown if a tensor cannot be created, if the session cannot be run, if the
   *     model declares no output of that name, if {@code outputShape} is not the shape the graph
   *     computes, if the output is not a float tensor, or if the reader rejects the shape.
   * @throws IllegalStateException Thrown if this instance is closed, if the run would take the
   *     reusable direct memory of this instance past one of its bounds, or if {@code stager} left an
   *     input short of the element count of {@code shape}, which would have let the inference read
   *     the previous run's values.
   * @throws IllegalArgumentException Thrown if {@code stager} or {@code reader} is {@code null}, or
   *     if {@code outputShape} is given without an {@code outputName}.
   */
  public <T> T run(final long[] shape, final InputStager stager, final String outputName,
      final long[] outputShape, final OutputReader<T> reader) throws OrtException {
    if (stager == null) {
      throw new IllegalArgumentException("The stager must not be null.");
    }
    if (reader == null) {
      throw new IllegalArgumentException("The reader must not be null.");
    }
    if (outputShape != null && outputName == null) {
      throw new IllegalArgumentException("A pinned output needs the name of the output to pin.");
    }
    return run(new HashMap<>(INPUT_CAPACITY), shape, stager, outputName, outputShape, reader);
  }

  /**
   * The implementation of the {@link InputStager} run, and the seam a test reaches through to observe
   * the tensors it created.
   *
   * @param <T> What the reader makes of the output.
   * @param inputs An empty, mutable map to stage the tensors into. Must not be {@code null}.
   * @param shape The tensor shape <code>{rows, width}</code> of the run.
   * @param stager The writer of the three inputs.
   * @param outputName The name of the output to read, or {@code null} for the first output.
   * @param outputShape The shape to pin the output at, or {@code null} to read it flat.
   * @param reader The reader of the output.
   * @return What the reader returned.
   *
   * @throws OrtException Thrown if a tensor cannot be created, if the session cannot be run, if the
   *     model declares no output named {@code outputName}, or if the reader rejects the output.
   */
  <T> T run(final Map<String, OnnxTensor> inputs, final long[] shape, final InputStager stager,
      final String outputName, final long[] outputShape, final OutputReader<T> reader)
      throws OrtException {
    return doRun(inputs, shape, null, null, null, stager, outputName, outputShape, reader, false);
  }

  /**
   * The implementation of both {@code Object} returning runs, and the seam a test reaches through to
   * observe the tensors a run created.
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
    return doRun(inputs, shape, ids, mask, types, null, outputName, null, OnnxInference::nested,
        true);
  }

  /**
   * The one place a session is run, and the seam a test reaches through to observe the tensors a run
   * created.
   *
   * <p>The input tensors are put into the map the caller passes and every one of them is closed
   * before this method returns, on every path out of it including an {@link Error}. A test that
   * wants to assert that release passes a map of its own, keeps the reference, and reads
   * {@link OnnxValue#isClosed()} off the tensors afterwards; production callers pass a fresh map and
   * never see it again. A pinned output tensor is not in that map and deliberately outlives the
   * run.</p>
   *
   * @param <T> What the reader makes of the output.
   * @param inputs An empty, mutable map to stage the tensors into. Must not be {@code null}.
   * @param shape The tensor shape <code>{rows, width}</code> of the run.
   * @param ids The token ids of every row, row after row.
   * @param mask The attention mask, or {@code null} to leave the input out.
   * @param types The token type ids, or {@code null} to leave the input out.
   * @param outputName The name of the output to read, or {@code null} for the first output.
   * @param outputShape The shape to pin the output at, or {@code null} to read it flat.
   * @param reader The reader of the output.
   * @return What the reader returned.
   *
   * @throws OrtException Thrown if a tensor cannot be created, if the session cannot be run, if the
   *     model declares no output named {@code outputName}, or if the reader rejects the output.
   */
  <T> T run(final Map<String, OnnxTensor> inputs, final long[] shape, final long[] ids,
      final long[] mask, final long[] types, final String outputName, final long[] outputShape,
      final OutputReader<T> reader) throws OrtException {
    return doRun(inputs, shape, ids, mask, types, null, outputName, outputShape, reader, false);
  }

  /**
   * Runs and reads, with the one difference between the reader paths and the {@code Object} paths:
   * whether an output that is not a float tensor falls back to {@link OnnxValue#getValue()} or is
   * reported as unreadable.
   *
   * @param <T> What the reader makes of the output.
   * @param inputs An empty, mutable map to stage the tensors into.
   * @param shape The tensor shape <code>{rows, width}</code> of the run.
   * @param ids The token ids of every row, row after row, or {@code null} when {@code stager} stages.
   * @param mask The attention mask, or {@code null} to leave the input out.
   * @param types The token type ids, or {@code null} to leave the input out.
   * @param stager The writer of all three inputs, or {@code null} to stage from the arrays.
   * @param outputName The name of the output to read, or {@code null} for the first output.
   * @param outputShape The shape to pin the output at, or {@code null} to read it flat.
   * @param reader The reader of the output.
   * @param rawWhenNotFloat Whether an output that is not a float tensor is handed back as the plain
   *     Java value of {@link OnnxValue#getValue()} rather than reported as unreadable.
   * @return What the reader returned.
   *
   * @throws OrtException Thrown if a tensor cannot be created, if the session cannot be run, if the
   *     model declares no output named {@code outputName}, or if the reader rejects the output.
   */
  private <T> T doRun(final Map<String, OnnxTensor> inputs, final long[] shape, final long[] ids,
      final long[] mask, final long[] types, final InputStager stager, final String outputName,
      final long[] outputShape, final OutputReader<T> reader, final boolean rawWhenNotFloat)
      throws OrtException {
    if (closed.get()) {
      throw new IllegalStateException(CLOSED);
    }
    // Reserved before any input tensor is staged, so a run that cannot have its output memory fails
    // without having allocated anything at all.
    final PinnedOutput pinned = outputShape == null ? null : pin(outputShape);
    try {
      stage(inputs, shape, ids, mask, types, stager);
      if (pinned == null) {
        try (OrtSession.Result result = session.run(inputs)) {
          return read(result, outputName, reader, rawWhenNotFloat);
        }
      }
      // The pinned name must not also be in the requested set: ONNX Runtime rejects a name that
      // appears in both. An empty requested set asks for the pinned output and nothing else, so an
      // output this run does not read is not computed either.
      final Map<String, OnnxValue> pins = new HashMap<>(INPUT_CAPACITY);
      pins.put(outputName, pinned.tensor);
      final OrtSession.Result result = session.run(inputs, Collections.emptySet(), pins);
      // Not try-with-resources, because nothing is read out of the result: a pinned output is not
      // owned by it and is not closed with it, so the values stay readable in the buffer this
      // instance owns whether the result is open or shut. Closed in a finally all the same, since
      // the result still holds a native handle of its own.
      try {
        return reader.read(pinned.values, pinned.shape);
      } finally {
        result.close();
      }
    } finally {
      // Closes what was staged whatever went wrong, including a failure part way through staging,
      // where the map holds the tensors created before the one that failed. The pinned output is
      // not closed here: it is reused by the next run and released by close().
      for (final OnnxTensor input : inputs.values()) {
        input.close();
      }
    }
  }

  /**
   * Stages the inputs of one run into the direct memory of the calling thread and creates a tensor
   * over each of them, the single place this package turns rows into native memory.
   *
   * <p>Every tensor created here is put into {@code inputs}, including on a path out that fails part
   * way, so that the caller's {@code finally} closes what was created before the failure.</p>
   *
   * @param inputs The map to put the tensors into.
   * @param shape The tensor shape of the run.
   * @param ids The token ids, or {@code null} when {@code stager} stages.
   * @param mask The attention mask, or {@code null} to leave the input out.
   * @param types The token type ids, or {@code null} to leave the input out.
   * @param stager The writer of all three inputs, or {@code null} to stage from the arrays.
   *
   * @throws OrtException Thrown if the shape describes no elements or more than an {@code int} can
   *     count, if an array does not hold exactly as many values as the shape describes, or if ONNX
   *     Runtime cannot create the tensor.
   * @throws IllegalStateException Thrown if this instance is closed, if the run would take the
   *     reusable input memory of this instance past its bound, or if {@code stager} left an input
   *     short of the element count of the shape.
   */
  private void stage(final Map<String, OnnxTensor> inputs, final long[] shape, final long[] ids,
      final long[] mask, final long[] types, final InputStager stager) throws OrtException {
    final int elements = inputElements(shape);
    final InputStaging area = staging(elements);
    if (stager != null) {
      final LongBuffer idsSlot = area.slot(IDS_SLOT);
      final LongBuffer maskSlot = area.slot(MASK_SLOT);
      final LongBuffer typesSlot = area.slot(TYPES_SLOT);
      stager.stage(idsSlot, maskSlot, typesSlot);
      // Checked before a single tensor is created, so a stager that wrote one input short fails the
      // run rather than leaving an inference to read the previous run's values out of the rest.
      fullyStaged(AbstractDL.INPUT_IDS, idsSlot, elements);
      fullyStaged(AbstractDL.ATTENTION_MASK, maskSlot, elements);
      fullyStaged(AbstractDL.TOKEN_TYPE_IDS, typesSlot, elements);
      inputs.put(AbstractDL.INPUT_IDS, tensor(idsSlot, shape));
      inputs.put(AbstractDL.ATTENTION_MASK, tensor(maskSlot, shape));
      inputs.put(AbstractDL.TOKEN_TYPE_IDS, tensor(typesSlot, shape));
      return;
    }
    inputs.put(AbstractDL.INPUT_IDS,
        tensor(AbstractDL.INPUT_IDS, ids, shape, elements, area.slot(IDS_SLOT)));
    if (mask != null) {
      inputs.put(AbstractDL.ATTENTION_MASK,
          tensor(AbstractDL.ATTENTION_MASK, mask, shape, elements, area.slot(MASK_SLOT)));
    }
    if (types != null) {
      inputs.put(AbstractDL.TOKEN_TYPE_IDS,
          tensor(AbstractDL.TOKEN_TYPE_IDS, types, shape, elements, area.slot(TYPES_SLOT)));
    }
  }

  /**
   * Copies one input array into its slot and creates the tensor over it.
   *
   * <p>The whole slot is written, because the array has to hold exactly as many values as the shape
   * describes and all of them are copied, so no position of the slot keeps the previous run's value.
   * The length is checked here rather than left to ONNX Runtime's own check, since the slot is cut to
   * the shape and would otherwise be handed a short array and quietly keep whatever the previous run
   * left in the tail.</p>
   *
   * @param name The name of the input, for the failure message.
   * @param data The values, row after row.
   * @param shape The tensor shape.
   * @param elements The element count of {@code shape}.
   * @param slot The slot to stage into, cut to {@code elements}.
   * @return A new tensor over {@code slot}, which the caller closes.
   *
   * @throws OrtException Thrown if {@code data} does not hold exactly {@code elements} values, or if
   *     ONNX Runtime cannot create the tensor.
   */
  private OnnxTensor tensor(final String name, final long[] data, final long[] shape,
      final int elements, final LongBuffer slot) throws OrtException {
    if (data.length != elements) {
      throw new OrtException("The " + name + " of this run holds " + data.length
          + " values where its shape " + Arrays.toString(shape) + " describes " + elements + ".");
    }
    // The slot arrived cleared, so this writes it from its first position to its limit.
    slot.put(data);
    return tensor(slot, shape);
  }

  /**
   * Creates one input tensor over a staged slot.
   *
   * <p>The slot is rewound first, so ONNX Runtime is handed it positioned at {@code 0} with exactly
   * the run's element count remaining, which is what it requires and what makes it wrap the buffer
   * where it lies instead of allocating a direct copy of it.</p>
   *
   * @param slot The staged slot, filled to its limit.
   * @param shape The tensor shape.
   * @return A new tensor the caller closes.
   *
   * @throws OrtException Thrown if ONNX Runtime cannot create the tensor.
   */
  private OnnxTensor tensor(final LongBuffer slot, final long[] shape) throws OrtException {
    slot.rewind();
    return OnnxTensor.createTensor(env, slot, shape);
  }

  /**
   * Checks that a stager filled one slot to its limit.
   *
   * @param name The name of the input, for the failure message.
   * @param slot The slot the stager wrote.
   * @param elements The element count it had to write.
   *
   * @throws IllegalStateException Thrown if fewer than {@code elements} values were written, which
   *     would have left the previous run's values in the rest of the slot.
   */
  private static void fullyStaged(final String name, final LongBuffer slot, final int elements) {
    if (slot.position() != elements) {
      throw new IllegalStateException("The stager of this run wrote " + slot.position()
          + " of the " + elements + " values of " + name + ", so " + (elements - slot.position())
          + " positions would have been read from the previous run of this thread; a stager has to"
          + " write every position of every input, with the relative put.");
    }
  }

  /**
   * {@return the element count of an input shape}
   *
   * @param shape The shape of the run.
   *
   * @throws OrtException Thrown if the shape describes no elements at all or more than an
   *     {@code int} can count, neither of which is a tensor ONNX Runtime can be handed.
   */
  private static int inputElements(final long[] shape) throws OrtException {
    if (shape.length == 0) {
      throw new OrtException("An input shape must have at least one dimension.");
    }
    long product = 1;
    for (final long dimension : shape) {
      product *= dimension;
      if (product <= 0 || product > Integer.MAX_VALUE) {
        throw new OrtException("The input shape " + Arrays.toString(shape) + " of this run describes "
            + product + " elements, which is not a tensor that can be staged; every dimension has to"
            + " be positive and the product has to fit an int.");
      }
    }
    return (int) product;
  }

  /**
   * {@return the input arena of the calling thread, its slots cut to {@code elements}}
   *
   * <p>Grows the arena of the thread if the extent needs more room than it has, re-cuts the slots if
   * the extent differs from the one they were cut to, and does neither when the extent repeats.</p>
   *
   * @param elements The element count one input of this run holds.
   *
   * @throws IllegalStateException Thrown if this instance is closed, or if growing the arena would
   *     take the reusable input memory of this instance past its bound.
   */
  private InputStaging staging(final int elements) {
    InputStaging area = stagingArea.get();
    if (area == null) {
      area = new InputStaging();
      stagingArea.set(area);
      stagingAreas.add(area);
    }
    area.reshape(elements);
    return area;
  }

  /**
   * Reads one output of an unpinned result, flat.
   *
   * <p>A float tensor is read with {@link OnnxTensor#getFloatBuffer()}, which is one flat copy of
   * the tensor, documented as a copy and implemented as an allocate plus a put. Anything else,
   * which means an output that is not a float tensor, has no flat float read, so it falls back to
   * {@link OnnxValue#getValue()} for the {@code Object} returning overloads and is reported as
   * unreadable for the reader overloads, which exist to be handed float values.</p>
   *
   * @param <T> What the reader makes of the output.
   * @param result The result of a run, still open.
   * @param outputName The name of the output to read, or {@code null} for the first output.
   * @param reader The reader of the output.
   * @param rawWhenNotFloat Whether an output that is not a float tensor is handed back as the plain
   *     Java value of {@link OnnxValue#getValue()} rather than reported as unreadable.
   * @return What the reader returned.
   *
   * @throws OrtException Thrown if the model declares no output named {@code outputName}, if the
   *     value cannot be read, or if the reader rejects the output.
   */
  @SuppressWarnings("unchecked")
  private <T> T read(final OrtSession.Result result, final String outputName,
      final OutputReader<T> reader, final boolean rawWhenNotFloat) throws OrtException {
    final OnnxValue output = outputName == null ? result.get(0)
        : result.get(outputName).orElseThrow(() -> new OrtException(
            "The model returned no output named " + outputName));
    if (output instanceof OnnxTensor && output.getInfo() instanceof TensorInfo
        && ((TensorInfo) output.getInfo()).type == OnnxJavaType.FLOAT) {
      final TensorInfo info = (TensorInfo) output.getInfo();
      return reader.read(((OnnxTensor) output).getFloatBuffer(), info.getShape());
    }
    if (rawWhenNotFloat) {
      // Only the Object returning overloads take this branch, so T is Object and the cast holds.
      return (T) output.getValue();
    }
    throw new OrtException("The output " + (outputName == null ? "read by position" : outputName)
        + " is not a float tensor, so it cannot be read as flat float values.");
  }

  /**
   * Shapes a flat read into the nested arrays the {@code Object} returning overloads hand back, so
   * that those overloads read the output the same way every other path does.
   *
   * <p>A {@code null} buffer means the output was not a float tensor and the reader was handed
   * nothing, which cannot happen through this method because the {@code Object} overloads route a
   * non-float output to {@link OnnxValue#getValue()} before reaching here.</p>
   *
   * @param values The values of the output, row major.
   * @param shape The shape of the output.
   * @return {@code float[]}, {@code float[][]} or {@code float[][][]} by the rank of the output.
   *
   * @throws OrtException Thrown if the output has a rank this cannot shape.
   */
  private static Object nested(final FloatBuffer values, final long[] shape) throws OrtException {
    if (values == null) {
      throw new OrtException("The output is not a float tensor.");
    }
    switch (shape.length) {
      case 1:
        return row(values, 0, (int) shape[0]);
      case 2: {
        final int rows = (int) shape[0];
        final int width = (int) shape[1];
        final float[][] nested = new float[rows][];
        for (int r = 0; r < rows; r++) {
          nested[r] = row(values, r * width, width);
        }
        return nested;
      }
      case 3: {
        final int rows = (int) shape[0];
        final int width = (int) shape[1];
        final int hidden = (int) shape[2];
        final float[][][] nested = new float[rows][width][];
        for (int r = 0; r < rows; r++) {
          for (int w = 0; w < width; w++) {
            nested[r][w] = row(values, (r * width + w) * hidden, hidden);
          }
        }
        return nested;
      }
      default:
        throw new OrtException("The model returned an output of rank " + shape.length
            + ", which this reader does not shape; ranks 1 to 3 are supported.");
    }
  }

  /** {@return {@code length} values of {@code values} from {@code offset}, as a new array} */
  private static float[] row(final FloatBuffer values, final int offset, final int length) {
    final float[] copy = new float[length];
    for (int i = 0; i < length; i++) {
      copy[i] = values.get(offset + i);
    }
    return copy;
  }

  /**
   * {@return the pinned output of the calling thread, its tensor shaped to {@code outputShape}}
   *
   * <p>Grows the arena of the thread if the shape needs more room than it has, rebuilds the tensor
   * if the shape differs from the one the tensor already has, and does neither when the shape
   * repeats.</p>
   *
   * @param outputShape The shape to pin at. Every dimension must be positive.
   *
   * @throws OrtException Thrown if the tensor cannot be created.
   * @throws IllegalStateException Thrown if growing the arena would take the reusable direct output
   *     memory of this instance past its bound.
   */
  private PinnedOutput pin(final long[] outputShape) throws OrtException {
    final int floats = elements(outputShape);
    PinnedOutput pinned = arena.get();
    if (pinned == null) {
      pinned = new PinnedOutput();
      arena.set(pinned);
      arenas.add(pinned);
    }
    pinned.reshape(outputShape, floats);
    return pinned;
  }

  /**
   * {@return the element count of a shape}
   *
   * @param shape The shape to count. Must not be empty, every dimension must be positive and the
   *     product must fit an {@code int}, which is the largest buffer ONNX Runtime can be handed.
   */
  private static int elements(final long[] shape) {
    if (shape.length == 0) {
      throw new IllegalArgumentException("An output shape must have at least one dimension.");
    }
    long product = 1;
    for (final long dimension : shape) {
      if (dimension <= 0) {
        throw new IllegalArgumentException("An output shape to pin must have positive dimensions"
            + " but one of them was " + dimension + "; a dynamic dimension has to be resolved"
            + " before the run.");
      }
      product *= dimension;
      if (product > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "An output of more than " + Integer.MAX_VALUE + " elements cannot be pinned.");
      }
    }
    return (int) product;
  }

  /**
   * Adds {@code delta} bytes to the direct output memory of this instance, or refuses.
   *
   * @param delta The bytes to add, never negative here since arenas only grow.
   *
   * @throws IllegalStateException Thrown if the total would pass the bound, in which case nothing
   *     is charged.
   */
  private void charge(final long delta) {
    final long total = pinnedOutputBytes.addAndGet(delta);
    if (total > maxPinnedOutputBytes) {
      pinnedOutputBytes.addAndGet(-delta);
      throw new IllegalStateException("Pinning this output would hold " + total
          + " bytes of direct memory for reusable output buffers, past the bound of "
          + maxPinnedOutputBytes + " bytes for this inference; run smaller batches, use fewer"
          + " threads, or read the output without pinning it.");
    }
  }

  /**
   * Adds {@code delta} bytes to the direct input memory of this instance, or refuses.
   *
   * @param delta The bytes to add, never negative here since arenas only grow.
   *
   * @throws IllegalStateException Thrown if the total would pass the bound, in which case nothing
   *     is charged.
   */
  private void chargeInputStaging(final long delta) {
    final long total = inputStagingBytes.addAndGet(delta);
    if (total > maxInputStagingBytes) {
      inputStagingBytes.addAndGet(-delta);
      throw new IllegalStateException("Staging the inputs of this run would hold " + total
          + " bytes of direct memory for reusable input buffers, past the bound of "
          + maxInputStagingBytes + " bytes for this inference; run smaller batches or use fewer"
          + " threads.");
    }
  }

  /**
   * Closes every pinned output tensor of every thread, releases every input arena of every thread,
   * and releases the accounting of both.
   *
   * <p>Idempotent, so a second close is a no-op, and a run after a close fails with an
   * {@link IllegalStateException} rather than reading released memory. Must not race a run, which is
   * the contract the components using this class already document for their own {@code close()}.
   * The {@link OrtEnvironment} and the {@link OrtSession} are not touched: they belong to the
   * {@link AbstractDL} that built them.</p>
   */
  @Override
  public void close() {
    if (closed.compareAndSet(false, true)) {
      for (final PinnedOutput pinned : arenas) {
        pinned.release();
      }
      arenas.clear();
      for (final InputStaging area : stagingAreas) {
        area.release();
      }
      stagingAreas.clear();
      // Whatever the arenas were holding is no longer held, so the budget of this instance is free
      // again even though a thread local reference may still exist on a live thread. The released
      // marker on the arena is what stops such a thread reusing it.
      pinnedOutputBytes.set(0);
      inputStagingBytes.set(0);
    }
  }

  /** {@return the direct bytes this instance currently holds in reusable pinned output arenas} */
  long pinnedOutputBytes() {
    return pinnedOutputBytes.get();
  }

  /** {@return the number of threads that have allocated a pinned output arena on this instance} */
  int pinnedOutputArenas() {
    return arenas.size();
  }

  /** {@return the direct bytes this instance currently holds in reusable input staging arenas} */
  long inputStagingBytes() {
    return inputStagingBytes.get();
  }

  /** {@return the number of threads that have allocated an input staging arena on this instance} */
  int inputStagingArenas() {
    return stagingAreas.size();
  }

  /**
   * {@return the slot of the calling thread one named input is staged in, or {@code null} if that
   * thread has staged nothing}
   *
   * <p>The seam a test reaches through to assert that the memory an input reaches ONNX Runtime in is
   * direct, and that it is the same memory from one run to the next. The buffer is positioned and
   * limited as the last run of this thread left it. Not for production use, which has no business
   * holding memory this class owns.</p>
   *
   * @param name One of {@link AbstractDL#INPUT_IDS}, {@link AbstractDL#ATTENTION_MASK} or
   *     {@link AbstractDL#TOKEN_TYPE_IDS}.
   */
  LongBuffer inputStagingBuffer(final String name) {
    final InputStaging area = stagingArea.get();
    if (area == null) {
      return null;
    }
    return area.staged(slotOf(name));
  }

  /**
   * {@return the slot index one named input is staged in}
   *
   * @param name The name of the input.
   */
  private static int slotOf(final String name) {
    if (AbstractDL.INPUT_IDS.equals(name)) {
      return IDS_SLOT;
    }
    if (AbstractDL.ATTENTION_MASK.equals(name)) {
      return MASK_SLOT;
    }
    if (AbstractDL.TOKEN_TYPE_IDS.equals(name)) {
      return TYPES_SLOT;
    }
    throw new IllegalArgumentException(name + " is not one of the three inputs staged here.");
  }

  /**
   * {@return the pinned output tensor of the calling thread, or {@code null} if that thread has not
   * pinned an output}
   *
   * <p>The seam a test reaches through to observe reuse and release: two runs of the same shape hand
   * back the same tensor, and {@link #close()} leaves the tensor it handed back
   * {@link OnnxValue#isClosed() closed}. Not for production use, which has no business holding a
   * tensor this class owns.</p>
   */
  OnnxTensor pinnedOutputTensor() {
    final PinnedOutput pinned = arena.get();
    return pinned == null ? null : pinned.tensor;
  }

  /**
   * The reusable pinned output of one thread: a grow-only direct arena, an exact slice of it, and
   * the tensor ONNX Runtime writes into.
   *
   * <p>Confined to the thread that created it, so nothing here is synchronized. The one exception is
   * {@link #release()}, which {@link OnnxInference#close()} calls from whichever thread closes, and
   * which the contract forbids racing a run.</p>
   */
  private final class PinnedOutput {

    /** The direct memory, never shrinking, sized in floats by {@link #capacity}. */
    private ByteBuffer bytes;

    /** How many floats {@link #bytes} holds. */
    private int capacity;

    /** The shape {@link #tensor} was created at, or {@code null} if there is no tensor. */
    private long[] shape;

    /** The exact slice of {@link #bytes} that {@link #tensor} was created over. */
    private FloatBuffer values;

    private OnnxTensor tensor;

    private boolean released;

    /**
     * Makes this arena hold a tensor of exactly {@code outputShape}, growing and rebuilding only as
     * far as it has to.
     *
     * @param outputShape The shape to pin at.
     * @param floats The element count of {@code outputShape}.
     *
     * @throws OrtException Thrown if the tensor cannot be created.
     */
    private void reshape(final long[] outputShape, final int floats) throws OrtException {
      if (released) {
        throw new IllegalStateException(CLOSED);
      }
      if (capacity < floats) {
        // Charged before the allocation, so a refusal leaves the arena exactly as it was.
        charge((long) (floats - capacity) * BYTES_PER_FLOAT);
        closeTensor();
        bytes = ByteBuffer.allocateDirect(floats * BYTES_PER_FLOAT).order(ByteOrder.nativeOrder());
        capacity = floats;
      }
      if (tensor != null && Arrays.equals(shape, outputShape)) {
        return;
      }
      closeTensor();
      // ONNX Runtime requires the buffer behind a tensor to hold exactly as many elements as the
      // shape describes, so the tensor goes over a slice of the arena rather than the arena itself.
      // duplicate, position, limit, slice rather than ByteBuffer.slice(int, int), which is Java 13.
      final ByteBuffer view = bytes.duplicate();
      view.position(0);
      view.limit(floats * BYTES_PER_FLOAT);
      values = view.slice().order(ByteOrder.nativeOrder()).asFloatBuffer();
      tensor = OnnxTensor.createTensor(env, values, outputShape);
      shape = outputShape.clone();
    }

    private void closeTensor() {
      if (tensor != null) {
        tensor.close();
        tensor = null;
        shape = null;
        values = null;
      }
    }

    /**
     * Releases the tensor and marks the arena unusable, which {@link OnnxInference#close()} does.
     */
    private void release() {
      released = true;
      closeTensor();
      bytes = null;
      capacity = 0;
    }
  }

  /**
   * The reusable input memory of one thread: a grow-only direct arena of three equal slots, one per
   * input, and an exact view of each slot cut to the extent of the last run.
   *
   * <p>Confined to the thread that created it, so nothing here is synchronized. The one exception is
   * {@link #release()}, which {@link OnnxInference#close()} calls from whichever thread closes, and
   * which the contract forbids racing a run.</p>
   *
   * <p>Three slots in one arena rather than three arenas, because growing then costs one
   * {@code allocateDirect} instead of three and the accounting is one number rather than three. The
   * slots are equal and are sized by {@link #capacity}, so slot {@code i} begins at
   * {@code i * capacity} longs and the views a run uses are its first {@code elements} longs.</p>
   */
  private final class InputStaging {

    /**
     * The direct memory, never shrinking, holding {@link OnnxInference#INPUT_SLOTS} slots of
     * {@link #capacity} longs each.
     */
    private ByteBuffer bytes;

    /** How many longs one slot of {@link #bytes} holds. */
    private int capacity;

    /** The extent {@link #views} were cut to, or {@code 0} if there are no views. */
    private int extent;

    /** An exact view of the first {@link #extent} longs of each slot. */
    private final LongBuffer[] views = new LongBuffer[INPUT_SLOTS];

    private boolean released;

    /**
     * Makes this arena hold three slots of exactly {@code elements} longs, growing and re-cutting
     * only as far as it has to.
     *
     * @param elements The element count one input of the run holds.
     *
     * @throws IllegalStateException Thrown if this arena has been released, or if growing it would
     *     take the reusable input memory of this instance past its bound.
     */
    private void reshape(final int elements) {
      if (released) {
        throw new IllegalStateException(CLOSED);
      }
      if (capacity < elements) {
        final long grown = (long) (elements - capacity) * INPUT_SLOTS * BYTES_PER_LONG;
        final long total = (long) elements * INPUT_SLOTS * BYTES_PER_LONG;
        if (total > Integer.MAX_VALUE) {
          throw new IllegalStateException("Staging an input of " + elements
              + " elements needs " + total + " bytes of direct memory in one buffer, which is more"
              + " than a buffer can hold; run smaller batches.");
        }
        // Charged before the allocation, so a refusal leaves the arena exactly as it was.
        chargeInputStaging(grown);
        bytes = ByteBuffer.allocateDirect((int) total).order(ByteOrder.nativeOrder());
        capacity = elements;
        // The views pointed into memory that is gone, and the extent they were cut to no longer
        // describes them, so the extent is cleared and they are cut again below.
        extent = 0;
      }
      if (extent == elements) {
        return;
      }
      for (int slot = 0; slot < INPUT_SLOTS; slot++) {
        // duplicate, position, limit, slice rather than ByteBuffer.slice(int, int), which is Java 13.
        final ByteBuffer view = bytes.duplicate();
        final int from = slot * capacity * BYTES_PER_LONG;
        view.position(from);
        view.limit(from + elements * BYTES_PER_LONG);
        views[slot] = view.slice().order(ByteOrder.nativeOrder()).asLongBuffer();
      }
      extent = elements;
    }

    /**
     * {@return the view of one slot, ready to be written from its first position}
     *
     * @param slot The slot to write.
     */
    private LongBuffer slot(final int slot) {
      final LongBuffer view = views[slot];
      view.clear();
      return view;
    }

    /**
     * {@return the view of one slot as the last run left it, or {@code null} if it was never cut}
     *
     * @param slot The slot to read.
     */
    private LongBuffer staged(final int slot) {
      return views[slot];
    }

    /**
     * Releases the arena and marks it unusable, which {@link OnnxInference#close()} does. The direct
     * memory itself is reclaimed once nothing refers to it, as any direct buffer is.
     */
    private void release() {
      released = true;
      bytes = null;
      capacity = 0;
      extent = 0;
      Arrays.fill(views, null);
    }
  }
}
