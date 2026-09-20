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

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import opennlp.dl.AbstractDL;

/**
 * The diagnostic that decides how much of the padding win is really an ONNX Runtime threading
 * effect: the same bars as {@link SentenceVectorsDLBatchBenchmark}, swept over the intra-op thread
 * count.
 *
 * <p>ONNX Runtime defaults {@code intra_op_param.thread_pool_size} to {@code 0}, which on this
 * machine starts 31 worker threads and spins them, and
 * {@link opennlp.dl.AbstractDL#sessionOptions(opennlp.dl.InferenceOptions)} is not on
 * {@link SentenceVectorsDL}'s construction path, so a shipped {@code SentenceVectorsDL} always
 * runs at that default. Spinning 31 threads for a {@code {3,10}} matrix multiply is
 * oversubscription on trivial work, and the fragmented {@link PaddingStrategy#EXACT_LENGTH} plan
 * does exactly that tens of times per call. If pinning the pool small lifts
 * {@code EXACT_LENGTH} or the per-input loop, then part of what looks like a batching win is a
 * threading misconfiguration and must be attributed there instead.</p>
 *
 * <p><b>Why this class reflects over a field.</b> {@link SentenceVectorsDL} builds its session
 * from a bare {@code new OrtSession.SessionOptions()} with no seam to pass options in, and main
 * code is frozen for this measurement, so the thread count cannot be set the supported way. This
 * benchmark therefore builds a second session with an explicit intra-op count and replaces the
 * {@code session} field of an otherwise normally constructed instance, so that every other part
 * of the measured code is the real shipped path. {@code taskset} was tried first and rejected:
 * ONNX Runtime starts 31 workers whatever the CPU affinity is, so restricting cores increases
 * oversubscription rather than reducing it and would answer the opposite question. Nothing here
 * changes main code, and the reflection is confined to this benchmark.</p>
 *
 * <p>The setup prints the worker thread count the replacement session actually started, so the
 * reader can confirm the setting took effect instead of trusting that it did.</p>
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5)
@Measurement(iterations = 20)
@Fork(1)
public class SentenceVectorsDLIntraOpBenchmark {

  @Param({"32", "128"})
  public int callSize;

  @Param({"LOOP", "EXACT_LENGTH", "LONGEST"})
  public SentenceVectorsDLBatchBenchmark.Bar bar;

  /** The intra-op thread count, where {@code 0} leaves ONNX Runtime's default in force. */
  @Param({"0", "1", "4", "8"})
  public int intraOp;

  private SentenceVectorsDL embedder;
  private OrtSession displaced;
  private List<String> inputs;

  /**
   * Loads the model, swaps in a session with the configured intra-op thread count, and gates the
   * bar on agreeing with a per-input loop.
   *
   * @throws Exception Thrown if the model cannot be loaded, the corpus cannot be read, or the
   *     session field cannot be replaced.
   */
  @Setup(Level.Trial)
  public void setUp() throws Exception {
    inputs = BatchBenchmarkInputs.call(BatchBenchmarkInputs.pool(), callSize);
    embedder = new SentenceVectorsDL(BatchBenchmarkInputs.model(),
        BatchBenchmarkInputs.vocabulary(), true, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, bar.strategy());

    if (intraOp > 0) {
      final int before = osThreads();
      final OrtSession replacement;
      try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
        options.setIntraOpNumThreads(intraOp);
        replacement = OrtEnvironment.getEnvironment()
            .createSession(BatchBenchmarkInputs.model().getPath(), options);
      }
      final Field field = AbstractDL.class.getDeclaredField("session");
      field.setAccessible(true);
      displaced = (OrtSession) field.get(embedder);
      field.set(embedder, replacement);
      if (field.get(embedder) != replacement) {
        throw new IllegalStateException("The session field was not replaced.");
      }
      System.out.println("# intraOp=" + intraOp + " workerThreadsStarted="
          + (osThreads() - before) + " (jvm thread delta for the replacement session)");
    } else {
      System.out.println("# intraOp=default ORT thread_pool_size 0, all cores");
    }

    final float[][] reference;
    try (SentenceVectorsDL loopEmbedder = new SentenceVectorsDL(BatchBenchmarkInputs.model(),
        BatchBenchmarkInputs.vocabulary(), true, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, PaddingStrategy.EXACT_LENGTH)) {
      reference = new float[inputs.size()][];
      for (int i = 0; i < reference.length; i++) {
        reference[i] = loopEmbedder.embed(inputs.get(i));
      }
    }
    final float deviation = BatchBenchmarkInputs.deviation(reference, embedder.embedAll(inputs));
    if (!(deviation <= BatchBenchmarkInputs.TOLERANCE)) {
      throw new IllegalStateException("Bar " + bar + " at call size " + callSize + " intraOp "
          + intraOp + " deviates from the per-input loop by " + deviation);
    }
    System.out.println("# gate bar=" + bar + " callSize=" + callSize + " intraOp=" + intraOp
        + " maxAbsDeviation=" + deviation
        + " sessionRunCalls=" + (bar == SentenceVectorsDLBatchBenchmark.Bar.LOOP
            ? callSize : embedder.batchShapes(inputs).length));
  }

  /** Closes the embedder's session and the session that was displaced, if any. */
  @TearDown(Level.Trial)
  public void tearDown() throws Exception {
    if (embedder != null) {
      embedder.close();
    }
    if (displaced != null) {
      displaced.close();
    }
  }

  /**
   * One call of {@link #callSize} inputs, embedded the way {@link #bar} says.
   *
   * @param blackhole The sink for the vectors.
   */
  @Benchmark
  public void call(final Blackhole blackhole) {
    if (bar == SentenceVectorsDLBatchBenchmark.Bar.LOOP) {
      for (final String input : inputs) {
        blackhole.consume(embedder.embed(input));
      }
    } else {
      blackhole.consume(embedder.embedAll(inputs));
    }
  }

  /** {@return the number of OS threads of this JVM, or {@code -1} where it cannot be read} */
  private static int osThreads() {
    try (var entries = Files.list(Path.of("/proc/self/task"))) {
      return (int) entries.count();
    } catch (final IOException | RuntimeException e) {
      return -1;
    }
  }

}
