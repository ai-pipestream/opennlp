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

import java.util.List;
import java.util.concurrent.TimeUnit;

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

import opennlp.dl.InferenceOptions;

/**
 * Throughput of the four ways to embed a call of several inputs with one
 * {@link SentenceVectorsDL}: a loop of single-input {@link SentenceVectorsDL#embed(CharSequence)}
 * calls, and {@link SentenceVectorsDL#embedAll(List)} under each {@link PaddingStrategy}.
 *
 * <p>One operation is one call of {@link #callSize} inputs, so the reported {@code ops/s} is
 * calls per second and embeddings per second is that times {@link #callSize}. Every bar at a
 * given call size runs over the same texts in the same order, read by
 * {@link BatchBenchmarkInputs}.</p>
 *
 * <p>The trial setup refuses to run a bar whose vectors differ from the per-input loop by more
 * than the tolerance of its {@link Provider}, so no timing can come out of a bar that computes the
 * wrong answer.</p>
 *
 * <p>Session options are whatever {@link opennlp.dl.AbstractDL} builds from the
 * {@link InferenceOptions} of the chosen {@link Provider}: no intra-op or inter-op thread count
 * and no graph optimization level are set, so ONNX Runtime's own defaults are in force and this
 * benchmark does not change them. {@link #provider} defaults to {@link Provider#CPU}, so the
 * numbers a plain run produces are the same measurement as before the provider became selectable;
 * pass {@code -p provider=CPU,CUDA} to compare the two, which needs the {@code onnxruntime_gpu}
 * runtime rather than the CPU-only one.</p>
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Fork(1)
public class SentenceVectorsDLBatchBenchmark {

  /** The bars, in the order the phase B specification lists them. */
  public enum Bar {

    /** A loop of {@link SentenceVectorsDL#embed(CharSequence)}, one inference per input. */
    LOOP(PaddingStrategy.EXACT_LENGTH),

    /** {@link SentenceVectorsDL#embedAll(List)} as main has it today. */
    EXACT_LENGTH(PaddingStrategy.EXACT_LENGTH),

    /** {@link SentenceVectorsDL#embedAll(List)} padding each batch to its longest row. */
    LONGEST(PaddingStrategy.LONGEST),

    /** {@link SentenceVectorsDL#embedAll(List)} padding every row to the maximum length. */
    MAX_LENGTH(PaddingStrategy.MAX_LENGTH);

    private final PaddingStrategy strategy;

    Bar(final PaddingStrategy strategy) {
      this.strategy = strategy;
    }

    /** {@return the padding strategy the bar configures the embedder with} */
    public PaddingStrategy strategy() {
      return strategy;
    }
  }

  /**
   * The execution providers the benchmark can run a bar on, each with the tolerance its own
   * correctness gate uses.
   *
   * <p>The gate compares a padded bar against the same inputs run one at a time on the same
   * provider, so it is measuring what padding does to that provider's arithmetic. On the CPU the
   * measured difference is exactly {@code 0}, so
   * {@link BatchBenchmarkInputs#TOLERANCE} stands unchanged. On the CUDA provider it is not:
   * cuBLAS picks its blocking from the tensor shape, so a row run at {@code [8, 34]} and the same
   * row run at {@code [1, 12]} sum in a different order and the measured difference is around
   * {@code 1.2e-4} on unit-length vectors. The CUDA gate is set an order of magnitude above that.
   * The vectors are still the same embedding: the eval test
   * {@code SentenceVectorsDLExecutionProviderEval} measures the angle rather than the components
   * and finds a worst cosine distance of {@code 5.1e-7}.</p>
   */
  public enum Provider {

    /** The default CPU execution provider. */
    CPU(false, BatchBenchmarkInputs.TOLERANCE),

    /** The CUDA execution provider on device {@code 0}. */
    CUDA(true, 1.0e-3f);

    private final boolean gpu;
    private final float tolerance;

    Provider(final boolean gpu, final float tolerance) {
      this.gpu = gpu;
      this.tolerance = tolerance;
    }

    /** {@return a fresh {@link InferenceOptions} selecting this provider} */
    public InferenceOptions inferenceOptions() {
      final InferenceOptions options = new InferenceOptions();
      options.setGpu(gpu);
      return options;
    }

    /** {@return how far a padded bar may sit from the per-input loop on this provider} */
    public float tolerance() {
      return tolerance;
    }
  }

  @Param({"1", "8", "32", "64", "128"})
  public int callSize;

  @Param({"CPU"})
  public Provider provider;

  @Param({"LOOP", "EXACT_LENGTH", "LONGEST", "MAX_LENGTH"})
  public Bar bar;

  private SentenceVectorsDL embedder;
  private List<String> inputs;

  /**
   * Loads the model and the input set, then gates the bar on agreeing with the per-input loop.
   *
   * @throws Exception Thrown if the model cannot be loaded or the corpus cannot be read.
   */
  @Setup(Level.Trial)
  public void setUp() throws Exception {
    inputs = BatchBenchmarkInputs.call(BatchBenchmarkInputs.pool(), callSize);
    embedder = new SentenceVectorsDL(BatchBenchmarkInputs.model(),
        BatchBenchmarkInputs.vocabulary(), true, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, bar.strategy(), provider.inferenceOptions());

    if (bar != Bar.LOOP) {
      // The reference is bar 1 itself: single-input embed calls on a default-strategy embedder,
      // not on this bar's embedder, since MAX_LENGTH also pads a single-input call. It runs on the
      // same execution provider as the bar, so the gate measures what padding does and nothing
      // else; that the providers agree with each other is asserted by
      // SentenceVectorsDLExecutionProviderEval, against a tolerance stated there.
      final float[][] reference;
      try (SentenceVectorsDL loopEmbedder = new SentenceVectorsDL(BatchBenchmarkInputs.model(),
          BatchBenchmarkInputs.vocabulary(), true, Pooling.MEAN, true,
          SentenceVectorsDL.DEFAULT_MAX_LENGTH, PaddingStrategy.EXACT_LENGTH,
          provider.inferenceOptions())) {
        reference = new float[inputs.size()][];
        for (int i = 0; i < reference.length; i++) {
          reference[i] = loopEmbedder.embed(inputs.get(i));
        }
      }
      final float deviation = BatchBenchmarkInputs.deviation(reference, embedder.embedAll(inputs));
      if (!(deviation <= provider.tolerance())) {
        throw new IllegalStateException("Bar " + bar + " on " + provider + " at call size "
            + callSize + " deviates from the per-input loop by " + deviation
            + ", above the tolerance " + provider.tolerance());
      }
      System.out.println("# gate bar=" + bar + " provider=" + provider + " callSize=" + callSize
          + " maxAbsDeviation=" + deviation);
    }
    final int inferences = bar == Bar.LOOP ? callSize : embedder.batchShapes(inputs).length;
    System.out.println("# bar=" + bar + " provider=" + provider + " callSize=" + callSize
        + " sessionRunCalls=" + inferences);
  }

  /** Closes the ONNX session. */
  @TearDown(Level.Trial)
  public void tearDown() {
    if (embedder != null) {
      embedder.close();
    }
  }

  /**
   * One call of {@link #callSize} inputs, embedded the way {@link #bar} says.
   *
   * @param blackhole The sink for the vectors.
   */
  @Benchmark
  public void call(final Blackhole blackhole) {
    if (bar == Bar.LOOP) {
      for (final String input : inputs) {
        blackhole.consume(embedder.embed(input));
      }
    } else {
      blackhole.consume(embedder.embedAll(inputs));
    }
  }

}
