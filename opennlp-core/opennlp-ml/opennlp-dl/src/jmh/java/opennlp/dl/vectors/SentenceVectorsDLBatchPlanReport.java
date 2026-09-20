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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import ai.onnxruntime.OrtException;

/**
 * Everything about the phase B bars that is a property of the inputs rather than of the clock:
 * the tokenized length distribution of the input set, the number of {@code session.run} calls
 * each bar makes at each call size, the padded against real token positions of each bar, and the
 * correctness gate that the padded bars reproduce the per-input loop's vectors.
 *
 * <p>The counts come from {@link SentenceVectorsDL#batchShapes(List)}, which returns the batch
 * plan without running the session, so the inference count is the plan's length and the padded
 * positions are the sum of {@code rows * width} over the plan. Under
 * {@link PaddingStrategy#EXACT_LENGTH} no row is padded, so that plan's row widths are the
 * tokenized lengths themselves and they give the length distribution and the real position
 * count.</p>
 *
 * <p>Run it as a plain main class; it takes no arguments and prints one report to standard
 * output.</p>
 */
public final class SentenceVectorsDLBatchPlanReport {

  private SentenceVectorsDLBatchPlanReport() {
  }

  /**
   * Prints the report.
   *
   * @param args Ignored.
   * @throws IOException Thrown if the corpus, the model or the vocabulary cannot be read.
   * @throws OrtException Thrown if the model cannot be loaded.
   */
  public static void main(final String[] args) throws IOException, OrtException {
    final List<String> pool = BatchBenchmarkInputs.pool();
    System.out.println("model       " + BatchBenchmarkInputs.model());
    System.out.println("vocabulary  " + BatchBenchmarkInputs.vocabulary());
    System.out.println("corpus      " + BatchBenchmarkInputs.corpus());
    System.out.println("inputs      first " + pool.size()
        + " non-blank sentences of the corpus, in file order, line number and tab removed");
    System.out.println("cores       " + Runtime.getRuntime().availableProcessors());
    System.out.println("maxLength   " + SentenceVectorsDL.DEFAULT_MAX_LENGTH);
    System.out.println("batch cap   " + SentenceVectorsDL.MAX_BATCH_TOKEN_POSITIONS
        + " token positions per inference");
    System.out.println();

    final int beforeSession = osThreads();
    final Map<PaddingStrategy, SentenceVectorsDL> embedders = new LinkedHashMap<>();
    try {
      for (final PaddingStrategy strategy : PaddingStrategy.values()) {
        embedders.put(strategy, new SentenceVectorsDL(BatchBenchmarkInputs.model(),
            BatchBenchmarkInputs.vocabulary(), true, Pooling.MEAN, true,
            SentenceVectorsDL.DEFAULT_MAX_LENGTH, strategy));
      }
      final SentenceVectorsDL reference = embedders.get(PaddingStrategy.EXACT_LENGTH);
      final int afterSessions = osThreads();
      reference.embed(pool.get(0));
      final int afterInference = osThreads();
      System.out.println("ORT session options are a default OrtSession.SessionOptions:"
          + " no intra-op count, no inter-op count, no graph optimization level set.");
      System.out.println("OS threads of this JVM: " + beforeSession + " before any session, "
          + afterSessions + " after " + embedders.size() + " sessions, "
          + afterInference + " after one inference.");
      System.out.println("dimension   " + reference.dimension());
      System.out.println();

      lengthDistribution(reference, pool);
      System.out.println();
      plans(embedders, pool);
      System.out.println();
      gate(embedders, pool);
    } finally {
      for (final SentenceVectorsDL embedder : embedders.values()) {
        embedder.close();
      }
    }
  }

  /**
   * Prints the tokenized length distribution of the whole pool and of each call size.
   *
   * @param reference An embedder configured with {@link PaddingStrategy#EXACT_LENGTH}.
   * @param pool The input pool.
   */
  private static void lengthDistribution(final SentenceVectorsDL reference,
      final List<String> pool) {
    System.out.println("Tokenized length distribution, [CLS] and [SEP] included");
    System.out.printf("%-10s %6s %7s %7s %5s %5s %9s %14s%n",
        "inputs", "count", "mean", "median", "min", "max", "distinct", "totalPositions");
    final List<Integer> sizes = new ArrayList<>();
    for (final int callSize : BatchBenchmarkInputs.CALL_SIZES) {
      sizes.add(callSize);
    }
    if (!sizes.contains(pool.size())) {
      sizes.add(pool.size());
    }
    for (final int callSize : sizes) {
      final int[] lengths = lengths(reference, BatchBenchmarkInputs.call(pool, callSize));
      Arrays.sort(lengths);
      long total = 0;
      for (final int length : lengths) {
        total += length;
      }
      final double median = lengths.length % 2 == 1 ? lengths[lengths.length / 2]
          : (lengths[lengths.length / 2 - 1] + lengths[lengths.length / 2]) / 2.0;
      final TreeSet<Integer> distinct = new TreeSet<>();
      for (final int length : lengths) {
        distinct.add(length);
      }
      System.out.printf("%-10s %6d %7.2f %7.1f %5d %5d %9d %14d%n",
          "first " + callSize, lengths.length, (double) total / lengths.length, median,
          lengths[0], lengths[lengths.length - 1], distinct.size(), total);
    }
    final int[] all = lengths(reference, pool);
    Arrays.sort(all);
    final TreeSet<Integer> distinct = new TreeSet<>();
    for (final int length : all) {
      distinct.add(length);
    }
    System.out.println("distinct lengths in the pool: " + distinct);
  }

  /**
   * {@return the tokenized lengths of the inputs, unsorted order not guaranteed}
   *
   * <p>Read from the {@link PaddingStrategy#EXACT_LENGTH} batch plan, whose batches hold rows of
   * one length each and whose row widths are therefore the tokenized lengths.</p>
   *
   * @param reference An embedder configured with {@link PaddingStrategy#EXACT_LENGTH}.
   * @param texts The inputs.
   */
  private static int[] lengths(final SentenceVectorsDL reference, final List<String> texts) {
    final int[] lengths = new int[texts.size()];
    int at = 0;
    for (final int[] shape : reference.batchShapes(texts)) {
      for (int row = 0; row < shape[0]; row++) {
        lengths[at++] = shape[1];
      }
    }
    if (at != texts.size()) {
      throw new IllegalStateException("The batch plan covered " + at + " of " + texts.size()
          + " inputs");
    }
    return lengths;
  }

  /**
   * Prints the inference count, the padded positions and the real positions of every bar at every
   * call size.
   *
   * @param embedders One embedder per strategy.
   * @param pool The input pool.
   */
  private static void plans(final Map<PaddingStrategy, SentenceVectorsDL> embedders,
      final List<String> pool) {
    System.out.println("Batch plans, from batchShapes, no session run");
    System.out.printf("%-22s %9s %12s %12s %12s %8s %s%n",
        "bar", "callSize", "sessionRuns", "padded", "real", "waste", "shapes");
    for (final int callSize : BatchBenchmarkInputs.CALL_SIZES) {
      final List<String> call = BatchBenchmarkInputs.call(pool, callSize);
      final long real = positions(embedders.get(PaddingStrategy.EXACT_LENGTH).batchShapes(call));
      System.out.printf("%-22s %9d %12d %12d %12d %7.1f%% %s%n",
          "1 loop embed()", callSize, callSize, real, real, 0.0,
          "callSize inferences of {1, own length}");
      int index = 2;
      for (final PaddingStrategy strategy : PaddingStrategy.values()) {
        final int[][] shapes = embedders.get(strategy).batchShapes(call);
        final long padded = positions(shapes);
        System.out.printf("%-22s %9d %12d %12d %12d %7.1f%% %s%n",
            index + " embedAll " + strategy, callSize, shapes.length, padded, real,
            100.0 * (padded - real) / padded, summarize(shapes));
        index++;
      }
    }
  }

  /**
   * {@return the total token positions of a batch plan, the sum of rows times width}
   *
   * @param shapes The batch plan.
   */
  private static long positions(final int[][] shapes) {
    long total = 0;
    for (final int[] shape : shapes) {
      total += (long) shape[0] * shape[1];
    }
    return total;
  }

  /**
   * {@return a short rendering of a batch plan, abbreviated when it has many batches}
   *
   * @param shapes The batch plan.
   */
  private static String summarize(final int[][] shapes) {
    final StringBuilder text = new StringBuilder();
    final int shown = Math.min(shapes.length, 6);
    for (int g = 0; g < shown; g++) {
      text.append('{').append(shapes[g][0]).append(',').append(shapes[g][1]).append('}');
    }
    if (shapes.length > shown) {
      text.append("... ").append(shapes.length).append(" batches");
    }
    return text.toString();
  }

  /**
   * Prints the correctness gate: the largest absolute deviation of each padded bar from the
   * per-input loop, and fails if any exceeds the tolerance.
   *
   * @param embedders One embedder per strategy.
   * @param pool The input pool.
   */
  private static void gate(final Map<PaddingStrategy, SentenceVectorsDL> embedders,
      final List<String> pool) {
    System.out.println("Correctness gate, tolerance " + BatchBenchmarkInputs.TOLERANCE
        + ", reference is a per-input embed() loop on an EXACT_LENGTH embedder");
    System.out.printf("%-22s %9s %18s %s%n", "bar", "callSize", "maxAbsDeviation", "verdict");
    final SentenceVectorsDL reference = embedders.get(PaddingStrategy.EXACT_LENGTH);
    boolean failed = false;
    for (final int callSize : BatchBenchmarkInputs.CALL_SIZES) {
      final List<String> call = BatchBenchmarkInputs.call(pool, callSize);
      final float[][] loop = new float[call.size()][];
      for (int i = 0; i < loop.length; i++) {
        loop[i] = reference.embed(call.get(i));
      }
      for (final PaddingStrategy strategy : PaddingStrategy.values()) {
        final float deviation =
            BatchBenchmarkInputs.deviation(loop, embedders.get(strategy).embedAll(call));
        final boolean ok = deviation <= BatchBenchmarkInputs.TOLERANCE;
        failed |= !ok;
        System.out.printf("%-22s %9d %18s %s%n", "embedAll " + strategy, callSize,
            deviation, ok ? "pass" : "FAIL");
      }
    }
    // The whole pool as one call as well, so the gate covers the widest plan each strategy makes.
    final float[][] loop = new float[pool.size()][];
    for (int i = 0; i < loop.length; i++) {
      loop[i] = reference.embed(pool.get(i));
    }
    for (final PaddingStrategy strategy : PaddingStrategy.values()) {
      final float deviation =
          BatchBenchmarkInputs.deviation(loop, embedders.get(strategy).embedAll(pool));
      final boolean ok = deviation <= BatchBenchmarkInputs.TOLERANCE;
      failed |= !ok;
      System.out.printf("%-22s %9d %18s %s%n", "embedAll " + strategy, pool.size(),
          deviation, ok ? "pass" : "FAIL");
    }
    System.out.println(failed ? "GATE FAILED" : "GATE PASSED");
    if (failed) {
      throw new IllegalStateException("The correctness gate failed; no timing is meaningful.");
    }
  }

  /**
   * {@return the number of OS threads this JVM currently has, or {@code -1} where that cannot be
   * read}
   *
   * <p>ONNX Runtime's thread pools are native, so they do not appear in the JVM's own thread
   * accounting. Counting {@code /proc/self/task} is the only way to see them from here, and it
   * is how this report reports the effective intra-op pool size rather than asserting a default.
   * </p>
   */
  private static int osThreads() {
    final File tasks = new File("/proc/self/task");
    try (var entries = Files.list(Path.of(tasks.getPath()))) {
      return (int) entries.count();
    } catch (final IOException | RuntimeException e) {
      return -1;
    }
  }

}
