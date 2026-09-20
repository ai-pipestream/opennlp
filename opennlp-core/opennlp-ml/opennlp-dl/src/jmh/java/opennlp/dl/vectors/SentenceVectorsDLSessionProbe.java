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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;

/**
 * Reports what ONNX Runtime actually does with the default
 * {@code OrtSession.SessionOptions} that {@link SentenceVectorsDL} builds: the execution
 * providers of the session and the size of the thread pool it starts.
 *
 * <p>{@code SessionOptions} exposes no getter for the intra-op or inter-op thread count, so the
 * pool size is measured rather than read: ONNX Runtime's threads are native and invisible to the
 * JVM's own thread accounting, but they are visible as entries of {@code /proc/self/task}. The
 * environment is initialized at verbose logging level first, which makes ONNX Runtime print the
 * session's effective configuration to standard error.</p>
 *
 * <p>Nothing here configures threads. It only observes the defaults, which is what the phase B
 * report has to state.</p>
 */
public final class SentenceVectorsDLSessionProbe {

  private SentenceVectorsDLSessionProbe() {
  }

  /**
   * Prints the probe.
   *
   * @param args Ignored.
   * @throws IOException Thrown if the model, vocabulary or corpus cannot be read.
   * @throws OrtException Thrown if the model cannot be loaded.
   */
  public static void main(final String[] args) throws IOException, OrtException {
    // Initializes the process-wide singleton that AbstractDL later picks up, so the session
    // creation below logs at verbose level.
    OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE, "opennlp-phaseb");
    final List<String> pool = BatchBenchmarkInputs.pool();
    System.out.println("cores " + Runtime.getRuntime().availableProcessors());
    System.out.println("threads before any session " + osThreads());
    try (SentenceVectorsDL embedder = new SentenceVectorsDL(BatchBenchmarkInputs.model(),
        BatchBenchmarkInputs.vocabulary(), true, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, PaddingStrategy.LONGEST)) {
      System.out.println("threads after one session " + osThreads());
      System.out.println("available providers " + OrtEnvironment.getAvailableProviders());
      System.out.println("dimension " + embedder.dimension());
      embedder.embedAll(BatchBenchmarkInputs.call(pool, 128));
      System.out.println("threads after one 128 input call " + osThreads());
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
