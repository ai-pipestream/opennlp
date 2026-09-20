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
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import opennlp.dl.CpuExecutionProviderConfigurer;
import opennlp.dl.CudaExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderRequest;
import opennlp.dl.ExecutionProviders;
import opennlp.dl.InferenceOptions;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link SentenceVectorsDL} built on a real ONNX session with execution providers requested through
 * {@link InferenceOptions#setExecutionProviders(List)} and with the session settings set.
 *
 * <p>{@code ExecutionProvidersTest} and {@code InferenceOptionsSessionConfigTest} observe the calls
 * a configuration makes without creating a session. These cases create one, which is what shows that
 * the configuration is one ONNX Runtime accepts and that the vectors do not change because of it.
 * Where a case needs to observe which execution provider a session ran on, it uses the technique
 * {@code SentenceVectorsDLExecutionProviderTest} established: a request ONNX Runtime must reject if
 * and only if it reached the runtime, namely a device id no machine answers to.</p>
 */
class SentenceVectorsDLSessionConfigTest {

  /**
   * A device id no machine this runs on answers to, the tracer that proves a request reached ONNX
   * Runtime. The value matches {@code SentenceVectorsDLExecutionProviderTest}: a plausible id is
   * rejected by the device rather than while the options are being built.
   */
  private static final int UNUSABLE_DEVICE_ID = 99;

  private static final String DEVICE_ID = CudaExecutionProviderConfigurer.DEVICE_ID_OPTION;

  private static final List<String> TEXTS =
      List.of("hello world", "hello", "world", "hello hello world", "");

  // Copied out of the classpath rather than resolved in place, because this test also runs from the
  // opennlp-dl test-jar in opennlp-dl-gpu, where the resource URI is not hierarchical.
  private static File model(final Path dir) throws IOException {
    final Path file = dir.resolve("tiny-vectors.onnx");
    try (InputStream is = Objects.requireNonNull(SentenceVectorsDLSessionConfigTest.class
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

  private static SentenceVectorsDL vectors(final Path dir, final PaddingStrategy padding,
      final InferenceOptions inferenceOptions) throws Exception {
    return new SentenceVectorsDL(model(dir), vocab(dir), true, Pooling.MEAN, false, 16, padding,
        inferenceOptions);
  }

  private static InferenceOptions requesting(final ExecutionProviderRequest... requests) {
    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(List.of(requests));
    return options;
  }

  private static ExecutionProviderRequest cuda(final int deviceId) {
    return ExecutionProviderRequest.of(ExecutionProviders.CUDA,
        Map.of(DEVICE_ID, Integer.toString(deviceId)));
  }

  /** {@return whether a CUDA session can actually be created on device {@code 0} here} */
  private static boolean cudaUsable() {
    if (!OrtEnvironment.getAvailableProviders().contains(OrtProvider.CUDA)) {
      return false;
    }
    OrtEnvironment.getEnvironment();
    try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
      options.addCUDA(0);
      return true;
    } catch (final OrtException | RuntimeException e) {
      return false;
    }
  }

  /**
   * An explicit request for the CPU builds a working session and does not change the vectors. This is
   * the case that would fail if requesting {@value ExecutionProviders#CPU} made ONNX Runtime do
   * something other than what it does on its own.
   */
  @Test
  void testExplicitCpuRequestProducesTheSameVectors(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL reference = vectors(dir, PaddingStrategy.EXACT_LENGTH,
             new InferenceOptions());
         SentenceVectorsDL explicit = vectors(dir, PaddingStrategy.EXACT_LENGTH,
             requesting(ExecutionProviderRequest.of(ExecutionProviders.CPU)))) {
      final float[][] expected = reference.embedAll(TEXTS);
      final float[][] actual = explicit.embedAll(TEXTS);
      assertEquals(expected.length, actual.length);
      for (int i = 0; i < expected.length; i++) {
        assertArrayEquals(expected[i], actual[i], 0f, "input " + i);
      }
    }
  }

  /**
   * Turning the CPU arena off builds a working session, which is what shows the provider option was
   * one ONNX Runtime accepts rather than one it only recorded.
   */
  @Test
  void testCpuWithoutTheArenaProducesTheSameVectors(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL reference = vectors(dir, PaddingStrategy.EXACT_LENGTH,
             new InferenceOptions());
         SentenceVectorsDL noArena = vectors(dir, PaddingStrategy.EXACT_LENGTH,
             requesting(ExecutionProviderRequest.of(ExecutionProviders.CPU,
                 Map.of(CpuExecutionProviderConfigurer.USE_ARENA_OPTION, "false"))))) {
      assertArrayEquals(reference.embed("hello world"), noArena.embed("hello world"), 0f);
    }
  }

  /**
   * The session settings reach a real session and the session still runs: a thread count and an
   * optimization level that ONNX Runtime rejected would fail here, and one it accepted must not
   * change the vectors.
   */
  @Test
  void testSessionSettingsBuildAWorkingSession(@TempDir final Path dir) throws Exception {
    final InferenceOptions configured = new InferenceOptions();
    configured.setIntraOpNumThreads(1);
    configured.setInterOpNumThreads(1);
    configured.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

    try (SentenceVectorsDL reference = vectors(dir, PaddingStrategy.EXACT_LENGTH,
             new InferenceOptions());
         SentenceVectorsDL singleThreaded = vectors(dir, PaddingStrategy.EXACT_LENGTH,
             configured)) {
      final float[][] expected = reference.embedAll(TEXTS);
      final float[][] actual = singleThreaded.embedAll(TEXTS);
      for (int i = 0; i < expected.length; i++) {
        assertArrayEquals(expected[i], actual[i], 0f, "input " + i);
      }
    }
  }

  /**
   * Taking every graph optimization away still builds a working session, which is the one setting
   * where the value that differs from ONNX Runtime's default is the interesting one.
   */
  @Test
  void testNoGraphOptimizationBuildsAWorkingSession(@TempDir final Path dir) throws Exception {
    final InferenceOptions unoptimized = new InferenceOptions();
    unoptimized.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT);

    try (SentenceVectorsDL reference = vectors(dir, PaddingStrategy.EXACT_LENGTH,
             new InferenceOptions());
         SentenceVectorsDL plain = vectors(dir, PaddingStrategy.EXACT_LENGTH, unoptimized)) {
      assertArrayEquals(reference.embed("hello world"), plain.embed("hello world"), 1e-5f);
    }
  }

  /**
   * Every {@link PaddingStrategy} composes with an explicit execution provider request and with the
   * session settings: the strategy shapes the tensors, the provider runs them and the thread counts
   * decide how, and none of the three changes the vectors. The reference pads nothing and is
   * configured with nothing, so a bug in the padding cannot cancel out a bug in the configuration.
   *
   * @param padding The strategy under test.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testEveryPaddingStrategyComposesWithTheConfiguration(final PaddingStrategy padding,
      @TempDir final Path dir) throws Exception {
    final InferenceOptions configured =
        requesting(ExecutionProviderRequest.of(ExecutionProviders.CPU));
    configured.setIntraOpNumThreads(2);
    configured.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

    try (SentenceVectorsDL reference = vectors(dir, PaddingStrategy.EXACT_LENGTH,
             new InferenceOptions());
         SentenceVectorsDL configuredVectors = vectors(dir, padding, configured)) {
      final float[][] actual = configuredVectors.embedAll(TEXTS);
      assertEquals(TEXTS.size(), actual.length);
      for (int i = 0; i < TEXTS.size(); i++) {
        assertArrayEquals(reference.embed(TEXTS.get(i)), actual[i], 1e-5f,
            "input " + i + " under " + padding);
      }
    }
  }

  /**
   * An id no configurer answers to fails while the component is being constructed, naming the id,
   * and no session is left behind.
   */
  @Test
  void testUnknownExecutionProviderIdFailsAtConstruction(@TempDir final Path dir) throws Exception {
    final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
        () -> vectors(dir, PaddingStrategy.LONGEST,
            requesting(ExecutionProviderRequest.of("tensorrt"))));

    assertTrue(failure.getMessage().contains("tensorrt"), failure.getMessage());
  }

  /**
   * An explicit CUDA request reaches ONNX Runtime, which is observed by asking for a device no
   * machine answers to. This is the case that fails if the request list is ever dropped on the way
   * to the session.
   */
  @Test
  void testExplicitCudaRequestReachesTheRuntime(@TempDir final Path dir) throws Exception {
    final OrtException failure = assertThrows(OrtException.class,
        () -> vectors(dir, PaddingStrategy.LONGEST, requesting(cuda(UNUSABLE_DEVICE_ID))));

    final String message = failure.getMessage();
    assertTrue(message.contains("CUDA") || message.contains("cuda") || message.contains("device"),
        "the failure should name the execution provider or the device: " + message);
  }

  /**
   * The deprecated flag and an explicit CUDA request are the same configuration, observed at the
   * component: both reach ONNX Runtime and both are rejected the same way for the same unusable
   * device.
   */
  @Test
  @SuppressWarnings({"deprecation", "removal"})
  void testDeprecatedFlagAndExplicitRequestFailTheSameWay(@TempDir final Path dir)
      throws Exception {
    final InferenceOptions deprecated = new InferenceOptions();
    deprecated.setGpu(true);
    deprecated.setGpuDeviceId(UNUSABLE_DEVICE_ID);

    final OrtException fromFlag = assertThrows(OrtException.class,
        () -> vectors(dir, PaddingStrategy.LONGEST, deprecated));
    final OrtException fromList = assertThrows(OrtException.class,
        () -> vectors(dir, PaddingStrategy.LONGEST, requesting(cuda(UNUSABLE_DEVICE_ID))));

    assertEquals(fromList.getMessage(), fromFlag.getMessage());
  }

  /**
   * The request list wins over the deprecated flag, and the observable consequence is that the
   * component constructs: the flag names a device no machine answers to, so a configuration that
   * read it would fail here instead.
   */
  @Test
  @SuppressWarnings({"deprecation", "removal"})
  void testTheListWinsOverTheDeprecatedFlagAtConstruction(@TempDir final Path dir)
      throws Exception {
    final InferenceOptions options =
        requesting(ExecutionProviderRequest.of(ExecutionProviders.CPU));
    options.setGpu(true);
    options.setGpuDeviceId(UNUSABLE_DEVICE_ID);

    try (SentenceVectorsDL constructed = vectors(dir, PaddingStrategy.LONGEST, options)) {
      assertEquals(3, constructed.dimension());
    }
  }

  /**
   * The vectors must not depend on the execution provider. This runs the same inputs on the CPU and
   * on a CUDA device requested through the new list, and needs a working device, so it is skipped
   * where there is none.
   */
  @Test
  void testExplicitCudaRequestProducesTheSameVectorsAsTheCpu(@TempDir final Path dir)
      throws Exception {
    Assumptions.assumeTrue(cudaUsable(), "a usable CUDA device is needed");
    try (SentenceVectorsDL cpu = vectors(dir, PaddingStrategy.LONGEST, new InferenceOptions());
         SentenceVectorsDL onCuda = vectors(dir, PaddingStrategy.LONGEST,
             requesting(cuda(0), ExecutionProviderRequest.of(ExecutionProviders.CPU)))) {
      final float[][] expected = cpu.embedAll(TEXTS);
      final float[][] actual = onCuda.embedAll(TEXTS);
      assertEquals(expected.length, actual.length);
      for (int i = 0; i < expected.length; i++) {
        assertArrayEquals(expected[i], actual[i], 1e-5f, "input " + i);
      }
    }
  }
}
