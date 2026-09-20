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

import opennlp.dl.InferenceOptions;
import opennlp.dl.SessionOptionsProbe;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which ONNX Runtime execution provider {@link SentenceVectorsDL} actually runs on.
 *
 * <p>Until the {@link InferenceOptions} constructor was added this class built its session with a
 * hardcoded default {@code OrtSession.SessionOptions}, so it could only ever run on the CPU even
 * though {@code opennlp-dl-gpu} advertises GPU acceleration. That failure was silent, so a test
 * asserting only that construction succeeds proves nothing. These tests observe the registered
 * provider instead.</p>
 *
 * <p><b>What onnxruntime 1.29.0 lets a test observe.</b> {@code OrtSession} has no
 * {@code getProviders()} and {@code OrtSession.SessionOptions} has no getter for the providers
 * added to it; {@code getConfigEntries()} stays empty because {@code addCUDA} is a native call
 * that does not go through the config map. {@link OrtEnvironment#getAvailableProviders()} reports
 * what the loaded native library was built with, not what a session registered, and it lists
 * {@link OrtProvider#CUDA} even on a machine whose CUDA installation the runtime cannot load. The
 * provider of a session is therefore observed here the only runtime-independent way there is:
 * through a request ONNX Runtime must reject if, and only if, it reached the runtime. An
 * unusable CUDA device id is such a request. On the CPU-only {@code onnxruntime} artifact it
 * fails while the options are being built ({@code ORT_EP_FAIL}, "Failed to find CUDA shared
 * provider"); on the {@code onnxruntime_gpu} artifact it fails while the session is being created
 * ({@code ORT_FAIL}, "invalid device ordinal"), which was observed on the 1.27.0 GPU build. Either
 * way it throws {@link OrtException}, and either way it cannot throw unless the constructor passed
 * the caller's choice through. ONNX Runtime was verified not to fall back to the CPU in any of
 * those cases, so OpenNLP needs no fallback detection of its own. The assertions below check that
 * an {@link OrtException} names the provider or the device rather than matching either exact
 * message, so a differently worded rejection from another build still passes.</p>
 *
 * <p>The cases that need a working CUDA device are skipped unless one is present. They do run in
 * {@code opennlp-dl-gpu}, which puts {@code onnxruntime_gpu} on the classpath and rescans this
 * test-jar.</p>
 */
class SentenceVectorsDLExecutionProviderTest {

  /**
   * A device id no machine this runs on answers to, used as the tracer that proves the caller's
   * {@link InferenceOptions} reached ONNX Runtime. It is a plausible-looking id rather than
   * {@code Integer.MAX_VALUE}: onnxruntime 1.27.0 rejects the extreme value while it is still
   * building the options, with the same "Failed to load shared library" message a missing CUDA
   * installation gives, whereas an ordinary out-of-range id reaches the device and is rejected as
   * an "invalid device ordinal" when the session is created, which is the behavior worth pinning
   * down.
   */
  private static final int UNUSABLE_DEVICE_ID = 99;

  /** The tolerance vectors from two execution providers are compared to. */
  private static final float DELTA = 1e-5f;

  private static final List<String> TEXTS =
      List.of("hello world", "hello", "world", "hello hello world", "");

  // Copied out of the classpath rather than resolved in place, because this test also runs from
  // the opennlp-dl test-jar in opennlp-dl-gpu, where the resource URI is not hierarchical.
  private static File model(final Path dir) throws IOException {
    final Path file = dir.resolve("tiny-vectors.onnx");
    try (InputStream is = Objects.requireNonNull(SentenceVectorsDLExecutionProviderTest.class
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
    return new SentenceVectorsDL(model(dir), vocab(dir), true, Pooling.MEAN, false,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, padding, inferenceOptions);
  }

  private static InferenceOptions gpu(final int deviceId) {
    final InferenceOptions options = new InferenceOptions();
    options.setGpu(true);
    options.setGpuDeviceId(deviceId);
    return options;
  }

  /**
   * {@return whether a CUDA session can actually be created on device {@code 0} here}
   *
   * <p>This is deliberately not {@code getAvailableProviders().contains(CUDA)}: that is true of
   * the {@code onnxruntime_gpu} artifact whatever CUDA the machine has, including a version whose
   * {@code libonnxruntime_providers_cuda.so} will not link. Building the options is the cheapest
   * step that tells the two apart.</p>
   *
   * <p>The environment is fetched first for the same reason
   * {@code AbstractDL.sessionOptions(InferenceOptions)} fetches it: ONNX Runtime cannot load a
   * shared execution provider library before its default logger exists, and it remembers the
   * failure, so a single probe in the wrong order would make every later CUDA request in this JVM
   * fail.</p>
   */
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
   * The tracer itself: requesting a GPU reaches ONNX Runtime and is rejected loudly. This is the
   * case that fails if {@link SentenceVectorsDL} ever stops honoring {@link InferenceOptions},
   * whichever ONNX Runtime artifact is on the classpath.
   */
  @Test
  void testGpuRequestReachesTheRuntimeAndFailsLoudly(@TempDir final Path dir) throws Exception {
    final OrtException failure = assertThrows(OrtException.class,
        () -> vectors(dir, SentenceVectorsDL.DEFAULT_PADDING, gpu(UNUSABLE_DEVICE_ID)));
    assertNotNull(failure.getMessage());
    // Whichever of the two sites rejects it, the message names CUDA or the device.
    final String message = failure.getMessage();
    assertTrue(message.contains("CUDA") || message.contains("cuda")
            || message.contains("device"),
        "the failure should name the execution provider or the device: " + message);
  }

  /**
   * The other half of the tracer: nothing on the default path asks for a GPU. Every constructor
   * that existed before {@link InferenceOptions} was accepted, and a default
   * {@link InferenceOptions}, all construct even with a device id no card answers to, because
   * that id is never read. If the default ever became the GPU, the CPU-only artifact would fail
   * here.
   */
  @Test
  void testDefaultSelectsTheCpuProvider(@TempDir final Path dir) throws Exception {
    final InferenceOptions cpuWithBadDeviceId = new InferenceOptions();
    cpuWithBadDeviceId.setGpuDeviceId(UNUSABLE_DEVICE_ID);
    assertFalse(cpuWithBadDeviceId.isGpu(), "a fresh InferenceOptions must not select the GPU");
    try (SentenceVectorsDL explicit = vectors(dir, SentenceVectorsDL.DEFAULT_PADDING,
             cpuWithBadDeviceId);
         SentenceVectorsDL twoArg = new SentenceVectorsDL(model(dir), vocab(dir));
         SentenceVectorsDL threeArg = new SentenceVectorsDL(model(dir), vocab(dir), true);
         SentenceVectorsDL sixArg = new SentenceVectorsDL(model(dir), vocab(dir), true,
             Pooling.MEAN, false, SentenceVectorsDL.DEFAULT_MAX_LENGTH);
         SentenceVectorsDL sevenArg = new SentenceVectorsDL(model(dir), vocab(dir), true,
             Pooling.MEAN, false, SentenceVectorsDL.DEFAULT_MAX_LENGTH,
             PaddingStrategy.LONGEST)) {
      assertEquals(3, explicit.dimension());
      assertEquals(3, twoArg.dimension());
      assertEquals(3, threeArg.dimension());
      assertEquals(3, sixArg.dimension());
      assertEquals(3, sevenArg.dimension());
    }
  }

  /**
   * A default {@link InferenceOptions} configures the session exactly as the hardcoded default
   * options did, so the constructor that existed before is bit-identical for its callers. The
   * vectors are compared with no tolerance at all: a different session configuration would have
   * to produce the same bits.
   */
  @Test
  void testDefaultInferenceOptionsMatchesHardcodedDefaultOptions(@TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL before = new SentenceVectorsDL(model(dir), vocab(dir), true,
             Pooling.MEAN, false, SentenceVectorsDL.DEFAULT_MAX_LENGTH, PaddingStrategy.LONGEST);
         SentenceVectorsDL after = vectors(dir, PaddingStrategy.LONGEST,
             new InferenceOptions())) {
      final float[][] expected = before.embedAll(TEXTS);
      final float[][] actual = after.embedAll(TEXTS);
      assertEquals(expected.length, actual.length);
      for (int i = 0; i < expected.length; i++) {
        assertArrayEquals(expected[i], actual[i], 0f,
            "input " + i + " must be bit-identical on the default path");
      }
    }
    // And the session options the two paths hand to ONNX Runtime carry the same configuration.
    try (OrtSession.SessionOptions hardcoded = new OrtSession.SessionOptions();
         OrtSession.SessionOptions fromOptions = SessionOptionsProbe.of(new InferenceOptions())) {
      assertEquals(hardcoded.getConfigEntries(), fromOptions.getConfigEntries());
    }
  }

  /** A {@code null} {@link InferenceOptions} is rejected before any session is created. */
  @Test
  void testNullInferenceOptionsRejected(@TempDir final Path dir) throws Exception {
    assertThrows(IllegalArgumentException.class,
        () -> vectors(dir, SentenceVectorsDL.DEFAULT_PADDING, null));
  }

  /**
   * {@link InferenceOptions#setLowerCase(boolean)} wins over the {@code lowerCase} parameter when
   * it is set, and the parameter stands when it is not. The tiny vocabulary has only the lower
   * case {@code hello}, so upper case input is {@code [UNK]} unless the text is lower cased.
   */
  @Test
  void testInferenceOptionsLowerCaseOverridesTheParameter(@TempDir final Path dir)
      throws Exception {
    final InferenceOptions lowerCasing = new InferenceOptions();
    lowerCasing.setLowerCase(true);
    try (SentenceVectorsDL parameterOnly = vectors(dir, PaddingStrategy.LONGEST,
             new InferenceOptions());
         SentenceVectorsDL overridden = new SentenceVectorsDL(model(dir), vocab(dir), false,
             Pooling.MEAN, false, SentenceVectorsDL.DEFAULT_MAX_LENGTH, PaddingStrategy.LONGEST,
             lowerCasing)) {
      assertArrayEquals(parameterOnly.embed("HELLO"), overridden.embed("HELLO"), 0f,
          "the InferenceOptions setting must decide the lower casing");
    }
  }

  /**
   * The vectors must not depend on the execution provider. This runs the same inputs on the CPU
   * and on the GPU and compares them to {@link #DELTA}, which is the assertion that matters most:
   * a GPU that produced different numbers would be worse than one that was never used.
   */
  @Test
  void testGpuProducesTheSameVectorsAsTheCpu(@TempDir final Path dir) throws Exception {
    Assumptions.assumeTrue(cudaUsable(), "a usable CUDA device is needed");
    try (SentenceVectorsDL cpu = vectors(dir, PaddingStrategy.LONGEST, new InferenceOptions());
         SentenceVectorsDL cuda = vectors(dir, PaddingStrategy.LONGEST, gpu(0))) {
      final float[][] expected = cpu.embedAll(TEXTS);
      final float[][] actual = cuda.embedAll(TEXTS);
      assertEquals(expected.length, actual.length);
      for (int i = 0; i < expected.length; i++) {
        assertArrayEquals(expected[i], actual[i], DELTA,
            "input " + i + " must embed the same on both providers");
      }
    }
  }

  /**
   * Every {@link PaddingStrategy} composes with the GPU: the strategy shapes the tensors and the
   * provider runs them, and neither choice changes the vectors. The reference is a CPU instance
   * under {@link PaddingStrategy#EXACT_LENGTH}, which pads nothing, so a padding bug cannot cancel
   * itself out.
   *
   * @param padding The strategy under test.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testEveryPaddingStrategyRunsOnTheGpu(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    Assumptions.assumeTrue(cudaUsable(), "a usable CUDA device is needed");
    try (SentenceVectorsDL reference = new SentenceVectorsDL(model(dir), vocab(dir), true,
             Pooling.MEAN, false, 16, PaddingStrategy.EXACT_LENGTH, new InferenceOptions());
         SentenceVectorsDL cuda = new SentenceVectorsDL(model(dir), vocab(dir), true,
             Pooling.MEAN, false, 16, padding, gpu(0))) {
      final float[][] actual = cuda.embedAll(TEXTS);
      for (int i = 0; i < TEXTS.size(); i++) {
        assertArrayEquals(reference.embed(TEXTS.get(i)), actual[i], DELTA,
            "input " + i + " under " + padding + " on the GPU");
      }
    }
  }

  /**
   * A device id no card answers to fails when the session is created rather than falling back to
   * another card or to the CPU. This case only asserts anything where CUDA is actually usable;
   * {@code testGpuRequestReachesTheRuntimeAndFailsLoudly} covers the CPU-only artifact, where
   * the same request is rejected earlier.
   */
  @Test
  void testUnusableDeviceIdFailsRatherThanFallingBack(@TempDir final Path dir) throws Exception {
    Assumptions.assumeTrue(cudaUsable(), "a usable CUDA device is needed");
    final OrtException failure = assertThrows(OrtException.class,
        () -> vectors(dir, SentenceVectorsDL.DEFAULT_PADDING, gpu(UNUSABLE_DEVICE_ID)));
    assertTrue(failure.getMessage().contains("device"),
        "the failure should name the device: " + failure.getMessage());
  }

}
