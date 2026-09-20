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

import opennlp.dl.ExecutionProviderRequest;
import opennlp.dl.InferenceOptions;
import opennlp.dl.openvino.OpenVinoExecutionProviderConfigurer;
import opennlp.dl.openvino.OpenVinoPlacementProbeConfigurer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The padding strategy a running {@link SentenceVectorsDL} ends up with when its execution provider
 * is the OpenVINO addon: derived from the device type, and overridden by a strategy the caller names.
 *
 * <p>The sessions here are created with {@link OpenVinoPlacementProbeConfigurer}, which derives its
 * placement with the real {@link OpenVinoExecutionProviderConfigurer} and then adds a session config
 * entry in place of the execution provider. That substitution is needed because no OpenVINO runtime
 * can be had: {@link #testNoOpenVinoRuntimeIsAvailableHere()} records what the pinned ONNX Runtime
 * does with a request for the real id. So what these tests establish is that the derivation reaches a
 * component and shapes its tensors, not that OpenVINO ran anything.</p>
 *
 * <p>The batch plan is read through {@link SentenceVectorsDL#batchShapes(List)}, as
 * {@code PaddingStrategyDefaultTest} in {@code opennlp-dl} reads it, because the strategies differ in
 * the shape of the tensors and the number of inferences while returning the same vectors. This class
 * sits in the package of that class for the same reason it does.</p>
 */
class OpenVinoPaddingDefaultTest {

  /** Inputs of tokenized length 3, 4, 5, 3, 4, 3, as in {@code PaddingStrategyDefaultTest}. */
  private static final List<String> DISTINCT = List.of("x", "hello hello",
      "hello hello world", "hello", "hello world", "world");

  /** One inference per distinct tokenized length, with no padding, in first-seen order. */
  private static final int[][] EXACT_SHAPES = {{3, 3}, {2, 4}, {1, 5}};

  /** One inference for the entire call, at the longest row of it. */
  private static final int[][] LONGEST_SHAPES = {{6, 5}};

  private static final int MAX_LENGTH = 8;

  private static final String DEVICE_TYPE = OpenVinoExecutionProviderConfigurer.DEVICE_TYPE_OPTION;

  // Copied out of the classpath rather than resolved in place, because it arrives in the opennlp-dl
  // test-jar, where the resource URI is not hierarchical.
  private static File model(final Path dir) throws IOException {
    final Path file = dir.resolve("tiny-vectors.onnx");
    try (InputStream is = Objects.requireNonNull(OpenVinoPaddingDefaultTest.class
        .getResourceAsStream("/opennlp/dl/vectors/tiny-vectors.onnx"))) {
      Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
    }
    return file.toFile();
  }

  /** The vocabulary of the vector tests of {@code opennlp-dl}, with {@code [PAD]} at id {@code 0}. */
  private static File vocab(final Path dir) throws IOException {
    final Path file = dir.resolve("vocab.txt");
    Files.write(file, List.of("[PAD]", "unused1", "[UNK]", "[SEP]", "hello", "world",
        "unused2", "[CLS]"));
    return file.toFile();
  }

  /** Options asking for the probe on one device type, or on none if {@code deviceType} is null. */
  private static InferenceOptions probing(final String deviceType) {
    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(List.of(deviceType == null
        ? ExecutionProviderRequest.of(OpenVinoPlacementProbeConfigurer.ID)
        : ExecutionProviderRequest.of(OpenVinoPlacementProbeConfigurer.ID,
            Map.of(DEVICE_TYPE, deviceType))));
    return options;
  }

  private static SentenceVectorsDL derived(final Path dir, final String deviceType)
      throws Exception {
    return SentenceVectorsDL.withDerivedPadding(model(dir), vocab(dir), true, Pooling.MEAN, false,
        MAX_LENGTH, probing(deviceType));
  }

  private static SentenceVectorsDL stated(final Path dir, final PaddingStrategy padding,
      final String deviceType) throws Exception {
    return new SentenceVectorsDL(model(dir), vocab(dir), true, Pooling.MEAN, false, MAX_LENGTH,
        padding, probing(deviceType));
  }

  /**
   * A GPU device type derives padding to the longest row of the batch, so one call of mixed-length
   * inputs becomes one inference. This is the case the stage exists for, and the case an execution
   * provider id on its own could not reach.
   */
  @Test
  void testAGpuDeviceTypeDerivesPaddingToTheLongestRow(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL onAGpu = derived(dir, "GPU.0");
         SentenceVectorsDL onAnNpu = derived(dir, "NPU")) {

      assertArrayEquals(LONGEST_SHAPES, onAGpu.batchShapes(DISTINCT));
      assertArrayEquals(LONGEST_SHAPES, onAnNpu.batchShapes(DISTINCT));
    }
  }

  /** A CPU device type derives grouping by exact tokenized length, on the very same id. */
  @Test
  void testACpuDeviceTypeDerivesExactLengthGrouping(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL onTheCpu = derived(dir, "CPU")) {
      assertArrayEquals(EXACT_SHAPES, onTheCpu.batchShapes(DISTINCT));
    }
  }

  /**
   * A device type that decides nothing leaves the conservative default in place, so the addon on the
   * class path changes no vectors until a device type asks it to.
   */
  @Test
  void testAnUndecidedDeviceTypeKeepsTheConservativeDefault(@TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL onNoDevice = derived(dir, null);
         SentenceVectorsDL onAuto = derived(dir, "AUTO")) {

      assertArrayEquals(EXACT_SHAPES, onNoDevice.batchShapes(DISTINCT));
      assertArrayEquals(EXACT_SHAPES, onAuto.batchShapes(DISTINCT));
    }
  }

  /**
   * A strategy the caller states wins over the derived one in both directions, so the derivation
   * never takes a choice away.
   */
  @Test
  void testAStatedStrategyWinsOverTheDerivedOne(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL exactOnAGpu = stated(dir, PaddingStrategy.EXACT_LENGTH, "GPU.0");
         SentenceVectorsDL longestOnTheCpu = stated(dir, PaddingStrategy.LONGEST, "CPU")) {

      assertArrayEquals(EXACT_SHAPES, exactOnAGpu.batchShapes(DISTINCT));
      assertArrayEquals(LONGEST_SHAPES, longestOnTheCpu.batchShapes(DISTINCT));
    }
  }

  /**
   * What the real id does here, which is why the tests above substitute the registration: the ONNX
   * Runtime this build pins reports {@code CPU} as its only execution provider and fails to find the
   * OpenVINO shared provider library, whatever the device type is. Neither published runtime artifact
   * carries one, so a component asking for the real id does not get a session at all.
   *
   * <p>This is skipped where a runtime built with the OpenVINO execution provider is on the class
   * path, which is what a deployment that wants this module supplies and where the substitution above
   * would no longer be needed.</p>
   */
  @Test
  void testNoOpenVinoRuntimeIsAvailableHere(@TempDir final Path dir) throws Exception {
    Assumptions.assumeFalse(OrtEnvironment.getAvailableProviders().contains(OrtProvider.OPEN_VINO),
        "this ONNX Runtime was built with the OpenVINO execution provider");

    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(List.of(ExecutionProviderRequest
        .of(OpenVinoExecutionProviderConfigurer.ID, Map.of(DEVICE_TYPE, "CPU"))));

    final OrtException failure = assertThrows(OrtException.class,
        () -> SentenceVectorsDL.withDerivedPadding(model(dir), vocab(dir), true, Pooling.MEAN,
            false, MAX_LENGTH, options));

    assertTrue(failure.getMessage().contains("OpenVINO"), failure.getMessage());
  }

  /** The same failure without a component around it, so the message is on the record on its own. */
  @Test
  void testTheRuntimeRejectsTheProviderRatherThanFallingBackToTheCpu() throws Exception {
    Assumptions.assumeFalse(OrtEnvironment.getAvailableProviders().contains(OrtProvider.OPEN_VINO),
        "this ONNX Runtime was built with the OpenVINO execution provider");

    OrtEnvironment.getEnvironment();
    try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
      final OrtException failure =
          assertThrows(OrtException.class, () -> options.addOpenVINO("GPU.0"));

      assertTrue(failure.getMessage().contains("OpenVINO"), failure.getMessage());
    }
  }
}
