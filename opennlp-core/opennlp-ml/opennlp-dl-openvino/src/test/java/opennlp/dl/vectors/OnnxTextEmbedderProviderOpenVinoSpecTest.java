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
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import opennlp.dl.openvino.OpenVinoExecutionProviderConfigurer;
import opennlp.dl.openvino.OpenVinoPlacementProbeConfigurer;
import opennlp.tools.embeddings.TextEmbedder;
import opennlp.tools.util.ext.ProviderSpec;

import static opennlp.dl.vectors.OnnxTextEmbedderProvider.EXECUTION_PROVIDERS_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.VOCABULARY_OPTION;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The OpenVINO addon requested from a {@link ProviderSpec}, which is the gap this stage closed: an
 * addon could register an execution provider and no spec could ask for it.
 *
 * <p>The device type travels as a provider option of one request inside the
 * {@value OnnxTextEmbedderProvider#EXECUTION_PROVIDERS_OPTION} value, so this is also where the
 * per-request part of that syntax is exercised against a real addon configurer rather than a test
 * double: the configurer accepts {@value OpenVinoExecutionProviderConfigurer#DEVICE_TYPE_OPTION} and
 * nothing else, and it derives a placement from the value, which becomes the padding strategy of the
 * constructed embedder.</p>
 *
 * <p>The sessions that are actually created go through
 * {@link OpenVinoPlacementProbeConfigurer}, for the reason that class documents: no ONNX Runtime with
 * the OpenVINO execution provider can be had here, so a session on the real id cannot be created at
 * all. {@link #testTheRealIdFromASpecReachesTheRuntimeAndFailsLoudly} records what the real id does
 * instead of assuming it.</p>
 */
class OnnxTextEmbedderProviderOpenVinoSpecTest {

  /** Inputs of tokenized length 3, 4, 5, 3, 4, 3, as in {@code OpenVinoPaddingDefaultTest}. */
  private static final List<String> DISTINCT = List.of("x", "hello hello",
      "hello hello world", "hello", "hello world", "world");

  /** One inference per distinct tokenized length, with no padding, in first-seen order. */
  private static final int[][] EXACT_SHAPES = {{3, 3}, {2, 4}, {1, 5}};

  /** One inference for the entire call, at the longest row of it. */
  private static final int[][] LONGEST_SHAPES = {{6, 5}};

  private static final String DEVICE_TYPE = OpenVinoExecutionProviderConfigurer.DEVICE_TYPE_OPTION;

  private final OnnxTextEmbedderProvider provider = new OnnxTextEmbedderProvider();

  // Copied out of the classpath rather than resolved in place, because it arrives in the opennlp-dl
  // test-jar, where the resource URI is not hierarchical.
  private static Path model(final Path dir) throws IOException {
    final Path file = dir.resolve("tiny-vectors.onnx");
    try (InputStream is = Objects.requireNonNull(OnnxTextEmbedderProviderOpenVinoSpecTest.class
        .getResourceAsStream("/opennlp/dl/vectors/tiny-vectors.onnx"))) {
      Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
    }
    return file;
  }

  /** The vocabulary of the vector tests of {@code opennlp-dl}, with {@code [PAD]} at id {@code 0}. */
  private static void vocab(final Path dir) throws IOException {
    Files.write(dir.resolve("vocab.txt"), List.of("[PAD]", "unused1", "[UNK]", "[SEP]", "hello",
        "world", "unused2", "[CLS]"));
  }

  /** A spec for the model in {@code dir} asking for one execution provider list. */
  private static ProviderSpec spec(final Path dir, final String executionProviders)
      throws IOException {
    vocab(dir);
    final Map<String, String> options = new LinkedHashMap<>();
    options.put(VOCABULARY_OPTION, "vocab.txt");
    options.put(EXECUTION_PROVIDERS_OPTION, executionProviders);
    return ProviderSpec.of(model(dir), options);
  }

  /** A request for the placement probe on one device type. */
  private static String probing(final String deviceType) {
    return OpenVinoPlacementProbeConfigurer.ID + "(" + DEVICE_TYPE + "=" + deviceType + ")";
  }

  /**
   * A GPU device type named in a spec derives padding to the longest row of the batch, so one call of
   * mixed-length inputs becomes one inference, and a CPU device type derives the grouping by exact
   * tokenized length. Same spec option, same addon, two batch plans, decided by a provider option
   * that only the new syntax can carry.
   */
  @Test
  void testTheDeviceTypeFromASpecDecidesTheBatchPlan(@TempDir final Path dir) throws Exception {
    final TextEmbedder onAGpu = provider.create(spec(dir, probing("GPU.0")));
    final TextEmbedder onACpu = provider.create(spec(dir, probing("CPU")));
    try (SentenceVectorsDL gpu = assertInstanceOf(SentenceVectorsDL.class, onAGpu);
         SentenceVectorsDL cpu = assertInstanceOf(SentenceVectorsDL.class, onACpu)) {
      assertArrayEquals(LONGEST_SHAPES, gpu.batchShapes(DISTINCT));
      assertArrayEquals(EXACT_SHAPES, cpu.batchShapes(DISTINCT));
      assertArrayEquals(cpu.embed("hello world"), gpu.embed("hello world"), 1e-5f,
          "the plan differs and the vectors do not");
    }
  }

  /** An NPU is an accelerator too, and the fallback behind it is an ordinary further list element. */
  @Test
  void testAnOrderedListWithTheAddonInFront(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL vectors = assertInstanceOf(SentenceVectorsDL.class,
        provider.create(spec(dir, probing("NPU") + ",cpu")))) {
      assertArrayEquals(LONGEST_SHAPES, vectors.batchShapes(DISTINCT),
          "the first element of the list decides the placement");
    }
  }

  /**
   * The CPU in front of the addon makes the session a CPU session, because ONNX Runtime offers each
   * node to the execution providers in the order they were appended and the CPU accepts every one.
   */
  @Test
  void testTheOrderOfTheListIsRead(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL vectors = assertInstanceOf(SentenceVectorsDL.class,
        provider.create(spec(dir, "cpu," + probing("GPU.0"))))) {
      assertArrayEquals(EXACT_SHAPES, vectors.batchShapes(DISTINCT));
    }
  }

  /**
   * A provider option the real addon configurer does not accept is reported by the configurer, and so
   * is a blank device type, so the per-request options of a spec reach the addon's own validation
   * rather than being dropped. Both fail before {@code addOpenVINO} is called, which is why they are
   * the two cases the real id can carry here.
   */
  @Test
  void testTheAddonValidatesTheProviderOptionsOfASpec(@TempDir final Path dir) throws Exception {
    final IllegalArgumentException foreign = assertThrows(IllegalArgumentException.class,
        () -> provider.create(spec(dir,
            OpenVinoExecutionProviderConfigurer.ID + "(cache_dir=/var/cache/ov)")));
    assertTrue(foreign.getMessage().contains(DEVICE_TYPE), foreign.getMessage());
    final IllegalArgumentException blank = assertThrows(IllegalArgumentException.class,
        () -> provider.create(spec(dir,
            OpenVinoExecutionProviderConfigurer.ID + "(" + DEVICE_TYPE + "= )")));
    assertTrue(blank.getMessage().contains(DEVICE_TYPE), blank.getMessage());
  }

  /**
   * The real {@value OpenVinoExecutionProviderConfigurer#ID} id from a spec reaches ONNX Runtime and
   * fails there, rather than being dropped or falling back to the CPU. That failure is the whole
   * evidence available here that the id travels from a spec to the runtime, since the pinned runtime
   * carries no OpenVINO execution provider.
   */
  @Test
  void testTheRealIdFromASpecReachesTheRuntimeAndFailsLoudly(@TempDir final Path dir)
      throws Exception {
    final IOException failure = assertThrows(IOException.class, () -> provider.create(
        spec(dir, OpenVinoExecutionProviderConfigurer.ID + "(" + DEVICE_TYPE + "=CPU)")));
    final String message = String.valueOf(failure.getCause());
    assertTrue(message.contains("OpenVINO") || message.contains("openvino"),
        "the failure must name the execution provider that could not be registered: " + message);
  }
}
