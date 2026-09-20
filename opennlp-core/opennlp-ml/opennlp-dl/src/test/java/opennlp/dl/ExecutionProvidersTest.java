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

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OrtException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * How {@link ExecutionProviders} resolves requests and what it does to a session's options.
 *
 * <p>The assertions observe the ONNX Runtime calls a configuration makes, through
 * {@link RecordingSessionOptions}, rather than reading the source back. That is the only way to
 * observe the order in onnxruntime 1.29.0: nothing on {@code OrtSession.SessionOptions} reports
 * which execution providers were appended to it, and a session cannot be asked either. It also means
 * these cases run on a machine without any accelerator, since a recorded call reaches no native
 * layer.</p>
 */
class ExecutionProvidersTest {

  /** A device id no machine answers to, used where a request has to reach the runtime. */
  private static final int UNUSABLE_DEVICE_ID = 99;

  private static final String DEVICE_ID = CudaExecutionProviderConfigurer.DEVICE_ID_OPTION;
  private static final String USE_ARENA = CpuExecutionProviderConfigurer.USE_ARENA_OPTION;

  private static InferenceOptions with(final ExecutionProviderRequest... requests) {
    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(List.of(requests));
    return options;
  }

  @SuppressWarnings({"deprecation", "removal"})
  private static InferenceOptions gpu(final int deviceId) {
    final InferenceOptions options = new InferenceOptions();
    options.setGpu(true);
    options.setGpuDeviceId(deviceId);
    return options;
  }

  // Resolution and precedence.

  /** A fresh {@link InferenceOptions} asks for nothing, which is what keeps the old path intact. */
  @Test
  void testDefaultOptionsResolveToNoRequest() {
    assertEquals(List.of(), ExecutionProviders.resolve(new InferenceOptions()));
  }

  @Test
  void testRequestedListResolvesToItself() {
    final List<ExecutionProviderRequest> requests = List.of(
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1")),
        ExecutionProviderRequest.of(ExecutionProviders.CPU));

    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(requests);

    assertEquals(requests, ExecutionProviders.resolve(options));
  }

  /**
   * The deprecated flag maps to exactly one CUDA request naming the configured device, which is what
   * makes the old path and an explicit request the same configuration rather than two similar ones.
   */
  @Test
  void testDeprecatedGpuFlagResolvesToACudaRequest() {
    assertEquals(
        List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "0"))),
        ExecutionProviders.resolve(gpu(0)));
    assertEquals(
        List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "3"))),
        ExecutionProviders.resolve(gpu(3)));
  }

  /** An unset flag leaves the device id unread, so a stale device id selects nothing. */
  @Test
  @SuppressWarnings({"deprecation", "removal"})
  void testDeviceIdWithoutTheFlagResolvesToNoRequest() {
    final InferenceOptions options = new InferenceOptions();
    options.setGpuDeviceId(UNUSABLE_DEVICE_ID);

    assertEquals(List.of(), ExecutionProviders.resolve(options));
  }

  /**
   * The list wins over the deprecated flag, and the observable consequence is that no CUDA call is
   * made at all: an unusable device id that the flag names never reaches ONNX Runtime.
   */
  @Test
  @SuppressWarnings({"deprecation", "removal"})
  void testRequestedListWinsOverTheDeprecatedFlag() throws Exception {
    final InferenceOptions options = gpu(UNUSABLE_DEVICE_ID);
    options.setExecutionProviders(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU)));

    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU)),
        ExecutionProviders.resolve(options));
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded, ExecutionProviders.resolve(options));
      assertEquals(List.of("addCPU(true)"), recorded.calls());
    }
  }

  /** Emptying the list again brings the deprecated flag back, so the precedence has no memory. */
  @Test
  void testEmptyingTheListRestoresTheDeprecatedFlag() {
    final InferenceOptions options = gpu(2);
    options.setExecutionProviders(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU)));
    options.setExecutionProviders(List.of());

    assertEquals(
        List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "2"))),
        ExecutionProviders.resolve(options));
  }

  @Test
  void testResolveRejectsNull() {
    assertThrows(IllegalArgumentException.class, () -> ExecutionProviders.resolve(null));
  }

  // Order.

  /**
   * The requests are appended in the order given, which is the order ONNX Runtime falls back along.
   * The same three ids in another order must produce the calls in that other order, so the test
   * cannot pass on a configuration that ignores the order.
   */
  @Test
  void testRequestsAreAppliedInTheOrderGiven() throws Exception {
    final List<ExecutionProviderRequest> forward = List.of(
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1")),
        ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID),
        ExecutionProviderRequest.of(ExecutionProviders.CPU, Map.of(USE_ARENA, "false")));
    final List<ExecutionProviderRequest> reversed = new ArrayList<>(forward);
    Collections.reverse(reversed);

    try (RecordingSessionOptions first = new RecordingSessionOptions();
         RecordingSessionOptions second = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(first, forward);
      ExecutionProviders.addTo(second, reversed);

      assertEquals(List.of("addCUDA(1)",
          "addConfigEntry(" + EchoExecutionProviderConfigurer.PREFIX + "ran="
              + EchoExecutionProviderConfigurer.ID + ")",
          "addCPU(false)"), first.calls());
      assertEquals(List.of("addCPU(false)",
          "addConfigEntry(" + EchoExecutionProviderConfigurer.PREFIX + "ran="
              + EchoExecutionProviderConfigurer.ID + ")",
          "addCUDA(1)"), second.calls());
    }
  }

  /** The same id may appear more than once, which is how two devices of one kind are requested. */
  @Test
  void testTheSameIdMayBeRequestedTwice() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded, List.of(
          ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "0")),
          ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1"))));

      assertEquals(List.of("addCUDA(0)", "addCUDA(1)"), recorded.calls());
    }
  }

  /**
   * An empty list touches the session options not at all, which is what keeps a default
   * {@link InferenceOptions} identical to the hardcoded default options of the code before
   * execution providers could be chosen.
   */
  @Test
  void testEmptyListMakesNoCall() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded, List.of());

      assertEquals(List.of(), recorded.calls());
      assertEquals(Map.of(), recorded.getConfigEntries());
    }
  }

  @Test
  void testAddToRejectsNullArgumentsAndNullElements() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      assertThrows(IllegalArgumentException.class,
          () -> ExecutionProviders.addTo(null, List.of()));
      assertThrows(IllegalArgumentException.class,
          () -> ExecutionProviders.addTo(recorded, null));
      final List<ExecutionProviderRequest> withNull = new ArrayList<>();
      withNull.add(null);
      assertThrows(IllegalArgumentException.class,
          () -> ExecutionProviders.addTo(recorded, withNull));
    }
  }

  // The loud failure.

  /**
   * An id no configurer answers to fails by name, and the message lists what is registered, because
   * the usual cause is a missing addon jar or a typo and neither one is guessable from "unknown
   * provider".
   */
  @Test
  void testUnknownIdFailsNamingTheIdAndWhatIsRegistered() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      final IllegalArgumentException failure =
          assertThrows(IllegalArgumentException.class, () -> ExecutionProviders.addTo(recorded,
              List.of(ExecutionProviderRequest.of("openvino"))));

      final String message = failure.getMessage();
      assertTrue(message.contains("openvino"), message);
      assertTrue(message.contains(ExecutionProviders.CPU), message);
      assertTrue(message.contains(ExecutionProviders.CUDA), message);
      assertTrue(message.contains(EchoExecutionProviderConfigurer.ID), message);
      assertTrue(message.contains(ExecutionProviderConfigurer.class.getName()), message);
      assertEquals(List.of(), recorded.calls(), "nothing may be appended before the failure");
    }
  }

  /** Ids are case-sensitive, so the upper case spelling of a built-in id is not a built-in id. */
  @Test
  void testIdResolutionIsCaseSensitive() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      final IllegalArgumentException failure =
          assertThrows(IllegalArgumentException.class, () -> ExecutionProviders.addTo(recorded,
              List.of(ExecutionProviderRequest.of("CUDA"))));

      assertTrue(failure.getMessage().contains("CUDA"), failure.getMessage());
    }
  }

  /** A failure part way through leaves the earlier requests appended and does not continue. */
  @Test
  void testAFailureStopsAtTheRequestThatFailed() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      assertThrows(IllegalArgumentException.class, () -> ExecutionProviders.addTo(recorded,
          List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU),
              ExecutionProviderRequest.of("nothing-answers-to-this"),
              ExecutionProviderRequest.of(ExecutionProviders.CUDA))));

      assertEquals(List.of("addCPU(true)"), recorded.calls());
    }
  }

  // The built-in CPU configurer.

  @Test
  void testCpuKeepsTheArenaByDefault() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded,
          List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU)));

      assertEquals(List.of("addCPU(true)"), recorded.calls());
    }
  }

  @Test
  void testCpuArenaCanBeTurnedOff() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded, List.of(
          ExecutionProviderRequest.of(ExecutionProviders.CPU, Map.of(USE_ARENA, "false"))));

      assertEquals(List.of("addCPU(false)"), recorded.calls());
    }
  }

  /**
   * The CPU option map is read here rather than by ONNX Runtime, so an unknown name is reported
   * with the name that is accepted instead of being dropped.
   */
  @Test
  void testCpuRejectsAnUnknownOptionName() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
          () -> ExecutionProviders.addTo(recorded, List.of(ExecutionProviderRequest
              .of(ExecutionProviders.CPU, Map.of("arena_extend_strategy", "kNextPowerOfTwo")))));

      final String message = failure.getMessage();
      assertTrue(message.contains("arena_extend_strategy"), message);
      assertTrue(message.contains(USE_ARENA), message);
      assertEquals(List.of(), recorded.calls());
    }
  }

  @Test
  void testCpuRejectsANonBooleanArenaValue() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
          () -> ExecutionProviders.addTo(recorded, List.of(
              ExecutionProviderRequest.of(ExecutionProviders.CPU, Map.of(USE_ARENA, "yes")))));

      assertTrue(failure.getMessage().contains(USE_ARENA), failure.getMessage());
    }
  }

  // The built-in CUDA configurer.

  @Test
  void testCudaWithoutOptionsUsesDeviceZero() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded,
          List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA)));

      assertEquals(List.of("addCUDA(0)"), recorded.calls());
    }
  }

  @Test
  void testCudaDeviceIdReachesTheCall() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded, List.of(
          ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "2"))));

      assertEquals(List.of("addCUDA(2)"), recorded.calls());
    }
  }

  @Test
  void testCudaRejectsAMalformedDeviceId() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      for (final String value : List.of("-1", "one", "", " 1", "1.0", "2147483648")) {
        final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
            () -> ExecutionProviders.addTo(recorded, List.of(ExecutionProviderRequest
                .of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, value)))),
            "'" + value + "' must not be accepted as a device id");
        assertTrue(failure.getMessage().contains(DEVICE_ID), failure.getMessage());
      }
      assertEquals(List.of(), recorded.calls());
    }
  }

  /**
   * A CUDA request that carries more than the device id goes through ONNX Runtime's own CUDA
   * provider options, which is also what decides whether an option name is known. Building those
   * needs the CUDA shared provider library, so on the CPU-only {@code onnxruntime} artifact this
   * fails loudly instead; either way the option map reached the runtime rather than being dropped,
   * which the plain {@code addCUDA(deviceId)} call of the other cases could not show.
   */
  @Test
  void testCudaPassesFurtherOptionsToTheRuntime() throws Exception {
    final Map<String, String> options = new LinkedHashMap<>();
    options.put(DEVICE_ID, "0");
    options.put("cudnn_conv_algo_search", "HEURISTIC");
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      try {
        ExecutionProviders.addTo(recorded,
            List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA, options)));
      } catch (final OrtException expected) {
        final String message = expected.getMessage();
        assertTrue(message.contains("CUDA") || message.contains("cuda"), message);
        return;
      }
      assertEquals(1, recorded.calls().size(), recorded.calls().toString());
      final String call = recorded.calls().get(0);
      assertTrue(call.startsWith("addCUDA("), call);
      assertTrue(call.contains("cudnn_conv_algo_search"), call);
    }
  }

  // The SPI.

  /**
   * A configurer registered in {@code META-INF/services} is resolved by its id, which is what an
   * addon does. {@link EchoExecutionProviderConfigurer} is registered in this module's test
   * resources and nothing in {@code opennlp-dl} knows its id, so resolving it proves the path an
   * addon would take. Its provider options are observed where they landed, in the session config
   * entries, which is the one part of the session options ONNX Runtime reads back.
   */
  @Test
  void testSpiResolvesARegisteredConfigurerByIdAndPassesItsOptions() throws Exception {
    final Map<String, String> options = new LinkedHashMap<>();
    options.put("device_type", "GPU.0");
    options.put("num_of_threads", "4");
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded,
          List.of(ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID, options)));

      assertEquals(Map.of(EchoExecutionProviderConfigurer.PREFIX + "ran",
              EchoExecutionProviderConfigurer.ID,
          EchoExecutionProviderConfigurer.PREFIX + "device_type", "GPU.0",
          EchoExecutionProviderConfigurer.PREFIX + "num_of_threads", "4"),
          recorded.getConfigEntries());
    }
  }

  /**
   * A configurer that passes its options on, as an accelerator's does, accepts a name it has never
   * heard of: the runtime owns that namespace, and a list kept here would go stale with every
   * runtime release.
   */
  @Test
  void testAConfigurerMayAcceptAnUnknownOptionName() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded, List.of(ExecutionProviderRequest
          .of(EchoExecutionProviderConfigurer.ID, Map.of("not_a_real_option", "value"))));

      assertEquals("value", recorded.getConfigEntries()
          .get(EchoExecutionProviderConfigurer.PREFIX + "not_a_real_option"));
    }
  }

  // The rendering used in the log entry.

  @Test
  void testDescribeStatesThatOnnxRuntimeDecidesWhenNothingIsRequested() {
    final String described = ExecutionProviders.describe(List.of());

    assertTrue(described.contains("CPU"), described);
    assertFalse(described.isBlank());
  }

  @Test
  void testDescribeNamesTheIdsInOrder() {
    final String described = ExecutionProviders.describe(List.of(
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1")),
        ExecutionProviderRequest.of(ExecutionProviders.CPU)));

    assertEquals("cuda[device_id], cpu[]", described);
  }

  @Test
  void testDescribeRejectsNull() {
    assertThrows(IllegalArgumentException.class, () -> ExecutionProviders.describe(null));
  }
}
