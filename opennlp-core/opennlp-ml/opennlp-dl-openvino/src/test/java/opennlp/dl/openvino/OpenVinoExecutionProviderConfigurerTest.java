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

package opennlp.dl.openvino;

import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.dl.EchoExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderPlacement;
import opennlp.dl.ExecutionProviderRequest;
import opennlp.dl.ExecutionProviders;
import opennlp.dl.RecordingSessionOptions;
import opennlp.dl.vectors.PaddingStrategy;
import opennlp.tools.util.ext.ProviderSpec;
import opennlp.tools.util.ext.Providers;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The OpenVINO execution provider addon: that {@code opennlp-dl} resolves it through the SPI although
 * nothing in {@code opennlp-dl} knows its id, that the device type of a request reaches
 * {@link ai.onnxruntime.OrtSession.SessionOptions#addOpenVINO(String)}, and that the device type is
 * what decides the padding default of a batching encoder.
 *
 * <p>The registration is observed and not performed. The ONNX Runtime this build pins was not built
 * with the OpenVINO execution provider, and no published Maven artifact carries one that was, so
 * {@code addOpenVINO} on a real set of session options fails with
 * {@code ORT_EP_FAIL: Failed to find OpenVINO shared provider} whatever the device type is.
 * {@link RecordingSessionOptions}, which {@code opennlp-dl}
 * uses for the same reason, records the call instead of making it, so what these tests establish is
 * that the configuration is the one an OpenVINO runtime would be handed, not that OpenVINO ran
 * anything.</p>
 */
class OpenVinoExecutionProviderConfigurerTest {

  private static final String DEVICE_TYPE = OpenVinoExecutionProviderConfigurer.DEVICE_TYPE_OPTION;

  private static ExecutionProviderRequest on(final String deviceType) {
    return ExecutionProviderRequest.of(OpenVinoExecutionProviderConfigurer.ID,
        Map.of(DEVICE_TYPE, deviceType));
  }

  private static ProviderSpec spec(final String deviceType) {
    return ProviderSpec.of(Map.of(DEVICE_TYPE, deviceType));
  }

  private static ExecutionProviderPlacement placementOn(final String deviceType) {
    return new OpenVinoExecutionProviderConfigurer().placement(spec(deviceType));
  }

  // The SPI, which is what this module exists to prove.

  /**
   * {@code opennlp-dl} resolves this configurer by the id it registered itself under, out of the
   * service file of this jar. No class of {@code opennlp-dl} holds the id, and none had to be edited
   * to make it usable.
   */
  @Test
  void testTheAddonIsResolvedThroughTheSpi() {
    final Optional<ExecutionProviderConfigurer> resolved =
        Providers.of(ExecutionProviderConfigurer.class)
            .byName(OpenVinoExecutionProviderConfigurer.ID);

    assertTrue(resolved.isPresent(), "the openvino id did not resolve through the SPI");
    assertInstanceOf(OpenVinoExecutionProviderConfigurer.class, resolved.get());
    assertTrue(resolved.get().isAvailable());
  }

  /**
   * A request for the id is registered through the resolved configurer, by the same
   * {@link ExecutionProviders#addTo} call every component makes. In {@code opennlp-dl}, where this
   * jar is absent, that same request fails naming the id, which is the test
   * {@code ExecutionProvidersTest.testUnknownIdFailsNamingTheIdAndWhatIsRegistered} makes.
   */
  @Test
  void testAnAddonRequestIsRegisteredThroughTheResolvedConfigurer() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded, List.of(on("GPU.0")));

      assertEquals(List.of("addOpenVINO(GPU.0)"), recorded.calls());
    }
  }

  /** The requests of a list are appended in order, so OpenVINO can be asked for with a CPU behind it. */
  @Test
  void testTheAddonTakesItsPlaceInAnOrderedList() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded,
          List.of(on("GPU.0"), ExecutionProviderRequest.of(ExecutionProviders.CPU)));

      assertEquals(List.of("addOpenVINO(GPU.0)", "addCPU(true)"), recorded.calls());
    }
  }

  // The device type on its way to ONNX Runtime.

  /**
   * The device type of the request is what reaches {@code addOpenVINO}, spelled as it was written:
   * ONNX Runtime and OpenVINO own that namespace, so the value is passed on rather than normalized.
   */
  @ParameterizedTest
  @ValueSource(strings = {"CPU", "GPU", "GPU.0", "GPU.1", "NPU", "AUTO", "MULTI:GPU,CPU",
      "HETERO:GPU,CPU", "AUTO:GPU,CPU", "gpu.0", "FPGA"})
  void testTheDeviceTypeReachesTheCall(final String deviceType) throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded, List.of(on(deviceType)));

      assertEquals(List.of("addOpenVINO(" + deviceType + ")"), recorded.calls());
    }
  }

  /**
   * A request without a device type registers the empty one, which is how ONNX Runtime is told to
   * leave the choice of device to OpenVINO. An explicitly empty value says the same thing.
   */
  @Test
  void testAMissingDeviceTypeRegistersTheEmptyOne() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      ExecutionProviders.addTo(recorded,
          List.of(ExecutionProviderRequest.of(OpenVinoExecutionProviderConfigurer.ID)));
      ExecutionProviders.addTo(recorded, List.of(on("")));

      assertEquals(List.of("addOpenVINO()", "addOpenVINO()"), recorded.calls());
    }
  }

  /**
   * A blank device type is a configuration mistake rather than a request for the default, and is
   * rejected where it is written instead of being passed to a runtime that would reject it later.
   */
  @Test
  void testABlankDeviceTypeIsRejected() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
          () -> ExecutionProviders.addTo(recorded, List.of(on("  "))));

      assertTrue(failure.getMessage().contains(DEVICE_TYPE), failure.getMessage());
      assertEquals(List.of(), recorded.calls(), "nothing may be appended before the failure");
    }
  }

  /** The device type is the only option, so a name ONNX Runtime would ignore is reported here. */
  @Test
  void testAnUnknownOptionNameIsRejected() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
          () -> ExecutionProviders.addTo(recorded, List.of(ExecutionProviderRequest
              .of(OpenVinoExecutionProviderConfigurer.ID, Map.of("num_of_threads", "4")))));

      assertTrue(failure.getMessage().contains("num_of_threads"), failure.getMessage());
      assertEquals(List.of(), recorded.calls());
    }
  }

  /** This execution provider has no location to read, so a spec that carries one is not supported. */
  @Test
  void testASpecWithALocationIsRejected() {
    final OpenVinoExecutionProviderConfigurer configurer =
        new OpenVinoExecutionProviderConfigurer();
    final ProviderSpec located = ProviderSpec.of(URI.create("file:/model.onnx"), Map.of());

    assertFalse(configurer.supports(located));
    assertThrows(IllegalArgumentException.class, () -> configurer.create(located));
    assertTrue(configurer.supports(spec("GPU.0")));
    assertFalse(configurer.supports(ProviderSpec.of(Map.of("num_of_threads", "4"))));
    assertThrows(IllegalArgumentException.class, () -> configurer.supports(null));
    assertThrows(IllegalArgumentException.class, () -> configurer.create(null));
  }

  // The placement, which is the decision this stage exists to settle.

  /** A GPU or NPU device type is an accelerator, with or without the index of one device. */
  @ParameterizedTest
  @ValueSource(strings = {"GPU", "GPU.0", "GPU.1", "NPU", "NPU.0", "gpu", "gpu.0", "Npu"})
  void testAGpuOrNpuDeviceTypeIsAnAccelerator(final String deviceType) {
    assertEquals(ExecutionProviderPlacement.ACCELERATOR, placementOn(deviceType));
  }

  /** The CPU device type is the host CPU, which is the half of this id an id-based list gets wrong. */
  @ParameterizedTest
  @ValueSource(strings = {"CPU", "cpu", "Cpu"})
  void testTheCpuDeviceTypeIsTheHostCpu(final String deviceType) {
    assertEquals(ExecutionProviderPlacement.CPU, placementOn(deviceType));
  }

  /**
   * A composite device type states nothing: OpenVINO picks the device while the session is created,
   * and it could be either kind. An unrecognized device type and a blank one state nothing either.
   */
  @ParameterizedTest
  @ValueSource(strings = {"AUTO", "AUTO:GPU,CPU", "MULTI:GPU,CPU", "HETERO:GPU,CPU", "FPGA",
      "VPUX", "  ", "GPUX", "gpu0"})
  void testAnUndecidedDeviceTypeStatesNoPlacement(final String deviceType) {
    assertEquals(ExecutionProviderPlacement.UNSPECIFIED, placementOn(deviceType));
  }

  /** A request without a device type states nothing, since the device is then OpenVINO's choice. */
  @Test
  void testAMissingDeviceTypeStatesNoPlacement() {
    assertEquals(ExecutionProviderPlacement.UNSPECIFIED,
        new OpenVinoExecutionProviderConfigurer().placement(ProviderSpec.of(Map.of())));
  }

  /** A null spec is rejected, as it is by every other method of the SPI. */
  @Test
  void testPlacementRejectsANullSpec() {
    assertThrows(IllegalArgumentException.class,
        () -> new OpenVinoExecutionProviderConfigurer().placement(null));
  }

  /**
   * The placement is answerable for a spec {@link OpenVinoExecutionProviderConfigurer#supports} says
   * no to, which its contract requires: a caller reading a placement is deciding a default, not
   * validating a configuration.
   */
  @Test
  void testThePlacementIsAnswerableForAnUnsupportedSpec() {
    final OpenVinoExecutionProviderConfigurer configurer =
        new OpenVinoExecutionProviderConfigurer();
    final ProviderSpec unsupported = ProviderSpec.of(URI.create("file:/model.onnx"),
        Map.of(DEVICE_TYPE, "GPU.0", "num_of_threads", "4"));

    assertFalse(configurer.supports(unsupported));
    assertEquals(ExecutionProviderPlacement.ACCELERATOR, configurer.placement(unsupported));
  }

  // The padding default the placement drives, with no id known to opennlp-dl.

  /** A GPU device type derives padding to the longest row of a batch, the strategy a GPU wants. */
  @Test
  void testAGpuDeviceTypeDerivesPaddingToTheLongestRow() {
    assertTrue(ExecutionProviders.runsOnAccelerator(List.of(on("GPU.0"))));
    assertEquals(PaddingStrategy.LONGEST, PaddingStrategy.defaultFor(List.of(on("GPU.0"))));
    assertEquals(PaddingStrategy.LONGEST, PaddingStrategy.defaultFor(List.of(on("NPU"))));
    assertEquals(PaddingStrategy.LONGEST,
        PaddingStrategy.defaultFor(List.of(on("GPU"), ExecutionProviderRequest.of("cpu"))));
  }

  /**
   * A CPU device type derives grouping by exact tokenized length, from the same execution provider
   * id. This is the pair a list of accelerator ids in {@code opennlp-dl} could not tell apart.
   */
  @Test
  void testACpuDeviceTypeDerivesExactLengthGrouping() {
    assertFalse(ExecutionProviders.runsOnAccelerator(List.of(on("CPU"))));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(List.of(on("CPU"))));
  }

  /**
   * Where the device type states no placement, the conservative default holds, so putting this jar on
   * the class path changes the vectors of a session that does not name a device.
   */
  @Test
  void testAnUndecidedDeviceTypeKeepsTheConservativeDefault() {
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(
        List.of(ExecutionProviderRequest.of(OpenVinoExecutionProviderConfigurer.ID))));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(List.of(on("AUTO"))));
    assertEquals(PaddingStrategy.EXACT_LENGTH,
        PaddingStrategy.defaultFor(List.of(on("MULTI:GPU,CPU"))));
  }

  /**
   * The built-in ids and a configurer that states no placement answer exactly as they do without
   * this jar, so the addon adds an id and changes nothing else.
   */
  @Test
  void testTheBuiltInIdsAndASilentConfigurerAreUnaffected() {
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(List.of()));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(
        List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU))));
    assertEquals(PaddingStrategy.LONGEST, PaddingStrategy.defaultFor(
        List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA))));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(
        List.of(ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID))));
  }

  /** Ids are case-sensitive, so the upper case spelling of this one is not this one. */
  @Test
  void testTheIdIsCaseSensitive() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      assertThrows(IllegalArgumentException.class, () -> ExecutionProviders.addTo(recorded,
          List.of(ExecutionProviderRequest.of("OpenVINO"))));
    }
  }
}
