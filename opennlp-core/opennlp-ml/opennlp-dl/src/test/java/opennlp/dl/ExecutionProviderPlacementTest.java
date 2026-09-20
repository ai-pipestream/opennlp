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

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import opennlp.dl.vectors.PaddingStrategy;
import opennlp.tools.util.ext.ProviderSpec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ExecutionProviderConfigurer#placement(ProviderSpec)}, the seam through which an execution
 * provider outside {@code opennlp-dl} states where its session runs, and
 * {@link ExecutionProviders#runsOnAccelerator(List)}, which reads it.
 *
 * <p>The question exists because the placement of several execution providers is not in the id.
 * ONNX Runtime registers OpenVINO with a device type, so {@code GPU.0} and {@code CPU} are one id on
 * two placements, and a list of accelerator ids in {@code opennlp-dl} would be right half the time.
 * The configurer answers instead, for the request it was handed.</p>
 *
 * <p>Two registered configurers stand in for addons here, both listed in
 * {@code META-INF/services/opennlp.dl.ExecutionProviderConfigurer} of this module's test resources:
 * {@link EchoExecutionProviderConfigurer}, which overrides nothing and so states nothing, and
 * {@link PlacementEchoExecutionProviderConfigurer}, which states whichever placement it is asked
 * for.</p>
 */
class ExecutionProviderPlacementTest {

  /** A configurer that overrides nothing beyond what the SPI requires, as one written before this. */
  private static final class SilentConfigurer implements ExecutionProviderConfigurer {

    @Override
    public String name() {
      return "opennlp-test-silent";
    }

    @Override
    public boolean supports(final ProviderSpec spec) {
      return true;
    }

    @Override
    public ExecutionProvider create(final ProviderSpec spec) {
      return sessionOptions -> {
      };
    }
  }

  private static ExecutionProviderRequest placing(final String placement) {
    return ExecutionProviderRequest.of(PlacementEchoExecutionProviderConfigurer.ID,
        Map.of(PlacementEchoExecutionProviderConfigurer.PLACEMENT_OPTION, placement));
  }

  // The default.

  /**
   * A configurer that says nothing about the placement states
   * {@link ExecutionProviderPlacement#UNSPECIFIED}, so the SPI addition forces no addon author to
   * answer and compiles against every implementation written before it.
   */
  @Test
  void testAConfigurerStatesNoPlacementByDefault() {
    assertEquals(ExecutionProviderPlacement.UNSPECIFIED,
        new SilentConfigurer().placement(ProviderSpec.of(Map.of())));
    assertEquals(ExecutionProviderPlacement.UNSPECIFIED,
        new EchoExecutionProviderConfigurer().placement(ProviderSpec.of(Map.of())));
  }

  /** The default implementation rejects a null spec, as every other method of the SPI does. */
  @Test
  void testTheDefaultRejectsANullSpec() {
    assertThrows(IllegalArgumentException.class, () -> new SilentConfigurer().placement(null));
    assertThrows(IllegalArgumentException.class,
        () -> new EchoExecutionProviderConfigurer().placement(null));
  }

  // The two built-in configurers.

  /** The CUDA configurer states an accelerator, whichever device its options name. */
  @Test
  void testTheCudaConfigurerStatesAnAccelerator() {
    final CudaExecutionProviderConfigurer cuda = new CudaExecutionProviderConfigurer();

    assertEquals(ExecutionProviderPlacement.ACCELERATOR, cuda.placement(ProviderSpec.of(Map.of())));
    assertEquals(ExecutionProviderPlacement.ACCELERATOR, cuda.placement(ProviderSpec.of(
        Map.of(CudaExecutionProviderConfigurer.DEVICE_ID_OPTION, "1"))));
    assertThrows(IllegalArgumentException.class, () -> cuda.placement(null));
  }

  /** The CPU configurer states the CPU, and its arena option does not change that. */
  @Test
  void testTheCpuConfigurerStatesTheCpu() {
    final CpuExecutionProviderConfigurer cpu = new CpuExecutionProviderConfigurer();

    assertEquals(ExecutionProviderPlacement.CPU, cpu.placement(ProviderSpec.of(Map.of())));
    assertEquals(ExecutionProviderPlacement.CPU, cpu.placement(ProviderSpec.of(
        Map.of(CpuExecutionProviderConfigurer.USE_ARENA_OPTION, "false"))));
    assertThrows(IllegalArgumentException.class, () -> cpu.placement(null));
  }

  // What runsOnAccelerator makes of it.

  /** The built-in ids answer as they did before the SPI could state a placement. */
  @Test
  void testTheBuiltInIdsAreClassifiedAsBefore() {
    assertFalse(ExecutionProviders.runsOnAccelerator(List.of()));
    assertFalse(ExecutionProviders.runsOnAccelerator(
        List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU))));
    assertTrue(ExecutionProviders.runsOnAccelerator(
        List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA))));
  }

  /**
   * An id nothing in {@code opennlp-dl} knows is classified as an accelerator because its own
   * configurer states so. This is the whole point of the addition: the accelerator group of
   * {@link ExecutionProviders} holds built-in ids only, and an addon needs no entry in it.
   */
  @Test
  void testARegisteredConfigurerClassifiesItsOwnId() {
    assertTrue(ExecutionProviders.runsOnAccelerator(
        List.of(ExecutionProviderRequest.of(PlacementEchoExecutionProviderConfigurer.ID))));
    assertTrue(ExecutionProviders.runsOnAccelerator(
        List.of(placing(ExecutionProviderPlacement.ACCELERATOR.name()))));
    assertEquals(PaddingStrategy.LONGEST, PaddingStrategy.defaultFor(
        List.of(placing(ExecutionProviderPlacement.ACCELERATOR.name()))));
  }

  /**
   * The same id states the CPU for another set of provider options, which is the case an id alone
   * cannot express and the reason the spec is passed to the configurer.
   */
  @Test
  void testTheSameIdCanStateEitherPlacement() {
    assertFalse(ExecutionProviders
        .runsOnAccelerator(List.of(placing(ExecutionProviderPlacement.CPU.name()))));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(
        List.of(placing(ExecutionProviderPlacement.CPU.name()))));
  }

  /**
   * A registered configurer that states nothing leaves the answer where it was: the id it answers to
   * is in no accelerator group, so the conservative default holds. This is what keeps a configurer
   * written before the SPI addition working unchanged.
   */
  @Test
  void testAConfigurerThatStatesNothingKeepsTheConservativeAnswer() {
    assertFalse(ExecutionProviders.runsOnAccelerator(
        List.of(ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID))));
    assertFalse(ExecutionProviders
        .runsOnAccelerator(List.of(placing(ExecutionProviderPlacement.UNSPECIFIED.name()))));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(
        List.of(ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID))));
  }

  /** An id no configurer answers to is not classified either, and reaching it does not fail. */
  @Test
  void testAnIdNoConfigurerAnswersToIsNotClassified() {
    assertFalse(ExecutionProviders
        .runsOnAccelerator(List.of(ExecutionProviderRequest.of("nothing-answers-to-this"))));
  }

  /**
   * A configurer that breaks the contract of the method, by returning {@code null} or by throwing,
   * leaves the placement unstated rather than failing the call. The caller is deriving a default, and
   * the request is registered a moment later, which is where a broken configurer does fail a
   * construction.
   */
  @Test
  void testABrokenConfigurerLeavesThePlacementUnstated() {
    assertFalse(ExecutionProviders
        .runsOnAccelerator(List.of(placing(PlacementEchoExecutionProviderConfigurer.NULL_PLACEMENT))));
    assertFalse(ExecutionProviders.runsOnAccelerator(
        List.of(placing(PlacementEchoExecutionProviderConfigurer.FAILING_PLACEMENT))));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(
        List.of(placing(PlacementEchoExecutionProviderConfigurer.FAILING_PLACEMENT))));
  }

  /**
   * Only the first request is asked, since ONNX Runtime offers each node to the execution providers
   * in the order they were appended and the CPU execution provider accepts every node.
   */
  @Test
  void testOnlyTheFirstRequestIsAsked() {
    final ExecutionProviderRequest cpu = ExecutionProviderRequest.of(ExecutionProviders.CPU);

    assertTrue(ExecutionProviders.runsOnAccelerator(
        List.of(placing(ExecutionProviderPlacement.ACCELERATOR.name()), cpu)));
    assertFalse(ExecutionProviders.runsOnAccelerator(
        List.of(cpu, placing(ExecutionProviderPlacement.ACCELERATOR.name()))));
  }
}
