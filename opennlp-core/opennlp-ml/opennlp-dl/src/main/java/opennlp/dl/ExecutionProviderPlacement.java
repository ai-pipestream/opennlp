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

import opennlp.tools.util.ext.ProviderSpec;

/**
 * Where an execution provider puts the arithmetic of a session: on an accelerator, on the host CPU,
 * or somewhere only the machine it runs on can say.
 *
 * <p>This is the one property of an execution provider that a setting outside this package needs to
 * read, and the padding strategy of a batching encoder is the first such setting: padding a batch to
 * its longest row is six to eight times faster on an accelerator and 0.66 to 0.81 times as fast on a
 * CPU, so the better default is the opposite one on the two placements. An id alone cannot answer
 * the question. {@code openvino} is the case that shows it: ONNX Runtime registers that execution
 * provider through
 * {@link ai.onnxruntime.OrtSession.SessionOptions#addOpenVINO(String)}, whose argument is a device
 * type, and {@code GPU.0} and {@code CPU} are the same execution provider on two placements.</p>
 *
 * <p>An {@link ExecutionProviderConfigurer} therefore answers for the request it was handed, through
 * {@link ExecutionProviderConfigurer#placement(ProviderSpec)}, and {@link #UNSPECIFIED} is the
 * answer of every configurer that says nothing. That keeps the question out of the id: nothing in
 * {@code opennlp-dl} holds a list of addon ids, and an addon classifies itself from its own provider
 * options.</p>
 *
 * @see ExecutionProviderConfigurer#placement(ProviderSpec)
 * @see ExecutionProviders#runsOnAccelerator(List)
 * @since 3.0.0
 */
public enum ExecutionProviderPlacement {

  /**
   * The session runs on an accelerator: a device with its own memory and enough parallelism that one
   * wide tensor beats a stream of narrow ones. CUDA, ROCm, TensorRT, DirectML and OpenVINO on a
   * {@code GPU} or {@code NPU} device are this.
   */
  ACCELERATOR,

  /**
   * The session runs on the host CPU. The ONNX Runtime CPU execution provider is this, and so is an
   * accelerator execution provider that was pointed at the CPU, which OpenVINO with
   * {@code device_type=CPU} is.
   */
  CPU,

  /**
   * The placement is not stated. This is the default of {@link ExecutionProviderConfigurer}, so an
   * addon author is never forced to answer, and it is also the honest answer where the answer
   * depends on the machine: OpenVINO's {@code AUTO}, {@code MULTI} and {@code HETERO} device types
   * pick a device while the session is being created, and finding out which one would mean loading
   * the runtime, which the {@link opennlp.tools.util.ext.Providers} contract forbids while providers
   * are being looked up.
   *
   * <p>A caller reading a placement treats this as "decide some other way" rather than as the CPU.
   * {@link ExecutionProviders#runsOnAccelerator(List)} falls back to the ids it knows itself, and
   * ends at the conservative answer for an id it does not know.</p>
   */
  UNSPECIFIED
}
