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

import ai.onnxruntime.OrtEnvironment;

import opennlp.dl.ExecutionProvider;
import opennlp.dl.ExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderPlacement;
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.ext.ProviderSpec;

/**
 * Registers the ONNX Runtime OpenVINO execution provider under the id {@value #ID}, on the device
 * {@value #DEVICE_TYPE_OPTION} names.
 *
 * <p>This module adds no class to {@code opennlp-dl} and changes none. It is a jar that lists this
 * class in {@code META-INF/services/opennlp.dl.ExecutionProviderConfigurer}, and putting it on the
 * class path is what makes {@code openvino} a usable execution provider id. Nothing in
 * {@code opennlp-dl} holds that id.</p>
 *
 * <h2>The device type, and why the placement is not in the id</h2>
 *
 * <p>ONNX Runtime registers this execution provider through
 * {@link ai.onnxruntime.OrtSession.SessionOptions#addOpenVINO(String)}, whose one argument is an
 * OpenVINO device type: {@code CPU}, {@code GPU}, {@code GPU.0}, {@code GPU.1}, {@code NPU}, and the
 * composite {@code AUTO}, {@code MULTI:GPU,CPU} and {@code HETERO:GPU,CPU}. So one execution provider
 * id covers a session on an accelerator and a session on the host CPU, and the id alone cannot say
 * which. {@link #placement(ProviderSpec)} reads the device type and answers for the request in hand:
 * {@link ExecutionProviderPlacement#ACCELERATOR} for a {@code GPU} or {@code NPU} device,
 * {@link ExecutionProviderPlacement#CPU} for {@code CPU}, and
 * {@link ExecutionProviderPlacement#UNSPECIFIED} for a composite device type, an absent one and one
 * this class does not recognize, since which device those run on is decided while the session is
 * created and finding it out would mean loading the runtime.</p>
 *
 * <p>That is what a batching encoder reads to pick its padding strategy, so
 * {@code device_type=GPU.0} derives padding to the longest row of a batch and
 * {@code device_type=CPU} derives grouping by exact tokenized length, from the same id.</p>
 *
 * <p>The device type is handed to ONNX Runtime as it stands, which is also what decides whether it is
 * a device type at all: OpenVINO's list of them grows with its releases and with what the machine
 * has, and a list kept here would go stale. Recognizing a device type for the placement and accepting
 * one for registration are therefore separate: an unrecognized device type is registered and states
 * no placement.</p>
 *
 * <h2>What running on it needs</h2>
 *
 * <p>{@code addOpenVINO} is part of the base {@code onnxruntime} API, so this module needs no further
 * artifact to compile or to register a request. Running a session on OpenVINO needs an ONNX Runtime
 * built with the OpenVINO execution provider, which no published Maven artifact carries: the
 * {@code onnxruntime} and {@code onnxruntime_gpu} jars both hold a runtime whose
 * {@code addOpenVINO} fails with {@code ORT_EP_FAIL: Failed to find OpenVINO shared provider}, because
 * {@code libonnxruntime_providers_openvino.so} is not in them. A deployment that wants this supplies
 * a runtime built with {@code --use_openvino} and the OpenVINO toolkit the build was made against.
 * The README of this module has the detail.</p>
 *
 * <p>As with every execution provider, a request that cannot be registered fails and is reported.
 * ONNX Runtime does not fall back to the CPU, and neither does this: a session that silently ran
 * somewhere else than the caller asked for is what this SPI exists to prevent.</p>
 *
 * @since 3.0.0
 */
public final class OpenVinoExecutionProviderConfigurer implements ExecutionProviderConfigurer {

  /** The id this configurer answers to. */
  public static final String ID = "openvino";

  /**
   * The option that names the OpenVINO device type to run on, for example {@code CPU},
   * {@code GPU.0}, {@code NPU} or {@code AUTO}. It carries ONNX Runtime's own name for it. Where it
   * is absent, the empty device type is registered, which leaves the choice of device to OpenVINO.
   */
  public static final String DEVICE_TYPE_OPTION = "device_type";

  /** The device type registered where {@value #DEVICE_TYPE_OPTION} is absent. */
  private static final String ANY_DEVICE = "";

  private static final String CPU_DEVICE = "cpu";
  private static final String GPU_DEVICE = "gpu";
  private static final String NPU_DEVICE = "npu";

  /** What separates a device type from the index of one device of that type, as in {@code GPU.1}. */
  private static final char DEVICE_INDEX_SEPARATOR = '.';

  /** {@inheritDoc} */
  @Override
  public String name() {
    return ID;
  }

  /**
   * {@inheritDoc}
   * Checks that the ONNX Runtime classes are present without initializing the runtime. This says
   * nothing about whether the runtime on the class path was built with the OpenVINO execution
   * provider, which cannot be answered without loading the shared provider library, so it is
   * answered by the attempt to register.
   */
  @Override
  public boolean isAvailable() {
    try {
      return OrtEnvironment.class.getName() != null;
    } catch (final LinkageError e) {
      return false;
    }
  }

  /** {@inheritDoc} */
  @Override
  public boolean supports(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    return spec.location().isEmpty() && spec.hasOnlyOptions(DEVICE_TYPE_OPTION);
  }

  /**
   * {@inheritDoc}
   * The device type of the request, which is the only thing
   * {@link ai.onnxruntime.OrtSession.SessionOptions#addOpenVINO(String)} takes. A
   * {@code GPU} or {@code NPU} device is an accelerator, {@code CPU} is the host CPU, and an absent,
   * composite or unrecognized device type is left unstated rather than guessed at, which keeps the
   * conservative default in force.
   */
  @Override
  public ExecutionProviderPlacement placement(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    return placementOfDevice(spec.option(DEVICE_TYPE_OPTION, ANY_DEVICE));
  }

  /**
   * {@inheritDoc}
   * The returned {@link ExecutionProvider} appends the OpenVINO execution provider on the requested
   * device type.
   *
   * @throws IllegalArgumentException Thrown if {@code spec} is {@code null}, carries a location,
   *     holds an option other than {@value #DEVICE_TYPE_OPTION}, or gives that option a blank
   *     value.
   */
  @Override
  public ExecutionProvider create(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    if (spec.location().isPresent()) {
      throw new IllegalArgumentException(
          "the " + ID + " execution provider takes no location: " + spec);
    }
    for (final String option : spec.options().keySet()) {
      if (!DEVICE_TYPE_OPTION.equals(option)) {
        throw new IllegalArgumentException("the " + ID + " execution provider accepts only the "
            + "option '" + DEVICE_TYPE_OPTION + "', not '" + option + "'");
      }
    }
    final String deviceType = spec.option(DEVICE_TYPE_OPTION, ANY_DEVICE);
    if (!deviceType.isEmpty() && deviceType.isBlank()) {
      throw new IllegalArgumentException("the " + ID + " execution provider option '"
          + DEVICE_TYPE_OPTION + "' must name a device type, such as CPU, GPU.0 or NPU, and must "
          + "not be blank; leave it out to let OpenVINO choose the device");
    }
    return sessionOptions -> {
      if (sessionOptions == null) {
        throw new IllegalArgumentException("sessionOptions must not be null");
      }
      sessionOptions.addOpenVINO(deviceType);
    };
  }

  /**
   * Classifies one OpenVINO device type.
   *
   * <p>The comparison is over the characters of the value and is case-insensitive, so that a
   * configuration written {@code gpu.0} is classified like {@code GPU.0}. It is only the
   * classification that is forgiving: the value itself reaches ONNX Runtime as it was written, and
   * whether ONNX Runtime accepts that spelling is the runtime's business.</p>
   *
   * @param deviceType The device type of the request, empty where the option is absent. Must not be
   *     {@code null}.
   * @return The placement it names, {@link ExecutionProviderPlacement#UNSPECIFIED} for an empty, a
   *     composite or an unrecognized one. Never {@code null}.
   */
  private static ExecutionProviderPlacement placementOfDevice(final String deviceType) {
    final String device = StringUtil.toLowerCase(deviceType);
    if (names(device, CPU_DEVICE)) {
      return ExecutionProviderPlacement.CPU;
    }
    if (names(device, GPU_DEVICE) || names(device, NPU_DEVICE)) {
      return ExecutionProviderPlacement.ACCELERATOR;
    }
    // Empty, so OpenVINO chooses; AUTO, MULTI or HETERO, so OpenVINO chooses among the devices the
    // value lists; or a device type a later OpenVINO release added. All three are the device's answer
    // to give, and asking it would mean loading the runtime.
    return ExecutionProviderPlacement.UNSPECIFIED;
  }

  /**
   * {@return whether a lower-cased device type names one device of a kind: the kind itself, as in
   * {@code gpu}, or one device of it, as in {@code gpu.1}}
   *
   * @param device The lower-cased device type of the request. Must not be {@code null}.
   * @param kind The lower-cased device kind to check for. Must not be {@code null} or empty.
   */
  private static boolean names(final String device, final String kind) {
    if (device.equals(kind)) {
      return true;
    }
    if (device.length() <= kind.length() || !device.startsWith(kind)) {
      return false;
    }
    return device.charAt(kind.length()) == DEVICE_INDEX_SEPARATOR;
  }
}
